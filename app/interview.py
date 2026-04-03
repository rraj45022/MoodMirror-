from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import subprocess
import sys
import time
import wave

import numpy as np
import requests
from PySide6.QtCore import QIODevice, QObject, QTimer, Signal
from PySide6.QtMultimedia import QAudioFormat, QAudioSource, QMediaDevices


DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_GROQ_TRANSCRIPTION_MODEL = "whisper-large-v3-turbo"
GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_AUDIO_TRANSCRIPTIONS_URL = "https://api.groq.com/openai/v1/audio/transcriptions"


@dataclass(frozen=True)
class InterviewMessage:
    role: str
    content: str


class GroqInterviewService:
    def __init__(self, project_dir: str | Path) -> None:
        self.project_dir = Path(project_dir)
        self.reload()

    def reload(self) -> None:
        dotenv_values = _read_dotenv(self.project_dir / ".env")
        self.api_key = os.getenv("GROQ_API_KEY") or dotenv_values.get("GROQ_API_KEY", "")
        self.model = os.getenv("GROQ_MODEL") or dotenv_values.get("GROQ_MODEL", DEFAULT_GROQ_MODEL)
        self.transcription_model = os.getenv("GROQ_TRANSCRIPTION_MODEL") or dotenv_values.get(
            "GROQ_TRANSCRIPTION_MODEL", DEFAULT_GROQ_TRANSCRIPTION_MODEL
        )

    def configuration_status(self) -> str:
        if self.api_key:
            return f"Groq connected. Interview model: {self.model}. Transcription model: {self.transcription_model}."
        return "Groq key missing in .env. Interview fallback can still ask prompts, but live speech transcription stays unavailable."

    def generate_turn(
        self,
        conversation: list[InterviewMessage],
        expression_summary: str,
        stage: str,
    ) -> str:
        self.reload()
        if not self.api_key:
            return self._fallback_question(expression_summary, stage, conversation)

        system_prompt = (
            "You are a sharp but supportive mock interviewer. "
            "Run a natural spoken mock interview for a software or technical candidate. "
            "Ask exactly one question at a time. "
            "Keep each response concise, usually under 85 words. "
            "Start broad, then adapt to the candidate's spoken answers. "
            "If live facial-expression notes are provided, you may lightly reference them once in a supportive way, "
            "but never sound clinical or judgmental. "
            "Do not answer on behalf of the candidate."
        )
        stage_prompt = {
            "start": "Open the interview with a short intro and the first focused question.",
            "reply": "Continue the interview with one focused follow-up question based on the candidate's latest reply.",
            "next": "The candidate is answering out loud off-screen. Move the interview forward with the next focused question.",
        }.get(stage, "Ask the next focused question.")

        messages = [{"role": "system", "content": system_prompt}]
        for item in conversation[-12:]:
            if item.role in {"user", "assistant"}:
                messages.append({"role": item.role, "content": item.content})

        messages.append(
            {
                "role": "system",
                "content": (
                    f"Recent expression summary: {expression_summary}\n"
                    f"Instruction: {stage_prompt}"
                ),
            }
        )

        response = requests.post(
            GROQ_CHAT_COMPLETIONS_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 180,
            },
            timeout=25,
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"].strip()
        return " ".join(content.split())

    def transcribe_audio(self, wav_bytes: bytes) -> str:
        self.reload()
        if not self.api_key:
            return ""

        response = requests.post(
            GROQ_AUDIO_TRANSCRIPTIONS_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
            },
            files={
                "file": ("interview-response.wav", wav_bytes, "audio/wav"),
            },
            data={
                "model": self.transcription_model,
                "temperature": "0",
                "response_format": "json",
                "language": "en",
            },
            timeout=45,
        )
        response.raise_for_status()
        data = response.json()
        return " ".join(data.get("text", "").split())

    def _fallback_question(
        self,
        expression_summary: str,
        stage: str,
        conversation: list[InterviewMessage],
    ) -> str:
        gentle_read = expression_summary.replace("Recent dominant expression:", "I'm reading")
        if stage == "start":
            return (
                f"Let's start your mock interview. {gentle_read} "
                "Tell me about yourself and the kind of technical work you have been doing recently."
            )
        if stage == "next":
            return (
                "Let's keep the flow moving. What deeper technical detail, tradeoff, or concrete example would you add next?"
            )
        latest_user_reply = next((item.content for item in reversed(conversation) if item.role == "user"), "your answer")
        return (
            f"You mentioned: {latest_user_reply[:120]}. What is the strongest technical detail or example "
            "you would add next to make that answer more convincing?"
        )


class AudioChunkRecorder(QObject):
    utterance_ready = Signal(bytes)
    level_changed = Signal(int)
    status_changed = Signal(str)
    error = Signal(str)

    def __init__(self, chunk_ms: int = 1600, max_utterance_ms: int = 40000) -> None:
        super().__init__()
        self.chunk_ms = chunk_ms
        self.max_utterance_ms = max_utterance_ms
        self.silence_timeout_seconds = 5.0
        self.audio_source: QAudioSource | None = None
        self.audio_input: QIODevice | None = None
        self.audio_device = None
        self.segment_pcm = bytearray()
        self.pending_pcm = bytearray()
        self.recording = False
        self.last_voice_at = 0.0
        self.segment_timer = QTimer(self)
        self.segment_timer.setInterval(self.chunk_ms)
        self.segment_timer.timeout.connect(self._finalize_segment)
        self.read_timer = QTimer(self)
        self.read_timer.setInterval(80)
        self.read_timer.timeout.connect(self._pull_audio_data)

    def default_input_name(self) -> str:
        device = QMediaDevices.defaultAudioInput()
        if device and not device.isNull():
            return device.description()
        inputs = QMediaDevices.audioInputs()
        return inputs[0].description() if inputs else "No microphone detected"

    def start(self) -> None:
        inputs = QMediaDevices.audioInputs()
        if not inputs:
            self.error.emit("No microphone input detected. Grant microphone access and try again.")
            return
        if self.recording:
            return

        self.audio_device = QMediaDevices.defaultAudioInput()
        if self.audio_device.isNull():
            self.audio_device = inputs[0]

        self.recording = True
        self.pending_pcm.clear()
        self.last_voice_at = 0.0
        self.status_changed.emit(f"Listening on {self.audio_device.description()}")
        self._begin_segment()
        self.segment_timer.start()

    def pause(self, status: str, discard_current: bool = True) -> None:
        if not self.recording:
            self.status_changed.emit(status)
            return

        self.segment_timer.stop()
        if discard_current:
            self._discard_segment()
        else:
            self._process_segment(self._take_segment_bytes())
        self.recording = False
        self.level_changed.emit(0)
        self.status_changed.emit(status)

    def stop(self) -> None:
        self.segment_timer.stop()
        self.read_timer.stop()
        self._discard_segment()
        self.pending_pcm.clear()
        self.recording = False
        self.level_changed.emit(0)
        self.status_changed.emit("Microphone idle")

    def _begin_segment(self) -> None:
        audio_format = QAudioFormat()
        audio_format.setSampleRate(16000)
        audio_format.setChannelCount(1)
        audio_format.setSampleFormat(QAudioFormat.Int16)

        self.audio_source = QAudioSource(self.audio_device, audio_format, self)
        self.audio_input = self.audio_source.start()
        self.segment_pcm.clear()
        self.read_timer.start()

    def _finalize_segment(self) -> None:
        if not self.recording:
            return

        self._process_segment(self._take_segment_bytes())
        if self.recording:
            self._begin_segment()

    def _take_segment_bytes(self) -> bytes:
        self._pull_audio_data()
        raw = bytes(self.segment_pcm)
        self.segment_pcm.clear()
        if self.audio_source is not None:
            self.audio_source.stop()
            self.audio_source.deleteLater()
            self.audio_source = None
        self.audio_input = None
        self.read_timer.stop()
        return raw

    def _discard_segment(self) -> None:
        self._take_segment_bytes()
        self.segment_pcm.clear()
        self.audio_source = None
        self.audio_input = None

    def _pull_audio_data(self) -> None:
        if self.audio_input is None:
            return

        available = self.audio_input.bytesAvailable()
        if available <= 0:
            return

        self.segment_pcm.extend(bytes(self.audio_input.read(available)))

    def _process_segment(self, raw: bytes) -> None:
        if not raw:
            return

        samples = np.frombuffer(raw, dtype=np.int16)
        if samples.size == 0:
            return

        normalized = samples.astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(np.square(normalized))))
        level = min(100, int(rms * 680))
        self.level_changed.emit(level)

        if rms >= 0.014:
            self.pending_pcm.extend(raw)
            self.last_voice_at = time.monotonic()
            self.status_changed.emit(f"Listening on {self.audio_device.description()} | hearing speech")
            if self._pending_duration_ms() >= self.max_utterance_ms:
                self._emit_pending_utterance()
            return

        if self.pending_pcm and time.monotonic() - self.last_voice_at > self.silence_timeout_seconds:
            self._emit_pending_utterance()
            return

        self.status_changed.emit(f"Listening on {self.audio_device.description()}")

    def _pending_duration_ms(self) -> int:
        bytes_per_second = 16000 * 2
        return int((len(self.pending_pcm) / bytes_per_second) * 1000)

    def _emit_pending_utterance(self) -> None:
        if not self.pending_pcm:
            return

        wav_bytes = _pcm_to_wav(bytes(self.pending_pcm), sample_rate=16000)
        self.pending_pcm.clear()
        self.utterance_ready.emit(wav_bytes)
        self.status_changed.emit("Captured your answer")


class PromptSpeaker:
    def __init__(self) -> None:
        self.process: subprocess.Popen | None = None

    def speak(self, text: str) -> bool:
        if sys.platform != "darwin" or not text.strip():
            return False
        self.stop()
        try:
            self.process = subprocess.Popen(["say", text])
        except OSError:
            self.process = None
        return self.is_speaking()

    def is_speaking(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
        self.process = None


def _pcm_to_wav(raw_pcm: bytes, sample_rate: int) -> bytes:
    path = Path("/tmp") / "mood-mirror-audio.wav"
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(raw_pcm)
    data = path.read_bytes()
    path.unlink(missing_ok=True)
    return data


def _read_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values