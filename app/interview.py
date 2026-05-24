from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import os
import re
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


@dataclass(frozen=True)
class SessionReview:
    overall_score: int
    answer_score: int
    expression_score: int
    summary: str
    strengths: list[str]
    brush_up_topics: list[str]
    answer_feedback: list[str]
    expression_feedback: list[str]


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

    def review_session(
        self,
        conversation: list[InterviewMessage],
        expression_summary: str,
        session_metrics: dict[str, object],
    ) -> SessionReview:
        self.reload()
        if not self.api_key:
            return self._fallback_review(conversation, expression_summary, session_metrics)

        system_prompt = (
            "You are an interview coach. Review a completed mock interview session and return strict JSON only. "
            "Score the session fairly, infer weak areas from the candidate's answers, and give expression coaching. "
            "Do not invent facts not supported by the transcript. If evidence is weak, say so briefly but still provide practical next steps."
        )
        review_prompt = {
            "transcript": self._format_conversation(conversation),
            "expression_summary": expression_summary,
            "session_metrics": session_metrics,
            "required_json_schema": {
                "overall_score": "integer 0-100",
                "answer_score": "integer 0-100",
                "expression_score": "integer 0-100",
                "summary": "short paragraph",
                "strengths": ["3 short bullets max"],
                "brush_up_topics": ["up to 4 concrete topics to revise"],
                "answer_feedback": ["up to 4 direct answer-improvement bullets"],
                "expression_feedback": ["up to 4 direct expression-improvement bullets"],
            },
        }

        try:
            response = requests.post(
                GROQ_CHAT_COMPLETIONS_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": json.dumps(review_prompt, indent=2)},
                    ],
                    "temperature": 0.2,
                    "max_tokens": 500,
                },
                timeout=35,
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"].strip()
            return self._parse_review_response(content, conversation, expression_summary, session_metrics)
        except (requests.RequestException, KeyError, ValueError, TypeError):
            return self._fallback_review(conversation, expression_summary, session_metrics)

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

    def _parse_review_response(
        self,
        content: str,
        conversation: list[InterviewMessage],
        expression_summary: str,
        session_metrics: dict[str, object],
    ) -> SessionReview:
        fallback = self._fallback_review(conversation, expression_summary, session_metrics)
        payload = self._extract_json_object(content)
        if payload is None:
            return fallback

        strengths = self._normalize_list(payload.get("strengths"), fallback.strengths)
        brush_up_topics = self._normalize_list(payload.get("brush_up_topics"), fallback.brush_up_topics)
        answer_feedback = self._normalize_list(payload.get("answer_feedback"), fallback.answer_feedback)
        expression_feedback = self._normalize_list(payload.get("expression_feedback"), fallback.expression_feedback)

        return SessionReview(
            overall_score=self._clamp_score(payload.get("overall_score"), fallback.overall_score),
            answer_score=self._clamp_score(payload.get("answer_score"), fallback.answer_score),
            expression_score=self._clamp_score(payload.get("expression_score"), fallback.expression_score),
            summary=self._normalize_text(payload.get("summary"), fallback.summary),
            strengths=strengths,
            brush_up_topics=brush_up_topics,
            answer_feedback=answer_feedback,
            expression_feedback=expression_feedback,
        )

    def _fallback_review(
        self,
        conversation: list[InterviewMessage],
        expression_summary: str,
        session_metrics: dict[str, object],
    ) -> SessionReview:
        user_answers = [item.content.strip() for item in conversation if item.role == "user" and item.content.strip()]
        answer_count = len(user_answers)
        if not answer_count:
            expression_score = self._estimate_expression_score(session_metrics)
            expression_feedback = self._expression_feedback(session_metrics)
            return SessionReview(
                overall_score=expression_score,
                answer_score=0,
                expression_score=expression_score,
                summary="The session ended before enough spoken answers were captured to score answer quality. Expression feedback is still available from the visible interview segment.",
                strengths=["The interview flow started successfully."],
                brush_up_topics=["Practice giving complete spoken answers before ending the session."],
                answer_feedback=["Aim for answers that explain context, action, and result in one pass."],
                expression_feedback=expression_feedback,
            )

        answer_word_counts = [len(answer.split()) for answer in user_answers]
        average_words = sum(answer_word_counts) / answer_count
        example_hits = sum(
            1
            for answer in user_answers
            if any(token in answer.lower() for token in ("for example", "for instance", "i built", "i used", "tradeoff", "because", "result"))
        )
        vague_hits = sum(
            1
            for answer in user_answers
            if len(answer.split()) < 18 or any(token in answer.lower() for token in ("not sure", "maybe", "kind of", "sort of", "i think"))
        )
        answer_score = int(
            max(
                28,
                min(
                    95,
                    34 + answer_count * 7 + average_words * 0.32 + example_hits * 5 - vague_hits * 4,
                ),
            )
        )
        expression_score = self._estimate_expression_score(session_metrics)
        overall_score = int(round(answer_score * 0.72 + expression_score * 0.28))

        strengths: list[str] = []
        if average_words >= 45:
            strengths.append("Your answers had enough length to show some reasoning instead of one-line replies.")
        if example_hits:
            strengths.append("You used concrete examples or tradeoff language in at least part of the session.")
        if expression_score >= 75:
            strengths.append("Your on-camera delivery stayed fairly steady for most of the interview window.")
        if not strengths:
            strengths.append("You stayed engaged through the interview flow and produced usable review data.")

        brush_up_topics = self._infer_brush_up_topics(conversation)
        answer_feedback = self._answer_feedback(user_answers, average_words, vague_hits, example_hits)
        expression_feedback = self._expression_feedback(session_metrics)
        calmness = int(session_metrics.get("calmness_percent", 0) or 0)

        return SessionReview(
            overall_score=overall_score,
            answer_score=answer_score,
            expression_score=expression_score,
            summary=(
                f"You completed {answer_count} spoken answer{'s' if answer_count != 1 else ''}. "
                f"Answer quality looked {'solid' if answer_score >= 70 else 'mixed'}, and expression stability was {'strong' if calmness >= 70 else 'still improvable'}."
            ),
            strengths=strengths[:3],
            brush_up_topics=brush_up_topics[:4],
            answer_feedback=answer_feedback[:4],
            expression_feedback=expression_feedback[:4],
        )

    def _answer_feedback(
        self,
        user_answers: list[str],
        average_words: float,
        vague_hits: int,
        example_hits: int,
    ) -> list[str]:
        feedback: list[str] = []
        if average_words < 35:
            feedback.append("Expand each answer with context, your decision, and the result so it does not sound cut short.")
        if vague_hits:
            feedback.append("Replace hedging language with specific ownership, metrics, or named tradeoffs whenever possible.")
        if example_hits < max(1, len(user_answers) // 2):
            feedback.append("Back more answers with a real project example instead of staying generic.")
        feedback.append("When you state a choice, add why you made it and what constraint or tradeoff drove it.")
        return feedback

    def _expression_feedback(self, session_metrics: dict[str, object]) -> list[str]:
        feedback: list[str] = []
        calmness = int(session_metrics.get("calmness_percent", 0) or 0)
        smiles_per_minute = float(session_metrics.get("smiles_per_minute", 0.0) or 0.0)
        surprise_events = int(session_metrics.get("surprise_events", 0) or 0)
        dominant_emotion = str(session_metrics.get("dominant_emotion", "neutral") or "neutral")

        if calmness < 60:
            feedback.append("Keep your face steadier between answers so nervous shifts do not read as uncertainty.")
        if surprise_events >= 3:
            feedback.append("Reduce startled reactions when you hear a question; pause first, then answer with a composed expression.")
        if dominant_emotion in {"sad", "angry"}:
            feedback.append("Aim for a more neutral-to-positive resting face so you appear more approachable on camera.")
        if smiles_per_minute < 0.35:
            feedback.append("Add a light natural smile at greetings and transitions to make delivery feel warmer.")
        if calmness >= 75 and dominant_emotion in {"neutral", "happy"}:
            feedback.append("Your expression baseline already reads interview-ready; keep that same calm posture and eye line.")
        if not feedback:
            feedback.append("Keep a calm neutral baseline, then use small smiles only when they fit the answer naturally.")
        return feedback

    def _estimate_expression_score(self, session_metrics: dict[str, object]) -> int:
        calmness = float(session_metrics.get("calmness_percent", 0) or 0)
        smiles_per_minute = float(session_metrics.get("smiles_per_minute", 0.0) or 0.0)
        surprise_events = float(session_metrics.get("surprise_events", 0) or 0)
        dominant_emotion = str(session_metrics.get("dominant_emotion", "neutral") or "neutral")

        score = 45 + calmness * 0.45
        if 0.35 <= smiles_per_minute <= 2.2:
            score += 10
        elif smiles_per_minute > 2.2:
            score += 4
        else:
            score -= 4
        score -= min(surprise_events * 4.5, 18)
        if dominant_emotion in {"neutral", "happy"}:
            score += 8
        elif dominant_emotion in {"sad", "angry"}:
            score -= 8
        return max(20, min(int(round(score)), 96))

    def _infer_brush_up_topics(self, conversation: list[InterviewMessage]) -> list[str]:
        topic_map = {
            "system design": ("design", "architecture", "scale", "scalability", "distributed", "cache", "latency"),
            "algorithms and data structures": ("algorithm", "complexity", "tree", "graph", "hash", "sorting", "data structure"),
            "testing and debugging": ("test", "testing", "debug", "bug", "quality", "coverage"),
            "apis and backend fundamentals": ("api", "http", "rest", "endpoint", "database", "sql", "backend", "service"),
            "projects and impact storytelling": ("project", "built", "impact", "ownership", "challenge", "result", "team"),
            "performance and concurrency": ("performance", "optimize", "thread", "async", "concurrency", "parallel", "throughput"),
        }
        score_by_topic = {topic: 0 for topic in topic_map}
        last_question = ""
        for item in conversation:
            if item.role == "assistant":
                last_question = item.content.lower()
                continue
            if item.role != "user":
                continue
            answer = item.content.lower()
            weak_answer = len(answer.split()) < 18 or any(token in answer for token in ("not sure", "maybe", "i think", "kind of", "sort of"))
            if not weak_answer:
                continue
            for topic, keywords in topic_map.items():
                if any(keyword in last_question for keyword in keywords):
                    score_by_topic[topic] += 1

        ranked = [topic for topic, score in sorted(score_by_topic.items(), key=lambda item: item[1], reverse=True) if score > 0]
        if ranked:
            return ranked
        return ["projects and impact storytelling", "technical depth in follow-up answers"]

    @staticmethod
    def _format_conversation(conversation: list[InterviewMessage]) -> str:
        lines = []
        for item in conversation:
            speaker = "Candidate" if item.role == "user" else "Interviewer"
            lines.append(f"{speaker}: {item.content}")
        return "\n".join(lines)

    @staticmethod
    def _extract_json_object(content: str) -> dict[str, object] | None:
        cleaned = content.strip()
        fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, re.DOTALL)
        if fence_match:
            cleaned = fence_match.group(1)
        else:
            object_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if object_match:
                cleaned = object_match.group(0)
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _normalize_list(value: object, fallback: list[str]) -> list[str]:
        if not isinstance(value, list):
            return fallback
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        return cleaned[:4] if cleaned else fallback

    @staticmethod
    def _normalize_text(value: object, fallback: str) -> str:
        text = str(value).strip() if value is not None else ""
        return text or fallback

    @staticmethod
    def _clamp_score(value: object, fallback: int) -> int:
        try:
            score = int(float(value))
        except (TypeError, ValueError):
            return fallback
        return max(0, min(score, 100))


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