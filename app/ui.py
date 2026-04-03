from __future__ import annotations

from dataclasses import dataclass
import html
from pathlib import Path
import math
import random
import sys

import cv2
import requests
from PySide6.QtCore import QObject, QThread, QTimer, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QLinearGradient, QPainter, QPainterPath, QPen, QRadialGradient, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .interview import AudioChunkRecorder, GroqInterviewService, InterviewMessage, PromptSpeaker
from .session import SessionTracker
from .vision import EmotionResult, FaceAnalyzer

try:
    from AVFoundation import AVCaptureDevice, AVMediaTypeVideo
except ImportError:
    AVCaptureDevice = None
    AVMediaTypeVideo = None


FACE_SEGMENTS = [
    list(range(0, 17)),
    list(range(17, 22)),
    list(range(22, 27)),
    list(range(27, 31)),
    list(range(31, 36)),
    [36, 37, 38, 39, 40, 41, 36],
    [42, 43, 44, 45, 46, 47, 42],
    list(range(48, 60)) + [48],
    list(range(60, 68)) + [60],
]


@dataclass(frozen=True)
class Theme:
    start: str
    end: str
    accent: str
    accent_soft: str
    text: str
    panel: str
    effect: str


@dataclass(frozen=True)
class CameraSource:
    index: int
    name: str
    width: int
    height: int
    mean_luma: float
    std_luma: float


THEMES = {
    "happy": Theme("#fff0a8", "#ff8a5b", "#ff5f45", "#ffd978", "#1e1a16", "rgba(255, 248, 229, 0.82)", "orbs"),
    "sad": Theme("#14213d", "#355c7d", "#6ec5ff", "#93c9ff", "#eef6ff", "rgba(20, 33, 61, 0.72)", "rain"),
    "angry": Theme("#2b0b0b", "#8f1d14", "#ff553e", "#ff8d6d", "#fff1ef", "rgba(59, 13, 13, 0.78)", "pulse"),
    "surprise": Theme("#dffcf2", "#4cc9f0", "#ffbe0b", "#a8ffe6", "#152022", "rgba(237, 255, 251, 0.82)", "confetti"),
    "neutral": Theme("#1f2a44", "#42526e", "#b8c3d9", "#8ca0be", "#f4f7fb", "rgba(20, 29, 46, 0.74)", "drift"),
}


class MetricCard(QFrame):
    def __init__(self, title: str, value: str) -> None:
        super().__init__()
        self.setObjectName("metricCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(6)
        self.title_label = QLabel(title)
        self.title_label.setObjectName("metricTitle")
        self.value_label = QLabel(value)
        self.value_label.setObjectName("metricValue")
        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)

    def update_text(self, title: str, value: str) -> None:
        self.title_label.setText(title)
        self.value_label.setText(value)


class TimelineWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.history: list[tuple[str, float]] = []
        self.setMinimumHeight(34)

    def set_history(self, history: list[tuple[str, float]]) -> None:
        self.history = history
        self.update()

    def paintEvent(self, event) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        rect = self.rect().adjusted(0, 0, -1, -1)
        painter.setBrush(QColor(255, 255, 255, 25))
        painter.drawRoundedRect(rect, 16, 16)
        if not self.history:
            return
        bar_width = rect.width() / max(len(self.history), 1)
        for index, (emotion, confidence) in enumerate(self.history):
            color = QColor(THEMES.get(emotion, THEMES["neutral"]).accent)
            color.setAlphaF(min(0.35 + confidence * 0.55, 0.95))
            painter.setBrush(color)
            x = rect.left() + (index * bar_width)
            painter.drawRoundedRect(QRectF(x + 0.5, rect.top() + 4, max(bar_width - 1.5, 3.0), rect.height() - 8), 6, 6)


class CameraWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(560, 360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.frame: QImage | None = None
        self.frame_size = (1, 1)
        self.result = EmotionResult()
        self.display_mode = "mirror"

    def set_display_mode(self, mode: str) -> None:
        self.display_mode = mode
        self.setMinimumSize(560, 360)
        self.update()

    def set_status(self, message: str) -> None:
        self.frame = None
        self.result = EmotionResult(status=message)
        self.update()

    def set_frame(self, frame_bgr, result: EmotionResult) -> None:
        enhanced = self._enhance_frame(frame_bgr)
        rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        bytes_per_line = channels * width
        self.frame = QImage(rgb.data, width, height, bytes_per_line, QImage.Format_RGB888).copy()
        self.frame_size = (width, height)
        self.result = result
        self.update()

    def _enhance_frame(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        mean_luma = float(gray.mean())

        if mean_luma < 60.0:
            return cv2.convertScaleAbs(frame_bgr, alpha=1.12, beta=10)
        if mean_luma < 85.0:
            return cv2.convertScaleAbs(frame_bgr, alpha=1.05, beta=5)
        return frame_bgr

    def paintEvent(self, event) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        bounds = QRectF(self.rect()).adjusted(2, 2, -2, -2)
        clip = QPainterPath()
        clip.addRoundedRect(bounds, 28, 28)
        painter.fillPath(clip, QColor(6, 10, 16, 250))
        painter.setPen(QPen(QColor(255, 255, 255, 24), 1.0))
        painter.drawRoundedRect(bounds, 28, 28)

        header_rect = QRectF(bounds.left() + 16, bounds.top() + 16, bounds.width() - 32, 56)
        viewport = bounds.adjusted(16, 84, -16, -16)
        viewport_clip = QPainterPath()
        viewport_clip.addRoundedRect(viewport, 22, 22)

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(10, 15, 22, 232))
        painter.drawRoundedRect(header_rect, 18, 18)

        painter.setPen(QColor("#dfe9f7"))
        painter.setFont(QFont("Avenir Next", 12, QFont.Medium))
        painter.drawText(header_rect.adjusted(16, 8, -16, -26), Qt.AlignLeft | Qt.AlignVCenter, "Detected mood")
        painter.setFont(QFont("Avenir Next", 20, QFont.Bold))
        painter.drawText(
            header_rect.adjusted(16, 20, -16, -6),
            Qt.AlignLeft | Qt.AlignVCenter,
            f"{self.result.emotion.title()}  {int(self.result.confidence * 100)}%",
        )

        painter.setClipPath(viewport_clip)
        painter.fillRect(viewport, QColor(5, 8, 12, 255))

        if self.frame is None:
            painter.setPen(QColor("#f3f7ff"))
            painter.setFont(QFont("Avenir Next", 18, QFont.DemiBold))
            painter.drawText(viewport.adjusted(40, 40, -40, -40), Qt.AlignCenter | Qt.TextWordWrap, self.result.status)
            painter.setClipping(False)
            return

        frame_width, frame_height = self.frame_size
        scale = min(viewport.width() / frame_width, viewport.height() / frame_height) * 0.97
        draw_width = frame_width * scale
        draw_height = frame_height * scale
        draw_x = viewport.center().x() - (draw_width / 2)
        draw_y = viewport.center().y() - (draw_height / 2)
        target = QRectF(draw_x, draw_y, draw_width, draw_height)

        painter.drawImage(target, self.frame)

        painter.setPen(QPen(QColor(255, 255, 255, 170), 2.0))
        if self.result.face_box is not None:
            x, y, width, height = self.result.face_box
            painter.drawRoundedRect(
                QRectF(draw_x + x * scale, draw_y + y * scale, width * scale, height * scale),
                18,
                18,
            )

        if self.result.landmarks:
            self._draw_landmarks(painter, draw_x, draw_y, scale)

        painter.setClipping(False)
        footer_rect = QRectF(viewport.left() + 16, viewport.bottom() - 54, 236, 38)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(8, 12, 18, 170))
        painter.drawRoundedRect(footer_rect, 14, 14)
        painter.setPen(QColor("#f1f6ff"))
        painter.setFont(QFont("Avenir Next", 11, QFont.Medium))
        footer_text = "Face detected" if self.result.face_box is not None else self.result.status
        painter.drawText(footer_rect.adjusted(14, 0, -14, 0), Qt.AlignLeft | Qt.AlignVCenter, footer_text)
        painter.setPen(QColor(255, 255, 255, 72))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(viewport, 22, 22)

    def _draw_landmarks(self, painter: QPainter, draw_x: float, draw_y: float, scale: float) -> None:
        line_pen = QPen(QColor(255, 255, 255, 145), 1.6)
        painter.setPen(line_pen)
        for segment in FACE_SEGMENTS:
            for left, right in zip(segment, segment[1:]):
                x1, y1 = self.result.landmarks[left]
                x2, y2 = self.result.landmarks[right]
                painter.drawLine(
                    QPointF(draw_x + x1 * scale, draw_y + y1 * scale),
                    QPointF(draw_x + x2 * scale, draw_y + y2 * scale),
                )
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 255, 255, 210))
        for px, py in self.result.landmarks:
            painter.drawEllipse(QPointF(draw_x + px * scale, draw_y + py * scale), 2.1, 2.1)


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    size: float
    alpha: float
    spin: float
    hue: QColor


class StageWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.theme_name = "neutral"
        self.particles: list[Particle] = []
        self.phase = 0.0
        self._seed_particles()

    def set_theme(self, emotion: str) -> None:
        emotion = emotion if emotion in THEMES else "neutral"
        if emotion == self.theme_name:
            return
        self.theme_name = emotion
        self._seed_particles()
        self.update()

    def advance(self) -> None:
        self.phase += 0.05
        width = max(self.width(), 1)
        height = max(self.height(), 1)
        for particle in self.particles:
            particle.x += particle.vx
            particle.y += particle.vy
            if THEMES[self.theme_name].effect == "rain":
                if particle.y > height + 20:
                    particle.y = random.uniform(-height * 0.3, 0)
                    particle.x = random.uniform(0, width)
            elif THEMES[self.theme_name].effect == "confetti":
                particle.vy = min(particle.vy + 0.015, 3.4)
                particle.spin += 0.08
                if particle.y > height + 18:
                    particle.y = random.uniform(-height * 0.25, -10)
                    particle.x = random.uniform(0, width)
            else:
                if particle.x < -30 or particle.x > width + 30:
                    particle.vx *= -1
                if particle.y < -30 or particle.y > height + 30:
                    particle.vy *= -1
        self.update()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._seed_particles()

    def paintEvent(self, event) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        theme = THEMES[self.theme_name]
        gradient = QLinearGradient(0, 0, self.width(), self.height())
        gradient.setColorAt(0, QColor(theme.start))
        gradient.setColorAt(1, QColor(theme.end))
        painter.fillRect(self.rect(), gradient)

        radial = QRadialGradient(self.width() * 0.12, self.height() * 0.1, self.width() * 0.75)
        radial.setColorAt(0, QColor(255, 255, 255, 80))
        radial.setColorAt(1, QColor(255, 255, 255, 0))
        painter.fillRect(self.rect(), radial)

        effect = theme.effect
        painter.setPen(Qt.NoPen)
        if effect == "rain":
            for particle in self.particles:
                color = QColor(theme.accent)
                color.setAlphaF(particle.alpha)
                painter.setBrush(color)
                painter.save()
                painter.translate(particle.x, particle.y)
                painter.rotate(-18)
                painter.drawRoundedRect(QRectF(0, 0, 2.2, particle.size * 10), 1.1, 1.1)
                painter.restore()
        elif effect == "pulse":
            for ring in range(4):
                radius = 110 + ring * 55 + math.sin(self.phase + ring) * 10
                color = QColor(theme.accent)
                color.setAlpha(58 - ring * 8)
                painter.setPen(QPen(color, 6))
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(QPointF(self.width() * 0.78, self.height() * 0.24), radius, radius)
            painter.setPen(Qt.NoPen)
            for particle in self.particles:
                ember = QColor(theme.accent_soft)
                ember.setAlphaF(particle.alpha)
                painter.setBrush(ember)
                painter.drawEllipse(QPointF(particle.x, particle.y), particle.size, particle.size)
        elif effect == "confetti":
            for particle in self.particles:
                color = QColor(particle.hue)
                color.setAlphaF(particle.alpha)
                painter.setBrush(color)
                painter.save()
                painter.translate(particle.x, particle.y)
                painter.rotate(math.degrees(particle.spin))
                painter.drawRoundedRect(QRectF(-particle.size, -particle.size * 0.3, particle.size * 2.0, particle.size * 0.7), 2, 2)
                painter.restore()
        else:
            for particle in self.particles:
                glow = QColor(theme.accent if effect == "orbs" else theme.accent_soft)
                glow.setAlphaF(particle.alpha)
                painter.setBrush(glow)
                painter.drawEllipse(QPointF(particle.x, particle.y), particle.size, particle.size)

    def _seed_particles(self) -> None:
        width = max(self.width(), 1280)
        height = max(self.height(), 780)
        theme = THEMES[self.theme_name]
        rng = random.Random(f"{self.theme_name}-{width}-{height}")
        self.particles = []
        count = 58 if theme.effect in {"orbs", "confetti"} else 42
        if theme.effect == "rain":
            count = 70
        for _ in range(count):
            hue = QColor(theme.accent if rng.random() > 0.4 else theme.accent_soft)
            if theme.effect == "rain":
                particle = Particle(rng.uniform(0, width), rng.uniform(0, height), -0.8, rng.uniform(5.8, 9.2), rng.uniform(1.6, 3.4), rng.uniform(0.28, 0.72), 0.0, hue)
            elif theme.effect == "confetti":
                particle = Particle(rng.uniform(0, width), rng.uniform(-height * 0.25, height), rng.uniform(-1.8, 1.8), rng.uniform(0.8, 2.3), rng.uniform(4.0, 9.5), rng.uniform(0.45, 0.92), rng.uniform(0, math.pi), hue)
            else:
                particle = Particle(rng.uniform(0, width), rng.uniform(0, height), rng.uniform(-0.7, 0.7), rng.uniform(-0.5, 0.5), rng.uniform(7.0, 18.0), rng.uniform(0.12, 0.32), 0.0, hue)
            self.particles.append(particle)


class InterviewRequestWorker(QObject):
    finished = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        service: GroqInterviewService,
        conversation: list[InterviewMessage],
        expression_summary: str,
        stage: str,
    ) -> None:
        super().__init__()
        self.service = service
        self.conversation = conversation
        self.expression_summary = expression_summary
        self.stage = stage

    def run(self) -> None:
        try:
            prompt = self.service.generate_turn(
                conversation=self.conversation,
                expression_summary=self.expression_summary,
                stage=self.stage,
            )
        except requests.RequestException as exc:
            self.failed.emit(f"Groq request failed: {exc}")
            return
        except Exception as exc:
            self.failed.emit(str(exc))
            return

        self.finished.emit(prompt)


class TranscriptionWorker(QObject):
    finished = Signal(str)
    failed = Signal(str)

    def __init__(self, service: GroqInterviewService, wav_bytes: bytes) -> None:
        super().__init__()
        self.service = service
        self.wav_bytes = wav_bytes

    def run(self) -> None:
        try:
            transcript = self.service.transcribe_audio(self.wav_bytes)
        except requests.RequestException as exc:
            self.failed.emit(f"Groq transcription failed: {exc}")
            return
        except Exception as exc:
            self.failed.emit(str(exc))
            return

        self.finished.emit(transcript)


class MoodMirrorWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Mood Mirror")
        self.resize(1500, 940)
        self.setMinimumSize(1120, 760)

        self.project_dir = Path(__file__).resolve().parent.parent
        self.analyzer = FaceAnalyzer(self.project_dir / "models")
        self.tracker = SessionTracker()
        self.interview_service = GroqInterviewService(self.project_dir)
        self.prompt_speaker = PromptSpeaker()
        self.audio_recorder = AudioChunkRecorder()
        self.camera_sources = self._probe_camera_sources()
        self.capture: cv2.VideoCapture | None = None
        self.active_camera_pos = -1
        self.no_face_frames = 0
        self.interview_active = False
        self.interview_pending = False
        self.transcription_pending = False
        self.interview_messages: list[InterviewMessage] = []
        self.audio_queue: list[bytes] = []
        self.interview_request_id = 0
        self.transcription_request_id = 0
        self.worker_thread: QThread | None = None
        self.worker_object: QObject | None = None

        self.speaker_poll_timer = QTimer(self)
        self.speaker_poll_timer.setInterval(250)
        self.speaker_poll_timer.timeout.connect(self._check_prompt_speaker)

        self.audio_recorder.utterance_ready.connect(self._handle_audio_utterance)
        self.audio_recorder.status_changed.connect(self._update_listening_status)
        self.audio_recorder.level_changed.connect(self._update_mic_level)
        self.audio_recorder.error.connect(self._handle_audio_error)

        self.stage = StageWidget()
        self.setCentralWidget(self.stage)
        self._build_layout()
        self._apply_theme("neutral")
        self._open_initial_camera()

        self.frame_timer = QTimer(self)
        self.frame_timer.timeout.connect(self._process_frame)
        self.frame_timer.start(40)

        self.scene_timer = QTimer(self)
        self.scene_timer.timeout.connect(self.stage.advance)
        self.scene_timer.start(33)

    def _build_layout(self) -> None:
        root = QVBoxLayout(self.stage)
        root.setContentsMargins(28, 26, 28, 28)
        root.setSpacing(18)

        header = QHBoxLayout()
        title_block = QVBoxLayout()
        title_block.setSpacing(2)
        title = QLabel("Mood Mirror")
        title.setObjectName("appTitle")
        subtitle = QLabel("Real-time webcam mood visualizer with interview and streamer overlays")
        subtitle.setObjectName("appSubtitle")
        title_block.addWidget(title)
        title_block.addWidget(subtitle)
        header.addLayout(title_block, 1)

        self.mode_buttons: dict[str, QPushButton] = {}
        for mode, label in (("mirror", "Mirror"), ("interview", "Interview Mode"), ("streamer", "Streamer Mode")):
            button = QPushButton(label)
            button.setCheckable(True)
            button.clicked.connect(lambda checked=False, selected=mode: self._set_mode(selected))
            self.mode_buttons[mode] = button
            header.addWidget(button)

        self.camera_button = QPushButton("Switch Camera")
        self.camera_button.clicked.connect(self._cycle_camera)
        header.addWidget(self.camera_button)

        reset_button = QPushButton("Reset Session")
        reset_button.clicked.connect(self._reset_session)
        header.addWidget(reset_button)
        root.addLayout(header)

        self.body_layout = QHBoxLayout()
        self.body_layout.setSpacing(18)
        root.addLayout(self.body_layout, 1)

        self.camera = CameraWidget()
        self.body_layout.addWidget(self.camera, 5)

        self.side_scroll = QScrollArea()
        self.side_scroll.setObjectName("sideScroll")
        self.side_scroll.setWidgetResizable(True)
        self.side_scroll.setFrameShape(QFrame.NoFrame)
        self.side_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.side_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.side_scroll.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        self.side_container = QWidget()
        self.side_scroll.setWidget(self.side_container)

        self.side_layout = QVBoxLayout(self.side_container)
        self.side_layout.setSpacing(14)
        self.side_layout.setContentsMargins(0, 0, 6, 0)
        self.body_layout.addWidget(self.side_scroll, 4)

        mood_panel = QFrame()
        mood_panel.setObjectName("panel")
        mood_layout = QVBoxLayout(mood_panel)
        mood_layout.setContentsMargins(20, 20, 20, 20)
        mood_layout.setSpacing(10)
        self.status_chip = QLabel("Scanning for a face")
        self.status_chip.setObjectName("statusChip")
        self.mood_label = QLabel("Neutral")
        self.mood_label.setObjectName("moodLabel")
        self.details_label = QLabel("Point the camera toward a single face")
        self.details_label.setObjectName("detailsLabel")
        self.source_label = QLabel("Camera source: scanning")
        self.source_label.setObjectName("detailsLabel")
        mood_layout.addWidget(self.status_chip)
        mood_layout.addWidget(self.mood_label)
        mood_layout.addWidget(self.details_label)
        mood_layout.addWidget(self.source_label)
        self.side_layout.addWidget(mood_panel)

        self.interview_panel = QFrame()
        self.interview_panel.setObjectName("interviewPanel")
        interview_layout = QVBoxLayout(self.interview_panel)
        interview_layout.setContentsMargins(22, 22, 22, 22)
        interview_layout.setSpacing(14)

        interview_title = QLabel("Live AI Interview")
        interview_title.setObjectName("interviewTitle")
        self.interview_status_label = QLabel(self.interview_service.configuration_status())
        self.interview_status_label.setObjectName("interviewDetailLabel")
        self.interview_intro_label = QLabel(
            "Press Start once. The app will ask questions, listen to your spoken answer, transcribe it, and reply in both text and audio."
        )
        self.interview_intro_label.setObjectName("interviewDetailLabel")
        self.interview_expression_label = QLabel("Expression read: waiting for a face")
        self.interview_expression_label.setObjectName("interviewDetailLabel")
        self.interview_expression_label.setWordWrap(True)

        control_row = QHBoxLayout()
        self.start_interview_button = QPushButton("Start")
        self.start_interview_button.clicked.connect(self._start_interview)
        self.stop_interview_button = QPushButton("End")
        self.stop_interview_button.clicked.connect(self._stop_interview)
        control_row.addWidget(self.start_interview_button)
        control_row.addWidget(self.stop_interview_button)
        control_row.addStretch(1)

        live_row = QHBoxLayout()
        self.listening_chip = QLabel("Microphone idle")
        self.listening_chip.setObjectName("statusChip")
        self.mic_level_label = QLabel("Mic level 0%")
        self.mic_level_label.setObjectName("interviewMinorLabel")
        live_row.addWidget(self.listening_chip)
        live_row.addStretch(1)
        live_row.addWidget(self.mic_level_label)

        heard_title = QLabel("Latest Candidate Transcript")
        heard_title.setObjectName("sectionTitle")
        self.last_heard_label = QLabel("Waiting for your first spoken answer.")
        self.last_heard_label.setObjectName("liveNote")
        self.last_heard_label.setWordWrap(True)

        self.interview_transcript = QTextEdit()
        self.interview_transcript.setObjectName("transcriptBox")
        self.interview_transcript.setReadOnly(True)
        self.interview_transcript.setMinimumHeight(220)
        self.interview_transcript.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.interview_transcript.setPlaceholderText("The live interview transcript will appear here.")

        self.interview_reset_hint = QLabel("Reset Session in the top bar clears the emotion metrics without closing the app.")
        self.interview_reset_hint.setObjectName("interviewMinorLabel")
        self.interview_reset_hint.setWordWrap(True)

        interview_layout.addWidget(interview_title)
        interview_layout.addWidget(self.interview_status_label)
        interview_layout.addWidget(self.interview_intro_label)
        interview_layout.addLayout(control_row)
        interview_layout.addLayout(live_row)
        interview_layout.addWidget(self.interview_expression_label)
        interview_layout.addWidget(heard_title)
        interview_layout.addWidget(self.last_heard_label)
        interview_layout.addWidget(self.interview_transcript)
        interview_layout.addWidget(self.interview_reset_hint)
        self.side_layout.addWidget(self.interview_panel, 3)

        cards_grid = QGridLayout()
        cards_grid.setSpacing(12)
        self.card_primary = MetricCard("Session", "00:00")
        self.card_secondary = MetricCard("Calmness", "0%")
        self.card_third = MetricCard("Smile Rate", "0.0/min")
        self.card_fourth = MetricCard("Surprise Moments", "0")
        cards_grid.addWidget(self.card_primary, 0, 0)
        cards_grid.addWidget(self.card_secondary, 0, 1)
        cards_grid.addWidget(self.card_third, 1, 0)
        cards_grid.addWidget(self.card_fourth, 1, 1)
        self.side_layout.addLayout(cards_grid)

        self.confidence_panel = QFrame()
        self.confidence_panel.setObjectName("panel")
        confidence_layout = QVBoxLayout(self.confidence_panel)
        confidence_layout.setContentsMargins(20, 18, 20, 18)
        confidence_layout.setSpacing(12)
        confidence_title = QLabel("Emotion Confidence")
        confidence_title.setObjectName("sectionTitle")
        confidence_layout.addWidget(confidence_title)
        self.bars: dict[str, QProgressBar] = {}
        for emotion in ("happy", "sad", "angry", "surprise", "neutral"):
            row = QHBoxLayout()
            label = QLabel(emotion.title())
            label.setMinimumWidth(78)
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setTextVisible(True)
            row.addWidget(label)
            row.addWidget(bar, 1)
            confidence_layout.addLayout(row)
            self.bars[emotion] = bar
        self.side_layout.addWidget(self.confidence_panel)

        self.streamer_panel = QFrame()
        self.streamer_panel.setObjectName("panel")
        streamer_layout = QVBoxLayout(self.streamer_panel)
        streamer_layout.setContentsMargins(20, 18, 20, 18)
        streamer_layout.setSpacing(10)
        label = QLabel("Live Overlay")
        label.setObjectName("sectionTitle")
        self.overlay_label = QLabel("No reaction spike yet")
        self.overlay_label.setObjectName("overlayText")
        self.overlay_label.setWordWrap(True)
        streamer_layout.addWidget(label)
        streamer_layout.addWidget(self.overlay_label)
        self.side_layout.addWidget(self.streamer_panel)

        self.timeline_panel = QFrame()
        self.timeline_panel.setObjectName("panel")
        timeline_layout = QVBoxLayout(self.timeline_panel)
        timeline_layout.setContentsMargins(20, 18, 20, 18)
        timeline_layout.setSpacing(10)
        timeline_label = QLabel("Mood Timeline")
        timeline_label.setObjectName("sectionTitle")
        self.timeline = TimelineWidget()
        timeline_layout.addWidget(timeline_label)
        timeline_layout.addWidget(self.timeline)
        self.side_layout.addWidget(self.timeline_panel)
        self.side_layout.addStretch(1)

        self._set_mode("mirror")

    def _set_mode(self, mode: str) -> None:
        if self.tracker.mode == "interview" and mode != "interview" and self.interview_active:
            self._stop_interview()
        self.tracker.set_mode(mode)
        for key, button in self.mode_buttons.items():
            button.setChecked(key == mode)
        self._update_cards()
        self._refresh_interview_controls()

    def _reset_session(self) -> None:
        self.tracker.reset()
        self._update_cards()
        self.interview_expression_label.setText("Expression read: waiting for fresh tracking data")
        if self.tracker.mode == "interview":
            self.interview_status_label.setText("Facial metrics reset. The interview loop stays available.")

    def _probe_camera_sources(self) -> list[CameraSource]:
        candidate_indices = self._candidate_camera_indices()
        sources: list[CameraSource] = []
        for index, name in candidate_indices:
            capture = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
            if not capture.isOpened():
                capture.release()
                continue
            ok, frame = capture.read()
            capture.release()
            if not ok or frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            sources.append(
                CameraSource(
                    index=index,
                    name=name,
                    width=width,
                    height=height,
                    mean_luma=float(gray.mean()),
                    std_luma=float(gray.std()),
                )
            )
        sources.sort(key=lambda source: (source.width * source.height, source.std_luma, source.mean_luma), reverse=True)
        return sources

    def _candidate_camera_indices(self) -> list[tuple[int, str]]:
        if sys.platform == "darwin" and AVCaptureDevice is not None and AVMediaTypeVideo is not None:
            devices = AVCaptureDevice.devicesWithMediaType_(AVMediaTypeVideo)
            return [(index, str(device.localizedName())) for index, device in enumerate(devices)]
        return [(index, f"Camera {index}") for index in range(2)]

    def _open_initial_camera(self) -> None:
        if not self.camera_sources:
            self.source_label.setText("Camera source: none detected")
            self.camera_button.setEnabled(False)
            self.camera.set_status("No camera devices were detected. Connect a webcam and relaunch Mood Mirror.")
            return
        preferred = 0
        for index, source in enumerate(self.camera_sources):
            if source.std_luma > 8.0 or source.mean_luma > 15.0:
                preferred = index
                break
        self._open_camera_by_position(preferred)
        self.camera_button.setEnabled(len(self.camera_sources) > 1)

    def _open_camera_by_position(self, position: int) -> None:
        if not self.camera_sources:
            return
        position = position % len(self.camera_sources)
        if self.capture is not None and self.capture.isOpened():
            self.capture.release()
        source = self.camera_sources[position]
        self.capture = cv2.VideoCapture(source.index, cv2.CAP_AVFOUNDATION)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.active_camera_pos = position
        self.no_face_frames = 0
        self.source_label.setText(
            f"Camera source: {source.name}  {source.width}x{source.height}"
        )

    def _cycle_camera(self) -> None:
        if not self.camera_sources:
            return
        self._open_camera_by_position(self.active_camera_pos + 1)
        self.status_chip.setText("Camera switched")
        self.details_label.setText("Switched to the next detected webcam source.")

    def _frame_looks_inactive(self, frame) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_luma = float(gray.mean())
        std_luma = float(gray.std())
        dark_ratio = float((gray < 20).mean())
        return dark_ratio > 0.985 and mean_luma < 12.0 and std_luma < 18.0

    def _process_frame(self) -> None:
        if self.capture is None or not self.capture.isOpened():
            self.status_chip.setText("Camera unavailable")
            self.mood_label.setText("Permission Needed")
            self.details_label.setText("Grant camera access to Terminal or VS Code in macOS Settings, then relaunch the app.")
            self.camera.set_status("Camera access is blocked or unavailable. Grant permission in macOS Settings and relaunch Mood Mirror.")
            return
        ok, frame = self.capture.read()
        if not ok:
            self.status_chip.setText("Frame read failed")
            self.details_label.setText("The webcam opened but did not return frames. Close other camera apps or switch sources.")
            self.camera.set_status("Webcam opened, but no frames were returned. Close other camera apps and try again.")
            if len(self.camera_sources) > 1:
                self._open_camera_by_position(self.active_camera_pos + 1)
            return
        frame = cv2.flip(frame, 1)
        result = self.analyzer.analyze(frame)
        if result.face_box is not None:
            self.no_face_frames = 0
        else:
            self.no_face_frames += 1
            if len(self.camera_sources) > 1 and self.no_face_frames >= 75 and self._frame_looks_inactive(frame):
                self._open_camera_by_position(self.active_camera_pos + 1)
                self.status_chip.setText("Trying another camera")
                self.details_label.setText("The current feed looks inactive, so Mood Mirror switched to another webcam source.")
                self.camera.set_status("Switching camera source because the current feed looks inactive.")
                return
        self.tracker.update(result.emotion, result.confidence, result.metrics, result.scores)
        self.camera.set_frame(frame, result)
        self.timeline.set_history(list(self.tracker.history))
        self._apply_theme(result.emotion)
        self._update_panels(result)

    def _apply_theme(self, emotion: str) -> None:
        theme = THEMES.get(emotion, THEMES["neutral"])
        self.stage.set_theme(emotion)
        panel_label_color = "rgba(15, 24, 34, 0.82)" if emotion in {"happy", "surprise"} else "rgba(255, 255, 255, 0.92)"
        panel_muted_color = "rgba(15, 24, 34, 0.68)" if emotion in {"happy", "surprise"} else "rgba(255, 255, 255, 0.76)"
        self.setStyleSheet(
            f"""
            QWidget {{
                color: {theme.text};
                font-family: 'Avenir Next', 'Segoe UI', sans-serif;
                font-size: 14px;
            }}
            QLabel#appTitle {{
                font-size: 36px;
                font-weight: 800;
            }}
            QLabel#appSubtitle {{
                font-size: 15px;
                color: rgba(255, 255, 255, 0.86);
            }}
            QFrame#panel, QFrame#metricCard {{
                background: {theme.panel};
                border: 1px solid rgba(255, 255, 255, 0.12);
                border-radius: 24px;
            }}
            QFrame#interviewPanel {{
                background: rgba(6, 10, 18, 0.74);
                border: 1px solid rgba(255, 255, 255, 0.12);
                border-radius: 28px;
            }}
            QLabel#statusChip {{
                background: rgba(0, 0, 0, 0.16);
                border-radius: 14px;
                padding: 6px 12px;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }}
            QLabel#moodLabel {{
                font-size: 34px;
                font-weight: 800;
            }}
            QLabel#interviewTitle {{
                font-size: 28px;
                font-weight: 800;
                color: rgba(244, 248, 255, 0.98);
            }}
            QLabel#detailsLabel, QLabel#overlayText {{
                font-size: 15px;
                color: {panel_label_color};
            }}
            QLabel#interviewDetailLabel {{
                font-size: 15px;
                color: rgba(244, 248, 255, 0.92);
            }}
            QLabel#interviewMinorLabel {{
                font-size: 13px;
                color: rgba(214, 224, 238, 0.74);
            }}
            QLabel#liveNote {{
                background: rgba(255, 255, 255, 0.08);
                border-radius: 16px;
                padding: 12px 14px;
                font-size: 15px;
                color: rgba(245, 248, 255, 0.96);
            }}
            QLabel#sectionTitle, QLabel#metricTitle {{
                font-size: 13px;
                font-weight: 700;
                color: {panel_muted_color};
                letter-spacing: 0.06em;
                text-transform: uppercase;
            }}
            QLabel#metricValue {{
                font-size: 26px;
                font-weight: 800;
            }}
            QPushButton {{
                background: rgba(255, 255, 255, 0.12);
                border: 1px solid rgba(255, 255, 255, 0.12);
                border-radius: 18px;
                padding: 10px 18px;
                font-size: 14px;
                font-weight: 700;
            }}
            QPushButton:checked {{
                background: {theme.accent};
                color: #151515;
                border-color: rgba(255, 255, 255, 0.0);
            }}
            QProgressBar {{
                min-height: 18px;
                border-radius: 9px;
                background: rgba(255, 255, 255, 0.14);
                text-align: center;
                font-weight: 700;
            }}
            QProgressBar::chunk {{
                border-radius: 9px;
                background: {theme.accent};
            }}
            QTextEdit {{
                background: rgba(7, 10, 16, 0.24);
                border: 1px solid rgba(255, 255, 255, 0.14);
                border-radius: 16px;
                padding: 10px 12px;
                selection-background-color: {theme.accent};
            }}
            QTextEdit#transcriptBox {{
                background: rgba(2, 6, 12, 0.58);
                color: rgba(246, 249, 255, 0.97);
            }}
            QScrollArea#sideScroll {{
                background: transparent;
            }}
            QScrollArea#sideScroll > QWidget > QWidget {{
                background: transparent;
            }}
            QScrollBar:vertical {{
                background: rgba(0, 0, 0, 0.14);
                width: 12px;
                margin: 4px 0 4px 0;
                border-radius: 6px;
            }}
            QScrollBar::handle:vertical {{
                background: rgba(255, 255, 255, 0.28);
                min-height: 40px;
                border-radius: 6px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            """
        )

    def _update_panels(self, result: EmotionResult) -> None:
        self.status_chip.setText(result.status)
        self.mood_label.setText(result.emotion.title())
        if self.tracker.mode == "interview":
            self.details_label.setText("Interview mode keeps the camera and live expression metrics visible while the AI interviewer listens and replies on the right.")
        elif self.tracker.mode == "streamer":
            self.details_label.setText("Streamer mode turns confidence spikes into live callouts and hype metrics.")
        else:
            self.details_label.setText("Mood Mirror maps your expression straight into the full scene theme in real time.")

        self.interview_expression_label.setText(f"Expression read: {self.tracker.live_signal_label()}")

        for emotion, bar in self.bars.items():
            bar.setValue(int(result.scores.get(emotion, 0.0) * 100))
            bar.setFormat(f"{int(result.scores.get(emotion, 0.0) * 100)}%")

        if self.tracker.overlay:
            self.overlay_label.setText(self.tracker.overlay.text)
        else:
            dominant = self.tracker.dominant_history_emotion().title()
            self.overlay_label.setText(f"Session trend: {dominant}")

        self._update_cards()

    def _refresh_interview_controls(self) -> None:
        interview_mode = self.tracker.mode == "interview"
        self.interview_panel.setVisible(interview_mode)
        self.streamer_panel.setVisible(self.tracker.mode == "streamer")
        busy = self.interview_pending or self.transcription_pending
        self.start_interview_button.setEnabled(interview_mode and not self.interview_active and not busy)
        self.stop_interview_button.setEnabled(interview_mode and self.interview_active)
        self.confidence_panel.setVisible(self.tracker.mode != "streamer")
        self.timeline_panel.setVisible(True)
        self.camera.set_display_mode("mirror")
        self.body_layout.setStretch(0, 5)
        self.body_layout.setStretch(1, 4)

    def _start_interview(self) -> None:
        self.prompt_speaker.stop()
        self.speaker_poll_timer.stop()
        self.audio_recorder.stop()
        self.audio_queue = []
        self.interview_messages = []
        self.interview_active = True
        self.interview_pending = False
        self.transcription_pending = False
        self.interview_transcript.clear()
        self.last_heard_label.setText("Waiting for your first spoken answer.")
        self.listening_chip.setText("Preparing interview")
        self.interview_status_label.setText("Starting the interviewer...")
        self._append_interview_entry("System", "Interview started. The AI interviewer will ask questions, then the microphone will listen automatically.")
        self._append_interview_entry("Interviewer", "Joining the interview...")
        self._queue_interview_request(stage="start")

    def _stop_interview(self) -> None:
        self.prompt_speaker.stop()
        self.speaker_poll_timer.stop()
        self.audio_recorder.stop()
        self.interview_active = False
        self.interview_pending = False
        self.transcription_pending = False
        self.audio_queue = []
        self.listening_chip.setText("Microphone idle")
        self.mic_level_label.setText("Mic level 0%")
        self.interview_status_label.setText("Interview session ended. Press Start to begin a fresh voice-driven round.")
        self._refresh_interview_controls()

    def _handle_audio_utterance(self, wav_bytes: bytes) -> None:
        if not self.interview_active:
            return
        self.audio_queue.append(wav_bytes)
        self.audio_recorder.pause("Processing your answer...", discard_current=True)
        self._start_next_transcription()

    def _start_next_transcription(self) -> None:
        if not self.interview_active or self.transcription_pending or self.interview_pending or not self.audio_queue:
            return

        self.transcription_pending = True
        self.transcription_request_id += 1
        request_id = self.transcription_request_id
        wav_bytes = self.audio_queue.pop(0)
        self.interview_status_label.setText("Transcribing your answer...")
        self.listening_chip.setText("Transcribing")
        self._refresh_interview_controls()

        worker = TranscriptionWorker(self.interview_service, wav_bytes)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_transcription_finished)
        worker.failed.connect(self._on_transcription_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(self._clear_worker_refs)
        thread.finished.connect(thread.deleteLater)
        self.worker_object = worker
        self.worker_thread = thread
        thread.start()

    def _queue_interview_request(self, stage: str) -> None:
        if not self.interview_active or self.interview_pending:
            return

        self.interview_pending = True
        self.interview_request_id += 1
        request_id = self.interview_request_id
        self.interview_status_label.setText("Interviewer is preparing the next question...")
        self.listening_chip.setText("Thinking")
        self._refresh_interview_controls()

        worker = InterviewRequestWorker(
            service=self.interview_service,
            conversation=list(self.interview_messages),
            expression_summary=self.tracker.recent_expression_summary(),
            stage=stage,
        )
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_interview_finished)
        worker.failed.connect(self._on_interview_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(self._clear_worker_refs)
        thread.finished.connect(thread.deleteLater)
        self.worker_object = worker
        self.worker_thread = thread
        thread.start()

    def _handle_transcription_success(self, request_id: int, text: str) -> None:
        if request_id != self.transcription_request_id or not self.interview_active:
            return

        self.transcription_pending = False
        if not text.strip():
            self.interview_status_label.setText("I did not catch a clear answer. Speak again and I will keep listening.")
            self.last_heard_label.setText("No clear speech was captured in the last turn.")
            self._refresh_interview_controls()
            self._resume_audio_listener()
            return

        self.last_heard_label.setText(text)
        self.interview_messages.append(InterviewMessage(role="user", content=text))
        self._append_interview_entry("You", text)
        self._queue_interview_request(stage="reply")

    def _handle_transcription_failure(self, request_id: int, message: str) -> None:
        if request_id != self.transcription_request_id:
            return

        self.transcription_pending = False
        self.interview_status_label.setText(message)
        self._refresh_interview_controls()
        self._resume_audio_listener()

    def _handle_interview_success(self, request_id: int, text: str) -> None:
        if request_id != self.interview_request_id or not self.interview_active:
            return

        self.interview_pending = False
        transcript_html = self.interview_transcript.toHtml()
        if "Joining the interview..." in transcript_html:
            self.interview_transcript.clear()
            for item in self.interview_messages:
                speaker = "You" if item.role == "user" else "Interviewer"
                self._append_interview_entry(speaker, item.content)
        self.interview_messages.append(InterviewMessage(role="assistant", content=text))
        self._append_interview_entry("Interviewer", text)
        self.interview_status_label.setText(self.interview_service.configuration_status())
        self._refresh_interview_controls()
        self.listening_chip.setText("Speaking")
        if self.prompt_speaker.speak(text):
            self.speaker_poll_timer.start()
        else:
            self._resume_audio_listener()

    def _handle_interview_failure(self, request_id: int, message: str) -> None:
        if request_id != self.interview_request_id:
            return

        self.interview_pending = False
        self.interview_status_label.setText(message)
        self._refresh_interview_controls()
        self._resume_audio_listener()

    def _resume_audio_listener(self) -> None:
        if not self.interview_active or self.interview_pending or self.transcription_pending or self.prompt_speaker.is_speaking():
            return

        if not self.interview_service.api_key:
            self.listening_chip.setText("Mic disabled")
            self.interview_status_label.setText("Groq speech transcription needs a key in .env before live listening can start.")
            return

        self.audio_recorder.start()
        self._refresh_interview_controls()

    def _check_prompt_speaker(self) -> None:
        if self.prompt_speaker.is_speaking():
            return
        self.speaker_poll_timer.stop()
        self._resume_audio_listener()

    def _update_listening_status(self, status: str) -> None:
        self.listening_chip.setText(status)

    def _update_mic_level(self, level: int) -> None:
        self.mic_level_label.setText(f"Mic level {level}%")

    def _handle_audio_error(self, message: str) -> None:
        self.listening_chip.setText("Microphone error")
        self.interview_status_label.setText(message)
        self._refresh_interview_controls()

    def _append_interview_entry(self, speaker: str, text: str) -> None:
        safe_text = html.escape(text).replace("\n", "<br>")
        self.interview_transcript.append(f"<b>{html.escape(speaker)}:</b> {safe_text}")
        self.interview_transcript.moveCursor(QTextCursor.End)

    def _on_transcription_finished(self, text: str) -> None:
        self._handle_transcription_success(self.transcription_request_id, text)

    def _on_transcription_failed(self, message: str) -> None:
        self._handle_transcription_failure(self.transcription_request_id, message)

    def _on_interview_finished(self, text: str) -> None:
        self._handle_interview_success(self.interview_request_id, text)

    def _on_interview_failed(self, message: str) -> None:
        self._handle_interview_failure(self.interview_request_id, message)

    def _clear_worker_refs(self) -> None:
        self.worker_object = None
        self.worker_thread = None

    def _update_cards(self) -> None:
        if self.tracker.mode == "interview":
            self.card_primary.update_text("Session", self.tracker.elapsed_text())
            self.card_secondary.update_text("Calmness", f"{self.tracker.calmness_percent()}%")
            self.card_third.update_text("Smile Rate", f"{self.tracker.smiles_per_minute():.1f}/min")
            self.card_fourth.update_text("Surprise Moments", str(self.tracker.surprise_events))
        elif self.tracker.mode == "streamer":
            self.card_primary.update_text("Live Time", self.tracker.elapsed_text())
            self.card_secondary.update_text("Reaction Spikes", str(self.tracker.reaction_spikes))
            self.card_third.update_text("Top Callout", self.tracker.latest_callout.title())
            self.card_fourth.update_text("Smiles Caught", str(self.tracker.smile_events))
        else:
            mix = self.tracker.mood_mix()
            dominant = self.tracker.dominant_history_emotion().title()
            self.card_primary.update_text("Session", self.tracker.elapsed_text())
            self.card_secondary.update_text("Trend", dominant)
            self.card_third.update_text("Happy Mix", f"{mix.get('happy', 0)}%")
            self.card_fourth.update_text("Surprise Mix", f"{mix.get('surprise', 0)}%")

    def closeEvent(self, event) -> None:
        self.frame_timer.stop()
        self.scene_timer.stop()
        self.speaker_poll_timer.stop()
        self.audio_recorder.stop()
        self.prompt_speaker.stop()
        self.interview_active = False
        if self.worker_thread is not None and self.worker_thread.isRunning():
            self.worker_thread.quit()
            self.worker_thread.wait(5000)
        if self.capture is not None and self.capture.isOpened():
            self.capture.release()
        super().closeEvent(event)


def run() -> None:
    app = QApplication(sys.argv)
    window = MoodMirrorWindow()
    window.show()
    sys.exit(app.exec())