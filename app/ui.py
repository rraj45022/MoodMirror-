from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import random
import sys

import cv2
from PySide6.QtCore import QTimer, QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QLinearGradient, QPainter, QPainterPath, QPen, QRadialGradient
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

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
        self.setMinimumSize(840, 520)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.frame: QImage | None = None
        self.frame_size = (1, 1)
        self.result = EmotionResult()

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
        scale = max(viewport.width() / frame_width, viewport.height() / frame_height)
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


class MoodMirrorWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Mood Mirror")
        self.resize(1500, 940)

        self.analyzer = FaceAnalyzer(Path(__file__).resolve().parent.parent / "models")
        self.tracker = SessionTracker()
        self.camera_sources = self._probe_camera_sources()
        self.capture: cv2.VideoCapture | None = None
        self.active_camera_pos = -1
        self.no_face_frames = 0

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

        body = QHBoxLayout()
        body.setSpacing(18)
        root.addLayout(body, 1)

        self.camera = CameraWidget()
        body.addWidget(self.camera, 3)

        side = QVBoxLayout()
        side.setSpacing(14)
        body.addLayout(side, 2)

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
        side.addWidget(mood_panel)

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
        side.addLayout(cards_grid)

        confidence_panel = QFrame()
        confidence_panel.setObjectName("panel")
        confidence_layout = QVBoxLayout(confidence_panel)
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
        side.addWidget(confidence_panel)

        streamer_panel = QFrame()
        streamer_panel.setObjectName("panel")
        streamer_layout = QVBoxLayout(streamer_panel)
        streamer_layout.setContentsMargins(20, 18, 20, 18)
        streamer_layout.setSpacing(10)
        label = QLabel("Live Overlay")
        label.setObjectName("sectionTitle")
        self.overlay_label = QLabel("No reaction spike yet")
        self.overlay_label.setObjectName("overlayText")
        self.overlay_label.setWordWrap(True)
        streamer_layout.addWidget(label)
        streamer_layout.addWidget(self.overlay_label)
        side.addWidget(streamer_panel)

        timeline_panel = QFrame()
        timeline_panel.setObjectName("panel")
        timeline_layout = QVBoxLayout(timeline_panel)
        timeline_layout.setContentsMargins(20, 18, 20, 18)
        timeline_layout.setSpacing(10)
        timeline_label = QLabel("Mood Timeline")
        timeline_label.setObjectName("sectionTitle")
        self.timeline = TimelineWidget()
        timeline_layout.addWidget(timeline_label)
        timeline_layout.addWidget(self.timeline)
        root.addWidget(timeline_panel)

        self._set_mode("mirror")

    def _set_mode(self, mode: str) -> None:
        self.tracker.set_mode(mode)
        for key, button in self.mode_buttons.items():
            button.setChecked(key == mode)
        self._update_cards()

    def _reset_session(self) -> None:
        self.tracker.reset()
        self._update_cards()

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
            QLabel#detailsLabel, QLabel#overlayText {{
                font-size: 15px;
                color: rgba(255, 255, 255, 0.92);
            }}
            QLabel#sectionTitle, QLabel#metricTitle {{
                font-size: 13px;
                font-weight: 700;
                color: rgba(255, 255, 255, 0.76);
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
            """
        )

    def _update_panels(self, result: EmotionResult) -> None:
        self.status_chip.setText(result.status)
        self.mood_label.setText(result.emotion.title())
        if self.tracker.mode == "interview":
            self.details_label.setText("Interview mode tracks calmness, smile rate, and surprise moments through the session.")
        elif self.tracker.mode == "streamer":
            self.details_label.setText("Streamer mode turns confidence spikes into live callouts and hype metrics.")
        else:
            self.details_label.setText("Mood Mirror maps your expression straight into the full scene theme in real time.")

        for emotion, bar in self.bars.items():
            bar.setValue(int(result.scores.get(emotion, 0.0) * 100))
            bar.setFormat(f"{int(result.scores.get(emotion, 0.0) * 100)}%")

        if self.tracker.overlay:
            self.overlay_label.setText(self.tracker.overlay.text)
        else:
            dominant = self.tracker.dominant_history_emotion().title()
            self.overlay_label.setText(f"Session trend: {dominant}")

        self._update_cards()

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
        if self.capture is not None and self.capture.isOpened():
            self.capture.release()
        super().closeEvent(event)


def run() -> None:
    app = QApplication(sys.argv)
    window = MoodMirrorWindow()
    window.show()
    sys.exit(app.exec())