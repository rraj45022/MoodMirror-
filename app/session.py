from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import time


EMOTION_PRIORITY = {
    "happy": 0.78,
    "neutral": 0.92,
    "sad": 0.38,
    "surprise": 0.16,
    "angry": 0.08,
}


@dataclass
class StreamerOverlay:
    text: str
    emotion: str
    expires_at: float


class SessionTracker:
    def __init__(self) -> None:
        self.mode = "mirror"
        self.reset()

    def reset(self) -> None:
        self.started_at = time.time()
        self.calm_score_total = 0.0
        self.total_samples = 0
        self.smile_events = 0
        self.surprise_events = 0
        self.reaction_spikes = 0
        self.latest_callout = "Waiting for a spike"
        self.overlay: StreamerOverlay | None = None
        self.history: deque[tuple[str, float]] = deque(maxlen=180)
        self.last_emotion = "neutral"
        self.last_confidence = 0.0
        self.last_smile_at = 0.0
        self.last_surprise_at = 0.0
        self.smile_active = False
        self.surprise_active = False

    def set_mode(self, mode: str) -> None:
        self.mode = mode

    def update(
        self,
        emotion: str,
        confidence: float,
        metrics: dict[str, float] | None = None,
        scores: dict[str, float] | None = None,
    ) -> None:
        now = time.time()
        metrics = metrics or {}
        scores = scores or {}
        self.total_samples += 1
        self.history.append((emotion, confidence))
        self.calm_score_total += EMOTION_PRIORITY.get(emotion, 0.5) * (1.0 - min(confidence, 1.0) * 0.18)

        smile_signal = max(
            scores.get("happy", 0.0),
            metrics.get("smile_curve", 0.0) * 5.6 + max(metrics.get("mouth_width", 0.0) - 0.34, 0.0) * 2.0,
        )
        surprise_signal = max(
            scores.get("surprise", 0.0),
            max(metrics.get("mouth_open", 0.0) - 0.028, 0.0) * 7.0
            + max(metrics.get("eye_open", 0.0) - 0.31, 0.0) * 2.4
            + max(metrics.get("brow_raise", 0.0) - 0.05, 0.0) * 14.0,
        )

        if smile_signal >= 0.54 and not self.smile_active and now - self.last_smile_at > 1.0:
            self.smile_events += 1
            self.last_smile_at = now
            self.smile_active = True
        elif smile_signal <= 0.34:
            self.smile_active = False

        if surprise_signal >= 0.52 and not self.surprise_active and now - self.last_surprise_at > 1.2:
            self.surprise_events += 1
            self.last_surprise_at = now
            self.surprise_active = True
        elif surprise_signal <= 0.3:
            self.surprise_active = False

        delta = confidence - self.last_confidence
        if delta > 0.2 and confidence >= 0.68:
            self.reaction_spikes += 1
            self.latest_callout = self._callout_for(emotion)
            self.overlay = StreamerOverlay(
                text=self.latest_callout,
                emotion=emotion,
                expires_at=now + 2.6,
            )

        if self.overlay and now > self.overlay.expires_at:
            self.overlay = None

        self.last_emotion = emotion
        self.last_confidence = confidence

    def _callout_for(self, emotion: str) -> str:
        mapping = {
            "happy": "chat loved that reaction",
            "surprise": "clip that face now",
            "angry": "the room felt that glare",
            "sad": "audience went quiet there",
            "neutral": "camera locked on your calm",
        }
        return mapping.get(emotion, "chat noticed that")

    def elapsed_seconds(self) -> int:
        return int(time.time() - self.started_at)

    def elapsed_text(self) -> str:
        seconds = self.elapsed_seconds()
        minutes, seconds = divmod(seconds, 60)
        return f"{minutes:02d}:{seconds:02d}"

    def calmness_percent(self) -> int:
        if not self.total_samples:
            return 0
        return int((self.calm_score_total / self.total_samples) * 100)

    def smiles_per_minute(self) -> float:
        minutes = max(self.elapsed_seconds() / 60.0, 1 / 60.0)
        return self.smile_events / minutes

    def mood_mix(self) -> dict[str, int]:
        counts = Counter(emotion for emotion, _ in self.history)
        total = max(sum(counts.values()), 1)
        return {emotion: int((count / total) * 100) for emotion, count in counts.items()}

    def dominant_history_emotion(self) -> str:
        if not self.history:
            return "neutral"
        counts = Counter(emotion for emotion, _ in self.history)
        return counts.most_common(1)[0][0]

    def recent_expression_summary(self, sample_size: int = 45) -> str:
        recent = list(self.history)[-sample_size:]
        if not recent:
            return "No recent expression data yet."

        counts = Counter(emotion for emotion, _ in recent)
        dominant, dominant_count = counts.most_common(1)[0]
        average_confidence = sum(confidence for _, confidence in recent) / len(recent)
        dominant_share = int((dominant_count / len(recent)) * 100)
        return (
            f"Recent dominant expression: {dominant} for about {dominant_share}% of frames, "
            f"average confidence {int(average_confidence * 100)}%, calmness {self.calmness_percent()}%, "
            f"smiles {self.smile_events}, surprise moments {self.surprise_events}."
        )

    def live_signal_label(self) -> str:
        return (
            f"{self.last_emotion.title()} at {int(self.last_confidence * 100)}% confidence | "
            f"calmness {self.calmness_percent()}% | smiles {self.smile_events} | surprises {self.surprise_events}"
        )