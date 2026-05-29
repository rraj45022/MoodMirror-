from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .schemas import EmotionSampleInput, SessionMessageResponse, SessionReviewResponse


EMOTION_PRIORITY = {
    "happy": 0.78,
    "neutral": 0.92,
    "sad": 0.38,
    "surprise": 0.16,
    "angry": 0.08,
}


@dataclass(frozen=True)
class SessionAggregate:
    duration_seconds: int
    calmness_percent: int
    smile_events: int
    surprise_events: int
    reaction_spikes: int
    smiles_per_minute: float
    dominant_emotion: str
    mood_mix: dict[str, int]
    total_samples: int
    transcript_turns: int
    latest_expression: str
    overall_score: int | None


class SessionAnalytics:
    @staticmethod
    def summarize(
        started_at: float,
        completed_at: float | None,
        samples: list[EmotionSampleInput],
        messages: list[SessionMessageResponse],
        review: SessionReviewResponse | None,
    ) -> SessionAggregate:
        if not samples:
            duration = max(int((completed_at or started_at) - started_at), 0)
            return SessionAggregate(
                duration_seconds=duration,
                calmness_percent=0,
                smile_events=0,
                surprise_events=0,
                reaction_spikes=0,
                smiles_per_minute=0.0,
                dominant_emotion="neutral",
                mood_mix={},
                total_samples=0,
                transcript_turns=len(messages),
                latest_expression="No samples yet",
                overall_score=review.overall_score if review else None,
            )

        ordered_samples = sorted(samples, key=lambda item: item.recorded_at or started_at)
        first_at = ordered_samples[0].recorded_at or started_at
        end_at = completed_at or ordered_samples[-1].recorded_at or first_at
        duration = max(int(end_at - min(first_at, started_at)), 0)

        calm_score_total = 0.0
        emotion_counts: Counter[str] = Counter()
        smile_events = 0
        surprise_events = 0
        reaction_spikes = 0
        last_confidence = 0.0
        last_smile_at = 0.0
        last_surprise_at = 0.0
        smile_active = False
        surprise_active = False

        for sample in ordered_samples:
            emotion_counts[sample.emotion] += 1
            confidence = min(max(sample.confidence, 0.0), 1.0)
            calm_score_total += EMOTION_PRIORITY.get(sample.emotion, 0.5) * (1.0 - confidence * 0.18)

            metrics = sample.metrics or {}
            scores = sample.scores or {}
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

            recorded_at = sample.recorded_at or started_at
            if smile_signal >= 0.54 and not smile_active and recorded_at - last_smile_at > 1.0:
                smile_events += 1
                last_smile_at = recorded_at
                smile_active = True
            elif smile_signal <= 0.34:
                smile_active = False

            if surprise_signal >= 0.52 and not surprise_active and recorded_at - last_surprise_at > 1.2:
                surprise_events += 1
                last_surprise_at = recorded_at
                surprise_active = True
            elif surprise_signal <= 0.3:
                surprise_active = False

            delta = confidence - last_confidence
            if delta > 0.2 and confidence >= 0.68:
                reaction_spikes += 1
            last_confidence = confidence

        total_samples = len(ordered_samples)
        total_emotions = sum(emotion_counts.values()) or 1
        mood_mix = {
            emotion: int((count / total_emotions) * 100)
            for emotion, count in emotion_counts.items()
        }
        dominant_emotion = emotion_counts.most_common(1)[0][0] if emotion_counts else "neutral"
        minutes = max(duration / 60.0, 1 / 60.0)
        last_sample = ordered_samples[-1]
        latest_expression = f"{last_sample.emotion.title()} at {int(last_sample.confidence * 100)}% confidence"

        return SessionAggregate(
            duration_seconds=duration,
            calmness_percent=int((calm_score_total / total_samples) * 100),
            smile_events=smile_events,
            surprise_events=surprise_events,
            reaction_spikes=reaction_spikes,
            smiles_per_minute=smile_events / minutes,
            dominant_emotion=dominant_emotion,
            mood_mix=mood_mix,
            total_samples=total_samples,
            transcript_turns=len(messages),
            latest_expression=latest_expression,
            overall_score=review.overall_score if review else None,
        )