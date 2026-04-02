from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import requests


LBF_MODEL_URL = "https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml"


@dataclass
class EmotionResult:
    emotion: str = "neutral"
    confidence: float = 0.0
    scores: dict[str, float] = field(default_factory=dict)
    face_box: tuple[int, int, int, int] | None = None
    landmarks: list[tuple[float, float]] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    status: str = "Looking for a face"


class FaceAnalyzer:
    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.face_cascade = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))
        self.eye_cascade = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_eye_tree_eyeglasses.xml"))
        self.smile_cascade = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_smile.xml"))
        self.facemark = None
        self.model_path = self.model_dir / "lbfmodel.yaml"
        self._prepare_facemark()

    def _prepare_facemark(self) -> None:
        if not hasattr(cv2, "face"):
            return
        if not self.model_path.exists():
            self._download_model()
        if not self.model_path.exists():
            return
        try:
            facemark = cv2.face.createFacemarkLBF()
            facemark.loadModel(str(self.model_path))
            self.facemark = facemark
        except cv2.error:
            self.facemark = None

    def _download_model(self) -> None:
        try:
            response = requests.get(LBF_MODEL_URL, timeout=12)
            response.raise_for_status()
            self.model_path.write_bytes(response.content)
        except requests.RequestException:
            return

    def analyze(self, frame: np.ndarray) -> EmotionResult:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=6, minSize=(140, 140))
        if len(faces) == 0:
            return EmotionResult(status="No face detected")

        face_box = max(faces, key=lambda box: box[2] * box[3])
        x, y, width, height = [int(value) for value in face_box]
        roi_gray = gray[y : y + height, x : x + width]
        eyes = self.eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=7, minSize=(24, 24))
        smiles = self.smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.6, minNeighbors=20, minSize=(36, 18))
        landmarks = self._fit_landmarks(gray, face_box)

        if landmarks:
            metrics = self._metrics_from_landmarks(landmarks)
        else:
            metrics = self._metrics_from_detections(face_box, eyes, smiles)

        scores = self._emotion_scores(metrics)
        emotion = max(scores, key=scores.get)
        confidence = float(scores[emotion])
        status = "Landmark tracking live" if landmarks else "Using fallback feature tracking"
        return EmotionResult(
            emotion=emotion,
            confidence=confidence,
            scores=scores,
            face_box=face_box,
            landmarks=landmarks,
            metrics=metrics,
            status=status,
        )

    def _fit_landmarks(self, gray: np.ndarray, face_box: Iterable[int]) -> list[tuple[float, float]]:
        if self.facemark is None:
            return []
        rects = np.array([face_box], dtype=np.int32)
        try:
            ok, result = self.facemark.fit(gray, rects)
        except cv2.error:
            return []
        if not ok or len(result) == 0:
            return []
        points = result[0][0]
        return [(float(px), float(py)) for px, py in points]

    @staticmethod
    def _distance(points: list[tuple[float, float]], left: int, right: int) -> float:
        point_a = np.array(points[left])
        point_b = np.array(points[right])
        return float(np.linalg.norm(point_a - point_b))

    def _metrics_from_landmarks(self, points: list[tuple[float, float]]) -> dict[str, float]:
        face_width = max(self._distance(points, 0, 16), 1.0)
        left_eye_open = self._distance(points, 37, 41) + self._distance(points, 38, 40)
        right_eye_open = self._distance(points, 43, 47) + self._distance(points, 44, 46)
        left_eye_width = max(self._distance(points, 36, 39), 1.0)
        right_eye_width = max(self._distance(points, 42, 45), 1.0)
        eye_open = ((left_eye_open / left_eye_width) + (right_eye_open / right_eye_width)) / 2.0
        mouth_width = self._distance(points, 48, 54) / face_width
        mouth_open = self._distance(points, 62, 66) / face_width
        brow_raise_left = ((points[37][1] + points[38][1]) / 2.0 - (points[19][1] + points[20][1]) / 2.0) / face_width
        brow_raise_right = ((points[43][1] + points[44][1]) / 2.0 - (points[23][1] + points[24][1]) / 2.0) / face_width
        brow_raise = (brow_raise_left + brow_raise_right) / 2.0
        inner_brow_gap = self._distance(points, 21, 22) / face_width
        mouth_center_y = (points[51][1] + points[57][1]) / 2.0
        mouth_corner_y = (points[48][1] + points[54][1]) / 2.0
        smile_curve = (mouth_center_y - mouth_corner_y) / face_width
        mouth_drop = (mouth_corner_y - mouth_center_y) / face_width
        return {
            "eye_open": eye_open,
            "mouth_width": mouth_width,
            "mouth_open": mouth_open,
            "brow_raise": brow_raise,
            "brow_furrow": max(0.0, 0.22 - inner_brow_gap),
            "smile_curve": smile_curve,
            "mouth_drop": max(0.0, mouth_drop),
        }

    def _metrics_from_detections(
        self,
        face_box: tuple[int, int, int, int],
        eyes: np.ndarray,
        smiles: np.ndarray,
    ) -> dict[str, float]:
        _, _, width, height = face_box
        eye_factor = min(len(eyes), 2) / 2.0
        smile_count = len(smiles)
        biggest_smile = max((sw * sh for _, _, sw, sh in smiles), default=0)
        smile_ratio = biggest_smile / max(width * height, 1)
        return {
            "eye_open": 0.18 + (eye_factor * 0.12),
            "mouth_width": 0.22 + (smile_ratio * 2.2),
            "mouth_open": 0.04 + (smile_ratio * 1.8),
            "brow_raise": 0.045 + (eye_factor * 0.02),
            "brow_furrow": 0.05 if eye_factor < 0.5 else 0.01,
            "smile_curve": 0.11 if smile_count else 0.01,
            "mouth_drop": 0.08 if smile_count == 0 else 0.01,
        }

    def _emotion_scores(self, metrics: dict[str, float]) -> dict[str, float]:
        happy = self._clamp(
            0.18
            + metrics["smile_curve"] * 4.5
            + max(metrics["mouth_width"] - 0.34, 0.0) * 2.2
            + max(metrics["mouth_open"] - 0.03, 0.0) * 1.5
        )
        surprise = self._clamp(
            max(metrics["mouth_open"] - 0.03, 0.0) * 6.3
            + max(metrics["eye_open"] - 0.3, 0.0) * 2.8
            + max(metrics["brow_raise"] - 0.055, 0.0) * 18.0
            - metrics["smile_curve"] * 0.8
        )
        angry = self._clamp(
            metrics["brow_furrow"] * 7.5
            + max(0.055 - metrics["brow_raise"], 0.0) * 10.0
            + max(0.27 - metrics["eye_open"], 0.0) * 1.4
            + max(0.05 - metrics["mouth_open"], 0.0) * 2.0
        )
        sad = self._clamp(
            metrics["mouth_drop"] * 7.5
            + max(0.31 - metrics["eye_open"], 0.0) * 1.8
            + max(0.05 - metrics["brow_raise"], 0.0) * 6.0
        )
        neutral = self._clamp(0.32 + max(0.22 - max(happy, surprise, angry, sad), 0.0) * 1.6)
        scores = {
            "happy": happy,
            "sad": sad,
            "angry": angry,
            "surprise": surprise,
            "neutral": neutral,
        }
        total = sum(scores.values()) or 1.0
        return {emotion: value / total for emotion, value in scores.items()}

    @staticmethod
    def _clamp(value: float, lower: float = 0.02, upper: float = 1.0) -> float:
        return float(max(lower, min(value, upper)))