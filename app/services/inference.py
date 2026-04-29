import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tensorflow as tf


@dataclass
class PredictionResult:
    recognized_name: str
    confidence: float
    face_box: tuple[int, int, int, int]


class FacePredictor:
    def __init__(self, model_path: Path, class_names_path: Path, image_size: int) -> None:
        self.model_path = model_path
        self.class_names_path = class_names_path
        self.image_size = image_size
        self.model: Optional[tf.keras.Model] = None
        self.class_names: list[str] = []
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if self.face_cascade.empty():
            raise RuntimeError("Gagal memuat Haar Cascade.")

    def load(self) -> None:
        if self.model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model belum ditemukan: {self.model_path}")
            if not self.class_names_path.exists():
                raise FileNotFoundError(f"Class names belum ditemukan: {self.class_names_path}")

            self.model = tf.keras.models.load_model(self.model_path)
            with open(self.class_names_path, "r", encoding="utf-8") as f:
                self.class_names = json.load(f)

    def detect_largest_face(self, frame_bgr: np.ndarray) -> Optional[tuple[int, int, int, int]]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        return int(x), int(y), int(w), int(h)

    def predict(self, frame_bgr: np.ndarray) -> Optional[PredictionResult]:
        self.load()

        face_box = self.detect_largest_face(frame_bgr)
        if face_box is None:
            return None

        x, y, w, h = face_box
        crop = frame_bgr[y : y + h, x : x + w]
        resized = cv2.resize(crop, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        x_input = np.expand_dims(rgb.astype(np.float32) / 255.0, axis=0)
        probs = self.model.predict(x_input, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])

        return PredictionResult(
            recognized_name=self.class_names[pred_idx],
            confidence=confidence,
            face_box=face_box,
        )
