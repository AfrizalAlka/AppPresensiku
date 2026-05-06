import threading
import time
from dataclasses import dataclass
from typing import Optional
from datetime import datetime
from pathlib import Path

import cv2

from config import AppConfig
from services.attendance import AttendanceService
from services.inference import FacePredictor


@dataclass
class CameraState:
    running: bool
    last_identity: Optional[str]
    last_confidence: float
    last_message: str


class CameraAttendanceRunner:
    def __init__(
        self,
        config: AppConfig,
        predictor: FacePredictor,
        attendance_service: AttendanceService,
    ) -> None:
        self.config = config
        self.predictor = predictor
        self.attendance_service = attendance_service
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_identity: Optional[str] = None
        self._last_confidence: float = 0.0
        self._last_message: str = "idle"

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def state(self) -> CameraState:
        return CameraState(
            running=self._running,
            last_identity=self._last_identity,
            last_confidence=self._last_confidence,
            last_message=self._last_message,
        )

    def _save_camera_frame(self, frame: cv2.Mat) -> str:
        """Save camera frame to disk and return relative path.
        
        Returns: Relative path like 'attendance_pictures/camera_2026-02-05_14-30-45_123.jpg'
        """
        picture_dir = self.config.project_root / "attendance_pictures"
        picture_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
        filename = f"camera_{timestamp}.jpg"
        filepath = picture_dir / filename
        
        cv2.imwrite(str(filepath), frame)
        
        return f"attendance_pictures/{filename}"

    def _run_loop(self) -> None:
        cap = cv2.VideoCapture(self.config.camera_id)
        if not cap.isOpened():
            self._last_message = "camera open failed"
            self._running = False
            return

        last_action_ts = 0.0

        try:
            while self._running:
                ok, frame = cap.read()
                if not ok:
                    self._last_message = "frame read failed"
                    time.sleep(0.2)
                    continue

                prediction = self.predictor.predict(frame)
                if prediction is None:
                    self._last_message = "no face detected"
                    continue

                self._last_identity = prediction.recognized_name
                self._last_confidence = prediction.confidence

                if prediction.confidence < self.config.threshold:
                    self._last_message = "low confidence"
                    continue

                now_ts = time.time()
                if now_ts - last_action_ts < self.config.camera_check_interval_sec:
                    continue

                # Save the frame before marking attendance
                picture_filename = self._save_camera_frame(frame)

                result = self.attendance_service.mark_attendance(
                    recognized_name=prediction.recognized_name,
                    confidence=prediction.confidence,
                    source="camera",
                    picture_filename=picture_filename,
                )
                self._last_message = result.message
                last_action_ts = now_ts
        finally:
            cap.release()

