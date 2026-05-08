"""GUI for camera-based attendance system using OpenCV."""

import cv2
import threading
import time
from datetime import datetime
from typing import Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import requests
from io import BytesIO

from config import AppConfig
from services.inference import FacePredictor
from services.attendance import AttendanceService
from services.attendance_utils import calculate_attendance_status, format_attendance_status


@dataclass
class DetectionResult:
    """Result dari face detection."""
    name: Optional[str]
    confidence: float
    status: Optional[str]  # 'tepat_waktu', 'terlambat', 'no_face', 'low_confidence', 'unknown'
    message: str
    timestamp: datetime


class CameraAttendanceGUI:
    """GUI untuk camera attendance dengan display face detection."""
    
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
        self._current_result: Optional[DetectionResult] = None
        self._frame_count = 0
        
        # UI settings
        self.window_name = "Sistem Presensi - Face Recognition"
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.7
        self.thickness = 2
        self.color_white = (255, 255, 255)
        self.color_green = (0, 255, 0)
        self.color_red = (0, 0, 255)
        self.color_yellow = (0, 255, 255)
        self.color_gray = (128, 128, 128)

    def start(self) -> None:
        """Start camera GUI."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=False)
        self._thread.start()

    def stop(self) -> None:
        """Stop camera GUI."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        
        cv2.destroyAllWindows()

    def _run_loop(self) -> None:
        """Main loop untuk camera capture dan display."""
        cap = cv2.VideoCapture(self.config.camera_id)
        
        if not cap.isOpened():
            print("❌ Gagal membuka camera")
            self._running = False
            return
        
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        last_detection_time = 0.0
        
        try:
            while self._running:
                ok, frame = cap.read()
                if not ok:
                    print("❌ Gagal membaca frame dari camera")
                    break
                
                # Horizontal flip untuk mirror effect
                frame = cv2.flip(frame, 1)
                
                # Get prediction
                prediction = self.predictor.predict(frame)
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                
                # Process detection result
                if prediction is None:
                    self._current_result = DetectionResult(
                        name=None,
                        confidence=0.0,
                        status="no_face",
                        message="Wajah tidak terdeteksi",
                        timestamp=now,
                    )
                elif prediction.confidence < self.config.threshold:
                    self._current_result = DetectionResult(
                        name=prediction.recognized_name,
                        confidence=prediction.confidence,
                        status="low_confidence",
                        message=f"Confidence rendah: {prediction.confidence:.2f}",
                        timestamp=now,
                    )
                else:
                    # Check if can mark attendance
                    can_attend, check_msg, student_id, student_name = (
                        self.attendance_service.can_mark_attendance(
                            recognized_name=prediction.recognized_name
                        )
                    )
                    
                    if not can_attend:
                        self._current_result = DetectionResult(
                            name=student_name,
                            confidence=prediction.confidence,
                            status="duplicate",
                            message=check_msg,
                            timestamp=now,
                        )
                    elif time.time() - last_detection_time >= self.config.camera_check_interval_sec:
                        # Calculate attendance status
                        attendance_status = calculate_attendance_status(
                            attendance_time=now,
                            cutoff_hour=self.config.attendance_cutoff_hour,
                            cutoff_minute=self.config.attendance_cutoff_minute,
                        )
                        
                        # Save frame
                        picture_filename = self._save_camera_frame(frame)
                        
                        # Mark attendance
                        result = self.attendance_service.mark_attendance(
                            recognized_name=prediction.recognized_name,
                            confidence=prediction.confidence,
                            source="camera",
                            picture_filename=picture_filename,
                            status=attendance_status,
                        )
                        
                        self._current_result = DetectionResult(
                            name=result.student_name,
                            confidence=prediction.confidence,
                            status=attendance_status if result.status == "marked" else "failed",
                            message=result.message,
                            timestamp=now,
                        )
                        
                        last_detection_time = time.time()
                    else:
                        self._current_result = DetectionResult(
                            name=prediction.recognized_name,
                            confidence=prediction.confidence,
                            status="waiting",
                            message="Tunggu untuk deteksi berikutnya...",
                            timestamp=now,
                        )
                
                # Draw frame dengan informasi
                frame = self._draw_frame_info(frame, current_time)
                
                # Display
                cv2.imshow(self.window_name, frame)
                
                # Handle key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # q or ESC
                    break
                
                self._frame_count += 1
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self._running = False
            print("✓ Camera attendance GUI ditutup")

    def _draw_frame_info(self, frame: cv2.Mat, current_time: str) -> cv2.Mat:
        """Draw frame info: detection info, timestamp, status."""
        h, w = frame.shape[:2]
        
        # Draw background panel di bagian atas
        cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)
        cv2.putText(
            frame,
            "SISTEM PRESENSI - FACE RECOGNITION",
            (20, 30),
            self.font,
            1.0,
            self.color_white,
            2,
        )
        cv2.putText(
            frame,
            f"Waktu: {current_time}",
            (20, 60),
            self.font,
            0.7,
            self.color_yellow,
            1,
        )
        
        # Draw detection info
        if self._current_result:
            result = self._current_result
            
            # Determine colors based on status
            if result.status == "tepat_waktu":
                color = self.color_green
                status_text = "✓ TEPAT WAKTU"
            elif result.status == "terlambat":
                color = self.color_red
                status_text = "⚠ TERLAMBAT"
            elif result.status == "no_face":
                color = self.color_gray
                status_text = "⚪ TIDAK ADA WAJAH"
            elif result.status == "low_confidence":
                color = self.color_yellow
                status_text = "⚠ CONFIDENCE RENDAH"
            elif result.status == "duplicate":
                color = self.color_yellow
                status_text = "⚠ SUDAH ABSEN"
            elif result.status == "waiting":
                color = self.color_gray
                status_text = "⏳ MENUNGGU..."
            else:
                color = self.color_gray
                status_text = result.status.upper()
            
            # Draw bottom panel dengan info deteksi
            panel_height = 150
            cv2.rectangle(frame, (0, h - panel_height), (w, h), (0, 0, 0), -1)
            
            y_offset = h - panel_height + 30
            
            # Name
            if result.name:
                cv2.putText(
                    frame,
                    f"Nama: {result.name}",
                    (20, y_offset),
                    self.font,
                    self.font_scale,
                    self.color_white,
                    self.thickness,
                )
                y_offset += 35
            
            # Confidence
            if result.confidence > 0:
                cv2.putText(
                    frame,
                    f"Confidence: {result.confidence:.2%}",
                    (20, y_offset),
                    self.font,
                    self.font_scale,
                    self.color_white,
                    self.thickness,
                )
                y_offset += 35
            
            # Status
            cv2.putText(
                frame,
                status_text,
                (20, y_offset),
                self.font,
                self.font_scale,
                color,
                self.thickness,
            )
            
            # Message
            y_offset += 35
            cv2.putText(
                frame,
                result.message[:60],
                (20, y_offset),
                self.font,
                0.6,
                color,
                1,
            )
        
        # Draw frame counter di corner
        cv2.putText(
            frame,
            f"Frame: {self._frame_count}",
            (w - 200, 30),
            self.font,
            0.6,
            self.color_gray,
            1,
        )
        
        return frame

    def _save_camera_frame(self, frame: cv2.Mat) -> str:
        """Save camera frame to Laravel storage or local disk.
        
        Returns: Relative path
        """
        try:
            # Generate filename
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
            filename = f"camera_{timestamp}.jpg"
            
            # Encode frame to JPG
            success, jpg_buffer = cv2.imencode(".jpg", frame)
            if not success:
                raise ValueError("Gagal encode gambar")
            
            jpg_bytes = jpg_buffer.tobytes()
            
            # Upload to Laravel via API
            url = f"{self.config.laravel_url}/api/attendance/upload-picture"
            
            files = {"image": (filename, BytesIO(jpg_bytes), "image/jpeg")}
            data = {"storage_path": self.config.laravel_attendance_pictures_path}
            
            response = requests.post(url, files=files, data=data, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                return result.get("path", f"{self.config.laravel_attendance_pictures_path}/{filename}")
            else:
                # Fallback to local
                return self._save_camera_frame_locally(frame, filename)
        
        except Exception as e:
            # Fallback to local
            return self._save_camera_frame_locally(frame)
    
    def _save_camera_frame_locally(self, frame: cv2.Mat, filename: str = None) -> str:
        """Fallback: Save camera frame to local disk."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
            filename = f"camera_{timestamp}.jpg"
        
        picture_dir = self.config.project_root / "attendance_pictures"
        picture_dir.mkdir(exist_ok=True)
        
        filepath = picture_dir / filename
        cv2.imwrite(str(filepath), frame)
        return f"attendance_pictures/{filename}"
