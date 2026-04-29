from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from database import Database


@dataclass
class AttendanceResult:
    status: str
    message: str
    attendance_id: Optional[int]
    student_id: Optional[int]


class AttendanceService:
    def __init__(self, db: Database) -> None:
        self.db = db

    def mark_attendance(self, recognized_name: str, confidence: float, source: str) -> AttendanceResult:
        student = self.db.find_student_by_name(recognized_name)
        student_id = int(student["id"]) if student else None

        if self.db.has_attendance_today(student_id=student_id, recognized_name=recognized_name):
            return AttendanceResult(
                status="duplicate",
                message="Absensi hari ini sudah tercatat.",
                attendance_id=None,
                student_id=student_id,
            )

        attendance_id = self.db.insert_attendance(
            student_id=student_id,
            recognized_name=recognized_name,
            confidence=confidence,
            source=source,
            attendance_time=datetime.now(),
        )

        return AttendanceResult(
            status="marked",
            message="Absensi berhasil dicatat.",
            attendance_id=attendance_id,
            student_id=student_id,
        )
