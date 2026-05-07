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
    student_name: Optional[str]
    class_id: Optional[int]


class AttendanceService:
    def __init__(self, db: Database) -> None:
        self.db = db

    def can_mark_attendance(self, recognized_name: str) -> tuple[bool, str, Optional[int], Optional[str]]:
        """Check if a student can mark attendance today (not already attended).
        
        Returns: (can_attend, message, student_id, student_name)
        """
        student = self.db.find_student_by_name(recognized_name)
        
        if not student:
            return False, f"Siswa '{recognized_name}' tidak ditemukan di database.", None, None

        student_id = int(student["id"])
        id_class = int(student["id_class"])
        student_name = student["name"]

        if self.db.has_attendance_today(student_id=student_id, id_class=id_class):
            return False, "Absensi hari ini sudah tercatat untuk kelas ini.", student_id, student_name

        return True, "Siswa bisa absensi.", student_id, student_name

    def mark_attendance(
        self,
        recognized_name: str,
        confidence: float,
        source: str,
        picture_filename: str,
        status: str = "tepat waktu",
    ) -> AttendanceResult:
        """Mark attendance from recognized face.
        
        Args:
        - recognized_name: Student name recognized by model (must match student name in DB)
        - confidence: Model confidence score
        - source: Source of detection ('camera' or 'upload')
        - picture_filename: Filename/path of the picture being processed
        - status: 'tepat waktu' (on-time) or 'terlambat' (late)
        
        Returns: AttendanceResult with details of the attendance record
        """
        student = self.db.find_student_by_name(recognized_name)
        
        if not student:
            return AttendanceResult(
                status="not_found",
                message=f"Siswa '{recognized_name}' tidak ditemukan di database.",
                attendance_id=None,
                student_id=None,
                student_name=None,
                class_id=None,
            )

        student_id = int(student["id"])
        id_class = int(student["id_class"])
        student_name = student["name"]

        if self.db.has_attendance_today(student_id=student_id, id_class=id_class):
            return AttendanceResult(
                status="duplicate",
                message="Absensi hari ini sudah tercatat untuk kelas ini.",
                attendance_id=None,
                student_id=student_id,
                student_name=student_name,
                class_id=id_class,
            )

        attendance_id = self.db.insert_attendance(
            id_student=student_id,
            id_class=id_class,
            picture_filename=picture_filename,
            status=status,
        )

        return AttendanceResult(
            status="marked",
            message=f"Absensi {student_name} berhasil dicatat ({status}).",
            attendance_id=attendance_id,
            student_id=student_id,
            student_name=student_name,
            class_id=id_class,
        )

