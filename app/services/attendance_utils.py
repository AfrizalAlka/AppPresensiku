"""Helper utilities for attendance system."""

from datetime import datetime, time
from typing import Literal


def calculate_attendance_status(
    attendance_time: datetime = None, 
    cutoff_hour: int = 7,
    cutoff_minute: int = 0,
) -> Literal["tepat_waktu", "terlambat"]:
    """Calculate attendance status based on cutoff time.
    
    Args:
        attendance_time: Time of attendance (default: now)
        cutoff_hour: Cutoff hour (default: 7)
        cutoff_minute: Cutoff minute (default: 0)
    
    Returns:
        'tepat_waktu' if before/at cutoff, 'terlambat' if after cutoff
    
    Example:
        >>> status = calculate_attendance_status()  # Uses current time, cutoff 07:00
        >>> status = calculate_attendance_status(cutoff_hour=8, cutoff_minute=30)  # Cutoff 08:30
    """
    if attendance_time is None:
        attendance_time = datetime.now()
    
    cutoff_time = time(cutoff_hour, cutoff_minute, 0)
    current_time = attendance_time.time()
    
    # Jika waktu <= cutoff_time → tepat_waktu
    if current_time <= cutoff_time:
        return "tepat_waktu"
    else:
        return "terlambat"


def format_attendance_status(status: str) -> str:
    """Format status for display.
    
    Args:
        status: 'tepat_waktu' or 'terlambat'
    
    Returns:
        Human-readable status
    """
    status_map = {
        "tepat_waktu": "Tepat Waktu ✓",
        "terlambat": "Terlambat ⚠",
        "izin": "Izin",
    }
    return status_map.get(status, status)
