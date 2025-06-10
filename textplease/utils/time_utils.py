from typing import Optional


def format_time(seconds: Optional[float]) -> str:
    """Convert float seconds to HH:MM:SS format, safe for None"""
    if seconds is None:
        return "??:??:??"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02}:{m:02}:{s:02}"