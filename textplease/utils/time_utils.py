def format_time(seconds: float) -> str:
    """Convert float seconds to HH:MM:SS format"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) % 60 // 1)
    s = int(seconds % 60)
    return f"{h:02}:{m:02}:{s:02}"