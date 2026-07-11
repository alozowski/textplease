import math
import logging


logger = logging.getLogger(__name__)


def parse_time_str(time_str: str) -> float:
    """Convert a time string in HH:MM:SS or HH:MM:SS.mmm format to total seconds."""
    try:
        parts = time_str.split(":")
        if len(parts) != 3:
            raise ValueError(f"Time string must be in HH:MM:SS format, got: {time_str}")

        h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
        if h < 0 or m < 0 or s < 0:
            raise ValueError(f"Time components must be non-negative: {time_str}")
        if m >= 60 or s >= 60:
            raise ValueError(f"Minutes and seconds must be < 60: {time_str}")

        return h * 3600 + m * 60 + s
    except (ValueError, IndexError) as e:
        raise ValueError(f"Failed to parse time string '{time_str}': {e}") from e


def format_time_precise(seconds: float | None) -> str:
    """Convert float seconds to HH:MM:SS.mmm format, safe for None/NaN/negative/inf input."""
    if seconds is None or math.isnan(seconds) or seconds < 0 or math.isinf(seconds):
        logger.warning(f"Invalid time value: {seconds}")
        return "00:00:00.000"

    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02}:{m:02}:{s:06.3f}"
