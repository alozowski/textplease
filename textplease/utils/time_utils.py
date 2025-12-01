import math
import logging


logger = logging.getLogger(__name__)


def format_time(seconds: float | None) -> str:
    """Convert float seconds to HH:MM:SS format, safe for None or NaN.

    Args:
        seconds: Time in seconds (can be None or NaN)

    Returns:
        Time string in HH:MM:SS format, or "00:00:00" for invalid input

    """
    # Check for None, NaN, or invalid types
    if seconds is None or not isinstance(seconds, (int, float)):
        logger.warning(f"Invalid time input to format_time: {seconds}")
        return "00:00:00"

    # Check for NaN using math.isnan (more reliable than seconds != seconds)
    if math.isnan(seconds):
        logger.warning("NaN time input to format_time")
        return "00:00:00"

    # Check for negative values
    if seconds < 0:
        logger.warning(f"Negative time input to format_time: {seconds}")
        return "00:00:00"

    # Check for infinite values
    if math.isinf(seconds):
        logger.warning(f"Infinite time input to format_time: {seconds}")
        return "00:00:00"

    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02}:{m:02}:{s:02}"


def parse_time_str(time_str: str) -> float:
    """Convert a time string in HH:MM:SS or HH:MM:SS.mmm format to total seconds.

    Args:
        time_str: Time string in HH:MM:SS or HH:MM:SS.mmm format

    Returns:
        Total seconds as float  # ← Changed from int

    """
    if not time_str or not isinstance(time_str, str):
        raise ValueError(f"Invalid time_str: {time_str}")

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
        raise ValueError(f"Failed to parse time string '{time_str}': {e}")

def format_time_precise(seconds: float | None) -> str:
    """Convert float seconds to HH:MM:SS.mmm format with milliseconds.

    Args:
        seconds: Time in seconds (can be None or NaN)

    Returns:
        Time string in HH:MM:SS.mmm format, or "00:00:00.000" for invalid input

    """
    if seconds is None or not isinstance(seconds, (int, float)):
        logger.warning(f"Invalid time input to format_time_precise: {seconds}")
        return "00:00:00.000"

    if math.isnan(seconds) or seconds < 0 or math.isinf(seconds):
        logger.warning(f"Invalid time value: {seconds}")
        return "00:00:00.000"

    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60  # Keep as float
    return f"{h:02}:{m:02}:{s:06.3f}"
