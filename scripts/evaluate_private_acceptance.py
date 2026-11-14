"""Score a local transcript against private timestamped speech windows."""

import csv
import json
import argparse
from pathlib import Path

from textplease.utils.time_utils import parse_time_str


def _load_windows(path: Path, duration_seconds: float) -> list[tuple[float, float]]:
    """Load ordered positive speech windows from the private annotation TSV."""
    windows: list[tuple[float, float]] = []
    with path.open(encoding="utf-8", newline="") as stream:
        for line_number, row in enumerate(csv.reader(stream, delimiter="\t"), start=1):
            if not row:
                continue
            if len(row) < 4:
                raise ValueError(f"Annotation line {line_number} must have at least four columns")
            start = parse_time_str(row[2])
            end = parse_time_str(row[3])
            if start < (windows[-1][1] if windows else 0) or end <= start or end > duration_seconds:
                raise ValueError(f"Annotation line {line_number} is unordered or outside the audio")
            windows.append((start, end))
    if not windows:
        raise ValueError("Annotation file contains no windows")
    return windows


def _load_segments(path: Path) -> list[tuple[float, float]]:
    """Load nonblank timestamped segments from a TextPlease transcript."""
    segments: list[tuple[float, float]] = []
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {"start_time", "end_time", "text"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("Transcript must contain start_time, end_time, and text columns")
        for row in reader:
            if row["text"].strip():
                segments.append((parse_time_str(row["start_time"]), parse_time_str(row["end_time"])))
    return segments


def score(windows: list[tuple[float, float]], segments: list[tuple[float, float]], duration_seconds: float) -> dict:
    """Return answer-window coverage and transcript timestamp validity."""
    previous_end = 0.0
    timestamp_violations = 0
    valid_segments: list[tuple[float, float]] = []
    for start, end in segments:
        if start < previous_end or end <= start or start < 0 or end > duration_seconds:
            timestamp_violations += 1
        else:
            valid_segments.append((start, end))
        previous_end = max(previous_end, end)

    covered = sum(
        any(segment_end > window_start and segment_start < window_end for segment_start, segment_end in valid_segments)
        for window_start, window_end in windows
    )
    return {
        "reference_windows": len(windows),
        "covered_windows": covered,
        "window_recall": covered / len(windows),
        "output_segments": len(segments),
        "timestamp_violations": timestamp_violations,
    }


def main() -> None:
    """Score one private transcript without copying private data into the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--duration-seconds", type=float, required=True)
    args = parser.parse_args()
    if args.duration_seconds <= 0:
        parser.error("--duration-seconds must be positive")

    result = score(
        _load_windows(args.annotations, args.duration_seconds),
        _load_segments(args.transcript),
        args.duration_seconds,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
