import csv
import sys
import json
import subprocess
from pathlib import Path


def test_private_acceptance_reports_window_recall_and_timestamp_violations(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations.tsv"
    annotations.write_text(
        "questions\t\t00:00:01.000\t00:00:02.000\t00:00:01.000\tone\n"
        "questions\t\t00:00:04.000\t00:00:05.000\t00:00:01.000\ttwo\n",
        encoding="utf-8",
    )
    transcript = tmp_path / "transcript.csv"
    with transcript.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["start_time", "end_time", "text"], delimiter="\t")
        writer.writeheader()
        writer.writerow({"start_time": "00:00:01.500", "end_time": "00:00:02.500", "text": "covered"})
        writer.writerow({"start_time": "00:00:02.000", "end_time": "00:00:03.000", "text": "overlap"})

    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_private_acceptance.py",
            "--annotations",
            str(annotations),
            "--transcript",
            str(transcript),
            "--duration-seconds",
            "10",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "covered_windows": 1,
        "output_segments": 2,
        "reference_windows": 2,
        "timestamp_violations": 1,
        "window_recall": 0.5,
    }
