import sys
import copy
import json
import hashlib
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def quality_evaluation_files(tmp_path: Path) -> dict[str, object]:
    """Create one complete scoring case without invoking a model."""
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"audio fixture")
    audio_sha256 = hashlib.sha256(audio.read_bytes()).hexdigest()

    manifest_case = {
        "id": "speech-case",
        "audio": audio.name,
        "sha256": audio_sha256,
        "duration_ms": 1000,
        "language": "en",
        "split": "acceptance",
        "strata": ["speech", "short_utterance"],
        "source": {"url": "https://example.test/audio", "item": "test audio", "revision": "1"},
        "license": {
            "id": "CC0-1.0",
            "url": "https://creativecommons.org/publicdomain/zero/1.0/",
            "attribution": "Test fixture",
        },
        "reference": {"text": "hello world", "speech_intervals_ms": [[100, 900]]},
    }
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(manifest_case) + "\n", encoding="utf-8")

    protocol_data = {
        "schema_version": 1,
        "random_seed": 0,
        "model": {
            "repository": "openai/whisper-large-v3",
            "revision": "06f233fe06e710322aca913c1bc4249a0d71fce1",
        },
        "audio_classifier": {
            "repository": "MIT/ast-finetuned-audioset-10-10-0.4593",
            "revision": "f826b80d28226b62986cc218e5cec390b1096902",
        },
        "pipeline": {},
        "normalization": {
            "unicode_form": "NFKC",
            "casefold": True,
            "punctuation_to_space": True,
            "collapse_whitespace": True,
        },
        "boundary_collar_ms": 100,
        "short_case_max_duration_ms": 1000,
        "gates": {
            "timestamp_violation_cases": {"enabled": True, "max": 0},
            "wer": {"enabled": True, "max": 1.0},
        },
    }
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(protocol_data), encoding="utf-8")

    prediction_data = {
        "schema_version": 1,
        "run": {
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
            "random_seed": protocol_data["random_seed"],
            "source": {"git_revision": "0" * 40, "git_dirty": False},
            "model": protocol_data["model"],
            "audio_classifier": protocol_data["audio_classifier"],
            "resolved_device": "cpu",
            "whisper_batch_size": 1,
            "environment": {"python": "test"},
        },
        "cases": [
            {
                "id": manifest_case["id"],
                "audio_sha256": audio_sha256,
                "segments": [{"start_ms": 50, "end_ms": 950, "text": "hello brave world"}],
                "error": None,
                "elapsed_seconds": 1.0,
                "rtf": 1.0,
                "peak_rss_bytes": 1048576,
                "peak_cuda_bytes": None,
            }
        ],
    }
    predictions = tmp_path / "predictions.json"
    evaluator = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_audio_quality.py"
    prediction_data["run"]["evaluator_sha256"] = hashlib.sha256(evaluator.read_bytes()).hexdigest()
    predictions.write_text(json.dumps(prediction_data), encoding="utf-8")

    return {
        "manifest": manifest,
        "manifest_case": manifest_case,
        "protocol": protocol,
        "protocol_data": protocol_data,
        "predictions": predictions,
        "prediction_data": prediction_data,
    }


def test_score_reports_text_boundary_and_activity_metrics(
    quality_evaluation_files: dict[str, object],
    tmp_path: Path,
) -> None:
    """Score exact edit counts and interval metrics through the CLI."""
    report = tmp_path / "report.md"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_audio_quality.py",
            "score",
            "--manifest",
            str(quality_evaluation_files["manifest"]),
            "--protocol",
            str(quality_evaluation_files["protocol"]),
            "--predictions",
            str(quality_evaluation_files["predictions"]),
            "--output",
            str(report),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    markdown = report.read_text(encoding="utf-8")
    assert "| WER | 0.5000 |" in markdown
    assert "| Word insertions | 1 |" in markdown
    assert "| False-alarm rate | 0.5000 |" in markdown
    assert "| Boundary median error (ms) | 50.0000 |" in markdown
    assert "| Timestamp violations | 0 |" in markdown
    assert "| Output segments | 1 |" in markdown
    assert "| speech-case | 1 | 17.0000 | 17.0000 | 17 | 900.0000 | 900.0000 | 900 |" in markdown


def test_score_marks_missing_activity_reference_unavailable(
    quality_evaluation_files: dict[str, object],
    tmp_path: Path,
) -> None:
    """Do not report zero activity error when no interval reference exists."""
    manifest = quality_evaluation_files["manifest"]
    manifest_case = quality_evaluation_files["manifest_case"]
    predictions = quality_evaluation_files["predictions"]
    prediction_data = quality_evaluation_files["prediction_data"]
    assert isinstance(manifest, Path)
    assert isinstance(manifest_case, dict)
    assert isinstance(predictions, Path)
    assert isinstance(prediction_data, dict)

    manifest_case["reference"]["speech_intervals_ms"] = None
    manifest.write_text(json.dumps(manifest_case) + "\n", encoding="utf-8")
    prediction_data["run"]["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    predictions.write_text(json.dumps(prediction_data), encoding="utf-8")

    report = tmp_path / "missing-activity-report.md"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_audio_quality.py",
            "score",
            "--manifest",
            str(manifest),
            "--protocol",
            str(quality_evaluation_files["protocol"]),
            "--predictions",
            str(predictions),
            "--output",
            str(report),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    case_row = next(
        line
        for line in reversed(report.read_text(encoding="utf-8").splitlines())
        if line.startswith("| speech-case |")
    )
    cells = [cell.strip() for cell in case_row.split("|")[1:-1]]
    assert cells[10:14] == ["—", "—", "—", "—"]


def test_short_exact_match_gate_covers_speech_inside_long_audio(
    quality_evaluation_files: dict[str, object],
    tmp_path: Path,
) -> None:
    manifest = quality_evaluation_files["manifest"]
    manifest_case = quality_evaluation_files["manifest_case"]
    protocol = quality_evaluation_files["protocol"]
    protocol_data = quality_evaluation_files["protocol_data"]
    predictions = quality_evaluation_files["predictions"]
    prediction_data = quality_evaluation_files["prediction_data"]
    assert isinstance(manifest, Path)
    assert isinstance(manifest_case, dict)
    assert isinstance(protocol, Path)
    assert isinstance(protocol_data, dict)
    assert isinstance(predictions, Path)
    assert isinstance(prediction_data, dict)

    manifest_case["duration_ms"] = 10_000
    manifest_case["reference"]["speech_intervals_ms"] = [[5000, 5500]]
    manifest.write_text(json.dumps(manifest_case) + "\n", encoding="utf-8")
    protocol_data["gates"] = {"short_exact_match_rate": {"enabled": True, "min": 1.0}}
    protocol.write_text(json.dumps(protocol_data), encoding="utf-8")
    prediction_data["cases"][0]["segments"] = []
    prediction_data["run"]["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    prediction_data["run"]["protocol_sha256"] = hashlib.sha256(protocol.read_bytes()).hexdigest()
    predictions.write_text(json.dumps(prediction_data), encoding="utf-8")

    report = tmp_path / "embedded-short-speech-report.md"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_audio_quality.py",
            "score",
            "--manifest",
            str(manifest),
            "--protocol",
            str(protocol),
            "--predictions",
            str(predictions),
            "--output",
            str(report),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1, result.stderr
    assert "| `short_exact_match_rate` | min 1.0000 | 0.0000 | FAIL |" in report.read_text(encoding="utf-8")


def test_score_returns_failure_for_an_enabled_gate(
    quality_evaluation_files: dict[str, object],
    tmp_path: Path,
) -> None:
    """Return a failing status when a versioned release gate is missed."""
    protocol = quality_evaluation_files["protocol"]
    protocol_data = quality_evaluation_files["protocol_data"]
    assert isinstance(protocol, Path)
    assert isinstance(protocol_data, dict)
    protocol_data["gates"]["wer"]["max"] = 0.0
    protocol.write_text(json.dumps(protocol_data), encoding="utf-8")

    predictions = quality_evaluation_files["predictions"]
    prediction_data = quality_evaluation_files["prediction_data"]
    assert isinstance(predictions, Path)
    assert isinstance(prediction_data, dict)
    prediction_data["run"]["protocol_sha256"] = hashlib.sha256(protocol.read_bytes()).hexdigest()
    predictions.write_text(json.dumps(prediction_data), encoding="utf-8")

    report = tmp_path / "failed-report.md"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_audio_quality.py",
            "score",
            "--manifest",
            str(quality_evaluation_files["manifest"]),
            "--protocol",
            str(protocol),
            "--predictions",
            str(predictions),
            "--output",
            str(report),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1, result.stderr
    assert "| `wer` | max 0.0000 | 0.5000 | FAIL |" in report.read_text(encoding="utf-8")


def test_score_excludes_tuning_cases_from_release_gates(
    quality_evaluation_files: dict[str, object],
    tmp_path: Path,
) -> None:
    """Evaluate release gates against held-out acceptance cases only."""
    manifest = quality_evaluation_files["manifest"]
    manifest_case = quality_evaluation_files["manifest_case"]
    protocol = quality_evaluation_files["protocol"]
    protocol_data = quality_evaluation_files["protocol_data"]
    predictions = quality_evaluation_files["predictions"]
    prediction_data = quality_evaluation_files["prediction_data"]
    assert isinstance(manifest, Path)
    assert isinstance(manifest_case, dict)
    assert isinstance(protocol, Path)
    assert isinstance(protocol_data, dict)
    assert isinstance(predictions, Path)
    assert isinstance(prediction_data, dict)

    tuning_case = copy.deepcopy(manifest_case)
    tuning_case["id"] = "tuning-case"
    tuning_case["split"] = "tuning"
    manifest.write_text(
        json.dumps(manifest_case) + "\n" + json.dumps(tuning_case) + "\n",
        encoding="utf-8",
    )
    protocol_data["gates"]["wer"]["max"] = 0.0
    protocol.write_text(json.dumps(protocol_data), encoding="utf-8")

    prediction_data["run"]["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    prediction_data["run"]["protocol_sha256"] = hashlib.sha256(protocol.read_bytes()).hexdigest()
    prediction_data["cases"][0]["segments"][0]["text"] = "hello world"
    tuning_prediction = copy.deepcopy(prediction_data["cases"][0])
    tuning_prediction["id"] = tuning_case["id"]
    tuning_prediction["segments"][0]["text"] = "wrong words with extra insertions"
    prediction_data["cases"].append(tuning_prediction)
    predictions.write_text(json.dumps(prediction_data), encoding="utf-8")

    report = tmp_path / "acceptance-gates.md"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_audio_quality.py",
            "score",
            "--manifest",
            str(manifest),
            "--protocol",
            str(protocol),
            "--predictions",
            str(predictions),
            "--output",
            str(report),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    markdown = report.read_text(encoding="utf-8")
    assert "Gates evaluate only manifest rows with `split=acceptance`." in markdown
    assert "| `wer` | max 0.0000 | 0.0000 | PASS |" in markdown


def test_score_rejects_predictions_with_a_different_random_seed(
    quality_evaluation_files: dict[str, object],
    tmp_path: Path,
) -> None:
    """Reject predictions produced with a different sampling seed."""
    predictions = quality_evaluation_files["predictions"]
    prediction_data = quality_evaluation_files["prediction_data"]
    assert isinstance(predictions, Path)
    assert isinstance(prediction_data, dict)
    prediction_data["run"]["random_seed"] = 1
    predictions.write_text(json.dumps(prediction_data), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_audio_quality.py",
            "score",
            "--manifest",
            str(quality_evaluation_files["manifest"]),
            "--protocol",
            str(quality_evaluation_files["protocol"]),
            "--predictions",
            str(predictions),
            "--output",
            str(tmp_path / "invalid-seed.md"),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Predictions random_seed does not match the protocol" in result.stderr


def test_score_rejects_duplicate_manifest_ids(
    quality_evaluation_files: dict[str, object],
    tmp_path: Path,
) -> None:
    """Reject an ambiguous corpus before any metrics are produced."""
    manifest = quality_evaluation_files["manifest"]
    manifest_case = quality_evaluation_files["manifest_case"]
    assert isinstance(manifest, Path)
    manifest.write_text(
        json.dumps(manifest_case) + "\n" + json.dumps(manifest_case) + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_audio_quality.py",
            "score",
            "--manifest",
            str(manifest),
            "--protocol",
            str(quality_evaluation_files["protocol"]),
            "--predictions",
            str(quality_evaluation_files["predictions"]),
            "--output",
            str(tmp_path / "invalid.md"),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Duplicate manifest id: speech-case" in result.stderr


def test_infer_explains_how_to_materialize_lfs_audio(
    quality_evaluation_files: dict[str, object],
    tmp_path: Path,
) -> None:
    """Reject an unresolved LFS pointer before model inference."""
    manifest_case = quality_evaluation_files["manifest_case"]
    assert isinstance(manifest_case, dict)
    audio = tmp_path / str(manifest_case["audio"])
    audio.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:0000000000000000000000000000000000000000000000000000000000000000\n"
        "size 1\n",
        encoding="utf-8",
    )

    protocol_data = quality_evaluation_files["protocol_data"]
    assert isinstance(protocol_data, dict)
    model_snapshot = tmp_path / protocol_data["model"]["revision"]
    model_snapshot.mkdir()

    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_audio_quality.py",
            "infer",
            "--manifest",
            str(quality_evaluation_files["manifest"]),
            "--protocol",
            str(quality_evaluation_files["protocol"]),
            "--model-snapshot",
            str(model_snapshot),
            "--device",
            "cpu",
            "--batch-size",
            "1",
            "--output",
            str(tmp_path / "predictions.json"),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "is a Git LFS pointer" in result.stderr
    assert "git lfs pull" in result.stderr


def test_score_preserves_empty_reference_insertion_counts(
    quality_evaluation_files: dict[str, object],
    tmp_path: Path,
) -> None:
    """Keep hallucination counts while leaving empty-reference rates undefined."""
    manifest = quality_evaluation_files["manifest"]
    manifest_case = quality_evaluation_files["manifest_case"]
    protocol = quality_evaluation_files["protocol"]
    protocol_data = quality_evaluation_files["protocol_data"]
    predictions = quality_evaluation_files["predictions"]
    prediction_data = quality_evaluation_files["prediction_data"]
    assert isinstance(manifest, Path)
    assert isinstance(manifest_case, dict)
    assert isinstance(protocol, Path)
    assert isinstance(protocol_data, dict)
    assert isinstance(predictions, Path)
    assert isinstance(prediction_data, dict)

    manifest_case["reference"] = {"text": "", "speech_intervals_ms": []}
    manifest.write_text(json.dumps(manifest_case) + "\n", encoding="utf-8")
    protocol_data["gates"]["wer"]["enabled"] = False
    protocol.write_text(json.dumps(protocol_data), encoding="utf-8")
    prediction_data["run"]["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    prediction_data["run"]["protocol_sha256"] = hashlib.sha256(protocol.read_bytes()).hexdigest()
    prediction_data["cases"][0]["segments"] = []
    predictions.write_text(json.dumps(prediction_data), encoding="utf-8")

    empty_report = tmp_path / "empty-report.md"
    empty_result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_audio_quality.py",
            "score",
            "--manifest",
            str(manifest),
            "--protocol",
            str(protocol),
            "--predictions",
            str(predictions),
            "--output",
            str(empty_report),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert empty_result.returncode == 0, empty_result.stderr
    empty_markdown = empty_report.read_text(encoding="utf-8")
    assert "| WER | — |" in empty_markdown
    assert "| Word insertions | 0 |" in empty_markdown
    assert "| CER | — |" in empty_markdown

    prediction_data["cases"][0]["segments"] = [{"start_ms": 100, "end_ms": 900, "text": "hallucinated words"}]
    predictions.write_text(json.dumps(prediction_data), encoding="utf-8")
    hallucination_report = tmp_path / "hallucination-report.md"
    hallucination_result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_audio_quality.py",
            "score",
            "--manifest",
            str(manifest),
            "--protocol",
            str(protocol),
            "--predictions",
            str(predictions),
            "--output",
            str(hallucination_report),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert hallucination_result.returncode == 0, hallucination_result.stderr
    hallucination_markdown = hallucination_report.read_text(encoding="utf-8")
    assert "| WER | — |" in hallucination_markdown
    assert "| Word insertions | 2 |" in hallucination_markdown
    assert "| Non-speech nonempty cases | 1 |" in hallucination_markdown
    assert "| Boundary precision | 0.0000 |" in hallucination_markdown


def test_score_reports_known_substitution_deletion_and_insertion(
    quality_evaluation_files: dict[str, object],
    tmp_path: Path,
) -> None:
    """Expose JiWER's exact operation counts through the report."""
    manifest = quality_evaluation_files["manifest"]
    manifest_case = quality_evaluation_files["manifest_case"]
    predictions = quality_evaluation_files["predictions"]
    prediction_data = quality_evaluation_files["prediction_data"]
    assert isinstance(manifest, Path)
    assert isinstance(manifest_case, dict)
    assert isinstance(predictions, Path)
    assert isinstance(prediction_data, dict)

    manifest_case["reference"]["text"] = "a b c d e f g h"
    manifest.write_text(json.dumps(manifest_case) + "\n", encoding="utf-8")
    prediction_data["run"]["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    prediction_data["cases"][0]["segments"] = [{"start_ms": 100, "end_ms": 900, "text": "a x c d z e f h"}]
    predictions.write_text(json.dumps(prediction_data), encoding="utf-8")

    report = tmp_path / "edit-counts-report.md"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_audio_quality.py",
            "score",
            "--manifest",
            str(manifest),
            "--protocol",
            str(quality_evaluation_files["protocol"]),
            "--predictions",
            str(predictions),
            "--output",
            str(report),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    markdown = report.read_text(encoding="utf-8")
    assert "| Word substitutions | 1 |" in markdown
    assert "| Word deletions | 1 |" in markdown
    assert "| Word insertions | 1 |" in markdown
