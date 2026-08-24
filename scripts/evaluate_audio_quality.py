import os
import re
import csv
import json
import math
import time
import hashlib
import argparse
import platform
import tempfile
import threading
import statistics
import subprocess
import unicodedata
from pathlib import Path
from importlib import metadata
from collections.abc import Sequence

import torch
import psutil
from jiwer import process_words, process_characters

from textplease.pipeline import run_transcription_pipeline
from textplease.utils.time_utils import parse_time_str
from textplease.utils.device_utils import detect_device


def _read_json_object(path: Path, label: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    with path.open(encoding="utf-8") as stream:
        try:
            value = json.load(stream)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid {label} JSON at line {error.lineno}, column {error.colno}: {error.msg}"
            ) from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read_manifest(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {path}")

    cases: list[dict] = []
    case_ids: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid manifest JSON at line {line_number}, column {error.colno}: {error.msg}"
                ) from error
            if not isinstance(case, dict):
                raise ValueError(f"Manifest line {line_number} must contain one JSON object")

            required = {
                "id",
                "audio",
                "sha256",
                "duration_ms",
                "language",
                "split",
                "strata",
                "source",
                "license",
                "reference",
            }
            missing = required - case.keys()
            if missing:
                raise ValueError(f"Manifest case at line {line_number} is missing: {', '.join(sorted(missing))}")

            case_id = case["id"]
            if not isinstance(case_id, str) or not case_id.strip():
                raise ValueError(f"Manifest id at line {line_number} must be a nonempty string")
            if case_id in case_ids:
                raise ValueError(f"Duplicate manifest id: {case_id}")
            case_ids.add(case_id)

            audio = case["audio"]
            if not isinstance(audio, str) or not audio.strip() or Path(audio).is_absolute():
                raise ValueError(f"Manifest audio for {case_id} must be a nonempty relative path")

            checksum = case["sha256"]
            if not isinstance(checksum, str) or re.fullmatch(r"[0-9a-f]{64}", checksum) is None:
                raise ValueError(f"Manifest sha256 for {case_id} must be 64 lowercase hexadecimal characters")

            duration_ms = case["duration_ms"]
            if isinstance(duration_ms, bool) or not isinstance(duration_ms, int) or duration_ms <= 0:
                raise ValueError(f"Manifest duration_ms for {case_id} must be a positive integer")

            language = case["language"]
            if not isinstance(language, str) or not language.strip():
                raise ValueError(f"Manifest language for {case_id} must be a nonempty string")
            if case["split"] not in {"tuning", "acceptance"}:
                raise ValueError(f"Manifest split for {case_id} must be 'tuning' or 'acceptance'")

            strata = case["strata"]
            if (
                not isinstance(strata, list)
                or any(not isinstance(stratum, str) or not stratum.strip() for stratum in strata)
                or len(strata) != len(set(strata))
            ):
                raise ValueError(f"Manifest strata for {case_id} must be unique nonempty strings")

            source = case["source"]
            if not isinstance(source, dict):
                raise ValueError(f"Manifest source for {case_id} must be an object")
            for field in ("url", "item", "revision"):
                if not isinstance(source.get(field), str) or not source[field].strip():
                    raise ValueError(f"Manifest source.{field} for {case_id} must be a nonempty string")

            license_data = case["license"]
            if not isinstance(license_data, dict):
                raise ValueError(f"Manifest license for {case_id} must be an object")
            for field in ("id", "url", "attribution"):
                if not isinstance(license_data.get(field), str) or not license_data[field].strip():
                    raise ValueError(f"Manifest license.{field} for {case_id} must be a nonempty string")

            reference = case["reference"]
            if not isinstance(reference, dict):
                raise ValueError(f"Manifest reference for {case_id} must be an object")
            if not isinstance(reference.get("text"), str):
                raise ValueError(f"Manifest reference.text for {case_id} must be a string")
            intervals = reference.get("speech_intervals_ms")
            if intervals is not None:
                if not isinstance(intervals, list):
                    raise ValueError(f"Manifest reference.speech_intervals_ms for {case_id} must be a list or null")
                previous_end = 0
                for interval in intervals:
                    if (
                        not isinstance(interval, list)
                        or len(interval) != 2
                        or any(isinstance(value, bool) or not isinstance(value, int) for value in interval)
                    ):
                        raise ValueError(f"Manifest speech intervals for {case_id} must be [start_ms, end_ms] pairs")
                    start_ms, end_ms = interval
                    if start_ms < previous_end or start_ms < 0 or end_ms <= start_ms or end_ms > duration_ms:
                        raise ValueError(
                            f"Manifest speech intervals for {case_id} must be ordered and within the audio"
                        )
                    previous_end = end_ms

            reference_segments = reference.get("segments")
            if reference_segments is not None:
                if not isinstance(reference_segments, list):
                    raise ValueError(f"Manifest reference.segments for {case_id} must be a list")
                previous_end = 0
                for segment in reference_segments:
                    if not isinstance(segment, dict):
                        raise ValueError(f"Manifest reference segments for {case_id} must be objects")
                    start_ms = segment.get("start_ms")
                    end_ms = segment.get("end_ms")
                    text = segment.get("text")
                    if (
                        isinstance(start_ms, bool)
                        or not isinstance(start_ms, int)
                        or isinstance(end_ms, bool)
                        or not isinstance(end_ms, int)
                        or not isinstance(text, str)
                    ):
                        raise ValueError(f"Manifest reference segments for {case_id} have invalid fields")
                    if start_ms < previous_end or start_ms < 0 or end_ms <= start_ms or end_ms > duration_ms:
                        raise ValueError(
                            f"Manifest reference segments for {case_id} must be ordered and within the audio"
                        )
                    previous_end = end_ms

            cases.append(case)

    if not cases:
        raise ValueError("Manifest must contain at least one case")
    return cases


def _validate_protocol(protocol: dict) -> None:
    if protocol.get("schema_version") != 1:
        raise ValueError("Protocol schema_version must be 1")

    random_seed = protocol.get("random_seed")
    if isinstance(random_seed, bool) or not isinstance(random_seed, int) or random_seed < 0 or random_seed > 2**64 - 1:
        raise ValueError("Protocol random_seed must be an integer from 0 through 2^64 - 1")

    for field in ("model", "audio_classifier"):
        model = protocol.get(field)
        if not isinstance(model, dict):
            raise ValueError(f"Protocol {field} must be an object")
        repository = model.get("repository")
        revision = model.get("revision")
        if not isinstance(repository, str) or not repository.strip():
            raise ValueError(f"Protocol {field}.repository must be a nonempty string")
        if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise ValueError(f"Protocol {field}.revision must be an exact 40-character commit SHA")

    pipeline = protocol.get("pipeline")
    if not isinstance(pipeline, dict):
        raise ValueError("Protocol pipeline must be an object")
    forbidden_pipeline_fields = {"input_path", "output_path", "model_name", "language"} & pipeline.keys()
    if forbidden_pipeline_fields:
        raise ValueError(
            "Protocol pipeline cannot override per-run fields: " + ", ".join(sorted(forbidden_pipeline_fields))
        )
    allowed_pipeline_fields = {
        "device",
        "pause_threshold",
        "similarity_threshold",
        "embedding_model",
        "min_segment_words",
        "min_segment_chars",
        "max_segment_words",
        "performance",
    }
    unknown_pipeline_fields = pipeline.keys() - allowed_pipeline_fields
    if unknown_pipeline_fields:
        raise ValueError("Unknown protocol pipeline fields: " + ", ".join(sorted(unknown_pipeline_fields)))

    normalization = protocol.get("normalization")
    if normalization != {
        "unicode_form": "NFKC",
        "casefold": True,
        "punctuation_to_space": True,
        "collapse_whitespace": True,
    }:
        raise ValueError(
            "Protocol normalization must use NFKC, casefolding, punctuation-to-space, and collapsed whitespace"
        )

    collar_ms = protocol.get("boundary_collar_ms")
    if isinstance(collar_ms, bool) or not isinstance(collar_ms, int) or collar_ms < 0:
        raise ValueError("Protocol boundary_collar_ms must be a nonnegative integer")
    short_max_ms = protocol.get("short_case_max_duration_ms")
    if isinstance(short_max_ms, bool) or not isinstance(short_max_ms, int) or short_max_ms <= 0:
        raise ValueError("Protocol short_case_max_duration_ms must be a positive integer")
    if not isinstance(protocol.get("gates", {}), dict):
        raise ValueError("Protocol gates must be an object")


def _sample_peak_rss(
    process_id: int,
    interval_seconds: float,
    stop_event: threading.Event,
    peak_rss: list[int],
) -> None:
    process = psutil.Process(process_id)
    while not stop_event.is_set():
        rss = 0
        try:
            rss += process.memory_info().rss
            children = process.children(recursive=True)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            children = []
        for child in children:
            try:
                rss += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        peak_rss[0] = max(peak_rss[0], rss)
        stop_event.wait(interval_seconds)


def _parse_tsv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames is None or set(reader.fieldnames) != {"start_time", "end_time", "text"}:
            raise ValueError(f"Transcript has invalid columns: {path}")
        segments: list[dict] = []
        for line_number, row in enumerate(reader, start=2):
            if row.get("start_time") is None or row.get("end_time") is None or row.get("text") is None:
                raise ValueError(f"Transcript row {line_number} has missing fields: {path}")
            segments.append(
                {
                    "start_ms": round(parse_time_str(row["start_time"]) * 1000),
                    "end_ms": round(parse_time_str(row["end_time"]) * 1000),
                    "text": row["text"],
                }
            )
    return segments


def _infer(
    manifest_path: Path,
    protocol_path: Path,
    model_snapshot: Path,
    output_path: Path,
    rss_sample_interval_ms: int,
    device: str,
    batch_size: int,
) -> None:
    cases = _read_manifest(manifest_path)
    protocol = _read_json_object(protocol_path, "protocol")
    _validate_protocol(protocol)
    if rss_sample_interval_ms <= 0:
        raise ValueError("RSS sample interval must be positive")
    if batch_size <= 0:
        raise ValueError("Batch size must be positive")

    snapshot = model_snapshot.resolve()
    if not snapshot.is_dir():
        raise FileNotFoundError(f"Model snapshot directory does not exist: {snapshot}")
    revision = protocol["model"]["revision"]
    if snapshot.name != revision:
        raise ValueError(f"Model snapshot directory name must match pinned revision {revision}: {snapshot}")

    repository_root = Path(__file__).resolve().parents[1]
    git_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    git_status = subprocess.run(
        ["git", "status", "--short", "--untracked-files=no"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    ffmpeg_version = subprocess.run(
        ["ffmpeg", "-version"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.partition("\n")[0]

    predictions: list[dict] = []
    resolved_device = detect_device(device)
    pipeline_config = {
        **protocol["pipeline"],
        "device": resolved_device,
        "performance": {
            **protocol["pipeline"].get("performance", {}),
            "whisper_batch_size": batch_size,
        },
    }
    with tempfile.TemporaryDirectory(prefix="textplease-evaluation-") as temporary_directory:
        temporary_root = Path(temporary_directory)
        for index, case in enumerate(cases):
            audio_path = (manifest_path.resolve().parent / case["audio"]).resolve()
            if not audio_path.is_file():
                raise FileNotFoundError(f"Audio for {case['id']} does not exist: {audio_path}")
            with audio_path.open("rb") as audio_stream:
                if audio_stream.readline(128) == b"version https://git-lfs.github.com/spec/v1\n":
                    raise ValueError(f"Audio for {case['id']} is a Git LFS pointer. Run `git lfs pull` first.")
            actual_checksum = _sha256(audio_path)
            if actual_checksum != case["sha256"]:
                raise ValueError(
                    f"Audio checksum mismatch for {case['id']}: expected {case['sha256']}, got {actual_checksum}"
                )

            case_directory = temporary_root / str(index)
            case_directory.mkdir()
            transcript_path = case_directory / "transcript.tsv"
            config = {
                **pipeline_config,
                "input_path": str(audio_path),
                "output_path": str(transcript_path),
                "model_name": str(snapshot),
                "language": case["language"],
            }

            cuda_metrics = resolved_device == "cuda"
            if cuda_metrics:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

            torch.manual_seed(protocol["random_seed"])
            peak_rss = [0]
            stop_event = threading.Event()
            sampler = threading.Thread(
                target=_sample_peak_rss,
                args=(os.getpid(), rss_sample_interval_ms / 1000, stop_event, peak_rss),
                daemon=True,
            )
            sampler.start()
            started = time.perf_counter()
            try:
                run_transcription_pipeline(config)
            finally:
                if cuda_metrics:
                    torch.cuda.synchronize()
                elapsed_seconds = time.perf_counter() - started
                stop_event.set()
                sampler.join()

            segments = _parse_tsv(transcript_path)
            peak_cuda_bytes = torch.cuda.max_memory_allocated() if cuda_metrics else None
            predictions.append(
                {
                    "id": case["id"],
                    "audio_sha256": actual_checksum,
                    "segments": segments,
                    "error": None,
                    "elapsed_seconds": elapsed_seconds,
                    "rtf": elapsed_seconds / (case["duration_ms"] / 1000),
                    "peak_rss_bytes": peak_rss[0],
                    "peak_cuda_bytes": peak_cuda_bytes,
                }
            )

    result = {
        "schema_version": 1,
        "run": {
            "manifest_sha256": _sha256(manifest_path),
            "protocol_sha256": _sha256(protocol_path),
            "evaluator_sha256": _sha256(Path(__file__).resolve()),
            "random_seed": protocol["random_seed"],
            "source": {"git_revision": git_revision, "git_dirty": bool(git_status)},
            "model": {
                "repository": protocol["model"]["repository"],
                "revision": revision,
                "snapshot": str(snapshot),
            },
            "audio_classifier": protocol["audio_classifier"],
            "pipeline": pipeline_config,
            "requested_device": device,
            "resolved_device": resolved_device,
            "whisper_batch_size": batch_size,
            "environment": {
                "ffmpeg": ffmpeg_version,
                "platform": platform.platform(),
                "numpy": metadata.version("numpy"),
                "python": platform.python_version(),
                "textplease": metadata.version("textplease"),
                "torch": metadata.version("torch"),
                "transformers": metadata.version("transformers"),
                "silero-vad": metadata.version("silero-vad"),
                "sentence-transformers": metadata.version("sentence-transformers"),
            },
            "rss_sampling_interval_ms": rss_sample_interval_ms,
        },
        "cases": predictions,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_predictions(
    path: Path,
    cases: list[dict],
    manifest_path: Path,
    protocol_path: Path,
    protocol: dict,
) -> dict:
    document = _read_json_object(path, "predictions")
    if document.get("schema_version") != 1:
        raise ValueError("Prediction schema_version must be 1")
    run = document.get("run")
    if not isinstance(run, dict):
        raise ValueError("Predictions run metadata must be an object")
    expected_hashes = {
        "manifest_sha256": _sha256(manifest_path),
        "protocol_sha256": _sha256(protocol_path),
    }
    for field, expected in expected_hashes.items():
        if run.get(field) != expected:
            raise ValueError(f"Predictions {field} does not match the current evaluation input")
    evaluator_sha256 = run.get("evaluator_sha256")
    if not isinstance(evaluator_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", evaluator_sha256) is None:
        raise ValueError("Predictions evaluator_sha256 is invalid")
    if run.get("random_seed") != protocol["random_seed"]:
        raise ValueError("Predictions random_seed does not match the protocol")
    if run.get("model") is None or not isinstance(run["model"], dict):
        raise ValueError("Predictions model metadata must be an object")
    if any(run["model"].get(field) != protocol["model"][field] for field in ("repository", "revision")):
        raise ValueError("Predictions model metadata does not match the protocol")
    if run.get("audio_classifier") != protocol["audio_classifier"]:
        raise ValueError("Predictions audio classifier metadata does not match the protocol")
    source = run.get("source")
    if (
        not isinstance(source, dict)
        or not isinstance(source.get("git_revision"), str)
        or re.fullmatch(r"[0-9a-f]{40}", source["git_revision"]) is None
        or not isinstance(source.get("git_dirty"), bool)
    ):
        raise ValueError("Predictions source metadata is invalid")
    if run.get("resolved_device") not in {"cpu", "cuda", "mps"}:
        raise ValueError("Predictions resolved_device is invalid")
    batch_size = run.get("whisper_batch_size")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Predictions whisper_batch_size must be a positive integer")
    if not isinstance(run.get("environment"), dict):
        raise ValueError("Predictions environment metadata must be an object")
    prediction_cases = document.get("cases")
    if not isinstance(prediction_cases, list):
        raise ValueError("Predictions cases must be a list")

    manifest_by_id = {case["id"]: case for case in cases}
    predictions_by_id: dict[str, dict] = {}
    for prediction in prediction_cases:
        if not isinstance(prediction, dict):
            raise ValueError("Every prediction case must be an object")
        case_id = prediction.get("id")
        if not isinstance(case_id, str) or case_id not in manifest_by_id:
            raise ValueError(f"Prediction has unknown or invalid id: {case_id}")
        if case_id in predictions_by_id:
            raise ValueError(f"Duplicate prediction id: {case_id}")
        if prediction.get("audio_sha256") != manifest_by_id[case_id]["sha256"]:
            raise ValueError(f"Prediction audio checksum does not match the manifest for {case_id}")

        segments = prediction.get("segments")
        if not isinstance(segments, list):
            raise ValueError(f"Prediction segments for {case_id} must be a list")
        for segment in segments:
            if not isinstance(segment, dict):
                raise ValueError(f"Prediction segments for {case_id} must be objects")
            start_ms = segment.get("start_ms")
            end_ms = segment.get("end_ms")
            text = segment.get("text")
            if (
                isinstance(start_ms, bool)
                or not isinstance(start_ms, int)
                or isinstance(end_ms, bool)
                or not isinstance(end_ms, int)
                or not isinstance(text, str)
            ):
                raise ValueError(f"Prediction segments for {case_id} have invalid fields")

        error = prediction.get("error")
        if error is not None:
            if (
                not isinstance(error, dict)
                or error.get("kind") != "empty_output"
                or not isinstance(error.get("message"), str)
            ):
                raise ValueError(f"Prediction error for {case_id} is invalid")
            if segments:
                raise ValueError(f"Prediction for {case_id} cannot contain both segments and an error")

        for field in ("elapsed_seconds", "rtf"):
            value = prediction.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Prediction {field} for {case_id} must be a finite nonnegative number")
        peak_rss_bytes = prediction.get("peak_rss_bytes")
        if isinstance(peak_rss_bytes, bool) or not isinstance(peak_rss_bytes, int) or peak_rss_bytes < 0:
            raise ValueError(f"Prediction peak_rss_bytes for {case_id} must be a nonnegative integer")
        peak_cuda_bytes = prediction.get("peak_cuda_bytes")
        if peak_cuda_bytes is not None and (
            isinstance(peak_cuda_bytes, bool) or not isinstance(peak_cuda_bytes, int) or peak_cuda_bytes < 0
        ):
            raise ValueError(f"Prediction peak_cuda_bytes for {case_id} must be null or a nonnegative integer")

        predictions_by_id[case_id] = prediction

    missing = manifest_by_id.keys() - predictions_by_id.keys()
    if missing:
        raise ValueError("Predictions are missing manifest cases: " + ", ".join(sorted(missing)))
    return {"run": run, "cases": predictions_by_id}


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    punctuation_as_spaces = [
        " " if unicodedata.category(character)[0] in {"P", "Z"} else character for character in normalized
    ]
    return " ".join("".join(punctuation_as_spaces).split())


def _merge_intervals(intervals: list[tuple[int, int]], duration_ms: int) -> list[tuple[int, int]]:
    clipped = sorted(
        (max(0, start_ms), min(duration_ms, end_ms))
        for start_ms, end_ms in intervals
        if end_ms > start_ms and end_ms > 0 and start_ms < duration_ms
    )
    merged: list[tuple[int, int]] = []
    for start_ms, end_ms in clipped:
        if merged and start_ms <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end_ms))
        else:
            merged.append((start_ms, end_ms))
    return merged


def _intersection_duration(first: list[tuple[int, int]], second: list[tuple[int, int]]) -> int:
    duration_ms = 0
    first_index = 0
    second_index = 0
    while first_index < len(first) and second_index < len(second):
        start_ms = max(first[first_index][0], second[second_index][0])
        end_ms = min(first[first_index][1], second[second_index][1])
        duration_ms += max(0, end_ms - start_ms)
        if first[first_index][1] <= second[second_index][1]:
            first_index += 1
        else:
            second_index += 1
    return duration_ms


def _match_boundaries(reference: list[int], prediction: list[int], collar_ms: int) -> list[int]:
    errors: list[int] = []
    reference_index = 0
    prediction_index = 0
    while reference_index < len(reference) and prediction_index < len(prediction):
        difference = prediction[prediction_index] - reference[reference_index]
        if difference < -collar_ms:
            prediction_index += 1
        elif difference > collar_ms:
            reference_index += 1
        else:
            errors.append(abs(difference))
            reference_index += 1
            prediction_index += 1
    return errors


def _percentile(values: Sequence[int | float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return float(ordered[max(0, math.ceil(percentile * len(ordered)) - 1)])


def _score_subset(cases: list[dict], predictions: dict[str, dict], protocol: dict) -> dict:
    word_substitutions = 0
    word_deletions = 0
    word_insertions = 0
    reference_words = 0
    character_substitutions = 0
    character_deletions = 0
    character_insertions = 0
    reference_characters = 0
    short_cases = 0
    short_exact_matches = 0
    non_speech_cases = 0
    non_speech_nonempty_cases = 0
    non_speech_error_cases = 0
    prediction_error_cases = 0
    activity_cases = 0
    reference_speech_ms = 0
    reference_non_speech_ms = 0
    missed_speech_ms = 0
    false_alarm_ms = 0
    reference_boundaries = 0
    predicted_boundaries = 0
    onset_errors: list[int] = []
    offset_errors: list[int] = []
    timestamp_violation_cases = 0
    timestamp_violations = 0
    rtfs: list[float] = []
    peak_rss_bytes: list[int] = []
    peak_cuda_bytes: list[int] = []

    for case in cases:
        prediction = predictions[case["id"]]
        normalized_reference = _normalize(case["reference"]["text"])
        normalized_prediction = _normalize(" ".join(segment["text"] for segment in prediction["segments"]))
        reference_interval_data = case["reference"]["speech_intervals_ms"]
        reference_intervals = (
            None if reference_interval_data is None else [tuple(interval) for interval in reference_interval_data]
        )
        merged_reference = (
            None if reference_intervals is None else _merge_intervals(reference_intervals, case["duration_ms"])
        )
        reference_speech_duration = (
            None if merged_reference is None else sum(end_ms - start_ms for start_ms, end_ms in merged_reference)
        )

        word_output = process_words(normalized_reference, normalized_prediction)
        word_substitutions += word_output.substitutions
        word_deletions += word_output.deletions
        word_insertions += word_output.insertions
        reference_words += word_output.hits + word_output.substitutions + word_output.deletions

        character_output = process_characters(normalized_reference, normalized_prediction)
        character_substitutions += character_output.substitutions
        character_deletions += character_output.deletions
        character_insertions += character_output.insertions
        reference_characters += character_output.hits + character_output.substitutions + character_output.deletions

        if normalized_reference and (
            case["duration_ms"] <= protocol["short_case_max_duration_ms"]
            or (
                reference_speech_duration is not None
                and 0 < reference_speech_duration <= protocol["short_case_max_duration_ms"]
            )
        ):
            short_cases += 1
            short_exact_matches += normalized_reference == normalized_prediction

        if reference_intervals == []:
            non_speech_cases += 1
            non_speech_nonempty_cases += bool(normalized_prediction)
            non_speech_error_cases += prediction["error"] is not None
        prediction_error_cases += prediction["error"] is not None

        predicted_intervals = [(segment["start_ms"], segment["end_ms"]) for segment in prediction["segments"]]
        if merged_reference is not None:
            activity_cases += 1
            merged_prediction = _merge_intervals(predicted_intervals, case["duration_ms"])
            reference_duration = sum(end_ms - start_ms for start_ms, end_ms in merged_reference)
            predicted_duration = sum(end_ms - start_ms for start_ms, end_ms in merged_prediction)
            overlap_duration = _intersection_duration(merged_reference, merged_prediction)
            reference_speech_ms += reference_duration
            reference_non_speech_ms += case["duration_ms"] - reference_duration
            missed_speech_ms += reference_duration - overlap_duration
            false_alarm_ms += predicted_duration - overlap_duration

        boundary_source = case["reference"].get("segments")
        if boundary_source is None:
            boundary_intervals = reference_intervals
        else:
            boundary_intervals = [(segment["start_ms"], segment["end_ms"]) for segment in boundary_source]
        if boundary_intervals is not None:
            reference_onsets = [interval[0] for interval in boundary_intervals]
            reference_offsets = [interval[1] for interval in boundary_intervals]
            boundary_predictions = sorted(
                (max(0, start_ms), min(case["duration_ms"], end_ms))
                for start_ms, end_ms in predicted_intervals
                if end_ms > start_ms and end_ms > 0 and start_ms < case["duration_ms"]
            )
            predicted_onsets = [interval[0] for interval in boundary_predictions]
            predicted_offsets = [interval[1] for interval in boundary_predictions]
            onset_errors.extend(_match_boundaries(reference_onsets, predicted_onsets, protocol["boundary_collar_ms"]))
            offset_errors.extend(
                _match_boundaries(reference_offsets, predicted_offsets, protocol["boundary_collar_ms"])
            )
            reference_boundaries += len(reference_onsets) + len(reference_offsets)
            predicted_boundaries += len(predicted_onsets) + len(predicted_offsets)

        case_violations = 0
        previous_start = -1
        previous_end = -1
        for segment in prediction["segments"]:
            violations = (
                int(segment["start_ms"] < 0)
                + int(segment["end_ms"] <= segment["start_ms"])
                + int(segment["end_ms"] > case["duration_ms"])
                + int(segment["start_ms"] < previous_start)
                + int(previous_end >= 0 and segment["start_ms"] < previous_end)
            )
            case_violations += violations
            previous_start = segment["start_ms"]
            previous_end = segment["end_ms"]
        timestamp_violations += case_violations
        timestamp_violation_cases += case_violations > 0

        rtfs.append(float(prediction["rtf"]))
        peak_rss_bytes.append(prediction["peak_rss_bytes"])
        if prediction["peak_cuda_bytes"] is not None:
            peak_cuda_bytes.append(prediction["peak_cuda_bytes"])

    word_errors = word_substitutions + word_deletions + word_insertions
    character_errors = character_substitutions + character_deletions + character_insertions
    boundary_matches = len(onset_errors) + len(offset_errors)
    boundary_errors = onset_errors + offset_errors
    return {
        "cases": len(cases),
        "reference_words": reference_words,
        "word_substitutions": word_substitutions,
        "word_deletions": word_deletions,
        "word_insertions": word_insertions,
        "wer": word_errors / reference_words if reference_words else None,
        "reference_characters": reference_characters,
        "character_substitutions": character_substitutions,
        "character_deletions": character_deletions,
        "character_insertions": character_insertions,
        "cer": character_errors / reference_characters if reference_characters else None,
        "short_cases": short_cases,
        "short_exact_matches": short_exact_matches,
        "short_exact_match_rate": short_exact_matches / short_cases if short_cases else None,
        "non_speech_cases": non_speech_cases,
        "non_speech_nonempty_cases": non_speech_nonempty_cases,
        "non_speech_nonempty_rate": non_speech_nonempty_cases / non_speech_cases if non_speech_cases else None,
        "non_speech_error_cases": non_speech_error_cases,
        "non_speech_error_rate": non_speech_error_cases / non_speech_cases if non_speech_cases else None,
        "prediction_error_cases": prediction_error_cases,
        "reference_speech_ms": reference_speech_ms if activity_cases else None,
        "missed_speech_ms": missed_speech_ms if activity_cases else None,
        "missed_speech_rate": missed_speech_ms / reference_speech_ms if reference_speech_ms else None,
        "reference_non_speech_ms": reference_non_speech_ms if activity_cases else None,
        "false_alarm_ms": false_alarm_ms if activity_cases else None,
        "false_alarm_rate": false_alarm_ms / reference_non_speech_ms if reference_non_speech_ms else None,
        "reference_boundaries": reference_boundaries,
        "predicted_boundaries": predicted_boundaries,
        "matched_boundaries": boundary_matches,
        "boundary_precision": (
            boundary_matches / predicted_boundaries if predicted_boundaries else 0.0 if reference_boundaries else None
        ),
        "boundary_recall": boundary_matches / reference_boundaries if reference_boundaries else None,
        "boundary_error_median_ms": float(statistics.median(boundary_errors)) if boundary_errors else None,
        "boundary_error_p95_ms": _percentile(boundary_errors, 0.95),
        "onset_error_median_ms": float(statistics.median(onset_errors)) if onset_errors else None,
        "onset_error_p95_ms": _percentile(onset_errors, 0.95),
        "offset_error_median_ms": float(statistics.median(offset_errors)) if offset_errors else None,
        "offset_error_p95_ms": _percentile(offset_errors, 0.95),
        "timestamp_violation_cases": timestamp_violation_cases,
        "timestamp_violations": timestamp_violations,
        "rtf_median": float(statistics.median(rtfs)),
        "rtf_p95": _percentile(rtfs, 0.95),
        "peak_rss_mb_max": max(peak_rss_bytes) / (1024 * 1024),
        "peak_cuda_mb_max": max(peak_cuda_bytes) / (1024 * 1024) if peak_cuda_bytes else None,
    }


def _evaluate_gates(metrics: dict, gates: dict) -> tuple[list[dict], bool]:
    comparisons = {
        "wer": "max",
        "cer": "max",
        "short_exact_match_rate": "min",
        "non_speech_nonempty_cases": "max",
        "non_speech_error_cases": "max",
        "missed_speech_rate": "max",
        "false_alarm_rate": "max",
        "boundary_precision": "min",
        "boundary_recall": "min",
        "timestamp_violation_cases": "max",
        "parity_mismatch_cases": "max",
        "rtf_median": "max",
        "peak_rss_mb_max": "max",
        "peak_cuda_mb_max": "max",
    }
    results: list[dict] = []
    failed = False
    for gate_name in sorted(gates):
        gate = gates[gate_name]
        if gate_name not in comparisons or not isinstance(gate, dict):
            raise ValueError(f"Unknown or invalid gate: {gate_name}")
        comparison = comparisons[gate_name]
        if set(gate) != {"enabled", comparison}:
            raise ValueError(f"Gate {gate_name} must contain only enabled and {comparison}")
        enabled = gate["enabled"]
        threshold = gate[comparison]
        if not isinstance(enabled, bool):
            raise ValueError(f"Gate {gate_name}.enabled must be a boolean")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError(f"Gate {gate_name}.{comparison} must be numeric")
        threshold = float(threshold)
        metric_name = gate_name
        if not math.isfinite(threshold):
            raise ValueError(f"Gate {gate_name} threshold must be finite")

        actual = metrics.get(metric_name)
        if not enabled:
            status = "DISABLED"
        elif actual is None:
            status = "FAIL"
            failed = True
        elif comparison == "max":
            status = "PASS" if actual <= threshold else "FAIL"
        else:
            status = "PASS" if actual >= threshold else "FAIL"
        if status == "FAIL":
            failed = True
        results.append(
            {
                "name": gate_name,
                "metric": metric_name,
                "comparison": comparison,
                "threshold": threshold,
                "actual": actual,
                "status": status,
            }
        )
    return results, failed


def _format_value(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _escape_markdown(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _score(
    manifest_path: Path,
    protocol_path: Path,
    predictions_path: Path,
    output_path: Path,
    parity_predictions_path: Path | None,
) -> bool:
    cases = _read_manifest(manifest_path)
    protocol = _read_json_object(protocol_path, "protocol")
    _validate_protocol(protocol)
    prediction_document = _read_predictions(predictions_path, cases, manifest_path, protocol_path, protocol)
    predictions = prediction_document["cases"]
    overall = _score_subset(cases, predictions, protocol)
    acceptance_cases = [case for case in cases if case["split"] == "acceptance"]
    if not acceptance_cases:
        raise ValueError("Manifest must contain at least one acceptance case")
    acceptance = _score_subset(acceptance_cases, predictions, protocol)

    if parity_predictions_path is None:
        overall["parity_mismatch_cases"] = None
        acceptance["parity_mismatch_cases"] = None
    else:
        parity_document = _read_predictions(
            parity_predictions_path,
            cases,
            manifest_path,
            protocol_path,
            protocol,
        )
        primary_run = prediction_document["run"]
        parity_run = parity_document["run"]
        if (
            any(primary_run["model"][field] != parity_run["model"][field] for field in ("repository", "revision"))
            or any(primary_run[field] != parity_run[field] for field in ("resolved_device", "evaluator_sha256"))
            or primary_run["source"]["git_revision"] != parity_run["source"]["git_revision"]
        ):
            raise ValueError("Parity predictions must use the same model, device, evaluator, and source revision")
        parity = parity_document["cases"]
        mismatches = 0
        acceptance_mismatches = 0
        acceptance_ids = {case["id"] for case in acceptance_cases}
        for case in cases:
            case_id = case["id"]
            primary = predictions[case_id]
            alternate = parity[case_id]
            primary_segments = [
                (segment["start_ms"], segment["end_ms"], _normalize(segment["text"]))
                for segment in primary["segments"]
            ]
            alternate_segments = [
                (segment["start_ms"], segment["end_ms"], _normalize(segment["text"]))
                for segment in alternate["segments"]
            ]
            primary_error = primary["error"]["kind"] if primary["error"] is not None else None
            alternate_error = alternate["error"]["kind"] if alternate["error"] is not None else None
            mismatch = primary_segments != alternate_segments or primary_error != alternate_error
            mismatches += mismatch
            acceptance_mismatches += case_id in acceptance_ids and mismatch
        overall["parity_mismatch_cases"] = mismatches
        acceptance["parity_mismatch_cases"] = acceptance_mismatches

    gates, failed = _evaluate_gates(acceptance, protocol.get("gates", {}))
    grouped_cases: list[tuple[str, list[dict]]] = []
    for split in ("tuning", "acceptance"):
        subset = [case for case in cases if case["split"] == split]
        if subset:
            grouped_cases.append((f"split={split}", subset))
    for language in sorted({case["language"] for case in cases}):
        grouped_cases.append((f"language={language}", [case for case in cases if case["language"] == language]))
    for stratum in sorted({stratum for case in cases for stratum in case["strata"]}):
        grouped_cases.append((f"stratum={stratum}", [case for case in cases if stratum in case["strata"]]))
    grouped_metrics = [(name, _score_subset(subset, predictions, protocol)) for name, subset in grouped_cases]

    run_metadata = prediction_document["run"]
    lines = [
        "# Audio quality evaluation",
        "",
        "This report scores the configured public `textplease` pipeline against the versioned manifest and protocol.",
        "",
        "## Run",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Manifest SHA-256 | `{_sha256(manifest_path)}` |",
        f"| Protocol SHA-256 | `{_sha256(protocol_path)}` |",
        f"| Inference evaluator SHA-256 | `{run_metadata['evaluator_sha256']}` |",
        f"| Scorer SHA-256 | `{_sha256(Path(__file__).resolve())}` |",
        f"| Scorer JiWER | `{metadata.version('jiwer')}` |",
        f"| Scorer RapidFuzz | `{metadata.version('rapidfuzz')}` |",
        f"| Random seed | `{run_metadata['random_seed']}` |",
    ]
    source_metadata = run_metadata["source"]
    lines.extend(
        [
            f"| Source revision | `{source_metadata['git_revision']}` |",
            f"| Source dirty | `{source_metadata['git_dirty']}` |",
            f"| Device | `{run_metadata['resolved_device']}` |",
            f"| Whisper batch size | `{run_metadata['whisper_batch_size']}` |",
        ]
    )
    model_metadata = run_metadata.get("model", {})
    classifier_metadata = run_metadata.get("audio_classifier", {})
    environment_metadata = run_metadata.get("environment", {})
    if isinstance(model_metadata, dict):
        for field in ("repository", "revision"):
            if field in model_metadata:
                lines.append(f"| Model {field} | `{_escape_markdown(model_metadata[field])}` |")
    if isinstance(classifier_metadata, dict):
        for field in ("repository", "revision"):
            if field in classifier_metadata:
                lines.append(f"| Audio classifier {field} | `{_escape_markdown(classifier_metadata[field])}` |")
    if isinstance(environment_metadata, dict):
        for field in sorted(environment_metadata):
            lines.append(f"| Environment {field} | `{_escape_markdown(environment_metadata[field])}` |")

    lines.extend(
        [
            "",
            "## Fixture sources",
            "",
            "| Case | Source | License | Attribution |",
            "|---|---|---|---|",
        ]
    )
    for case in cases:
        source = case["source"]
        license_data = case["license"]
        lines.append(
            f"| {_escape_markdown(case['id'])} | "
            f"[{_escape_markdown(source['item'])}]({source['url']}) with `{_escape_markdown(source['revision'])}` | "
            f"[{_escape_markdown(license_data['id'])}]({license_data['url']}) | "
            f"{_escape_markdown(license_data['attribution'])} |"
        )

    metric_order = [
        ("Cases", "cases"),
        ("WER", "wer"),
        ("Word substitutions", "word_substitutions"),
        ("Word deletions", "word_deletions"),
        ("Word insertions", "word_insertions"),
        ("CER", "cer"),
        ("Short exact-match rate", "short_exact_match_rate"),
        ("Non-speech nonempty cases", "non_speech_nonempty_cases"),
        ("Non-speech error cases", "non_speech_error_cases"),
        ("Prediction error cases", "prediction_error_cases"),
        ("Reference speech (ms)", "reference_speech_ms"),
        ("Missed speech (ms)", "missed_speech_ms"),
        ("Missed speech rate", "missed_speech_rate"),
        ("Reference non-speech (ms)", "reference_non_speech_ms"),
        ("False alarm (ms)", "false_alarm_ms"),
        ("False-alarm rate", "false_alarm_rate"),
        ("Boundary precision", "boundary_precision"),
        ("Boundary recall", "boundary_recall"),
        ("Boundary median error (ms)", "boundary_error_median_ms"),
        ("Boundary p95 error (ms)", "boundary_error_p95_ms"),
        ("Onset median error (ms)", "onset_error_median_ms"),
        ("Onset p95 error (ms)", "onset_error_p95_ms"),
        ("Offset median error (ms)", "offset_error_median_ms"),
        ("Offset p95 error (ms)", "offset_error_p95_ms"),
        ("Timestamp violation cases", "timestamp_violation_cases"),
        ("Timestamp violations", "timestamp_violations"),
        ("Parity mismatch cases", "parity_mismatch_cases"),
        ("Median RTF", "rtf_median"),
        ("p95 RTF", "rtf_p95"),
        ("Peak RSS (MiB)", "peak_rss_mb_max"),
        ("Peak CUDA allocation (MiB)", "peak_cuda_mb_max"),
    ]
    lines.extend(["", "## Overall", "", "| Metric | Value |", "|---|---:|"])
    lines.extend(f"| {label} | {_format_value(overall[key])} |" for label, key in metric_order)

    lines.extend(
        [
            "",
            "## Per stratum",
            "",
            "| Group | Cases | WER | CER | Short exact | Non-speech nonempty | Miss (ms) | Miss rate | False alarm (ms) | False-alarm rate | Boundary P | Boundary R | Timestamp violations | RTF |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, metrics in grouped_metrics:
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_markdown(name),
                    _format_value(metrics["cases"]),
                    _format_value(metrics["wer"]),
                    _format_value(metrics["cer"]),
                    _format_value(metrics["short_exact_match_rate"]),
                    _format_value(metrics["non_speech_nonempty_cases"]),
                    _format_value(metrics["missed_speech_ms"]),
                    _format_value(metrics["missed_speech_rate"]),
                    _format_value(metrics["false_alarm_ms"]),
                    _format_value(metrics["false_alarm_rate"]),
                    _format_value(metrics["boundary_precision"]),
                    _format_value(metrics["boundary_recall"]),
                    _format_value(metrics["timestamp_violations"]),
                    _format_value(metrics["rtf_median"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Gates",
            "",
            "Gates evaluate only manifest rows with `split=acceptance`.",
            "",
            "| Gate | Rule | Actual | Status |",
            "|---|---:|---:|---|",
        ]
    )
    if gates:
        for gate in gates:
            lines.append(
                f"| `{gate['name']}` | {gate['comparison']} {_format_value(gate['threshold'])} | "
                f"{_format_value(gate['actual'])} | {gate['status']} |"
            )
    else:
        lines.append("| _No gates configured_ | — | — | — |")

    lines.extend(
        [
            "",
            "## Cases",
            "",
            "| Case | Split | Strata | Duration (s) | Inference (s) | RTF | Peak RSS (MiB) | Peak CUDA (MiB) | WER | CER | Miss (ms) | Miss rate | False alarm (ms) | False-alarm rate | Timestamp violations | Error |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for case in cases:
        case_metrics = _score_subset([case], predictions, protocol)
        error = predictions[case["id"]]["error"]
        error_kind = error["kind"] if error is not None else "—"
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_markdown(case["id"]),
                    case["split"],
                    _escape_markdown(", ".join(case["strata"])),
                    _format_value(case["duration_ms"] / 1000),
                    _format_value(predictions[case["id"]]["elapsed_seconds"]),
                    _format_value(case_metrics["rtf_median"]),
                    _format_value(case_metrics["peak_rss_mb_max"]),
                    _format_value(case_metrics["peak_cuda_mb_max"]),
                    _format_value(case_metrics["wer"]),
                    _format_value(case_metrics["cer"]),
                    _format_value(case_metrics["missed_speech_ms"]),
                    _format_value(case_metrics["missed_speech_rate"]),
                    _format_value(case_metrics["false_alarm_ms"]),
                    _format_value(case_metrics["false_alarm_rate"]),
                    _format_value(case_metrics["timestamp_violations"]),
                    error_kind,
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- Boundary and speech-duration metrics compare final TSV intervals with the references. They are end-to-end output metrics, not direct Silero VAD measurements.",
            "- Activity duration metrics exclude cases whose speech interval reference is null. An em dash means no interval reference was scored.",
            "- Short exact match includes recordings whose total annotated speech is within the configured short duration, even when the surrounding recording is longer.",
            "- WER and CER score final post-processed text. The current pipeline does not expose raw decoder text, so this report cannot isolate decoder fidelity from later text mutation.",
            "- Audio-classifier identity is declared by the protocol and is not mechanically queried from the backend. Source and evaluator metadata aid auditing but do not hash an uncommitted runtime diff.",
            "- Pipeline settings are defined by the protocol and may differ from application defaults; interpret results only for the recorded configuration.",
            "- CER includes spaces after NFKC, casefolding, punctuation-to-space conversion, and whitespace collapse.",
            "- Peak RSS is sampled for this process and its children, so spikes shorter than the sampling interval may be missed. CUDA memory is PyTorch's peak allocated memory.",
            "- The first case that loads Whisper includes cold model loading; later cases may reuse in-process model caches.",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return failed


def main() -> int:
    """Run reproducible local audio inference or score existing predictions."""
    parser = argparse.ArgumentParser(description="Run and score the versioned textplease audio-quality evaluation.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    infer_parser = subparsers.add_parser("infer", help="Run the public pipeline on every manifest case.")
    infer_parser.add_argument("--manifest", type=Path, required=True)
    infer_parser.add_argument("--protocol", type=Path, required=True)
    infer_parser.add_argument("--model-snapshot", type=Path, required=True)
    infer_parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), required=True)
    infer_parser.add_argument("--batch-size", type=int, required=True)
    infer_parser.add_argument("--output", type=Path, required=True)
    infer_parser.add_argument("--rss-sample-interval-ms", type=int, default=50)

    score_parser = subparsers.add_parser("score", help="Score predictions and write a deterministic Markdown report.")
    score_parser.add_argument("--manifest", type=Path, required=True)
    score_parser.add_argument("--protocol", type=Path, required=True)
    score_parser.add_argument("--predictions", type=Path, required=True)
    score_parser.add_argument("--parity-predictions", type=Path)
    score_parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "infer":
        _infer(
            args.manifest,
            args.protocol,
            args.model_snapshot,
            args.output,
            args.rss_sample_interval_ms,
            args.device,
            args.batch_size,
        )
        return 0

    failed = _score(
        args.manifest,
        args.protocol,
        args.predictions,
        args.output,
        args.parity_predictions,
    )
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
