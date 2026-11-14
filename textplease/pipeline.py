import os
import json
import math
import time
import hashlib
import logging
import tempfile
from pathlib import Path
from contextlib import nullcontext

import pandas as pd

from textplease.transcriber import transcribe_audio
from textplease.utils.device_utils import detect_device


logger = logging.getLogger(__name__)


def _debug_json_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _debug_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_debug_json_value(item) for item in value]
    return value


def save_to_csv(
    segments: list[dict[str, str]],
    output_path: str,
    temporary_directory: str | Path,
    diagnostics: list[dict] | None = None,
) -> str:
    """Save segments to a tab-separated CSV file."""
    if segments:
        df = pd.DataFrame(segments, columns=["start_time", "end_time", "text"])
        df = df[df["text"].astype(str).str.strip() != ""]
    else:
        df = pd.DataFrame(columns=["start_time", "end_time", "text"])

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = Path(temporary_directory) / "transcript.tsv"
    try:
        df.to_csv(temporary_output, index=False, sep="\t")
        temporary_output.chmod(0o600)
        if diagnostics is not None:
            debug_output = Path(temporary_directory) / "debug.json"
            try:
                debug_output.touch(mode=0o600)
                debug_output.write_text(
                    json.dumps(
                        {
                            "transcript_sha256": hashlib.sha256(temporary_output.read_bytes()).hexdigest(),
                            "schema_version": 1,
                            "events": _debug_json_value(diagnostics),
                        },
                        ensure_ascii=False,
                        indent=2,
                        allow_nan=False,
                    ),
                    encoding="utf-8",
                )
                os.replace(debug_output, Path(str(output) + ".debug.json"))
            finally:
                debug_output.unlink(missing_ok=True)
        os.replace(temporary_output, output)
    finally:
        temporary_output.unlink(missing_ok=True)
    if df.empty:
        logger.info(f"No speech was transcribed; saved an empty transcript to {output_path}")
    else:
        logger.info(f"Saved {len(df)} segments to {output_path}")
    return output_path


def _validate_pipeline_config(config: dict) -> None:
    """Validate configuration for the transcription pipeline."""
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a dictionary, got {type(config)}")

    missing = {"input_path", "output_path", "model_name"} - set(config.keys())
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")

    supported = {
        "input_path",
        "output_path",
        "model_name",
        "device",
        "language",
        "log_level",
        "performance",
        "_temporary_directory",
    }
    unknown = set(config) - supported
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")

    performance = config.get("performance", {})
    if not isinstance(performance, dict):
        raise ValueError("Config performance must be a dictionary")
    unknown_performance = set(performance) - {"whisper_batch_size"}
    if unknown_performance:
        raise ValueError(f"Unknown performance config keys: {unknown_performance}")

    input_path = config["input_path"]
    if not input_path or not isinstance(input_path, str):
        raise ValueError(f"Invalid input_path: {input_path}")

    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = config["output_path"]
    if not output_path or not isinstance(output_path, str):
        raise ValueError(f"Invalid output_path: {output_path}")

    if str(config.get("log_level", "INFO")).upper() == "DEBUG":
        debug_file = Path(str(output_path) + ".debug.json")
        if Path(input_path).resolve() == debug_file.resolve() or (
            debug_file.exists() and Path(input_path).samefile(debug_file)
        ):
            raise ValueError("Input and debug output paths must refer to different files")

    input_file = Path(input_path)
    output_file = Path(output_path)
    if input_file.resolve() == output_file.resolve() or (output_file.exists() and input_file.samefile(output_file)):
        raise ValueError("Input and output paths must refer to different files")


def run_transcription_pipeline(config: dict) -> None:
    """Run the complete transcription pipeline."""
    start = time.time()
    _validate_pipeline_config(config)
    input_path = config["input_path"]
    output_path = config["output_path"]
    model_name = config["model_name"]
    device = detect_device(config.get("device", "cpu"))
    language = config.get("language", "en")
    performance = config.get("performance", {})
    whisper_batch_size = performance.get("whisper_batch_size", 1)

    logger.info(f"Input: {input_path} → Output: {output_path}")
    logger.info(f"ASR: {model_name} | Device: {device}")

    output_parent = Path(output_path).resolve().parent
    output_parent.mkdir(parents=True, exist_ok=True)
    provided_directory = config.get("_temporary_directory")
    if provided_directory is not None and (
        not isinstance(provided_directory, Path) or provided_directory.resolve().parent != output_parent
    ):
        raise ValueError("Invalid managed temporary directory")
    temporary_directory = (
        nullcontext(provided_directory)
        if provided_directory is not None
        else tempfile.TemporaryDirectory(prefix=".textplease-", dir=output_parent)
    )
    with temporary_directory as work_directory:
        t0 = time.time()
        diagnostics: list[dict] | None = [] if str(config.get("log_level", "INFO")).upper() == "DEBUG" else None
        debug_arguments = {} if diagnostics is None else {"diagnostics": diagnostics}
        segments = transcribe_audio(
            input_path,
            model_name,
            device,
            temporary_directory=work_directory,
            language=language,
            batch_size=whisper_batch_size,
            **debug_arguments,
        )
        logger.info(f"Transcription: {len(segments)} segments in {time.time() - t0:.2f}s")

        t0 = time.time()
        save_to_csv(segments, output_path, work_directory, diagnostics=diagnostics)
        logger.info(f"Save: {time.time() - t0:.2f}s")

    logger.info(f"Total processing time: {time.time() - start:.2f}s")
