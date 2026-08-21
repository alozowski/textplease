import os
import shutil
import logging
import argparse
from pathlib import Path

import yaml

from textplease.pipeline import run_transcription_pipeline
from textplease.gradio_ui import launch_gradio
from textplease.utils.audio_utils import FFMPEG_INSTALL_ERROR
from textplease.utils.logging_config import configure_logging


logger = logging.getLogger(__name__)
__all__ = ["main"]


def load_config(path: str) -> dict:
    """Load and return configuration from a YAML file."""
    config_file = Path(path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with config_file.open() as config_stream:
        config = yaml.safe_load(config_stream)

    if config is None:
        raise ValueError(f"Config file is empty: {path}")
    return config


def apply_environment_config(env_config: dict) -> None:
    """Set environment variables from config if not already set."""
    if not isinstance(env_config, dict):
        logger.warning(f"Invalid environment config (expected dict): {type(env_config)}")
        return
    for key, value in env_config.items():
        if key not in os.environ:
            os.environ[key] = str(value)
            logger.info(f"Applied env var: {key}={value}")
        else:
            logger.debug(f"Env var {key} already set, skipping.")


def main() -> None:
    """Provide main entry point for the textplease CLI."""
    parser = argparse.ArgumentParser(description="Transcribe and segment audio using open-source ASR models.")
    parser.add_argument("--config", help="Path to YAML config file.")
    parser.add_argument("--gradio", action="store_true", help="Launch the Gradio UI instead of CLI pipeline.")

    args = parser.parse_args()

    if args.gradio:
        launch_gradio()
        return

    if not args.config:
        parser.error("--config is required when not using --gradio")

    config = load_config(args.config)

    env_config = config.get("environment", {})
    if env_config:
        apply_environment_config(env_config)

    if shutil.which("ffmpeg") is None:
        parser.error(FFMPEG_INSTALL_ERROR)

    log_level = config.get("log_level", "INFO").upper()
    configure_logging(level=getattr(logging, log_level, logging.INFO))

    run_transcription_pipeline(config)


if __name__ == "__main__":
    main()
