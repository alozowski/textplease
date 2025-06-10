import os
import logging
import argparse
from typing import Dict

import yaml
import pandas as pd
from sentence_transformers import SentenceTransformer

from textplease.segmenter import segment_coherently, post_process_segments
from textplease.transcriber import transcribe_audio
from textplease.utils.audio_utils import cleanup_temp, extract_audio
from textplease.utils.device_utils import detect_device
from textplease.utils.logging_config import configure_logging
from textplease.utils.deduplicate_segments import deduplicate_segments


logger = logging.getLogger(__name__)


def load_config(path: str) -> Dict:
    """Load configuration from a YAML file"""
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load configuration file: {e}")
        raise


def apply_environment_config(env_config: Dict) -> None:
    """Set environment variables from config if not already set"""
    for key, value in env_config.items():
        if key not in os.environ:
            os.environ[key] = str(value)
            logger.info(f"Applied env var: {key}={value}")
        else:
            logger.debug(f"Env var {key} already set, skipping.")


def save_to_csv(segments: list, output_path: str) -> str:
    """Save segments to a CSV file"""
    df = pd.DataFrame(segments)
    df = df[df["text"].astype(str).str.strip() != ""]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, sep="\t")
    return output_path


def run_transcription_pipeline(config: Dict) -> None:
    """Execute the full transcription and segmentation pipeline"""
    input_path = config["input_path"]
    output_path = config["output_path"]
    model_name = config["model_name"]
    device = config.get("device", "cpu")
    batch_size = config.get("max_batch_size", 1)
    chunk_duration = config.get("chunk_duration_minutes", 10)
    pause_threshold = config.get("pause_threshold", 2.0)
    similarity_threshold = config.get("similarity_threshold", 0.75)
    embedding_model_name = config.get("embedding_model", "all-MiniLM-L6-v2")
    min_segment_words = config.get("min_segment_words", 3)
    min_segment_chars = config.get("min_segment_chars", 15)
    max_segment_words = config.get("max_segment_words", 100)

    logger.info(f"Input: {input_path} → Output: {output_path}")
    logger.info(
        f"ASR Model: {model_name} | Device: {device} | Chunk: {chunk_duration} min | Pause: {pause_threshold}s"
    )
    logger.info(f"Segmentation Model: {embedding_model_name} | Similarity Thr: {similarity_threshold}")

    audio_path = extract_audio(input_path)
    logger.info(f"Extracted audio to: {audio_path}")

    segments = transcribe_audio(audio_path, model_name, device, chunk_duration, pause_threshold, batch_size=batch_size)
    logger.info(f"Raw segments: {len(segments)}")

    segments = deduplicate_segments(segments)
    logger.info(f"Segments after deduplication: {len(segments)}")

    resolved_device = detect_device(device)
    logger.info(f"Loading SentenceTransformer on: {resolved_device}")
    embedding_model = SentenceTransformer(embedding_model_name, device=resolved_device)

    coherent_segments = segment_coherently(
        segments,
        similarity_threshold=similarity_threshold,
        pause_threshold=int(pause_threshold),
        model=embedding_model,
        max_words=max_segment_words,
        min_words=min_segment_words,
        min_chars=min_segment_chars,
    )
    logger.info(f"Coherent segments: {len(coherent_segments)}")

    final_segments = post_process_segments(
        coherent_segments,
        min_words=min_segment_words,
        min_chars=min_segment_chars,
        max_words=max_segment_words,
    )
    logger.info(f"Final segments after post-processing: {len(final_segments)}")

    # Delete existing output file if it exists
    if os.path.exists(output_path):
        try:
            os.remove(output_path)
            logger.info(f"Removed existing output file: {output_path}")
        except Exception as e:
            logger.warning(f"Could not delete existing output file: {e}")
            
    saved_path = save_to_csv(final_segments, output_path)
    logger.info(f"Saved segments to: {saved_path}")

    if cleanup_temp(input_path, audio_path):
        try:
            os.remove(audio_path)
            logger.info(f"Deleted temporary audio file: {audio_path}")
        except Exception as e:
            logger.warning(f"Could not delete temp file: {e}")


def main():
    """Command-line interface entry point"""
    parser = argparse.ArgumentParser(description="Transcribe and segment audio using open-source ASR models.")
    parser.add_argument("--config", help="Path to YAML config file.")
    parser.add_argument("--gradio", action="store_true", help="Launch the Gradio UI.")
    args = parser.parse_args()

    if args.gradio:
        from textplease.gradio_ui import launch_gradio

        launch_gradio()
        return

    if not args.config:
        raise ValueError("You must provide --config or use --gradio.")

    config = load_config(args.config)
    apply_environment_config(config.get("environment", {}))

    log_level = getattr(logging, config.get("log_level", "INFO").upper(), logging.INFO)
    configure_logging(level=log_level)

    if "model_name" not in config or "input_path" not in config or "output_path" not in config:
        raise ValueError("Config must include 'model_name', 'input_path', and 'output_path'.")

    run_transcription_pipeline(config)


if __name__ == "__main__":
    main()
