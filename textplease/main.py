import os
import time
import logging
import argparse
from pathlib import Path

import yaml
import pandas as pd
from sentence_transformers import SentenceTransformer

from textplease.gradio_ui import launch_gradio
from textplease.segmenter import segment_transcript, post_process_segments
from textplease.transcriber import transcribe_audio
from textplease.utils.audio_utils import cleanup_temp, extract_audio
from textplease.utils.device_utils import detect_device
from textplease.utils.deduplicate_segments import deduplicate_segments


logger = logging.getLogger(__name__)
__all__ = ["main"]


def load_config(path: str) -> dict:
    """Load configuration from a YAML file."""
    if not path or not isinstance(path, str):
        raise ValueError(f"Invalid config path: {path}")

    config_file = Path(path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    try:
        with open(path, "r") as f:
            config = yaml.safe_load(f)
            if config is None:
                raise ValueError(f"Config file is empty: {path}")
            return config
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse YAML config: {e}")
        raise ValueError(f"Invalid YAML format in {path}: {e}") from e
    except Exception as e:
        logger.error(f"Failed to load configuration file: {e}")
        raise


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


def save_to_csv(segments: list, output_path: str) -> str:
    """Save segments to a CSV file.

    Args:
        segments: List of segment dictionaries to save
        output_path: Path where the CSV file should be saved

    Returns:
        Path to the saved CSV file

    Raises:
        ValueError: If segments is empty or output_path is invalid
        IOError: If file writing fails

    """
    if not segments:
        raise ValueError("Cannot save empty segments list")

    if not output_path or not isinstance(output_path, str):
        raise ValueError(f"Invalid output_path: {output_path}")

    try:
        df = pd.DataFrame(segments)

        # Filter out empty text segments
        df = df[df["text"].astype(str).str.strip() != ""]

        if df.empty:
            logger.warning("All segments were empty after filtering")
            raise ValueError("No valid segments to save (all were empty)")

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        df.to_csv(output_path, index=False, sep="\t")
        logger.info(f"Saved {len(df)} segments to {output_path}")
        return output_path

    except (ValueError, IOError):
        raise
    except Exception as e:
        logger.error(f"Failed to save CSV to {output_path}: {e}")
        raise IOError(f"Failed to save CSV: {e}") from e


def estimate_processing_time(num_segments: int) -> str:
    """Estimate processing time based on segment count."""
    if num_segments < 100:
        return "< 1 minute"
    elif num_segments < 500:
        return "1–3 minutes"
    elif num_segments < 1000:
        return "3–5 minutes"
    elif num_segments < 5000:
        return "5–15 minutes"
    else:
        return "15+ minutes"


def _validate_pipeline_config(config: dict) -> None:
    """Validate configuration for the transcription pipeline.

    Args:
        config: Configuration dictionary to validate

    Raises:
        ValueError: If required config keys are missing or invalid

    """
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a dictionary, got {type(config)}")

    required_keys = {"input_path", "output_path", "model_name"}
    missing_keys = required_keys - set(config.keys())
    if missing_keys:
        raise ValueError(f"Config missing required keys: {missing_keys}")

    # Verify input file exists
    input_path = config["input_path"]
    if not input_path or not isinstance(input_path, str):
        raise ValueError(f"Invalid input_path: {input_path}")

    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")


def _extract_config_params(config: dict) -> dict:
    """Extract and organize pipeline parameters from config."""
    return {
        'input_path': config["input_path"],
        'output_path': config["output_path"],
        'model_name': config["model_name"],
        'device': config.get("device", "cpu"),
        'batch_size': config.get("max_batch_size", 1),
        'chunk_duration': config.get("chunk_duration_minutes", 10),
        'pause_threshold': config.get("pause_threshold", 2.0),
        'similarity_threshold': config.get("similarity_threshold", 0.75),
        'embedding_model_name': config.get("embedding_model", "all-MiniLM-L6-v2"),
        'min_segment_words': config.get("min_segment_words", 3),
        'min_segment_chars': config.get("min_segment_chars", 15),
        'max_segment_words': config.get("max_segment_words", 100),
        'language': config.get("language", "en"),
        'similarity_batch_size': config.get("performance", {}).get("similarity_batch_size", 32),
        'chunk_size': config.get("performance", {}).get("chunk_size", 1000),
    }


def _log_pipeline_config(params: dict) -> None:
    """Log pipeline configuration parameters."""
    logger.info(f"Input: {params['input_path']} → Output: {params['output_path']}")
    logger.info(
        f"ASR Model: {params['model_name']} | Device: {params['device']} | "
        f"Chunk: {params['chunk_duration']} min | Pause: {params['pause_threshold']}s"
    )
    logger.info(
        f"Segmentation Model: {params['embedding_model_name']} | "
        f"Similarity Thr: {params['similarity_threshold']}"
    )
    logger.info(
        f"Performance: Batch size: {params['similarity_batch_size']}, "
        f"Chunk size: {params['chunk_size']}"
    )


def _load_embedding_model(config: dict, device: str) -> SentenceTransformer:
    """Load the embedding model for semantic segmentation.

    Args:
        config: Configuration dictionary
        device: Target device for model loading

    Returns:
        Loaded SentenceTransformer model

    """
    model_load_start = time.time()
    logger.info(f"Loading SentenceTransformer on: {device}")
    embedding_model = SentenceTransformer(config['embedding_model_name'], device=device)
    logger.info(f"Model loading took {time.time() - model_load_start:.2f}s")
    return embedding_model


def _execute_transcription_stage(config: dict, audio_path: str) -> list[dict]:
    """Execute transcription stage with deduplication.

    Args:
        config: Configuration parameters
        audio_path: Path to extracted audio file

    Returns:
        List of transcribed segments

    """
    transcription_start = time.time()
    segments = transcribe_audio(
        audio_path,
        config['model_name'],
        config['device'],
        config['chunk_duration'],
        config['pause_threshold'],
        batch_size=config['batch_size'],
        language=config['language'],
    )
    logger.info(f"Transcription took {time.time() - transcription_start:.2f}s for {len(segments)} segments")

    dedup_start = time.time()
    segments = deduplicate_segments(segments)
    logger.info(f"Deduplication took {time.time() - dedup_start:.2f}s, {len(segments)} segments remaining")

    return segments


def _execute_segmentation_stage(segments: list, config: dict, model: SentenceTransformer) -> list[dict]:
    """Execute semantic segmentation stage.

    Args:
        segments: Initial transcription segments
        config: Configuration parameters
        model: Loaded embedding model

    Returns:
        List of semantically coherent segments

    """
    logger.info(f"Estimated segmentation time: {estimate_processing_time(len(segments))}")

    segmentation_start = time.time()

    # Use segment count + 1 if chunk_size is 0 or negative (process all at once)
    effective_chunk_size = (
        config['chunk_size'] if config['chunk_size'] > 0
        else len(segments) + 1
    )

    coherent_segments = segment_transcript(
        segments,
        similarity_threshold=config['similarity_threshold'],
        pause_threshold=config['pause_threshold'],
        model=model,
        max_words=config['max_segment_words'],
        min_words=config['min_segment_words'],
        min_chars=config['min_segment_chars'],
        embedding_model_name=config['embedding_model_name'],
        preferred_device=config['device'],
        batch_size=config['similarity_batch_size'],
        chunk_size=effective_chunk_size,
    )
    logger.info(f"Segmentation took {time.time() - segmentation_start:.2f}s, {len(coherent_segments)} segments")

    return coherent_segments


def _execute_post_processing(segments: list, config: dict) -> list[dict]:
    """Execute post-processing stage.

    Args:
        segments: Segmented transcript
        config: Configuration parameters

    Returns:
        List of final processed segments

    """
    postprocess_start = time.time()
    final_segments = post_process_segments(
        segments,
        min_words=config['min_segment_words'],
        min_chars=config['min_segment_chars'],
        max_words=config['max_segment_words'],
    )
    logger.info(f"Post-processing took {time.time() - postprocess_start:.2f}s, {len(final_segments)} final segments")
    return final_segments


def _save_and_cleanup(final_segments: list, config: dict, audio_path: str) -> None:
    """Save results and cleanup temporary files.

    Args:
        final_segments: Final processed segments
        config: Configuration parameters
        audio_path: Path to temporary audio file

    """
    save_start = time.time()

    # Remove existing output file if it exists
    if os.path.exists(config['output_path']):
        try:
            os.remove(config['output_path'])
            logger.info(f"Removed existing output file: {config['output_path']}")
        except Exception as e:
            logger.warning(f"Could not delete existing output file: {e}")

    save_to_csv(final_segments, config['output_path'])
    logger.info(f"Saving took {time.time() - save_start:.2f}s")

    # Cleanup temporary audio file if it was created
    if cleanup_temp(config['input_path'], audio_path):
        try:
            os.remove(audio_path)
            logger.info(f"Deleted temporary audio file: {audio_path}")
        except Exception as e:
            logger.warning(f"Could not delete temp file: {e}")


def run_transcription_pipeline(config: dict) -> None:
    """Run the complete transcription pipeline.

    Args:
        config: Configuration dictionary with required keys

    Raises:
        ValueError: If required config keys are missing
        Exception: If any pipeline stage fails

    """
    start_time = time.time()

    # Validate configuration
    _validate_pipeline_config(config)

    # Extract and log configuration
    params = _extract_config_params(config)
    _log_pipeline_config(params)

    # Extract audio (converts video to audio if needed)
    extraction_start = time.time()
    audio_path = extract_audio(params['input_path'])
    logger.info(f"Audio extraction took {time.time() - extraction_start:.2f}s")

    # Load embedding model for semantic segmentation
    resolved_device = detect_device(params['device'])
    embedding_model = _load_embedding_model(params, resolved_device)

    # Execute pipeline stages
    segments = _execute_transcription_stage(params, audio_path)
    coherent_segments = _execute_segmentation_stage(segments, params, embedding_model)
    final_segments = _execute_post_processing(coherent_segments, params)
    _save_and_cleanup(final_segments, params, audio_path)

    logger.info(f"Total processing time: {time.time() - start_time:.2f}s")


def main():
    """Provide main entry point for the textplease CLI."""
    parser = argparse.ArgumentParser(description="Transcribe and segment audio using open-source ASR models.")
    parser.add_argument("--config", help="Path to YAML config file.")
    parser.add_argument("--gradio", action="store_true", help="Launch the Gradio UI instead of CLI pipeline.")

    try:
        args = parser.parse_args()

        if args.gradio:
            try:
                launch_gradio()
            except NameError:
                raise ImportError("Gradio UI dependencies not installed. Install with: uv add gradio") from None
            return

        # Validate config argument is provided
        if not args.config:
            parser.error("--config is required when not using --gradio")

        config = load_config(args.config)

        # Apply environment config if present
        env_config = config.get("environment", {})
        if env_config:
            apply_environment_config(env_config)

        # Configure logging level from config
        log_level = config.get("log_level", "INFO").upper()
        logging.basicConfig(
            level=getattr(logging, log_level, logging.INFO),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        run_transcription_pipeline(config)

    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        return
    except Exception as e:
        logger.error(f"Application failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
