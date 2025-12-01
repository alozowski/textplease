import shutil
import logging
import traceback
from pathlib import Path

import yaml
import gradio as gr
import pandas as pd
from pydub.utils import mediainfo  # type: ignore

from textplease.utils.device_utils import detect_device


logger = logging.getLogger(__name__)

# Ensure required directories exist
OUTPUT_DIR = Path("output")
INPUT_DIR = Path("input")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
INPUT_DIR.mkdir(parents=True, exist_ok=True)


def _prepare_input_files(audio_file) -> tuple[Path, Path]:
    """Prepare input files for transcription."""
    if audio_file is None:
        raise ValueError("No audio file provided")

    if not hasattr(audio_file, "name") or not audio_file.name:
        raise ValueError("Invalid audio file object")

    source_path = Path(audio_file.name)
    if not source_path.exists():
        raise FileNotFoundError(f"Uploaded file not found: {audio_file.name}")

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    input_path = INPUT_DIR / source_path.name

    try:
        shutil.copyfile(audio_file.name, input_path)
    except Exception as e:
        logger.error(f"Failed to copy file: {e}")
        raise IOError(f"Failed to copy audio file: {e}") from e

    output_path = OUTPUT_DIR / f"{input_path.stem}_transcript.csv"
    return input_path, output_path


def _create_transcription_config(
    input_path: Path,
    output_path: Path,
    chunk_duration_minutes: float,
    max_batch_size: int,
    similarity_threshold: float,
    pause_threshold: float,
    max_segment_words: int,
    min_segment_words: int,
    min_segment_chars: int,
    language: str,
    model_name: str,
    device: str,
) -> dict:
    """Create transcription configuration dictionary."""
    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "model_name": model_name,
        "device": device,
        "chunk_duration_minutes": chunk_duration_minutes,
        "max_batch_size": max_batch_size,
        "similarity_threshold": similarity_threshold,
        "pause_threshold": pause_threshold,
        "max_segment_words": max_segment_words,
        "min_segment_words": min_segment_words,
        "min_segment_chars": min_segment_chars,
        "embedding_model": "all-mpnet-base-v2",
        "log_level": "INFO",
        "language": language,
    }


def _execute_and_report(
    config: dict,
    input_path: Path,
    output_path: Path,
) -> tuple[str, str | None]:
    """Execute transcription and report results."""
    config_path = OUTPUT_DIR / f"{input_path.stem}_config.yaml"

    try:
        with open(config_path, "w") as f:
            yaml.dump(config, f)
    except Exception as e:
        logger.error(f"Failed to save config: {e}")
        return f"❌ Error saving config: {e}", None

    try:
        # Late import to avoid circular dependency
        from textplease.main import run_transcription_pipeline
        run_transcription_pipeline(config)

        if not output_path.exists():
            raise RuntimeError(
                f"Transcription completed but output file not found: {output_path}"
            )

        return (
            f"✅ Transcription complete!\n"
            f"📄 Transcript saved to: `{output_path}`\n"
            f"🛠️ Configuration saved to: `{config_path}`",
            str(output_path),
        )
    except Exception as e:
        logger.error(f"Transcription failed: {e}", exc_info=True)
        logger.debug(traceback.format_exc())
        return f"❌ Error: {e}", None
    finally:
        if INPUT_DIR.exists():
            try:
                shutil.rmtree(INPUT_DIR, ignore_errors=True)
            except Exception as e:
                logger.warning(f"Failed to cleanup input directory: {e}")


def start_transcription(
    audio_file,
    chunk_duration_minutes,
    max_batch_size,
    similarity_threshold,
    pause_threshold,
    max_segment_words,
    min_segment_words,
    min_segment_chars,
    language,
    model_name,
    device,
):
    """Start transcription process with proper file handling and configuration."""
    if audio_file is None:
        return (
            "❌ Please upload an audio file.",
            gr.update(visible=False),                    # download_button
            gr.update(interactive=False, value=False),   # show_transcript
            gr.update(visible=False, value=None),        # csv_preview
            None,                                        # transcript_state
        )

    try:
        input_path, output_path = _prepare_input_files(audio_file)
        config = _create_transcription_config(
            input_path,
            output_path,
            chunk_duration_minutes,
            max_batch_size,
            similarity_threshold,
            pause_threshold,
            max_segment_words,
            min_segment_words,
            min_segment_chars,
            language,
            model_name,
            device,
        )
        status_msg, output_file = _execute_and_report(config, input_path, output_path)

        if output_file:
            # Success: show download button, enable and check preview checkbox, show preview
            # Load and display the preview immediately
            try:
                df = pd.read_csv(output_file, sep="\t")
                head = df.head(10)
                preview_data = gr.update(
                    visible=True,
                    value={
                        "data": head.values.tolist(),
                        "headers": list(head.columns),
                    },
                )
            except Exception as e:
                logger.error(f"Failed to load preview: {e}")
                preview_data = gr.update(
                    visible=True,
                    value=[[f"Error loading preview: {str(e)}"]],
                )

            return (
                status_msg,
                gr.update(value=output_file, visible=True),  # Show download button
                gr.update(interactive=True, value=True),      # Enable AND check the checkbox
                preview_data,                                 # Show the preview immediately
                output_file,  # state with transcript path
            )
        else:
            return (
                status_msg,
                gr.update(visible=False),  # Hide download button
                gr.update(interactive=False, value=False),
                gr.update(visible=False, value=None),
                None,
            )

    except (ValueError, FileNotFoundError, IOError) as e:
        return (
            f"❌ File error: {e}",
            gr.update(visible=False),
            gr.update(interactive=False, value=False),
            gr.update(visible=False, value=None),
            None,
        )
    except Exception as e:
        logger.error(f"Unexpected error in start_transcription: {e}", exc_info=True)
        return (
            f"❌ Unexpected error: {e}",
            gr.update(visible=False),
            gr.update(interactive=False, value=False),
            gr.update(visible=False, value=None),
            None,
        )


def show_audio_info(file):
    """Display audio file information."""
    if file is None:
        return "🧹 Input cleared!"

    try:
        info = mediainfo(file.name)
        duration = round(float(info.get("duration", 0)), 2)
        sample_rate = info.get("sample_rate", "Unknown")
        channels = info.get("channels", "Unknown")
        return (
            f"🕒 Duration: {duration}s\n"
            f"📊 Sample rate: {sample_rate} Hz\n"
            f"🔊 Channels: {channels}"
        )
    except Exception as e:
        logger.error(f"Failed to get audio info: {e}")
        return f"⚠️ Could not read audio info: {e}"


def preview_transcript(show: bool, file_path: str | None):
    r"""Preview transcript file.

    The file is named *.csv but content is tab-separated, so we load with sep="\t".
    """
    logger.info(
        f"preview_transcript called: show={show}, file_path={file_path}, type={type(file_path)}"
    )

    if not show:
        return gr.update(visible=False)

    if not file_path:
        logger.warning("Preview requested but no file path provided")
        return gr.update(
            visible=True,
            value=[["Output file not found. Please run transcription first."]],
        )

    path = Path(file_path)
    logger.info(f"Checking file path for preview: {path}")

    if not path.exists():
        logger.warning(f"Preview requested but file doesn't exist: {path}")
        return gr.update(
            visible=True,
            value=[["Output file not found. Please wait for transcription to complete."]],
        )

    try:
        logger.info(f"Reading TSV/CSV from: {path}")
        df = pd.read_csv(path, sep="\t")  # tab-delimited content

        if df.empty:
            return gr.update(
                visible=True,
                value=[["Transcript is empty"]],
            )

        head = df.head(10)
        logger.info(f"Successfully loaded {len(df)} segments, showing first {len(head)}")

        return gr.update(
            visible=True,
            value={
                "data": head.values.tolist(),
                "headers": list(head.columns),
            },
        )

    except Exception as e:
        logger.error(f"CSV preview error: {e}", exc_info=True)
        return gr.update(
            visible=True,
            value=[[f"Error loading CSV: {str(e)}"]],
        )


def update_model_choices(language):
    """Update available models based on selected language."""
    if language == "en":
        choices = [
            "openai/whisper-large-v3",
            "nvidia/parakeet-ctc-1.1b",
        ]
        value = "openai/whisper-large-v3"
    else:
        choices = ["openai/whisper-large-v3"]
        value = "openai/whisper-large-v3"

    return gr.update(choices=choices, value=value)


def launch_gradio():
    """Launch the Gradio web interface."""
    try:
        best_device = detect_device("cuda")
        logger.info(f"Best available device detected: {best_device}")
    except Exception as e:
        logger.warning(f"Device detection failed, defaulting to CPU: {e}")
        best_device = "cpu"

    with gr.Blocks(title="textplease transcriber") as demo:
        gr.Markdown("## 🎙️ text, please!")
        gr.Markdown("Upload an audio file, configure settings, and receive a transcript 📝")

        with gr.Row():
            audio_input = gr.File(
                label="Upload Audio (.mp3/.wav/.mp4)",
                file_types=[".mp3", ".wav", ".mp4"],
            )
            audio_preview = gr.Audio(label="Audio Preview", interactive=False)

        audio_input.change(lambda f: f, inputs=audio_input, outputs=audio_preview)
        audio_info_box = gr.Textbox(label="Audio Info", lines=3)
        audio_input.change(show_audio_info, inputs=audio_input, outputs=audio_info_box)

        gr.Markdown("### Settings")
        with gr.Row():
            language = gr.Dropdown(
                choices=["en", "ru"],
                value="en",
                label="Language",
                info="Select transcription language",
            )
            model_name = gr.Dropdown(
                choices=["openai/whisper-large-v3", "nvidia/parakeet-ctc-1.1b"],
                value="openai/whisper-large-v3",
                label="Model",
                info="Whisper: Multi-language | NeMo (Parakeet): English only",
            )
            device = gr.Dropdown(
                choices=["cpu", "cuda", "mps"],
                value=best_device,
                label="Device",
                info="CPU: Universal (slow) | CUDA: NVIDIA GPU | MPS: Apple Silicon GPU",
            )

        with gr.Accordion("⚙️ Advanced Settings", open=False):
            chunk_duration_minutes = gr.Slider(
                1,
                60,
                value=10,
                step=1,
                label="Chunk Duration (minutes)",
                info="Audio split size for processing (default: 10 min)",
            )
            max_batch_size = gr.Slider(
                1,
                8,
                value=1,
                step=1,
                label="Max Batch Size",
                info="Ignored by Whisper backend, kept for compatibility",
            )
            similarity_threshold = gr.Slider(
                0.0,
                1.0,
                step=0.01,
                value=0.75,
                label="Similarity Threshold",
                info="Higher = more segments split (0.75 recommended)",
            )
            pause_threshold = gr.Slider(
                0.0,
                10.0,
                step=0.1,
                value=2.0,
                label="Pause Threshold (seconds)",
                info="Max pause duration to allow merging segments (longer pauses force splits, default: 2s)",
            )
            max_segment_words = gr.Slider(
                10,
                200,
                step=5,
                value=100,
                label="Max Segment Words",
                info="Maximum words per segment (default: 100)",
            )
            min_segment_words = gr.Slider(
                1,
                20,
                value=3,
                step=1,
                label="Min Segment Words",
                info="Minimum words per segment (default: 3)",
            )
            min_segment_chars = gr.Slider(
                1,
                100,
                value=15,
                step=1,
                label="Min Segment Characters",
                info="Minimum characters per segment (default: 15)",
            )

        run_button = gr.Button("🚀 Start Transcription", variant="primary")
        clear_btn = gr.Button("🧹 Clear Inputs")
        status_text = gr.Textbox(label="Status", value="Waiting...", interactive=False, lines=3)

        # Use DownloadButton for file downloads
        download_button = gr.DownloadButton(
            label="📥 Download Transcript",
            visible=False,
        )

        show_transcript = gr.Checkbox(
            label="📄 Show Transcript Preview",
            value=False,
            interactive=False,
        )

        csv_preview = gr.Dataframe(
            label="Transcript Preview",
            visible=False,
            interactive=False,
            datatype="str",
        )

        transcript_state = gr.State(value=None)

        # Update model choices when language changes
        language.change(
            update_model_choices,
            inputs=[language],
            outputs=[model_name],
        )

        # Main transcription event
        run_button.click(
            start_transcription,
            inputs=[
                audio_input,
                chunk_duration_minutes,
                max_batch_size,
                similarity_threshold,
                pause_threshold,
                max_segment_words,
                min_segment_words,
                min_segment_chars,
                language,
                model_name,
                device,
            ],
            outputs=[
                status_text,
                download_button,
                show_transcript,
                csv_preview,
                transcript_state,
            ],
            show_progress="full",
        )

        def clear_all():
            return (
                None,                                   # audio_input
                None,                                   # audio_preview
                "Waiting...",                           # status_text
                gr.update(visible=False),               # download_button
                gr.update(value=False, interactive=False),  # show_transcript
                gr.update(visible=False, value=None),   # csv_preview
                None,                                   # transcript_state
            )

        clear_btn.click(
            clear_all,
            outputs=[
                audio_input,
                audio_preview,
                status_text,
                download_button,
                show_transcript,
                csv_preview,
                transcript_state,
            ],
        )

        # Preview transcript when checkbox is toggled
        show_transcript.change(
            preview_transcript,
            inputs=[show_transcript, transcript_state],
            outputs=csv_preview,
        )

        download_button.click(
            lambda file_path: file_path,  # Return the file path from state
            inputs=[transcript_state],
            outputs=[download_button],
        )

    demo.launch(share=True)


if __name__ == "__main__":
    launch_gradio()
