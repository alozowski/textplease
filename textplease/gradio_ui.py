import io
import os
import sys
import shutil
import signal
import logging
import subprocess
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


class SubprocessManager:
    """Manages a transcription subprocess with start/stop/kill functionality."""

    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self.process: subprocess.Popen | None = None
        self.output_stream = io.StringIO()
        self.exit_code: int | None = None
        self.config_path: Path | None = None

    def start_process(self, config_path: Path, custom_env: dict | None = None):
        """Start the transcription subprocess."""
        if self.is_running():
            logger.info("Process is already running")
            return

        self.output_stream = io.StringIO()
        self.exit_code = None
        self.config_path = config_path

        command = [sys.executable, "-m", "textplease.main", "--config", str(config_path)]

        env = os.environ.copy()
        if custom_env:
            env.update(custom_env)

        try:
            logger.info(f"Starting process with command: {' '.join(command)}")
            # Use start_new_session to create a process group for proper cleanup
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
                start_new_session=True,  # Creates a new process group
            )
            # Set non-blocking mode for reading
            os.set_blocking(self.process.stdout.fileno(), False)
            logger.info(f"Started process with PID: {self.process.pid}")
        except Exception as e:
            logger.error(f"Failed to start process: {str(e)}")
            raise

    def read_output(self) -> str:
        """Read subprocess output and return current log."""
        if self.process and self.process.stdout:
            try:
                while True:
                    line = self.process.stdout.readline()
                    if line:
                        self.output_stream.write(line)
                    else:
                        break
            except BlockingIOError:
                pass

        return self.output_stream.getvalue()

    def stop_process(self):
        """Terminate the subprocess gracefully."""
        if not self.is_running():
            logger.info("Process is not running")
            return

        logger.info("Sending SIGTERM to the process group")
        try:
            # Send SIGTERM to the entire process group
            pid = self.process.pid
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
                logger.info(f"Sent SIGTERM to process group {pid}")
            except ProcessLookupError:
                logger.warning(f"Process group {pid} not found, trying direct termination")
                self.process.terminate()

            self.exit_code = self.process.wait(timeout=5)
            logger.info(f"Process terminated with exit code {self.exit_code}")
        except subprocess.TimeoutExpired:
            logger.warning("Process did not terminate within timeout, sending SIGKILL")
            self.kill_process()

    def kill_process(self):
        """Forcefully kill the subprocess and all its children."""
        if not self.is_running():
            logger.info("Process is not running")
            return

        logger.info("Sending SIGKILL to the process group")
        try:
            # Send SIGKILL to the entire process group to ensure all children are killed
            pid = self.process.pid
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
                logger.info(f"Sent SIGKILL to process group {pid}")
            except ProcessLookupError:
                logger.warning(f"Process group {pid} not found, trying direct kill")
                self.process.kill()

            self.exit_code = self.process.wait(timeout=5)
            logger.info(f"Process killed with exit code {self.exit_code}")
        except subprocess.TimeoutExpired:
            logger.error("Process could not be killed within timeout")
        except Exception as e:
            logger.error(f"Error during kill: {e}")
            # Fallback to direct kill
            try:
                self.process.kill()
                self.exit_code = self.process.wait(timeout=2)
            except:
                pass

    def is_running(self) -> bool:
        """Check if the subprocess is still running."""
        if self.process is None:
            return False
        return self.process.poll() is None

    def get_exit_details(self) -> tuple[int | None, str]:
        """Return exit code and reason if process has terminated."""
        if self.process is None:
            return None, "Process was never started"

        if self.is_running():
            return None, "Process is still running"

        if self.exit_code is not None and self.exit_code != 0:
            return self.exit_code, "Process exited abnormally"

        return self.exit_code, "Process exited normally"

    def __del__(self):
        """Cleanup when object is deleted."""
        if self.process and self.is_running():
            try:
                # Try to kill the entire process group
                pid = self.process.pid
                try:
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                except:
                    self.process.kill()
            except:
                pass


# Global subprocess manager
PROCESS_MANAGER = SubprocessManager()


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


def start_task(
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
    """Start transcription process in subprocess."""
    if audio_file is None:
        return "❌ Please upload an audio file.", gr.update(visible=False), None

    if PROCESS_MANAGER.is_running():
        return "⚠️ A transcription is already running. Please stop it first.", gr.update(visible=False), None

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

        config_path = OUTPUT_DIR / f"{input_path.stem}_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        PROCESS_MANAGER.start_process(config_path)

        return (
            f"✅ Transcription started!\n📄 Output will be saved to: {output_path}",
            gr.update(visible=False),
            str(output_path),
        )

    except (ValueError, FileNotFoundError, IOError) as e:
        return f"❌ File error: {e}", gr.update(visible=False), None
    except Exception as e:
        logger.error(f"Unexpected error starting transcription: {e}", exc_info=True)
        return f"❌ Unexpected error: {e}", gr.update(visible=False), None


def stop_task():
    """Stop the running transcription process."""
    if not PROCESS_MANAGER.is_running():
        return "⚠️ No process is currently running."

    PROCESS_MANAGER.stop_process()
    return "🛑 Process stopped."


def update_process_status():
    """Update process status for display."""
    if not PROCESS_MANAGER.is_running():
        exit_code, exit_reason = PROCESS_MANAGER.get_exit_details()
        if exit_reason == "Process was never started":
            status_text = "Process Status: Not started"
        else:
            status_text = f"Process Status: Stopped - {exit_reason}"
            if exit_code is not None:
                status_text += f", exit code: {exit_code}"
        return gr.update(value=False, label=status_text)

    return gr.update(value=True, label="Process Status: Running")


def update_log_output(transcript_state):
    """Update log output and check for completion."""
    log_text = PROCESS_MANAGER.read_output()

    # Check if process completed successfully
    if not PROCESS_MANAGER.is_running() and transcript_state:
        output_path = Path(transcript_state)
        if output_path.exists():
            # Process completed successfully
            return (
                log_text,
                gr.update(value=output_path, visible=True),
                gr.update(interactive=True, value=True),
            )

    return log_text, gr.update(visible=False), gr.update(interactive=False, value=False)


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
    """Preview transcript file.
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
                step=5,
                label="Min Segment Characters",
                info="Minimum characters per segment (default: 15)",
            )

        gr.Markdown("### Process Control")
        with gr.Row():
            start_button = gr.Button("🚀 Start Task", variant="primary")
            stop_button = gr.Button("🛑 Stop Task")

        process_status = gr.Checkbox(label="Process Status: Not started", interactive=False)
        status_text = gr.Textbox(label="Status", value="Ready to start...", interactive=False, lines=2)

        # Log output with auto-scroll
        with gr.Accordion("📋 Process Log", open=False):
            log_output = gr.Code(language=None, lines=15, interactive=False)

        # Download and preview
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

        # Start task
        start_button.click(
            start_task,
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
                transcript_state,
            ],
        )

        # Stop task
        stop_button.click(stop_task, outputs=status_text)

        # Auto-update status and logs
        status_timer = gr.Timer(2.0, active=True)
        status_timer.tick(update_process_status, outputs=process_status)

        log_timer = gr.Timer(1.0, active=True)
        log_timer.tick(
            update_log_output,
            inputs=[transcript_state],
            outputs=[log_output, download_button, show_transcript],
        )

        # Preview transcript when checkbox is toggled
        show_transcript.change(
            preview_transcript,
            inputs=[show_transcript, transcript_state],
            outputs=csv_preview,
        )

        # Clear functionality
        clear_btn = gr.Button("🧹 Clear All")

        def clear_all():
            # Stop any running process
            if PROCESS_MANAGER.is_running():
                PROCESS_MANAGER.stop_process()

            return (
                None,                                       # audio_input
                None,                                       # audio_preview
                "Ready to start...",                        # status_text
                gr.update(visible=False),                   # download_button
                gr.update(value=False, interactive=False),  # show_transcript
                gr.update(visible=False, value=None),       # csv_preview
                None,                                       # transcript_state
                "",                                         # log_output
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
                log_output,
            ],
        )

    demo.launch(share=True)


if __name__ == "__main__":
    launch_gradio()
