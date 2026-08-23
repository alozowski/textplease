import re
import time
import shutil
import logging
import tempfile
from pathlib import Path
from functools import partial

import yaml
import gradio as gr
import pandas as pd
from pydub.utils import mediainfo
from pydub.exceptions import CouldntDecodeError

from textplease.pipeline import DEFAULT_EMBEDDING_MODEL
from textplease.gradio_worker import CANCELLED_ERROR, PersistentPipelineWorker
from textplease.utils.device_utils import detect_device


logger = logging.getLogger(__name__)
DEFAULT_MODEL = "openai/whisper-large-v3"

# Whisper backend logs one line per VAD segment — the child's log file is the progress signal.
_PROGRESS_RE = re.compile(r"Transcribing speech segment (\d+)/(\d+)")

# Gradio has no native button tooltips, set browser title attributes on load.
_TOOLTIP_JS = """
() => {
    const tips = {
        "run-btn": "Transcribe the uploaded audio",
        "cancel-btn": "Cancel the running transcription",
        "clear-btn": "Delete this job's local artifacts and reset the form",
    };
    const apply = () => {
        let missing = false;
        for (const [id, tip] of Object.entries(tips)) {
            const el = document.getElementById(id);
            if (el) {
                el.title = tip;
                el.querySelectorAll("button").forEach((b) => (b.title = tip));
            } else {
                missing = true;
            }
        }
        if (missing) setTimeout(apply, 500);
    };
    apply();
}
"""

LANGUAGE_CHOICES = (
    ("English", "en"),
    ("Russian", "ru"),
    ("Spanish", "es"),
    ("French", "fr"),
    ("Italian", "it"),
    ("German", "de"),
    ("Turkish", "tr"),
    ("Chinese", "zh"),
    ("Korean", "ko"),
    ("Japanese", "ja"),
)


def _delete_job_workspace(output_directory: Path, workspace_path: Path) -> None:
    resolved_output_directory = output_directory.resolve()
    if workspace_path.parent.resolve() != resolved_output_directory or not workspace_path.name.startswith("job-"):
        raise ValueError(f"Refusing to delete unowned job workspace: {workspace_path}")
    if workspace_path.is_symlink():
        workspace_path.unlink()
    elif workspace_path.exists():
        shutil.rmtree(workspace_path)


def start_transcription(
    worker: PersistentPipelineWorker,
    output_directory: Path,
    audio_file: str | None,
    similarity_threshold: float,
    pause_threshold: float,
    max_segment_words: int,
    min_segment_words: int,
    min_segment_chars: int,
    language: str,
    device: str,
    run: dict | None,
) -> tuple[object, ...]:
    """Launch one isolated transcription job."""

    def error_result(message: str) -> tuple[object, ...]:
        return (
            message,
            gr.update(visible=False),
            gr.update(interactive=False, value=False),
            gr.update(visible=False, value=None),
            None,
            gr.update(interactive=True),
            gr.update(visible=False),
            gr.update(active=False),
            None,
        )

    if audio_file is None:
        return error_result("❌ Please upload an audio file.")

    input_path = Path(audio_file)
    if not input_path.is_file():
        return error_result(f"❌ Uploaded file not found: {audio_file}")

    if run is not None and worker.is_running(run["job_id"]):
        return (
            "⏳ A transcription is already running.",
            gr.update(),
            gr.update(),
            gr.update(),
            None,
            gr.update(interactive=False),
            gr.update(visible=True),
            gr.update(active=True),
            run,
        )

    output_directory.mkdir(parents=True, exist_ok=True)
    workspace_path = Path(tempfile.mkdtemp(prefix="job-", dir=output_directory))
    output_path = workspace_path / f"{input_path.stem}_transcript.csv"
    config_path = workspace_path / "config.yaml"
    log_path = workspace_path / "run.log"
    config = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "model_name": DEFAULT_MODEL,
        "device": device,
        "similarity_threshold": similarity_threshold,
        "pause_threshold": pause_threshold,
        "max_segment_words": max_segment_words,
        "min_segment_words": min_segment_words,
        "min_segment_chars": min_segment_chars,
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
        "log_level": "INFO",
        "language": language,
    }
    try:
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        config_path.chmod(0o600)
        log_path.touch(mode=0o600)
        job_id = worker.submit(config, str(log_path))
    except (OSError, RuntimeError, yaml.YAMLError) as error:
        _delete_job_workspace(output_directory, workspace_path)
        return error_result(f"❌ Could not start transcription: {error}")

    active_run = {
        "job_id": job_id,
        "workspace_path": workspace_path,
        "output_path": output_path,
        "config_path": config_path,
        "log_path": log_path,
        "started": time.time(),
    }
    logger.info("Transcription process started (pid=%s)", worker.pid)

    return (
        "⏳ Transcription starting...",
        gr.update(visible=False),
        gr.update(interactive=False, value=False),
        gr.update(visible=False, value=None),
        None,
        gr.update(interactive=False),
        gr.update(visible=True),
        gr.update(active=True),
        active_run,
    )


def check_completion(
    worker: PersistentPipelineWorker,
    run: dict | None,
    transcript_path: str | None,
) -> tuple[object, ...]:
    """Poll the transcription process; report progress, then surface results and hide run controls."""
    noop = (
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        transcript_path,
        gr.update(),
        gr.update(),
        gr.update(),
        run,
    )
    if run is None:
        return noop

    done, error = worker.result(run["job_id"])
    if not done:
        elapsed = int(time.time() - run["started"])
        try:
            log_text = run["log_path"].read_text()
        except OSError:
            log_text = ""

        if "Starting segmentation" in log_text:
            status = f"⏳ Segmenting transcript... · {elapsed}s elapsed"
        elif matches := _PROGRESS_RE.findall(log_text):
            # The log line marks a segment *starting*, so one fewer is actually complete.
            current, total = int(matches[-1][0]), int(matches[-1][1])
            completed = current - 1
            filled = round(10 * completed / total)
            bar = "▰" * filled + "▱" * (10 - filled)
            eta = f" · ~{int(elapsed * (total - completed) / completed)}s left" if completed else ""
            status = f"⏳ Transcribing {bar} segment {current}/{total} · {elapsed}s elapsed{eta}"
        else:
            status = f"⏳ Loading model... · {elapsed}s elapsed"
        return (status, *noop[1:])

    output_path = run["output_path"]

    completed = (
        gr.update(interactive=True),
        gr.update(visible=False),
        gr.update(active=False),
        run,
    )

    if error is None and output_path.exists():
        try:
            transcript = pd.read_csv(output_path, sep="\t", nrows=1)
        except (OSError, UnicodeError, pd.errors.EmptyDataError, pd.errors.ParserError) as output_error:
            error = f"could not read transcript: {output_error}"
        else:
            required_columns = ["start_time", "end_time", "text"]
            if list(transcript.columns[:3]) != required_columns:
                error = f"invalid transcript columns: expected {', '.join(required_columns)}"
            else:
                no_speech = transcript.empty
                status = (
                    f"✅ No speech was transcribed.\n📄 Empty transcript saved to: `{output_path}`"
                    if no_speech
                    else f"✅ Transcription complete!\n📄 Transcript saved to: `{output_path}`"
                )
                return (
                    f"{status}\n🛠️ Configuration saved to: `{run['config_path']}`",
                    gr.update(value=str(output_path), visible=True),
                    gr.update(interactive=True, value=True),
                    preview_transcript(True, str(output_path)),
                    str(output_path),
                    *completed,
                )

    if error == CANCELLED_ERROR:
        status = "🛑 Transcription cancelled"
    else:
        status = f"❌ Transcription failed ({error}) — see `{run['log_path']}`"
    return (
        status,
        gr.update(visible=False),
        gr.update(interactive=False, value=False),
        gr.update(visible=False, value=None),
        None,
        *completed,
    )


def cancel_transcription(worker: PersistentPipelineWorker, run: dict | None) -> str:
    """Cancel the active transcription process."""
    if run and worker.is_running(run["job_id"]):
        worker.terminate()
        return "🛑 Transcription cancelled"
    return "Process is not running"


def clear_transcription(
    worker: PersistentPipelineWorker,
    output_directory: Path,
    run: dict | None,
) -> tuple[object, ...]:
    """Cancel a job, delete its local artifacts, and reset the UI."""
    if run and worker.is_running(run["job_id"]):
        worker.terminate()
    if run:
        _delete_job_workspace(output_directory, run["workspace_path"])

    return (
        None,
        None,
        "Waiting...",
        gr.update(visible=False),
        gr.update(value=False, interactive=False),
        gr.update(visible=False, value=None),
        None,
        gr.update(interactive=True),
        gr.update(visible=False),
        gr.update(active=False),
        None,
    )


def show_audio_info(file_path: str | None) -> tuple[str | None, str]:
    """Return the uploaded file for preview with its media information."""
    if file_path is None:
        return None, "🧹 Input cleared!"

    try:
        info = mediainfo(file_path)
        duration = round(float(info.get("duration", 0)), 2)
        sample_rate = info.get("sample_rate", "Unknown")
        channels = info.get("channels", "Unknown")
        details = f"🕒 Duration: {duration}s\n📊 Sample rate: {sample_rate} Hz\n🔊 Channels: {channels}"
        return file_path, details
    except (CouldntDecodeError, OSError, TypeError, ValueError) as error:
        logger.error("Failed to get audio info: %s", error)
        return file_path, f"⚠️ Could not read audio info: {error}"


def preview_transcript(show: bool, file_path: str | None):
    """Preview the transcript file (tab-separated despite the .csv name)."""
    if not show:
        return gr.update(visible=False)

    if not file_path:
        return gr.update(
            visible=True,
            value=[["Output file not found. Please run transcription first."]],
        )

    path = Path(file_path)
    if not path.exists():
        return gr.update(
            visible=True,
            value=[["Output file not found. Please wait for transcription to complete."]],
        )

    try:
        df = pd.read_csv(path, sep="\t")

        if df.empty:
            return gr.update(
                visible=True,
                value=[["No speech was transcribed"]],
            )

        head = df.head(10)
        return gr.update(
            visible=True,
            value={
                "data": head.values.tolist(),
                "headers": list(head.columns),
            },
        )

    except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as error:
        logger.error("CSV preview error: %s", error)
        return gr.update(
            visible=True,
            value=[[f"Error loading CSV: {error}"]],
        )


def launch_gradio(
    output_directory: Path | None = None,
    worker: PersistentPipelineWorker | None = None,
) -> None:
    """Launch the Gradio web interface."""
    output_directory = output_directory or Path("output")
    worker = worker or PersistentPipelineWorker()
    output_directory.mkdir(parents=True, exist_ok=True)
    concurrency_id = "transcription-lifecycle"

    best_device = detect_device("auto")
    logger.info(f"Best available device detected: {best_device}")

    with gr.Blocks(
        title="textplease transcriber",
        analytics_enabled=False,
        delete_cache=(3600, 86400),
    ) as demo:
        gr.Markdown("# 🎙️ text, please!")
        gr.Markdown("Upload an audio file and receive a structured transcript 📝")

        with gr.Row(equal_height=True):
            with gr.Column(scale=3):
                audio_input = gr.File(
                    label="Upload Audio (.mp3/.wav/.mp4/.m4a/.ogg)",
                    file_types=[".mp3", ".wav", ".mp4", ".m4a", ".ogg"],
                    type="filepath",
                    height=180,
                )
            with gr.Column(scale=2):
                audio_preview = gr.Audio(label="Preview", interactive=False)
                audio_info_box = gr.Textbox(label="File Info", lines=3)

        upload_outputs = [audio_preview, audio_info_box]
        audio_input.upload(
            show_audio_info,
            inputs=audio_input,
            outputs=upload_outputs,
            queue=False,
            show_progress="hidden",
            api_visibility="private",
        )
        audio_input.clear(
            show_audio_info,
            inputs=audio_input,
            outputs=upload_outputs,
            queue=False,
            show_progress="hidden",
            api_visibility="private",
        )

        with gr.Accordion("⚙️ Advanced Settings", open=False):
            with gr.Row():
                device = gr.Dropdown(
                    choices=["auto", "cpu", "cuda", "mps"],
                    value=best_device,
                    label="Device",
                    info="Auto: best available | CPU: universal | CUDA: NVIDIA GPU | MPS: Apple Silicon",
                )
                similarity_threshold = gr.Slider(
                    0.0,
                    1.0,
                    step=0.01,
                    value=0.75,
                    label="Similarity Threshold",
                    info="Higher = more segments split",
                )
                pause_threshold = gr.Slider(
                    0.0,
                    10.0,
                    step=0.1,
                    value=2.0,
                    label="Pause Threshold (seconds)",
                    info="Silence that splits segments (also the Silero-VAD boundary)",
                )
            with gr.Row():
                max_segment_words = gr.Slider(
                    10,
                    200,
                    step=5,
                    value=100,
                    label="Max Segment Words",
                )
                min_segment_words = gr.Slider(
                    1,
                    20,
                    value=3,
                    step=1,
                    label="Min Segment Words",
                )
                min_segment_chars = gr.Slider(
                    1,
                    100,
                    value=15,
                    step=1,
                    label="Min Segment Characters",
                )

        with gr.Row(equal_height=True):
            language = gr.Dropdown(
                choices=LANGUAGE_CHOICES,
                value="en",
                label="Language",
                scale=1,
            )
            run_button = gr.Button("🚀 Start Transcription", variant="primary", size="lg", scale=2, elem_id="run-btn")
            cancel_button = gr.Button(
                "🛑 Cancel", visible=False, variant="stop", size="lg", scale=1, elem_id="cancel-btn"
            )
            clear_btn = gr.Button("🧹 Clear", size="lg", scale=1, elem_id="clear-btn")

        status_text = gr.Textbox(label="Status", value="Waiting...", interactive=False, lines=3)

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
        run_state = gr.State(value=None)
        poll_timer = gr.Timer(2.0, active=False)

        run_outputs = [
            status_text,
            download_button,
            show_transcript,
            csv_preview,
            transcript_state,
            run_button,
            cancel_button,
            poll_timer,
            run_state,
        ]

        run_button.click(
            partial(start_transcription, worker, output_directory),
            inputs=[
                audio_input,
                similarity_threshold,
                pause_threshold,
                max_segment_words,
                min_segment_words,
                min_segment_chars,
                language,
                device,
                run_state,
            ],
            outputs=run_outputs,
            trigger_mode="once",
            concurrency_limit=1,
            concurrency_id=concurrency_id,
            api_visibility="private",
        )

        poll_timer.tick(
            partial(check_completion, worker),
            inputs=[run_state, transcript_state],
            outputs=run_outputs,
            concurrency_limit=1,
            concurrency_id=concurrency_id,
            api_visibility="private",
        )

        cancel_button.click(
            partial(cancel_transcription, worker),
            inputs=[run_state],
            outputs=[status_text],
            concurrency_limit=1,
            concurrency_id=concurrency_id,
            api_visibility="private",
        )

        clear_btn.click(
            partial(clear_transcription, worker, output_directory),
            inputs=[run_state],
            outputs=[
                audio_input,
                audio_preview,
                status_text,
                download_button,
                show_transcript,
                csv_preview,
                transcript_state,
                run_button,
                cancel_button,
                poll_timer,
                run_state,
            ],
            concurrency_limit=1,
            concurrency_id=concurrency_id,
            api_visibility="private",
        )

        show_transcript.change(
            preview_transcript,
            inputs=[show_transcript, transcript_state],
            outputs=csv_preview,
        )

        download_button.click(
            lambda file_path: file_path,
            inputs=[transcript_state],
            outputs=[download_button],
        )

        demo.load(None, js=_TOOLTIP_JS)

    try:
        demo.launch(
            theme=gr.themes.Base(),
            share=False,
            server_name="127.0.0.1",
            enable_monitoring=False,
        )
    finally:
        worker.terminate()


if __name__ == "__main__":
    launch_gradio()
