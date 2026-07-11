import re
import time
import shutil
import logging
import multiprocessing
from pathlib import Path

import yaml
import gradio as gr
import pandas as pd
from pydub.utils import mediainfo

from textplease.pipeline import DEFAULT_EMBEDDING_MODEL, run_transcription_pipeline
from textplease.utils.device_utils import detect_device
from textplease.utils.logging_config import configure_logging


logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output")
INPUT_DIR = Path("input")

DEFAULT_MODEL = "openai/whisper-large-v3"

# Whisper backend logs one line per VAD segment — the child's log file is the progress signal.
_PROGRESS_RE = re.compile(r"Transcribing speech segment (\d+)/(\d+)")

# Gradio has no native button tooltips (gradio-app/gradio#3050); set browser title attributes on load.
_TOOLTIP_JS = """
() => {
    const tips = {
        "run-btn": "Transcribe the uploaded audio",
        "stop-btn": "Gracefully stop the running transcription",
        "kill-btn": "Force-kill the transcription immediately",
        "clear-btn": "Reset all inputs and results",
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

LANGUAGE_CHOICES = [
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
]


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
    except OSError as e:
        raise IOError(f"Failed to copy audio file: {e}") from e

    output_path = OUTPUT_DIR / f"{input_path.stem}_transcript.csv"
    return input_path, output_path


def _create_transcription_config(
    input_path: Path,
    output_path: Path,
    similarity_threshold: float,
    pause_threshold: float,
    max_segment_words: int,
    min_segment_words: int,
    min_segment_chars: int,
    language: str,
    device: str,
) -> dict:
    """Create transcription configuration dictionary."""
    return {
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


def _run_pipeline_process(config: dict, log_path: str) -> None:
    """Run the pipeline in a child process, logging to the terminal and a progress file."""
    configure_logging()
    logging.getLogger().addHandler(logging.FileHandler(log_path))
    run_transcription_pipeline(config)


def start_transcription(
    audio_file,
    similarity_threshold,
    pause_threshold,
    max_segment_words,
    min_segment_words,
    min_segment_chars,
    language,
    device,
):
    """Launch transcription in a child process and show run controls."""

    def _err(msg):
        return (
            msg,
            gr.update(visible=False),  # download_button
            gr.update(interactive=False, value=False),  # show_transcript
            gr.update(visible=False, value=None),  # csv_preview
            None,  # transcript_state
            gr.update(visible=False),  # stop_button
            gr.update(visible=False),  # kill_button
            gr.update(active=False),  # poll_timer
            None,  # run_state
        )

    if audio_file is None:
        return _err("❌ Please upload an audio file.")

    try:
        input_path, output_path = _prepare_input_files(audio_file)
        config = _create_transcription_config(
            input_path,
            output_path,
            similarity_threshold,
            pause_threshold,
            max_segment_words,
            min_segment_words,
            min_segment_chars,
            language,
            device,
        )
        config_path = OUTPUT_DIR / f"{input_path.stem}_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        log_path = OUTPUT_DIR / f"{input_path.stem}_run.log"
        log_path.unlink(missing_ok=True)
        process = multiprocessing.Process(target=_run_pipeline_process, args=(config, str(log_path)), daemon=True)
        process.start()
        logger.info(f"Transcription process started (pid={process.pid})")

        return (
            "⏳ Transcription starting...",
            gr.update(visible=False),
            gr.update(interactive=False, value=False),
            gr.update(visible=False, value=None),
            None,
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(active=True),
            {
                "process": process,
                "input_path": input_path,
                "output_path": output_path,
                "config_path": config_path,
                "log_path": log_path,
                "started": time.time(),
            },
        )

    except (ValueError, FileNotFoundError, IOError) as e:
        return _err(f"❌ File error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in start_transcription: {e}", exc_info=True)
        return _err(f"❌ Unexpected error: {e}")


def check_completion(run, transcript_path):
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

    if run["process"].is_alive():
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

    process = run["process"]
    output_path = run["output_path"]

    try:
        run["input_path"].unlink(missing_ok=True)
    except OSError as e:
        logger.warning(f"Could not remove input file {run['input_path']}: {e}")

    done = (gr.update(visible=False), gr.update(visible=False), gr.update(active=False), None)

    if process.exitcode == 0 and output_path.exists():
        return (
            f"✅ Transcription complete!\n"
            f"📄 Transcript saved to: `{output_path}`\n"
            f"🛠️ Configuration saved to: `{run['config_path']}`",
            gr.update(value=str(output_path), visible=True),
            gr.update(interactive=True, value=True),
            preview_transcript(True, str(output_path)),
            str(output_path),
            *done,
        )

    if process.exitcode is not None and process.exitcode < 0:
        status = f"🛑 Transcription stopped (exit code {process.exitcode})"
    else:
        status = f"❌ Transcription failed (exit code {process.exitcode}) — see `{run['log_path']}`"
    return (
        status,
        gr.update(visible=False),
        gr.update(interactive=False, value=False),
        gr.update(visible=False, value=None),
        None,
        *done,
    )


def stop_transcription(run):
    """Terminate the transcription process gracefully."""
    if run and run["process"].is_alive():
        run["process"].terminate()
        return "🛑 Stopping..."
    return "Process is not running"


def kill_transcription(run):
    """Kill the transcription process immediately."""
    if run and run["process"].is_alive():
        run["process"].kill()
        return "💀 Killed"
    return "Process is not running"


def show_audio_info(file):
    """Display audio file information."""
    if file is None:
        return "🧹 Input cleared!"

    try:
        info = mediainfo(file.name)
        duration = round(float(info.get("duration", 0)), 2)
        sample_rate = info.get("sample_rate", "Unknown")
        channels = info.get("channels", "Unknown")
        return f"🕒 Duration: {duration}s\n📊 Sample rate: {sample_rate} Hz\n🔊 Channels: {channels}"
    except Exception as e:
        logger.error(f"Failed to get audio info: {e}")
        return f"⚠️ Could not read audio info: {e}"


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
                value=[["Transcript is empty"]],
            )

        head = df.head(10)
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


def launch_gradio():
    """Launch the Gradio web interface."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        best_device = detect_device("cuda")
        logger.info(f"Best available device detected: {best_device}")
    except Exception as e:
        logger.warning(f"Device detection failed, defaulting to CPU: {e}")
        best_device = "cpu"

    with gr.Blocks(title="textplease transcriber", theme=gr.themes.Base()) as demo:
        gr.Markdown("# 🎙️ text, please!")
        gr.Markdown("Upload an audio file and receive a structured transcript 📝")

        with gr.Row(equal_height=True):
            with gr.Column(scale=3):
                audio_input = gr.File(
                    label="Upload Audio (.mp3/.wav/.mp4/.m4a/.ogg)",
                    file_types=[".mp3", ".wav", ".mp4", ".m4a", ".ogg"],
                    height=180,
                )
            with gr.Column(scale=2):
                audio_preview = gr.Audio(label="Preview", interactive=False)
                audio_info_box = gr.Textbox(label="File Info", lines=3)

        audio_input.change(lambda f: f, inputs=audio_input, outputs=audio_preview)
        audio_input.change(show_audio_info, inputs=audio_input, outputs=audio_info_box)

        with gr.Accordion("⚙️ Advanced Settings", open=False):
            with gr.Row():
                device = gr.Dropdown(
                    choices=["cpu", "cuda", "mps"],
                    value=best_device,
                    label="Device",
                    info="CPU: universal | CUDA: NVIDIA GPU | MPS: Apple Silicon",
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
            stop_button = gr.Button(
                "🛑 Stop", visible=False, variant="secondary", size="lg", scale=1, elem_id="stop-btn"
            )
            kill_button = gr.Button("💀 Kill", visible=False, variant="stop", size="lg", scale=1, elem_id="kill-btn")
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
            stop_button,
            kill_button,
            poll_timer,
            run_state,
        ]

        run_button.click(
            start_transcription,
            inputs=[
                audio_input,
                similarity_threshold,
                pause_threshold,
                max_segment_words,
                min_segment_words,
                min_segment_chars,
                language,
                device,
            ],
            outputs=run_outputs,
        )

        poll_timer.tick(
            check_completion,
            inputs=[run_state, transcript_state],
            outputs=run_outputs,
        )

        stop_button.click(stop_transcription, inputs=[run_state], outputs=[status_text])
        kill_button.click(kill_transcription, inputs=[run_state], outputs=[status_text])

        def clear_all(run):
            if run and run["process"].is_alive():
                run["process"].terminate()
            return (
                None,  # audio_input
                None,  # audio_preview
                "Waiting...",  # status_text
                gr.update(visible=False),  # download_button
                gr.update(value=False, interactive=False),  # show_transcript
                gr.update(visible=False, value=None),  # csv_preview
                None,  # transcript_state
                gr.update(visible=False),  # stop_button
                gr.update(visible=False),  # kill_button
                gr.update(active=False),  # poll_timer
                None,  # run_state
            )

        clear_btn.click(
            clear_all,
            inputs=[run_state],
            outputs=[
                audio_input,
                audio_preview,
                status_text,
                download_button,
                show_transcript,
                csv_preview,
                transcript_state,
                stop_button,
                kill_button,
                poll_timer,
                run_state,
            ],
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

    demo.launch()


if __name__ == "__main__":
    launch_gradio()
