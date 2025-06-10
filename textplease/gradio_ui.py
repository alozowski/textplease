import shutil
import traceback
from pathlib import Path

import yaml
import gradio as gr
import pandas as pd
from pydub.utils import mediainfo

from textplease.main import run_transcription_pipeline


# Ensure folders exist
Path("output").mkdir(parents=True, exist_ok=True)
Path("input").mkdir(parents=True, exist_ok=True)


class Tracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.data = {
            "done": False,
            "output_path": None,
        }

    def update(self, **kwargs):
        self.data.update(kwargs)

    def __getitem__(self, key):
        return self.data.get(key)

    def get(self, key, default=None):
        return self.data.get(key, default)


def start_transcription(
    audio_file,
    chunk_duration_minutes,
    max_batch_size,
    similarity_threshold,
    pause_threshold,
    max_segment_words,
    min_segment_words,
    min_segment_chars,
    tracker,
):
    if audio_file is None:
        return "❌ Please upload an audio file.", None

    Path("input").mkdir(parents=True, exist_ok=True)
    tracker.reset()

    input_path = Path("input") / Path(audio_file.name).name
    shutil.copyfile(audio_file.name, input_path)

    output_path = Path("output") / f"{input_path.stem}_transcript.csv"
    config_path = Path("output") / f"{input_path.stem}_config.yaml"

    config = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "model_name": "nvidia/parakeet-tdt-0.6b-v2",
        "device": "cpu",
        "chunk_duration_minutes": chunk_duration_minutes,
        "max_batch_size": max_batch_size,
        "similarity_threshold": similarity_threshold,
        "pause_threshold": pause_threshold,
        "max_segment_words": max_segment_words,
        "min_segment_words": min_segment_words,
        "min_segment_chars": min_segment_chars,
        "embedding_model": "all-mpnet-base-v2",
        "log_level": "INFO",
    }

    with open(config_path, "w") as f:
        yaml.dump(config, f)

    try:
        run_transcription_pipeline(config)
        tracker.update(done=True, output_path=str(output_path))
        return (
            f"✅ Transcription complete!\n"
            f"📄 Transcript saved to: `{output_path}`\n"
            f"🛠️ Configuration saved to: `{config_path}`",
            str(output_path),
        )
    except Exception as e:
        print(traceback.format_exc())
        return f"❌ Error: {e}", None
    finally:
        if Path("input").exists():
            shutil.rmtree("input", ignore_errors=True)


def poll_status(tracker):
    return None, gr.update(visible=tracker["done"] and tracker["output_path"])


def show_audio_info(file):
    if file is None:
        return "🧹 Input cleared!"
    info = mediainfo(file.name)
    duration = round(float(info["duration"]), 2)
    return f"🕒 Duration: {duration}s"


def preview_transcript(show, file):
    if show and file and Path(file).exists():
        try:
            df = pd.read_csv(file, sep="	")
            return gr.update(visible=True, value=df.head())
        except Exception as e:
            print("CSV Preview Error:", e)
            return gr.update(visible=True, value=[["Error loading CSV"]])
    return gr.update(visible=False)


def launch_gradio():
    with gr.Blocks(title="textplease transcriber").queue() as demo:
        tracker = gr.State(Tracker())

        gr.Markdown("## 🎙️ text, please!")
        gr.Markdown("Upload an audio file, configure settings, and receive a transcript 📝")

        with gr.Row():
            audio_input = gr.File(label="Upload Audio (.mp3/.wav/.mp4)", file_types=[".mp3", ".wav", ".mp4"])
            audio_preview = gr.Audio(label="Audio Preview", interactive=False)

        audio_input.change(lambda f: f, inputs=audio_input, outputs=audio_preview)
        audio_input.change(show_audio_info, inputs=audio_input, outputs=gr.Textbox(label="Audio Info"))

        with gr.Accordion("⚙️ Advanced Settings", open=False):
            chunk_duration_minutes = gr.Slider(1, 60, value=1, label="Chunk Duration (minutes)")
            max_batch_size = gr.Slider(1, 8, value=1, label="Max Batch Size")
            similarity_threshold = gr.Slider(0.0, 1.0, step=0.01, value=0.65, label="Similarity Threshold")
            pause_threshold = gr.Slider(0.5, 10.0, step=0.5, value=3.0, label="Pause Threshold (seconds)")
            max_segment_words = gr.Slider(10, 500, step=10, value=100, label="Max Segment Words")
            min_segment_words = gr.Slider(1, 20, value=3, label="Min Segment Words")
            min_segment_chars = gr.Slider(1, 100, value=20, label="Min Segment Characters")

        run_button = gr.Button("🚀 Start Transcription")
        clear_btn = gr.Button("🧹 Clear Inputs")
        status_text = gr.Textbox(label="Status", value="Waiting...", interactive=False, lines=3)
        download_output = gr.File(label="Download Transcript", visible=False)
        show_transcript = gr.Checkbox(label="📄 Show Transcript Preview", value=False)
        csv_preview = gr.Dataframe(label="Transcript", visible=False)

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
                tracker,
            ],
            outputs=[status_text, download_output],
            show_progress="full",
        )

        poll_timer = gr.Timer(value=1.0, active=True, render=False)
        poll_timer.tick(poll_status, inputs=[tracker], outputs=[download_output])

        clear_btn.click(
            lambda: (None, None, "Waiting...", None),
            outputs=[audio_input, audio_preview, status_text, download_output],
        )

        show_transcript.change(preview_transcript, inputs=[show_transcript, download_output], outputs=csv_preview)

    demo.launch(share=True)


if __name__ == "__main__":
    launch_gradio()
