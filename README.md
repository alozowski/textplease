# text, please!

textplease turns audio and video files into text with timestamps. It runs open-source Whisper models on your computer, so your media is not sent to a cloud API. It downloads each model from Hugging Face when first needed.

## Features

- Local web app and YAML-based command line.
- Start and end times for each text segment.
- Automatic speech detection and filtering of music without speech.
- Support for long files and compatible multilingual Whisper models.

## Install

You need:

- [Python 3.12](https://www.python.org/downloads/)
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- [FFmpeg](https://ffmpeg.org/download.html), with `ffmpeg` available on `PATH`

Then install the project:

```bash
ffmpeg -version
git clone https://github.com/alozowski/textplease.git
cd textplease
uv sync --locked --no-dev
```

## Run

### Web interface

```bash
uv run --locked --no-dev textplease --gradio
```

Open the local address shown in the terminal. Upload an audio or video file, then select **Start Transcription**. The app accepts `.mp3`, `.wav`, `.mp4`, `.m4a`, and `.ogg` files.

### Command line

```bash
uv run --locked --no-dev textplease --config examples/config_example.yaml
```

The app writes the transcript to the config's `output_path`. See [Configuration](CONFIGURATION.md) for common settings.
See [config_example.yaml](examples/config_example.yaml) for every option.

## Privacy and stored data

Your audio and transcript are not sent to Hugging Face or another cloud API. Hugging Face receives model and metadata requests when a model is first used. Later runs reuse the local cache. Disconnect the network or use a firewall if you need to enforce offline use once the required models are cached.

The web app listens only on `127.0.0.1`. It disables public share links, analytics, and monitoring. Each job stores its transcript, config, and log under `output/`. "Clear" deletes those files. The app removes temporary audio after each job. It also clears the upload cache on restart and removes uploaded files older than 24 hours. Downloaded models stay in the model cache for reuse.

## Output

TextPlease writes tab-separated columns. The example config uses a `.csv` name, which keeps commas in spoken text intact. A file with no speech contains only the header.

| start_time | end_time | text                    |
|------------|----------|-------------------------|
| 00:00:00.000 | 00:00:06.000 | Welcome to the demo ... |

## Quality status

The English tests cover silence, rain, music, speech over music, short speech, and clean read speech at three speeds. They also cover continuous 30 and 60 minute meetings. The active non-speech and acceptance short-text gates pass. The timestamp gate still fails and is a known issue.

The [baseline](evaluation/BASELINE.md) has all results and audio credits. The [protocol](evaluation/protocol.json) defines the gates. The [manifest](evaluation/manifest.jsonl) records the source, license, credit, changes, and hash for each file. Use `uv run --locked python scripts/evaluate_audio_quality.py --help` to find the inference and scoring commands.

## How it works

FFmpeg makes mono 16 kHz PCM for Whisper. Silero VAD finds likely speech. A local AudioSet model suppresses output that it rates as music without speech. Whisper transcribes what remains. The app groups the text by pauses, meaning, and length, then writes the tab-separated file.

## License

The code uses the [MIT license](LICENSE). Evaluation media keep the licenses listed in the manifest and baseline. The bundled example is from the public-domain [LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset/), recorded by Linda Johnson and aligned by Keith Ito.
