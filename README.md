# text, please!

textplease turns audio and video files into text with timestamps. It runs open-source Whisper models on your computer, so your media is not sent to a cloud API. It downloads each model from Hugging Face when first needed.

## Features

- Local web app and YAML-based command line.
- Start and end times for each text segment.
- Local speech and music checks before transcription.
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

The published evaluation is English-only. It covers silence, rain, music, speech over music, short speech, clean read speech at three speeds, and continuous 30 and 60 minute meetings. The enabled non-speech, acceptance short-text, and timestamp gates pass. WER, boundary, resource, and broader quality thresholds remain disabled while evidence grows.

The [baseline](evaluation/BASELINE.md) has all results and audio credits. The [protocol](evaluation/protocol.json) defines the gates. The [manifest](evaluation/manifest.jsonl) records the source, license, credit, changes, and hash for each file. Use `uv run --locked python scripts/evaluate_audio_quality.py --help` to find the inference and scoring commands.

## Known limitations

Whisper can write convincing text that nobody spoke. This happens most often with quiet, unclear, or non-speech audio. We found subtitle credits and a long unclear passage in a private Russian interview. Our public test set is English-only, so it does not validate Russian or the other languages in the web app. Always review transcripts before using them for research, publication, or sensitive decisions.

The app uses the first audio stream and mixes its channels to mono. This can lose sound when speakers are on separate channels or channels have opposite phase. Keep the original media.

## How it works

FFmpeg converts the media to mono 16 kHz PCM. Silero VAD marks likely speech. A local AudioSet model rejects windows that appear to contain music without speech. The remaining intervals are grouped by `whisper_batch_size`. Each batch is padded to its longest interval and passed through one top-level `model.generate` call. Whisper uses float16 on CUDA and float32 on MPS or CPU. If a batch with several intervals runs out of accelerator memory, the app retries each interval alone.

When a batch contains an interval longer than about 30 seconds, Transformers uses Whisper's long-form mode. It starts with greedy decoding at temperature `0.0`. If an item fails a quality check, only that item is retried at `0.2`, `0.4`, `0.6`, `0.8`, then `1.0`.

- A token-byte zlib compression ratio above `1.35` flags repeated output.
- An average token log-probability below `-1.0` flags low-confidence output.
- A no-speech probability above `0.6`, combined with an average log-probability below `-1.0`, skips the interval as silence.

Transformers measures compression from token ID bytes. OpenAI Whisper measures UTF-8 text and uses a `2.4` default. Finally, the app clips each timestamp to its source interval, shifts it to the full recording, and writes the model's nonblank text without rewriting it.

## License

The code uses the [MIT license](LICENSE). Evaluation media keep the licenses listed in the manifest and baseline. The bundled example is from the public-domain [LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset/), recorded by Linda Johnson and aligned by Keith Ito.
