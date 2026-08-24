# Configuration

The web interface is the easiest way to use TextPlease:

```bash
textplease --gradio
```

Upload a file, choose its language, and start the transcription. The app chooses the best available device. Each job
stores its transcript, configuration, and log in a private directory under `output/`. Clear deletes that directory.

## Command line

Create a YAML file such as `my_config.yaml`:

```yaml
input_path: "input/recording.mp3"
output_path: "output/recording_transcript.csv"
model_name: "openai/whisper-large-v3"
device: "auto"
language: "en"
```

Then run:

```bash
textplease --config my_config.yaml
```

Paths are resolved from the directory where you run the command. The input must exist. The output directory is
created automatically.

| Setting | Default | Notes |
|---------|---------|-------|
| `input_path` | Required | An existing audio or video file |
| `output_path` | Required | Replaced if it already exists |
| `model_name` | Required | A Hugging Face model ID downloaded on first use, or a local model directory |
| `device` | `cpu` | `auto`, `cpu`, `cuda`, or `mps` |
| `language` | `en` | A Whisper language code, or `null` for automatic detection |
| `log_level` | `INFO` | `DEBUG`, `INFO`, `WARNING`, or `ERROR` |

`auto` prefers CUDA, then MPS, then CPU. An unavailable accelerator falls back in the same order. The web interface
offers English, Russian, Spanish, French, Italian, German, Turkish, Chinese, Korean, and Japanese. YAML files can use
other language codes supported by the selected Whisper model.

The transcript preserves each retained, nonblank Whisper span and its timestamp. TextPlease does not merge, split,
deduplicate, or rewrite that text.

## Performance

Most users should keep the default:

```yaml
performance:
  whisper_batch_size: 1
```

This controls how many detected speech chunks Whisper processes together. The default is `1` because parity for larger
real-model batches is not established. If an explicitly configured accelerator batch runs out of memory, TextPlease
retries one chunk at a time.

The web interface keeps its loaded Whisper model after a successful job. Cancelling or failing a job discards the worker
and model. Changing the model or device also starts a new worker. Cancel removes temporary PCM. Clear also deletes the
job's transcript, configuration, and log.

## Privacy and model downloads

Audio and transcripts stay on the machine. Hugging Face receives model and metadata requests when a required model is
not cached. Models download automatically and later runs reuse the local cache.

Configuration files do not set environment variables. Set machine-specific values in the shell before starting
TextPlease so they are not copied into saved job configurations or logs.

## Troubleshooting

| If you see this | Try this |
|-----------------|----------|
| Whisper runs out of memory | Keep `performance.whisper_batch_size: 1`, select `cpu`, or use a smaller compatible Whisper model |
| A model download fails | Check the network connection and Hugging Face access, then try again |
| Speech is missing | Confirm the language and review the quality baseline before changing code |
| Music produces text | Record the file and expected silence as a new credited evaluation case |

The output uses tab-separated columns even when its name ends in `.csv`. This preserves commas in spoken text. Unknown
settings produce an error so misspellings and removed options cannot silently change expectations.

See [`examples/config_example.yaml`](examples/config_example.yaml) for a complete example.
