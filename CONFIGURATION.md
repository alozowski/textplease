# Configuration

The defaults are a good starting point for most recordings. If this is your first time using `textplease`, try them before changing anything.

You can run the app in two ways:

- Web interface: `textplease --gradio`
- Command line: `textplease --config path/to/config.yaml`

The web interface is the easiest option. Upload a file, choose a language, and start the transcription. A copy of the configuration is saved in `output/` with the transcript.

## Quick start from the command line

Create a YAML file, for example `my_config.yaml`:

```yaml
input_path: "input/recording.mp3"
output_path: "output/recording_transcript.csv"
model_name: "openai/whisper-large-v3"

device: "cpu"
language: "en"

pause_threshold: 2.0
similarity_threshold: 0.75
min_segment_words: 3
min_segment_chars: 15
max_segment_words: 100
```

Then run:

```bash
textplease --config my_config.yaml
```

Paths are read from the directory where you run the command. The input file must already exist. The output folder is created automatically.

Only `input_path`, `output_path`, and `model_name` are required. Everything else has a default value.

> The output file has a `.csv` extension, but its columns are separated by tabs. This makes transcript text containing commas safe to open and process.

## The settings you are most likely to change

### `pause_threshold`

Default: `2.0` seconds

This decides how long a silence must be before it counts as a break. It affects both the speech recognition step and the final transcript segments.

- Use a **lower** value to create more breaks.
- Use a **higher** value to keep speech together across longer pauses.

Good starting points:

| Recording | Try |
|-----------|-----|
| Subtitles or fast speech | `0.5`-`1.0` |
| Conversation or interview | `2.0` |
| Slow speech with thinking pauses | `3.0`-`4.0` |

### `similarity_threshold`

Default: `0.75`

This controls whether nearby pieces of text are similar enough to join together.

- Use a **higher** value to keep more segments separate.
- Use a **lower** value to allow more merging.

| Value | What to expect |
|-------|----------------|
| `0.0` | Very permissive; many segments may merge |
| `0.75` | A balanced default |
| `0.9` | Keeps more topic and sentence boundaries |
| `1.0` | Effectively turns off similarity-based merging |

`0.0` does **not** turn merging off. Short fragments may still be joined to a neighbour even when this is set to `1.0`.

### Segment length

These three settings keep the transcript from becoming too fragmented or too dense:

```yaml
min_segment_words: 3
min_segment_chars: 15
max_segment_words: 100
```

- A segment below either minimum is treated as a fragment and is usually joined to a neighbour.
- A segment above the maximum is split into smaller pieces.
- A fragment is kept when it cannot be merged without exceeding the maximum.

Set both minimums to `1` if short replies such as “Yes” or “No” should stay on their own.

## Useful presets

Add one of these blocks to your config, replacing the same settings if they are already there.

### Subtitles

```yaml
pause_threshold: 1.0
similarity_threshold: 1.0
min_segment_words: 1
min_segment_chars: 1
max_segment_words: 40
```

### Interview or podcast

```yaml
pause_threshold: 3.0
similarity_threshold: 0.75
min_segment_words: 3
min_segment_chars: 15
max_segment_words: 150
```

### Meeting notes

```yaml
pause_threshold: 2.0
similarity_threshold: 0.8
min_segment_words: 3
min_segment_chars: 15
max_segment_words: 80
```

### Keep Whisper's segments mostly unchanged

```yaml
similarity_threshold: 1.0
min_segment_words: 1
min_segment_chars: 1
max_segment_words: 100000
```

This turns off the normal segmentation rules as far as practical. The result is not completely raw Whisper output: `textplease` still splits sentences, removes repeated overlap and known false phrases, and drops empty segments.

## Input and model settings

| Setting | Default | Notes |
|---------|---------|-------|
| `input_path` | Required | An existing audio or video file |
| `output_path` | Required | Replaced if it already exists |
| `model_name` | Required | The web interface uses `openai/whisper-large-v3` |
| `device` | `cpu` | Use `auto` for the best available device, `cuda` for NVIDIA, or `mps` for Apple Silicon |
| `language` | `en` | Language code passed to Whisper |
| `embedding_model` | `all-MiniLM-L6-v2` | Model used to compare segment meaning |
| `log_level` | `INFO` | Also accepts `DEBUG`, `WARNING`, and `ERROR` |

`auto` prefers CUDA, then MPS, then CPU. An unavailable explicit accelerator uses the same fallback order. The web
interface automatically chooses the best available device. It currently offers these languages:

- English (`en`), Russian (`ru`), Spanish (`es`), French (`fr`), Italian (`it`)
- German (`de`), Turkish (`tr`), Chinese (`zh`), Korean (`ko`), Japanese (`ja`)

The command line can use other language codes supported by the selected Whisper model.

## Performance settings

Most users can leave these alone.

```yaml
performance:
  similarity_batch_size: 32
  chunk_size: 1000
```

`similarity_batch_size` controls how many text embeddings are created at once. Lower it if the embedding step runs out of memory.

`chunk_size` controls how many pieces of the Whisper transcript are handled at once during merging. Set it to `0` to turn chunked merging off. Embeddings are created before this stage, so lowering `chunk_size` does not reduce the memory used to create them.

## Environment variables

You can set environment variables from the YAML file:

```yaml
environment:
  PYTORCH_MPS_HIGH_WATERMARK_RATIO: "0.0"
  OMP_NUM_THREADS: "8"
  TOKENIZERS_PARALLELISM: "false"
```

This section is optional. If a variable is already set in your shell, the shell value wins.

## What the web interface exposes

The web interface lets you change:

- device and language
- pause and similarity thresholds
- minimum words, minimum characters, and maximum words

It chooses the input and output paths for you and uses the default Whisper and embedding models. Performance, logging, and environment settings are available only in a CLI YAML file.

## How merging works

In the usual case, two neighbouring segments are joined only when:

1. The pause is no longer than `pause_threshold`.
2. Their similarity is greater than `similarity_threshold`.
3. Their combined length is no more than `max_segment_words`.

Short fragments get an extra cleanup pass. They may be joined without passing the similarity check, but a merge is never allowed to create a segment longer than `max_segment_words`. If no suitable neighbour is available, the fragment is left as it is.

## Troubleshooting

| If you see this | Try this |
|-----------------|----------|
| Too many tiny segments | Raise `min_segment_words` or `min_segment_chars` |
| Segments are too long | Lower `max_segment_words` |
| Not enough breaks | Lower `pause_threshold` or raise `similarity_threshold` |
| Too many breaks | Raise `pause_threshold` or lower `similarity_threshold` |
| Unrelated sentences are joined | Raise `similarity_threshold` to about `0.85` |
| Short replies disappear into nearby text | Set both minimums to `1` |
| Embedding runs out of memory | Lower `performance.similarity_batch_size` |
| GPU transcription runs out of memory | Use `device: "cpu"` or a smaller compatible Whisper model |

## A note about CLI values

The command line checks that the required fields are present and that the input file exists, but it does not validate every numeric value. Use non-negative thresholds and positive length and batch values. Keep `min_segment_words` lower than or equal to `max_segment_words`. Unknown settings are ignored.

For a complete example with every supported option, see [`examples/config_example.yaml`](examples/config_example.yaml).
