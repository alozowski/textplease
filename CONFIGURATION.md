# Configuration Guide

All settings are available in both the YAML config file (CLI) and the Advanced Settings panel (web UI). The defaults work well for most recordings — start there, then adjust the two core settings below if needed.

---

## The two settings that matter most

### Pause Threshold (default: 2.0s)

The single most impactful parameter. Controls when a silence forces a new segment boundary.

- Pause **below** threshold: segments *can* merge if they are semantically similar
- Pause **above** threshold: segments *always* split, regardless of content

This also sets the minimum silence duration that Silero-VAD uses to split the audio before transcription, so it affects both speech recognition quality and final segmentation.

| Value | When to use |
|-------|-------------|
| `0.5` | Dense speech, few natural pauses — lectures, monologues |
| `2.0` | Default — conversational recordings, interviews |
| `4.0` | Very slow speech with long thinking pauses |

### Similarity Threshold (default: 0.75)

Controls semantic merging. Two segments separated by a short pause will only merge if their cosine similarity exceeds this value.

| Value | Effect |
|-------|--------|
| `0.0` | No semantic merging — keep raw ASR output as-is |
| `0.75` | Default — merge only clearly related sentences |
| `0.9` | Strict — preserve topic changes as distinct segments |

Setting this to `0.0` is the fastest way to debug: you get exactly what the ASR model produced, with no merging.

---

## Presets for common workflows

**Subtitles and captions**
```yaml
pause_threshold: 1.0
max_segment_words: 40
similarity_threshold: 0.0
min_segment_words: 1
min_segment_chars: 1
```

**Podcast or interview transcript**
```yaml
pause_threshold: 3.0
max_segment_words: 150
similarity_threshold: 0.75
```

**Meeting notes**
```yaml
pause_threshold: 2.0
max_segment_words: 80
similarity_threshold: 0.8
```

**Raw ASR output (no post-processing)**
```yaml
pause_threshold: 0.5
max_segment_words: 100
similarity_threshold: 0.0
min_segment_words: 1
min_segment_chars: 1
```

---

<details>
<summary>Advanced settings</summary>

### Max Segment Words (default: 100)

Hard ceiling on segment length. Any segment exceeding this is split regardless of other settings, with timestamps distributed proportionally.

```yaml
max_segment_words: 40   # subtitles
max_segment_words: 100  # default
max_segment_words: 150  # long-form transcripts
```

### Min Segment Words and Min Segment Chars (defaults: 3 words, 15 chars)

Minimum length constraints. Segments below either threshold are forcibly merged with a neighbour. Raise these to suppress fragments; lower them to allow very short segments.

```yaml
# Allow any length — useful with similarity_threshold: 0.0
min_segment_words: 1
min_segment_chars: 1

# Stricter — suppress short responses and filler
min_segment_words: 5
min_segment_chars: 30
```

Min constraints take priority over pause threshold: a segment below the minimum will merge with its neighbour even if the pause exceeds the threshold.

### Chunk Duration (default: 10 min) — NeMo only

Applies only to the NeMo (Parakeet) backend. Whisper uses Silero-VAD to determine speech segment boundaries and ignores this setting.

Increase if you have plenty of RAM; decrease if you encounter memory errors during NeMo transcription.

### Max Batch Size (default: 1) — NeMo only

Applies only to the NeMo (Parakeet) backend. Whisper uses `model.generate()` and ignores this setting. Leave at 1 unless you have a specific reason to change it.

</details>

<details>
<summary>Troubleshooting</summary>

| Symptom | Fix |
|---------|-----|
| Too many tiny segments | Raise `min_segment_words` to 5 |
| Segments not splitting enough | Lower `pause_threshold` to 1.0 |
| Segments splitting too aggressively | Raise `pause_threshold` to 3.0 |
| Unrelated sentences getting merged | Raise `similarity_threshold` to 0.85 |
| Want to inspect raw ASR output | Set `similarity_threshold: 0.0` |
| Segments too long | Lower `max_segment_words` to 50 |
| Memory errors (NeMo only) | Lower `chunk_duration_minutes` to 5 |

</details>

<details>
<summary>How the merge decision works</summary>

Two consecutive segments merge when **all three** conditions are met:

1. The pause between them does not exceed `pause_threshold`
2. Their cosine similarity exceeds `similarity_threshold`
3. The combined word count does not exceed `max_segment_words`

One override: if either segment is below the minimum length constraints, it merges with its neighbour regardless of pause duration or similarity. The priority is to avoid outputting fragments.

Priority order: minimum constraints > pause threshold > similarity > maximum length.

</details>
