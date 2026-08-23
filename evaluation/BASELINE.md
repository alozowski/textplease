# Audio quality evaluation

This report scores the configured public `textplease` pipeline against the versioned manifest and protocol.

## Run

| Field | Value |
|---|---|
| Manifest SHA-256 | `99d58806e89aeaad55a7684568bb40b5c34365c655ebef7b78b099c26fdebe3d` |
| Protocol SHA-256 | `ec747c0eff5a05dc6f0debe543f73bf2c3c23f5ad8039b09acdbdc5e38ba5fc3` |
| Inference evaluator SHA-256 | `d743887c8442b0da24d6e63b5a56b296e993f7d1b709c034ef823534574ea5ec` |
| Scorer SHA-256 | `fb4a63ab3267519aa4e28adfb78b1663e4b8722e70f800af5fbb3bed16055d64` |
| Scorer JiWER | `4.0.0` |
| Scorer RapidFuzz | `3.14.5` |
| Random seed | `0` |
| Source revision | `1837ff849fff9e14f6e68cdee0fc532ee0f46fcb` |
| Source dirty | `True` |
| Device | `mps` |
| Whisper batch size | `1` |
| Model repository | `openai/whisper-large-v3` |
| Model revision | `06f233fe06e710322aca913c1bc4249a0d71fce1` |
| Environment ffmpeg | `ffmpeg version 9.0 Copyright (c) 2000-2026 the FFmpeg developers` |
| Environment numpy | `2.5.1` |
| Environment platform | `macOS-26.6.2-arm64-arm-64bit` |
| Environment python | `3.12.12` |
| Environment sentence-transformers | `5.6.0` |
| Environment silero-vad | `6.2.1` |
| Environment textplease | `0.1.0` |
| Environment torch | `2.13.0` |
| Environment transformers | `5.13.1` |

## Fixture sources

| Case | Source | License | Attribution |
|---|---|---|---|
| silence-5s | [FFmpeg anullsrc, mono 16 kHz PCM16, 5 seconds](https://ffmpeg.org/ffmpeg-filters.html#anullsrc) with `ffmpeg-9.0` | [CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/) | Generated for TextPlease with FFmpeg anullsrc. |
| rain-10s | [Rain.ogg](https://commons.wikimedia.org/w/index.php?title=File:Rain.ogg&oldid=597184901) with `597184901` | [Public-Domain](https://commons.wikimedia.org/w/index.php?title=File:Rain.ogg&oldid=597184901#Licensing) | Recorded by Wikimedia Commons user ジダネ. |
| music-36s | [Greensleaves.ogg](https://commons.wikimedia.org/w/index.php?title=File:Greensleaves.ogg&oldid=845754359) with `845754359` | [Public-Domain](https://commons.wikimedia.org/w/index.php?title=File:Greensleaves.ogg&oldid=845754359#Licensing) | Performed and recorded by Wikimedia Commons user Rv87. |
| short-word-ear | [En-uk-ear.ogg](https://commons.wikimedia.org/w/index.php?title=File:En-uk-ear.ogg&oldid=1229077824) with `1229077824` | [Public-Domain](https://commons.wikimedia.org/w/index.php?title=File:En-uk-ear.ogg&oldid=1229077824#Licensing) | Spoken and recorded by Wikimedia Commons user Chris Melville. |
| short-name-john | [En-au-John.ogg](https://commons.wikimedia.org/w/index.php?title=File:En-au-John.ogg&oldid=724849560) with `724849560` | [CC-BY-SA-4.0](https://creativecommons.org/licenses/by-sa/4.0/) | Spoken and recorded by Commander Keane on Wikimedia Commons. |
| librispeech-sample-1 | [Hugging Face LibriSpeech sample 1](https://cdn-media.huggingface.co/speech_samples/sample1.flac) with `sha256:cb5c48a2d1d6f7dedd0330f088a4cbe76de1a86e6a6109c06d255bb1ca2f7542` | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | LibriSpeech ASR Corpus by Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur. Hosted by Hugging Face. |
| librispeech-sample-2 | [Hugging Face LibriSpeech sample 2](https://cdn-media.huggingface.co/speech_samples/sample2.flac) with `sha256:4e82c7e879bce92c1d3bc99ddb7bdf681611bc251b6d244430e54fe44b86e75e` | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | LibriSpeech ASR Corpus by Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur. Hosted by Hugging Face. |
| ami-meeting-30m | [Continuous 30-minute excerpt from AMI ES2002b scenario meeting headset mix, half-open range [360.9545, 2160.9545), converted to mono 16 kHz FLAC with FFmpeg 9.0](https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2002b/audio/ES2002b.Mix-Headset.wav) with `source-sha256:977fbf6cd473cfb1984b41755762eea4d7ffdad3fb15adcfdebcb8842163ec66` | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | AMI Meeting Corpus by the AMI Consortium. Excerpted and converted to FLAC by TextPlease. Reference text and intervals are from AMI manual annotations v1.6.2. |
| ami-meeting-60m | [Continuous 60-minute excerpt from AMI EN2001a natural meeting headset mix, half-open range [201.2265, 3801.2265), converted to mono 16 kHz FLAC with FFmpeg 9.0](https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/EN2001a/audio/EN2001a.Mix-Headset.wav) with `source-sha256:81e06be816e9d94d0bee410bef1b158b2cdfba8e2b80f44a6e62cf6b9fd780f9` | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | AMI Meeting Corpus by the AMI Consortium. Excerpted and converted to FLAC by TextPlease. Reference text and intervals are from AMI manual annotations v1.6.2. |

## Overall

| Metric | Value |
|---|---:|
| Cases | 9 |
| WER | 0.2432 |
| Word substitutions | 857 |
| Word deletions | 3572 |
| Word insertions | 171 |
| CER | 0.1725 |
| Short exact-match rate | 0.5000 |
| Non-speech nonempty cases | 1 |
| Non-speech error cases | 0 |
| Prediction error cases | 0 |
| Reference speech (ms) | 5025344 |
| Missed speech (ms) | 468625 |
| Missed speech rate | 0.0933 |
| Reference non-speech (ms) | 455599 |
| False alarm (ms) | 135538 |
| False-alarm rate | 0.2975 |
| Boundary precision | 0.0929 |
| Boundary recall | 0.5187 |
| Boundary median error (ms) | 99.0000 |
| Boundary p95 error (ms) | 234.0000 |
| Onset median error (ms) | 57.0000 |
| Onset p95 error (ms) | 230.0000 |
| Offset median error (ms) | 142.0000 |
| Offset p95 error (ms) | 234.0000 |
| Timestamp violation cases | 2 |
| Timestamp violations | 2 |
| Parity mismatch cases | 0 |
| Median RTF | 0.1135 |
| p95 RTF | 2.8986 |
| Peak RSS (MiB) | 7023.4688 |
| Peak CUDA allocation (MiB) | — |

## Per stratum

| Group | Cases | WER | CER | Short exact | Non-speech nonempty | Miss (ms) | Miss rate | False alarm (ms) | False-alarm rate | Boundary P | Boundary R | Timestamp violations | RTF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| split=tuning | 4 | 0.2718 | 0.2040 | 0.0000 | 0 | 231381 | 0.1325 | 5640 | 0.0721 | 0.0366 | 0.4600 | 1 | 0.1237 |
| split=acceptance | 5 | 0.2278 | 0.1555 | 1.0000 | 1 | 237244 | 0.0724 | 129898 | 0.3442 | 0.1233 | 0.5295 | 1 | 0.1135 |
| language=en | 9 | 0.2432 | 0.1725 | 0.5000 | 1 | 468625 | 0.0933 | 135538 | 0.2975 | 0.0929 | 0.5187 | 2 | 0.1135 |
| stratum=30_minute | 1 | 0.2733 | 0.2055 | — | 0 | 231289 | 0.1335 | 5450 | 0.0805 | 0.0335 | 0.4375 | 0 | 0.1090 |
| stratum=60_minute | 1 | 0.2280 | 0.1555 | — | 0 | 237128 | 0.0726 | 92921 | 0.2777 | 0.1224 | 0.5279 | 1 | 0.1135 |
| stratum=background_noise | 1 | — | — | — | 0 | 0 | — | 0 | 0.0000 | — | — | 0 | 0.0102 |
| stratum=clean | 2 | 0.0241 | 0.0023 | — | 0 | 92 | 0.0034 | 732 | 0.6393 | 0.7500 | 0.7500 | 0 | 0.1315 |
| stratum=long_form | 2 | 0.2438 | 0.1729 | — | 0 | 468417 | 0.0937 | 98371 | 0.2445 | 0.0912 | 0.5142 | 1 | 0.1112 |
| stratum=meeting | 2 | 0.2438 | 0.1729 | — | 0 | 468417 | 0.0937 | 98371 | 0.2445 | 0.0912 | 0.5142 | 1 | 0.1112 |
| stratum=mono_16khz_flac | 4 | 0.2429 | 0.1721 | — | 0 | 468509 | 0.0932 | 99103 | 0.2457 | 0.0920 | 0.5157 | 1 | 0.1190 |
| stratum=multi_speaker | 2 | 0.2438 | 0.1729 | — | 0 | 468417 | 0.0937 | 98371 | 0.2445 | 0.0912 | 0.5142 | 1 | 0.1112 |
| stratum=music | 1 | — | — | — | 1 | 0 | — | 36380 | 0.9992 | 0.0000 | — | 0 | 0.0908 |
| stratum=name | 1 | 0.0000 | 0.0000 | 1.0000 | 0 | 116 | 0.2755 | 55 | 0.1250 | 1.0000 | 1.0000 | 0 | 0.8638 |
| stratum=non_speech | 3 | — | — | — | 1 | 0 | — | 36380 | 0.7031 | 0.0000 | — | 0 | 0.0194 |
| stratum=pcm_wav | 1 | — | — | — | 0 | 0 | — | 0 | 0.0000 | — | — | 0 | 0.0194 |
| stratum=read_speech | 2 | 0.0241 | 0.0023 | — | 0 | 92 | 0.0034 | 732 | 0.6393 | 0.7500 | 0.7500 | 0 | 0.1315 |
| stratum=short_utterance | 2 | 0.5000 | 0.2857 | 0.5000 | 0 | 116 | 0.1352 | 55 | 0.1250 | 1.0000 | 1.0000 | 1 | 1.8812 |
| stratum=silence | 1 | — | — | — | 0 | 0 | — | 0 | 0.0000 | — | — | 0 | 0.0194 |
| stratum=speech | 6 | 0.2429 | 0.1721 | 0.5000 | 0 | 468625 | 0.0933 | 99158 | 0.2455 | 0.0930 | 0.5187 | 2 | 0.1315 |
| stratum=spontaneous_speech | 2 | 0.2438 | 0.1729 | — | 0 | 468417 | 0.0937 | 98371 | 0.2445 | 0.0912 | 0.5142 | 1 | 0.1112 |
| stratum=stereo_44khz_ogg | 4 | 3.5000 | 5.2857 | 0.5000 | 1 | 116 | 0.1352 | 36435 | 0.7723 | 0.5000 | 1.0000 | 1 | 0.4773 |
| stratum=word | 1 | 1.0000 | 0.6667 | 0.0000 | 0 | 0 | 0.0000 | 0 | — | 1.0000 | 1.0000 | 1 | 2.8986 |

## Gates

Gates evaluate only manifest rows with `split=acceptance`.

| Gate | Rule | Actual | Status |
|---|---:|---:|---|
| `boundary_precision` | min 0.9500 | 0.1233 | DISABLED |
| `boundary_recall` | min 0.9500 | 0.5295 | DISABLED |
| `cer` | max 0.0500 | 0.1555 | DISABLED |
| `false_alarm_rate` | max 0.0500 | 0.3442 | DISABLED |
| `missed_speech_rate` | max 0.0500 | 0.0724 | DISABLED |
| `non_speech_error_cases` | max 0.0000 | 0 | PASS |
| `non_speech_nonempty_cases` | max 0.0000 | 1 | FAIL |
| `parity_mismatch_cases` | max 0.0000 | 0 | DISABLED |
| `peak_cuda_mb_max` | max 16384.0000 | — | DISABLED |
| `peak_rss_mb_max` | max 16384.0000 | 7023.4688 | DISABLED |
| `rtf_median` | max 1.0000 | 0.1135 | DISABLED |
| `short_exact_match_rate` | min 1.0000 | 1.0000 | PASS |
| `timestamp_violation_cases` | max 0.0000 | 1 | FAIL |
| `wer` | max 0.1000 | 0.2278 | DISABLED |

## Cases

| Case | Split | Strata | Duration (s) | Inference (s) | RTF | Peak RSS (MiB) | Peak CUDA (MiB) | WER | CER | Miss (ms) | Miss rate | False alarm (ms) | False-alarm rate | Timestamp violations | Error |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| silence-5s | acceptance | non_speech, silence, pcm_wav | 5.0000 | 0.0971 | 0.0194 | 426.9062 | — | — | — | 0 | — | 0 | 0.0000 | 0 | — |
| rain-10s | tuning | non_speech, background_noise, stereo_44khz_ogg | 10.3310 | 0.1049 | 0.0102 | 433.4688 | — | — | — | 0 | — | 0 | 0.0000 | 0 | — |
| music-36s | acceptance | non_speech, music, stereo_44khz_ogg | 36.4090 | 3.3045 | 0.0908 | 7023.4688 | — | — | — | 0 | — | 36380 | 0.9992 | 0 | — |
| short-word-ear | tuning | speech, short_utterance, word, stereo_44khz_ogg | 0.4370 | 1.2667 | 2.8986 | 2046.8438 | — | 1.0000 | 0.6667 | 0 | 0.0000 | 0 | — | 1 | — |
| short-name-john | acceptance | speech, short_utterance, name, stereo_44khz_ogg | 0.8610 | 0.7437 | 0.8638 | 1454.7812 | — | 0.0000 | 0.0000 | 116 | 0.2755 | 55 | 0.1250 | 0 | — |
| librispeech-sample-1 | tuning | speech, read_speech, clean, mono_16khz_flac | 13.6900 | 1.8953 | 0.1384 | 1488.5469 | — | 0.0455 | 0.0041 | 92 | 0.0068 | 190 | 0.9135 | 0 | — |
| librispeech-sample-2 | acceptance | speech, read_speech, clean, mono_16khz_flac | 14.2150 | 1.7712 | 0.1246 | 1338.4844 | — | 0.0000 | 0.0000 | 0 | 0.0000 | 542 | 0.5784 | 0 | — |
| ami-meeting-30m | tuning | speech, meeting, multi_speaker, spontaneous_speech, long_form, 30_minute, mono_16khz_flac | 1800.0000 | 196.2109 | 0.1090 | 2483.5781 | — | 0.2733 | 0.2055 | 231289 | 0.1335 | 5450 | 0.0805 | 0 | — |
| ami-meeting-60m | acceptance | speech, meeting, multi_speaker, spontaneous_speech, long_form, 60_minute, mono_16khz_flac | 3600.0000 | 408.5622 | 0.1135 | 3305.4062 | — | 0.2280 | 0.1555 | 237128 | 0.0726 | 92921 | 0.2777 | 1 | — |

## Interpretation limits

- Boundary and speech-duration metrics compare final TSV intervals with the references. They are end-to-end output metrics, not direct Silero VAD measurements.
- WER and CER score final post-processed text. The current pipeline does not expose raw decoder text, so this report cannot isolate decoder fidelity from later text mutation.
- Pipeline settings are defined by the protocol and may differ from application defaults; interpret results only for the recorded configuration.
- CER includes spaces after NFKC, casefolding, punctuation-to-space conversion, and whitespace collapse.
- Peak RSS is sampled for this process and its children, so spikes shorter than the sampling interval may be missed. CUDA memory is PyTorch's peak allocated memory.
- The first case that loads Whisper includes cold model loading; later cases may reuse in-process model caches.
