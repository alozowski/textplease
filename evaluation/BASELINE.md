# Audio quality evaluation

This report scores the configured public `textplease` pipeline against the versioned manifest and protocol.

## Run

| Field | Value |
|---|---|
| Manifest SHA-256 | `99d58806e89aeaad55a7684568bb40b5c34365c655ebef7b78b099c26fdebe3d` |
| Protocol SHA-256 | `ec747c0eff5a05dc6f0debe543f73bf2c3c23f5ad8039b09acdbdc5e38ba5fc3` |
| Inference evaluator SHA-256 | `c475c2f0ab378cca509284723894d1663f5e31a5761ceb2402132a0276436092` |
| Scorer SHA-256 | `46903ad0bedba32c2e8270310e0f2c2b7c578643af862ed9d25041a3ee598531` |
| Scorer JiWER | `4.0.0` |
| Scorer RapidFuzz | `3.14.5` |
| Random seed | `0` |
| Source revision | `90df73689e37f3f7f7daa5a7ff76e76a9d94a446` |
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
| WER | 0.2442 |
| Word substitutions | 863 |
| Word deletions | 3556 |
| Word insertions | 199 |
| CER | 0.1743 |
| Short exact-match rate | 0.5000 |
| Non-speech nonempty cases | 3 |
| Non-speech error cases | 0 |
| Prediction error cases | 1 |
| Reference speech (ms) | 5025344 |
| Missed speech (ms) | 425553 |
| Missed speech rate | 0.0847 |
| Reference non-speech (ms) | 455599 |
| False alarm (ms) | 143009 |
| False-alarm rate | 0.3139 |
| Boundary precision | 0.0958 |
| Boundary recall | 0.5452 |
| Boundary median error (ms) | 105.5000 |
| Boundary p95 error (ms) | 231.0000 |
| Onset median error (ms) | 70.0000 |
| Onset p95 error (ms) | 231.0000 |
| Offset median error (ms) | 136.0000 |
| Offset p95 error (ms) | 231.0000 |
| Timestamp violation cases | 1 |
| Timestamp violations | 1 |
| Parity mismatch cases | 0 |
| Median RTF | 0.1526 |
| p95 RTF | 0.9149 |
| Peak RSS (MiB) | 7220.7344 |
| Peak CUDA allocation (MiB) | — |

## Per stratum

| Group | Cases | WER | CER | Short exact | Non-speech nonempty | Miss (ms) | Miss rate | False alarm (ms) | False-alarm rate | Boundary P | Boundary R | Timestamp violations | RTF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| split=tuning | 4 | 0.2791 | 0.2136 | 0.0000 | 1 | 203728 | 0.1167 | 16050 | 0.2052 | 0.0336 | 0.4400 | 0 | 0.2401 |
| split=acceptance | 5 | 0.2253 | 0.1533 | 1.0000 | 2 | 221825 | 0.0676 | 126959 | 0.3364 | 0.1305 | 0.5646 | 1 | 0.1488 |
| language=en | 9 | 0.2442 | 0.1743 | 0.5000 | 3 | 425553 | 0.0847 | 143009 | 0.3139 | 0.0958 | 0.5452 | 1 | 0.1526 |
| stratum=30_minute | 1 | 0.2789 | 0.2135 | — | 0 | 203191 | 0.1173 | 6052 | 0.0894 | 0.0322 | 0.4375 | 0 | 0.1526 |
| stratum=60_minute | 1 | 0.2253 | 0.1530 | — | 0 | 221721 | 0.0679 | 85014 | 0.2541 | 0.1298 | 0.5632 | 0 | 0.1488 |
| stratum=background_noise | 1 | — | — | — | 1 | 0 | — | 9800 | 0.9486 | 0.0000 | — | 0 | 0.3276 |
| stratum=clean | 2 | 0.0482 | 0.0046 | — | 0 | 100 | 0.0037 | 720 | 0.6288 | 0.7500 | 0.7500 | 0 | 0.1338 |
| stratum=long_form | 2 | 0.2440 | 0.1741 | — | 0 | 424912 | 0.0850 | 91066 | 0.2264 | 0.0948 | 0.5442 | 0 | 0.1507 |
| stratum=meeting | 2 | 0.2440 | 0.1741 | — | 0 | 424912 | 0.0850 | 91066 | 0.2264 | 0.0948 | 0.5442 | 0 | 0.1507 |
| stratum=mono_16khz_flac | 4 | 0.2431 | 0.1733 | — | 0 | 425012 | 0.0846 | 91786 | 0.2275 | 0.0955 | 0.5455 | 0 | 0.1436 |
| stratum=multi_speaker | 2 | 0.2440 | 0.1741 | — | 0 | 424912 | 0.0850 | 91066 | 0.2264 | 0.0948 | 0.5442 | 0 | 0.1507 |
| stratum=music | 1 | — | — | — | 1 | 0 | — | 36380 | 0.9992 | 0.0000 | — | 0 | 0.0645 |
| stratum=name | 1 | 0.0000 | 0.0000 | 1.0000 | 0 | 104 | 0.2470 | 43 | 0.0977 | 1.0000 | 1.0000 | 0 | 0.9149 |
| stratum=non_speech | 3 | — | — | — | 3 | 0 | — | 51180 | 0.9892 | 0.0000 | — | 1 | 0.3276 |
| stratum=pcm_wav | 1 | — | — | — | 1 | 0 | — | 5000 | 1.0000 | 0.0000 | — | 1 | 0.3915 |
| stratum=read_speech | 2 | 0.0482 | 0.0046 | — | 0 | 100 | 0.0037 | 720 | 0.6288 | 0.7500 | 0.7500 | 0 | 0.1338 |
| stratum=short_utterance | 2 | 0.5000 | 0.4286 | 0.5000 | 0 | 541 | 0.6305 | 43 | 0.0977 | 1.0000 | 0.5000 | 0 | 0.6281 |
| stratum=silence | 1 | — | — | — | 1 | 0 | — | 5000 | 1.0000 | 0.0000 | — | 1 | 0.3915 |
| stratum=speech | 6 | 0.2432 | 0.1733 | 0.5000 | 0 | 425553 | 0.0847 | 91829 | 0.2274 | 0.0960 | 0.5452 | 0 | 0.1507 |
| stratum=spontaneous_speech | 2 | 0.2440 | 0.1741 | — | 0 | 424912 | 0.0850 | 91066 | 0.2264 | 0.0948 | 0.5442 | 0 | 0.1507 |
| stratum=stereo_44khz_ogg | 4 | 9.0000 | 12.7143 | 0.5000 | 2 | 541 | 0.6305 | 46223 | 0.9797 | 0.2500 | 0.5000 | 0 | 0.3344 |
| stratum=word | 1 | 1.0000 | 1.0000 | 0.0000 | 0 | 437 | 1.0000 | 0 | — | 0.0000 | 0.0000 | 0 | 0.3413 |

## Gates

Gates evaluate only manifest rows with `split=acceptance`.

| Gate | Rule | Actual | Status |
|---|---:|---:|---|
| `boundary_precision` | min 0.9500 | 0.1305 | DISABLED |
| `boundary_recall` | min 0.9500 | 0.5646 | DISABLED |
| `cer` | max 0.0500 | 0.1533 | DISABLED |
| `false_alarm_rate` | max 0.0500 | 0.3364 | DISABLED |
| `missed_speech_rate` | max 0.0500 | 0.0676 | DISABLED |
| `non_speech_error_cases` | max 0.0000 | 0 | PASS |
| `non_speech_nonempty_cases` | max 0.0000 | 2 | FAIL |
| `parity_mismatch_cases` | max 0.0000 | 0 | DISABLED |
| `peak_cuda_mb_max` | max 16384.0000 | — | DISABLED |
| `peak_rss_mb_max` | max 16384.0000 | 7220.7344 | DISABLED |
| `rtf_median` | max 1.0000 | 0.1488 | DISABLED |
| `short_exact_match_rate` | min 1.0000 | 1.0000 | PASS |
| `timestamp_violation_cases` | max 0.0000 | 1 | FAIL |
| `wer` | max 0.1000 | 0.2253 | DISABLED |

## Cases

| Case | Split | Strata | Duration (s) | Inference (s) | RTF | Peak RSS (MiB) | Peak CUDA (MiB) | WER | CER | Miss (ms) | Miss rate | False alarm (ms) | False-alarm rate | Timestamp violations | Error |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| silence-5s | acceptance | non_speech, silence, pcm_wav | 5.0000 | 1.9576 | 0.3915 | 7220.7344 | — | — | — | 0 | — | 5000 | 1.0000 | 1 | — |
| rain-10s | tuning | non_speech, background_noise, stereo_44khz_ogg | 10.3310 | 3.3843 | 0.3276 | 1968.3750 | — | — | — | 0 | — | 9800 | 0.9486 | 0 | — |
| music-36s | acceptance | non_speech, music, stereo_44khz_ogg | 36.4090 | 2.3487 | 0.0645 | 1156.8750 | — | — | — | 0 | — | 36380 | 0.9992 | 0 | — |
| short-word-ear | tuning | speech, short_utterance, word, stereo_44khz_ogg | 0.4370 | 0.1491 | 0.3413 | 1122.5312 | — | 1.0000 | 1.0000 | 437 | 1.0000 | 0 | — | 0 | empty_output |
| short-name-john | acceptance | speech, short_utterance, name, stereo_44khz_ogg | 0.8610 | 0.7877 | 0.9149 | 1480.8125 | — | 0.0000 | 0.0000 | 104 | 0.2470 | 43 | 0.0977 | 0 | — |
| librispeech-sample-1 | tuning | speech, read_speech, clean, mono_16khz_flac | 13.6900 | 1.8946 | 0.1384 | 1445.7812 | — | 0.0455 | 0.0041 | 100 | 0.0074 | 198 | 0.9519 | 0 | — |
| librispeech-sample-2 | acceptance | speech, read_speech, clean, mono_16khz_flac | 14.2150 | 1.8373 | 0.1292 | 1491.2969 | — | 0.0513 | 0.0052 | 0 | 0.0000 | 522 | 0.5571 | 0 | — |
| ami-meeting-30m | tuning | speech, meeting, multi_speaker, spontaneous_speech, long_form, 30_minute, mono_16khz_flac | 1800.0000 | 274.7498 | 0.1526 | 2639.2500 | — | 0.2789 | 0.2135 | 203191 | 0.1173 | 6052 | 0.0894 | 0 | — |
| ami-meeting-60m | acceptance | speech, meeting, multi_speaker, spontaneous_speech, long_form, 60_minute, mono_16khz_flac | 3600.0000 | 535.5197 | 0.1488 | 3465.5469 | — | 0.2253 | 0.1530 | 221721 | 0.0679 | 85014 | 0.2541 | 0 | — |

## Interpretation limits

- Boundary and speech-duration metrics compare final TSV intervals with the references. They are end-to-end output metrics, not direct Silero VAD measurements.
- WER and CER score final post-processed text. The current pipeline does not expose raw decoder text, so this report cannot isolate decoder fidelity from later text mutation.
- Pipeline settings are defined by the protocol and may differ from application defaults; interpret results only for the recorded configuration.
- CER includes spaces after NFKC, casefolding, punctuation-to-space conversion, and whitespace collapse.
- Peak RSS is sampled for this process and its children, so spikes shorter than the sampling interval may be missed. CUDA memory is PyTorch's peak allocated memory.
- The first inference case includes cold model loading; later cases may reuse in-process model caches.
