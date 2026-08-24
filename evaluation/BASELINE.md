# Audio quality evaluation

This report scores the configured public `textplease` pipeline against the versioned manifest and protocol.

## Run

| Field | Value |
|---|---|
| Manifest SHA-256 | `d172a05be7283bf43c496f9686262764ef23c449f4b6a22e01e3b4c8c0483d65` |
| Protocol SHA-256 | `84a14380f14e18c5912d557018f6f942d940e0caa4182e86097481e2404e6048` |
| Inference evaluator SHA-256 | `4475d7dbc7340f795e872245b2539f29443b2df6e5609734d17bb7d0432f796e` |
| Scorer SHA-256 | `7a508a172bfcf7f9a6f6d844a5d9286d16d2a43b2d9e816d7ba2481756f8be89` |
| Scorer JiWER | `4.0.0` |
| Scorer RapidFuzz | `3.14.5` |
| Random seed | `0` |
| Source revision | `a4bf9773a2ad618d82ebe2a87c5bf417e7cf2c34` |
| Source dirty | `True` |
| Device | `mps` |
| Whisper batch size | `1` |
| Model repository | `openai/whisper-large-v3` |
| Model revision | `06f233fe06e710322aca913c1bc4249a0d71fce1` |
| Audio classifier repository | `MIT/ast-finetuned-audioset-10-10-0.4593` |
| Audio classifier revision | `f826b80d28226b62986cc218e5cec390b1096902` |
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
| music-jazz-sax-24s | [Jazz Tenor Sax](https://commons.wikimedia.org/w/index.php?title=File:Jazz-Sax.ogg&oldid=971661804) with `971661804` | [CC-BY-2.5](https://creativecommons.org/licenses/by/2.5/) | Jazz Tenor Sax by Wikimedia Commons user Serolillo. |
| music-36s | [Greensleaves.ogg](https://commons.wikimedia.org/w/index.php?title=File:Greensleaves.ogg&oldid=845754359) with `845754359` | [Public-Domain](https://commons.wikimedia.org/w/index.php?title=File:Greensleaves.ogg&oldid=845754359#Licensing) | Performed and recorded by Wikimedia Commons user Rv87. |
| speech-over-music-ear | [TextPlease speech-over-music mix of Jazz Tenor Sax and En-uk-ear.ogg](https://commons.wikimedia.org/w/index.php?title=File:Jazz-Sax.ogg&oldid=971661804) with `components-sha256:312a88774e7dd9d18e20420c2a1f3721156709032d55ba1e8cb630f433255a5b+34d9e3db4cac7a9091362cbabd59983cf4cd6d5d463a19bf481254a32799b356` | [CC-BY-2.5](https://creativecommons.org/licenses/by/2.5/) | Jazz Tenor Sax by Wikimedia Commons user Serolillo, mixed with the public-domain En-uk-ear recording by Wikimedia Commons user Chris Melville. |
| speech-over-music-john | [TextPlease speech-over-music mix of Greensleaves.ogg and En-au-John.ogg](https://commons.wikimedia.org/w/index.php?title=File:Greensleaves.ogg&oldid=845754359) with `components-sha256:1c31668ade22bb83f067ce178a2282ebfdf320bbca9b86b4413fedba64025ec9+8a7272b49e5c7b73aa13a50023a193118488c859d887efca6fe23022e524a9b9` | [CC-BY-SA-4.0](https://creativecommons.org/licenses/by-sa/4.0/) | Greensleaves performed and recorded by public-domain contributor Rv87, mixed with En-au-John spoken and recorded by Commander Keane under CC BY-SA 4.0 on Wikimedia Commons. |
| short-word-ear | [En-uk-ear.ogg](https://commons.wikimedia.org/w/index.php?title=File:En-uk-ear.ogg&oldid=1229077824) with `1229077824` | [Public-Domain](https://commons.wikimedia.org/w/index.php?title=File:En-uk-ear.ogg&oldid=1229077824#Licensing) | Spoken and recorded by Wikimedia Commons user Chris Melville. |
| short-name-john | [En-au-John.ogg](https://commons.wikimedia.org/w/index.php?title=File:En-au-John.ogg&oldid=724849560) with `724849560` | [CC-BY-SA-4.0](https://creativecommons.org/licenses/by-sa/4.0/) | Spoken and recorded by Commander Keane on Wikimedia Commons. |
| librispeech-sample-1 | [Hugging Face LibriSpeech sample 1](https://cdn-media.huggingface.co/speech_samples/sample1.flac) with `sha256:cb5c48a2d1d6f7dedd0330f088a4cbe76de1a86e6a6109c06d255bb1ca2f7542` | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | LibriSpeech ASR Corpus by Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur. Hosted by Hugging Face. |
| librispeech-sample-2 | [Hugging Face LibriSpeech sample 2](https://cdn-media.huggingface.co/speech_samples/sample2.flac) with `sha256:4e82c7e879bce92c1d3bc99ddb7bdf681611bc251b6d244430e54fe44b86e75e` | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | LibriSpeech ASR Corpus by Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur. Hosted by Hugging Face. |
| pace-slow-tuning | [LibriSpeech validation.clean speaker 6319, chapter 57405, utterances 0008 through 0011 concatenated in source order](https://huggingface.co/datasets/openslr/librispeech_asr/tree/71cacbfb7e2354c4226d01e70d77d5fca3d04ba1) with `71cacbfb7e2354c4226d01e70d77d5fca3d04ba1` | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | LibriSpeech ASR Corpus by Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur. Source rows mirrored by Hugging Face and concatenated losslessly by TextPlease. |
| pace-normal-tuning | [LibriSpeech validation.clean speaker 5338, chapter 24640, utterances 0007 through 0009 concatenated in source order](https://huggingface.co/datasets/openslr/librispeech_asr/tree/71cacbfb7e2354c4226d01e70d77d5fca3d04ba1) with `71cacbfb7e2354c4226d01e70d77d5fca3d04ba1` | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | LibriSpeech ASR Corpus by Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur. Source rows mirrored by Hugging Face and concatenated losslessly by TextPlease. |
| pace-fast-tuning | [LibriSpeech validation.clean speaker 2277, chapter 149896, utterances 0007 through 0013 concatenated in source order](https://huggingface.co/datasets/openslr/librispeech_asr/tree/71cacbfb7e2354c4226d01e70d77d5fca3d04ba1) with `71cacbfb7e2354c4226d01e70d77d5fca3d04ba1` | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | LibriSpeech ASR Corpus by Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur. Source rows mirrored by Hugging Face and concatenated losslessly by TextPlease. |
| pace-slow-acceptance | [LibriSpeech test.clean speaker 8455, chapter 210777, utterances 0054 through 0058 concatenated in source order](https://huggingface.co/datasets/openslr/librispeech_asr/tree/71cacbfb7e2354c4226d01e70d77d5fca3d04ba1) with `71cacbfb7e2354c4226d01e70d77d5fca3d04ba1` | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | LibriSpeech ASR Corpus by Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur. Source rows mirrored by Hugging Face and concatenated losslessly by TextPlease. |
| pace-normal-acceptance | [LibriSpeech test.clean speaker 6930, chapter 75918, utterances 0001 through 0004 concatenated in source order](https://huggingface.co/datasets/openslr/librispeech_asr/tree/71cacbfb7e2354c4226d01e70d77d5fca3d04ba1) with `71cacbfb7e2354c4226d01e70d77d5fca3d04ba1` | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | LibriSpeech ASR Corpus by Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur. Source rows mirrored by Hugging Face and concatenated losslessly by TextPlease. |
| pace-fast-acceptance | [LibriSpeech test.clean speaker 61, chapter 70970, utterances 0024 through 0028 concatenated in source order](https://huggingface.co/datasets/openslr/librispeech_asr/tree/71cacbfb7e2354c4226d01e70d77d5fca3d04ba1) with `71cacbfb7e2354c4226d01e70d77d5fca3d04ba1` | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | LibriSpeech ASR Corpus by Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur. Source rows mirrored by Hugging Face and concatenated losslessly by TextPlease. |
| ami-meeting-30m | [Continuous 30-minute excerpt from AMI ES2002b scenario meeting headset mix, half-open range [360.9545, 2160.9545), converted to mono 16 kHz FLAC with FFmpeg 9.0](https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2002b/audio/ES2002b.Mix-Headset.wav) with `source-sha256:977fbf6cd473cfb1984b41755762eea4d7ffdad3fb15adcfdebcb8842163ec66` | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | AMI Meeting Corpus by the AMI Consortium. Excerpted and converted to FLAC by TextPlease. Reference text and intervals are from AMI manual annotations v1.6.2. |
| ami-meeting-60m | [Continuous 60-minute excerpt from AMI EN2001a natural meeting headset mix, half-open range [201.2265, 3801.2265), converted to mono 16 kHz FLAC with FFmpeg 9.0](https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/EN2001a/audio/EN2001a.Mix-Headset.wav) with `source-sha256:81e06be816e9d94d0bee410bef1b158b2cdfba8e2b80f44a6e62cf6b9fd780f9` | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | AMI Meeting Corpus by the AMI Consortium. Excerpted and converted to FLAC by TextPlease. Reference text and intervals are from AMI manual annotations v1.6.2. |

## Overall

| Metric | Value |
|---|---:|
| Cases | 18 |
| WER | 0.2359 |
| Word substitutions | 875 |
| Word deletions | 3576 |
| Word insertions | 169 |
| CER | 0.1659 |
| Short exact-match rate | 0.5000 |
| Non-speech nonempty cases | 0 |
| Non-speech error cases | 0 |
| Prediction error cases | 0 |
| Reference speech (ms) | 5026202 |
| Missed speech (ms) | 468652 |
| Missed speech rate | 0.0932 |
| Reference non-speech (ms) | 539150 |
| False alarm (ms) | 115947 |
| False-alarm rate | 0.2151 |
| Boundary precision | 0.0934 |
| Boundary recall | 0.5186 |
| Boundary median error (ms) | 98.0000 |
| Boundary p95 error (ms) | 234.0000 |
| Onset median error (ms) | 58.0000 |
| Onset p95 error (ms) | 230.0000 |
| Offset median error (ms) | 142.0000 |
| Offset p95 error (ms) | 234.0000 |
| Timestamp violation cases | 3 |
| Timestamp violations | 3 |
| Parity mismatch cases | 0 |
| Median RTF | 0.1331 |
| p95 RTF | 4.6037 |
| Peak RSS (MiB) | 8059.9062 |
| Peak CUDA allocation (MiB) | — |

## Per stratum

| Group | Cases | WER | CER | Short exact | Non-speech nonempty | Miss (ms) | Miss rate | False alarm (ms) | False-alarm rate | Boundary P | Boundary R | Timestamp violations | RTF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| split=tuning | 9 | 0.2607 | 0.1942 | 0.0000 | 0 | 231408 | 0.1325 | 5730 | 0.0456 | 0.0381 | 0.4706 | 1 | 0.1313 |
| split=acceptance | 9 | 0.2222 | 0.1504 | 1.0000 | 0 | 237244 | 0.0723 | 110217 | 0.2666 | 0.1234 | 0.5276 | 2 | 0.1349 |
| language=en | 18 | 0.2359 | 0.1659 | 0.5000 | 0 | 468652 | 0.0932 | 115947 | 0.2151 | 0.0934 | 0.5186 | 3 | 0.1331 |
| stratum=30_minute | 1 | 0.2733 | 0.2055 | — | 0 | 231289 | 0.1335 | 5450 | 0.0805 | 0.0335 | 0.4375 | 0 | 0.1299 |
| stratum=60_minute | 1 | 0.2280 | 0.1555 | — | 0 | 237128 | 0.0726 | 92921 | 0.2777 | 0.1224 | 0.5279 | 1 | 0.1355 |
| stratum=background_noise | 1 | — | — | — | 0 | 0 | — | 0 | 0.0000 | — | — | 0 | 0.0102 |
| stratum=clean | 8 | 0.0358 | 0.0136 | — | 0 | 92 | 0.0034 | 732 | 0.6393 | 0.7500 | 0.7500 | 1 | 0.1495 |
| stratum=long_form | 2 | 0.2438 | 0.1729 | — | 0 | 468417 | 0.0937 | 98371 | 0.2445 | 0.0912 | 0.5142 | 1 | 0.1327 |
| stratum=meeting | 2 | 0.2438 | 0.1729 | — | 0 | 468417 | 0.0937 | 98371 | 0.2445 | 0.0912 | 0.5142 | 1 | 0.1327 |
| stratum=mono_16khz_flac | 12 | 0.2358 | 0.1659 | 0.5000 | 0 | 468536 | 0.0932 | 115892 | 0.2503 | 0.0924 | 0.5156 | 2 | 0.1352 |
| stratum=multi_speaker | 2 | 0.2438 | 0.1729 | — | 0 | 468417 | 0.0937 | 98371 | 0.2445 | 0.0912 | 0.5142 | 1 | 0.1327 |
| stratum=music | 4 | 0.5000 | 0.2857 | 0.5000 | 0 | 27 | 0.0315 | 16789 | 0.1400 | 0.5000 | 0.5000 | 0 | 0.0306 |
| stratum=name | 1 | 0.0000 | 0.0000 | 1.0000 | 0 | 116 | 0.2755 | 55 | 0.1250 | 1.0000 | 1.0000 | 0 | 1.3946 |
| stratum=non_speech | 4 | — | — | — | 0 | 0 | — | 0 | 0.0000 | — | — | 0 | 0.0155 |
| stratum=pace_fast | 2 | 0.0403 | 0.0094 | — | 0 | — | — | — | — | — | — | 0 | 0.1631 |
| stratum=pace_normal | 2 | 0.0232 | 0.0203 | — | 0 | — | — | — | — | — | — | 1 | 0.1362 |
| stratum=pace_slow | 2 | 0.0545 | 0.0139 | — | 0 | — | — | — | — | — | — | 0 | 0.1301 |
| stratum=pcm_wav | 1 | — | — | — | 0 | 0 | — | 0 | 0.0000 | — | — | 0 | 0.0193 |
| stratum=read_speech | 8 | 0.0358 | 0.0136 | — | 0 | 92 | 0.0034 | 732 | 0.6393 | 0.7500 | 0.7500 | 1 | 0.1495 |
| stratum=saxophone | 1 | — | — | — | 0 | 0 | — | 0 | 0.0000 | — | — | 0 | 0.0222 |
| stratum=short_utterance | 4 | 0.5000 | 0.2857 | 0.5000 | 0 | 143 | 0.0833 | 16844 | 0.2808 | 0.7500 | 0.7500 | 1 | 0.7624 |
| stratum=silence | 1 | — | — | — | 0 | 0 | — | 0 | 0.0000 | — | — | 0 | 0.0193 |
| stratum=single_speaker | 6 | 0.0372 | 0.0149 | — | 0 | — | — | — | — | — | — | 1 | 0.1362 |
| stratum=speech | 14 | 0.2359 | 0.1659 | 0.5000 | 0 | 468652 | 0.0932 | 115947 | 0.2502 | 0.0934 | 0.5186 | 3 | 0.1365 |
| stratum=speech_over_music | 2 | 0.5000 | 0.2857 | 0.5000 | 0 | 27 | 0.0315 | 16789 | 0.2819 | 0.5000 | 0.5000 | 0 | 0.0846 |
| stratum=spontaneous_speech | 2 | 0.2438 | 0.1729 | — | 0 | 468417 | 0.0937 | 98371 | 0.2445 | 0.0912 | 0.5142 | 1 | 0.1327 |
| stratum=stereo_44khz_ogg | 5 | 0.5000 | 0.2857 | 0.5000 | 0 | 116 | 0.1352 | 55 | 0.0008 | 1.0000 | 1.0000 | 1 | 0.0222 |
| stratum=word | 1 | 1.0000 | 0.6667 | 0.0000 | 0 | 0 | 0.0000 | 0 | — | 1.0000 | 1.0000 | 1 | 4.6037 |

## Gates

Gates evaluate only manifest rows with `split=acceptance`.

| Gate | Rule | Actual | Status |
|---|---:|---:|---|
| `boundary_precision` | min 0.9500 | 0.1234 | DISABLED |
| `boundary_recall` | min 0.9500 | 0.5276 | DISABLED |
| `cer` | max 0.0500 | 0.1504 | DISABLED |
| `false_alarm_rate` | max 0.0500 | 0.2666 | DISABLED |
| `missed_speech_rate` | max 0.0500 | 0.0723 | DISABLED |
| `non_speech_error_cases` | max 0.0000 | 0 | PASS |
| `non_speech_nonempty_cases` | max 0.0000 | 0 | PASS |
| `parity_mismatch_cases` | max 0.0000 | 0 | DISABLED |
| `peak_cuda_mb_max` | max 16384.0000 | — | DISABLED |
| `peak_rss_mb_max` | max 16384.0000 | 2700.9062 | DISABLED |
| `rtf_median` | max 1.0000 | 0.1349 | DISABLED |
| `short_exact_match_rate` | min 1.0000 | 1.0000 | PASS |
| `timestamp_violation_cases` | max 0.0000 | 2 | FAIL |
| `wer` | max 0.1000 | 0.2222 | DISABLED |

## Cases

| Case | Split | Strata | Duration (s) | Inference (s) | RTF | Peak RSS (MiB) | Peak CUDA (MiB) | WER | CER | Miss (ms) | Miss rate | False alarm (ms) | False-alarm rate | Timestamp violations | Error |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| silence-5s | acceptance | non_speech, silence, pcm_wav | 5.0000 | 0.0965 | 0.0193 | 427.4219 | — | — | — | 0 | — | 0 | 0.0000 | 0 | — |
| rain-10s | tuning | non_speech, background_noise, stereo_44khz_ogg | 10.3310 | 0.1054 | 0.0102 | 433.9375 | — | — | — | 0 | — | 0 | 0.0000 | 0 | — |
| music-jazz-sax-24s | tuning | non_speech, music, saxophone, stereo_44khz_ogg | 24.0000 | 0.5318 | 0.0222 | 558.6875 | — | — | — | 0 | — | 0 | 0.0000 | 0 | — |
| music-36s | acceptance | non_speech, music, stereo_44khz_ogg | 36.4090 | 0.4277 | 0.0117 | 568.1562 | — | — | — | 0 | — | 0 | 0.0000 | 0 | — |
| speech-over-music-ear | tuning | speech, music, speech_over_music, short_utterance, mono_16khz_flac | 24.0000 | 3.1219 | 0.1301 | 8059.9062 | — | 1.0000 | 0.6667 | 27 | 0.0618 | 90 | 0.0038 | 0 | — |
| speech-over-music-john | acceptance | speech, music, speech_over_music, short_utterance, mono_16khz_flac | 36.4090 | 1.4224 | 0.0391 | 1363.5156 | — | 0.0000 | 0.0000 | 0 | 0.0000 | 16699 | 0.4640 | 0 | — |
| short-word-ear | tuning | speech, short_utterance, word, stereo_44khz_ogg | 0.4370 | 2.0118 | 4.6037 | 2040.9062 | — | 1.0000 | 0.6667 | 0 | 0.0000 | 0 | — | 1 | — |
| short-name-john | acceptance | speech, short_utterance, name, stereo_44khz_ogg | 0.8610 | 1.2008 | 1.3946 | 1206.2656 | — | 0.0000 | 0.0000 | 116 | 0.2755 | 55 | 0.1250 | 0 | — |
| librispeech-sample-1 | tuning | speech, read_speech, clean, mono_16khz_flac | 13.6900 | 2.4534 | 0.1792 | 1586.2969 | — | 0.0455 | 0.0041 | 92 | 0.0068 | 190 | 0.9135 | 0 | — |
| librispeech-sample-2 | acceptance | speech, read_speech, clean, mono_16khz_flac | 14.2150 | 2.3456 | 0.1650 | 1491.2188 | — | 0.0000 | 0.0000 | 0 | 0.0000 | 542 | 0.5784 | 0 | — |
| pace-slow-tuning | tuning | speech, read_speech, clean, single_speaker, pace_slow, mono_16khz_flac | 34.9400 | 4.5868 | 0.1313 | 1170.0938 | — | 0.0127 | 0.0049 | — | — | — | — | 0 | — |
| pace-normal-tuning | tuning | speech, read_speech, clean, single_speaker, pace_normal, mono_16khz_flac | 39.7550 | 5.4665 | 0.1375 | 1159.1562 | — | 0.0273 | 0.0265 | — | — | — | — | 0 | — |
| pace-fast-tuning | tuning | speech, read_speech, clean, single_speaker, pace_fast, mono_16khz_flac | 36.4550 | 5.8908 | 0.1616 | 1176.4531 | — | 0.0441 | 0.0088 | — | — | — | — | 0 | — |
| pace-slow-acceptance | acceptance | speech, read_speech, clean, single_speaker, pace_slow, mono_16khz_flac | 35.5900 | 4.5872 | 0.1289 | 1170.1875 | — | 0.0930 | 0.0208 | — | — | — | — | 0 | — |
| pace-normal-acceptance | acceptance | speech, read_speech, clean, single_speaker, pace_normal, mono_16khz_flac | 53.6300 | 7.2369 | 0.1349 | 1175.0469 | — | 0.0201 | 0.0155 | — | — | — | — | 1 | — |
| pace-fast-acceptance | acceptance | speech, read_speech, clean, single_speaker, pace_fast, mono_16khz_flac | 31.7750 | 5.2293 | 0.1646 | 1163.7969 | — | 0.0357 | 0.0100 | — | — | — | — | 0 | — |
| ami-meeting-30m | tuning | speech, meeting, multi_speaker, spontaneous_speech, long_form, 30_minute, mono_16khz_flac | 1800.0000 | 233.7446 | 0.1299 | 2495.3750 | — | 0.2733 | 0.2055 | 231289 | 0.1335 | 5450 | 0.0805 | 0 | — |
| ami-meeting-60m | acceptance | speech, meeting, multi_speaker, spontaneous_speech, long_form, 60_minute, mono_16khz_flac | 3600.0000 | 487.6979 | 0.1355 | 2700.9062 | — | 0.2280 | 0.1555 | 237128 | 0.0726 | 92921 | 0.2777 | 1 | — |

## Interpretation limits

- Boundary and speech-duration metrics compare final TSV intervals with the references. They are end-to-end output metrics, not direct Silero VAD measurements.
- Activity duration metrics exclude cases whose speech interval reference is null. An em dash means no interval reference was scored.
- Short exact match includes recordings whose total annotated speech is within the configured short duration, even when the surrounding recording is longer.
- WER and CER score final post-processed text. The current pipeline does not expose raw decoder text, so this report cannot isolate decoder fidelity from later text mutation.
- Audio-classifier identity is declared by the protocol and is not mechanically queried from the backend. Source and evaluator metadata aid auditing but do not hash an uncommitted runtime diff.
- Pipeline settings are defined by the protocol and may differ from application defaults; interpret results only for the recorded configuration.
- CER includes spaces after NFKC, casefolding, punctuation-to-space conversion, and whitespace collapse.
- Peak RSS is sampled for this process and its children, so spikes shorter than the sampling interval may be missed. CUDA memory is PyTorch's peak allocated memory.
- The first case that loads Whisper includes cold model loading; later cases may reuse in-process model caches.
