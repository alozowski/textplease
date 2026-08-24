# Audio quality evaluation

This report scores the configured public `textplease` pipeline against the versioned manifest and protocol.

## Run

| Field | Value |
|---|---|
| Manifest SHA-256 | `58a4e462e4c2fcbbd61615d6506328ca524e43485615bc1d696283ef4704be9c` |
| Protocol SHA-256 | `ec77f64af4222ce40b0164f303d71744f1c26c194e7352dca7bc46faae3ed6e2` |
| Inference evaluator SHA-256 | `f0816ffb181ba47dc5e0ef2c068a2c8c4a416807b4143603b906156c3bd1e891` |
| Scorer SHA-256 | `e48a1b88a07a9555be748c6d0027493c18f53cd37da5613638b7a946682d521a` |
| Scorer JiWER | `4.0.0` |
| Scorer RapidFuzz | `3.14.5` |
| Random seed | `0` |
| Source revision | `2939d32b1cd5f3030bd1043fb82caf7299e31711` |
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
| WER | 0.2355 |
| Word substitutions | 882 |
| Word deletions | 3559 |
| Word insertions | 172 |
| CER | 0.1655 |
| Short exact-match rate | 0.5000 |
| Non-speech nonempty cases | 0 |
| Non-speech error cases | 0 |
| Prediction error cases | 0 |
| Reference speech (ms) | 5026202 |
| Missed speech (ms) | 467530 |
| Missed speech rate | 0.0930 |
| Reference non-speech (ms) | 539150 |
| False alarm (ms) | 102493 |
| False-alarm rate | 0.1901 |
| Boundary precision | 0.0961 |
| Boundary recall | 0.5201 |
| Boundary median error (ms) | 98.5000 |
| Boundary p95 error (ms) | 234.0000 |
| Onset median error (ms) | 55.5000 |
| Onset p95 error (ms) | 223.0000 |
| Offset median error (ms) | 143.0000 |
| Offset p95 error (ms) | 234.0000 |
| Timestamp violation cases | 0 |
| Timestamp violations | 0 |
| Output segments | 1794 |
| Segment characters median | 43.0000 |
| Segment characters p95 | 94.0000 |
| Segment characters max | 241 |
| Segment duration median (ms) | 2220.0000 |
| Segment duration p95 (ms) | 6540.0000 |
| Segment duration max (ms) | 17120 |
| Parity mismatch cases | 0 |
| Median RTF | 0.1427 |
| p95 RTF | 4.8415 |
| Peak RSS (MiB) | 7337.9531 |
| Peak CUDA allocation (MiB) | — |

## Per stratum

| Group | Cases | WER | CER | Short exact | Non-speech nonempty | Miss (ms) | Miss rate | False alarm (ms) | False-alarm rate | Boundary P | Boundary R | Timestamp violations | RTF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| split=tuning | 9 | 0.2595 | 0.1931 | 0.0000 | 0 | 230082 | 0.1317 | 4696 | 0.0373 | 0.0404 | 0.4902 | 0 | 0.1423 |
| split=acceptance | 9 | 0.2223 | 0.1503 | 1.0000 | 0 | 237448 | 0.0724 | 97797 | 0.2366 | 0.1267 | 0.5257 | 0 | 0.1430 |
| language=en | 18 | 0.2355 | 0.1655 | 0.5000 | 0 | 467530 | 0.0930 | 102493 | 0.1901 | 0.0961 | 0.5201 | 0 | 0.1427 |
| stratum=30_minute | 1 | 0.2720 | 0.2044 | — | 0 | 229963 | 0.1327 | 4416 | 0.0652 | 0.0357 | 0.4583 | 0 | 0.1670 |
| stratum=60_minute | 1 | 0.2281 | 0.1554 | — | 0 | 237332 | 0.0727 | 80501 | 0.2406 | 0.1257 | 0.5260 | 0 | 0.1472 |
| stratum=background_noise | 1 | — | — | — | 0 | 0 | — | 0 | 0.0000 | — | — | 0 | 0.0107 |
| stratum=clean | 8 | 0.0358 | 0.0136 | — | 0 | 92 | 0.0034 | 732 | 0.6393 | 0.7500 | 0.7500 | 0 | 0.1586 |
| stratum=long_form | 2 | 0.2435 | 0.1724 | — | 0 | 467295 | 0.0935 | 84917 | 0.2111 | 0.0939 | 0.5158 | 0 | 0.1571 |
| stratum=meeting | 2 | 0.2435 | 0.1724 | — | 0 | 467295 | 0.0935 | 84917 | 0.2111 | 0.0939 | 0.5158 | 0 | 0.1571 |
| stratum=mono_16khz_flac | 12 | 0.2355 | 0.1655 | 0.5000 | 0 | 467414 | 0.0930 | 102438 | 0.2213 | 0.0951 | 0.5171 | 0 | 0.1451 |
| stratum=mono_44khz_ogg | 1 | — | — | — | 0 | 0 | — | 0 | 0.0000 | — | — | 0 | 0.0107 |
| stratum=multi_speaker | 2 | 0.2435 | 0.1724 | — | 0 | 467295 | 0.0935 | 84917 | 0.2111 | 0.0939 | 0.5158 | 0 | 0.1571 |
| stratum=music | 4 | 0.5000 | 0.2857 | 0.5000 | 0 | 27 | 0.0315 | 16789 | 0.1400 | 0.5000 | 0.5000 | 0 | 0.0383 |
| stratum=name | 1 | 0.0000 | 0.0000 | 1.0000 | 0 | 116 | 0.2755 | 55 | 0.1250 | 1.0000 | 1.0000 | 0 | 1.4878 |
| stratum=non_speech | 4 | — | — | — | 0 | 0 | — | 0 | 0.0000 | — | — | 0 | 0.0205 |
| stratum=pace_fast | 2 | 0.0403 | 0.0094 | — | 0 | — | — | — | — | — | — | 0 | 0.1765 |
| stratum=pace_normal | 2 | 0.0232 | 0.0203 | — | 0 | — | — | — | — | — | — | 0 | 0.1427 |
| stratum=pace_slow | 2 | 0.0545 | 0.0139 | — | 0 | — | — | — | — | — | — | 0 | 0.1370 |
| stratum=pcm_wav | 1 | — | — | — | 0 | 0 | — | 0 | 0.0000 | — | — | 0 | 0.0225 |
| stratum=read_speech | 8 | 0.0358 | 0.0136 | — | 0 | 92 | 0.0034 | 732 | 0.6393 | 0.7500 | 0.7500 | 0 | 0.1586 |
| stratum=saxophone | 1 | — | — | — | 0 | 0 | — | 0 | 0.0000 | — | — | 0 | 0.0343 |
| stratum=short_utterance | 4 | 0.5000 | 0.2857 | 0.5000 | 0 | 143 | 0.0833 | 16844 | 0.2808 | 0.7500 | 0.7500 | 0 | 0.8118 |
| stratum=silence | 1 | — | — | — | 0 | 0 | — | 0 | 0.0000 | — | — | 0 | 0.0225 |
| stratum=single_speaker | 6 | 0.0372 | 0.0149 | — | 0 | — | — | — | — | — | — | 0 | 0.1427 |
| stratum=speech | 14 | 0.2355 | 0.1655 | 0.5000 | 0 | 467530 | 0.0930 | 102493 | 0.2212 | 0.0961 | 0.5201 | 0 | 0.1571 |
| stratum=speech_over_music | 2 | 0.5000 | 0.2857 | 0.5000 | 0 | 27 | 0.0315 | 16789 | 0.2819 | 0.5000 | 0.5000 | 0 | 0.0891 |
| stratum=spontaneous_speech | 2 | 0.2435 | 0.1724 | — | 0 | 467295 | 0.0935 | 84917 | 0.2111 | 0.0939 | 0.5158 | 0 | 0.1571 |
| stratum=stereo_44khz_ogg | 4 | 0.5000 | 0.2857 | 0.5000 | 0 | 116 | 0.1352 | 55 | 0.0009 | 1.0000 | 1.0000 | 0 | 0.7610 |
| stratum=word | 1 | 1.0000 | 0.6667 | 0.0000 | 0 | 0 | 0.0000 | 0 | — | 1.0000 | 1.0000 | 0 | 4.8415 |

## Segment shape

These descriptive metrics expose transcript line compactness. They are not release gates.

| Case | Segments | Characters median | Characters p95 | Characters max | Duration median (ms) | Duration p95 (ms) | Duration max (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| silence-5s | 0 | — | — | — | — | — | — |
| rain-10s | 0 | — | — | — | — | — | — |
| music-jazz-sax-24s | 0 | — | — | — | — | — | — |
| music-36s | 0 | — | — | — | — | — | — |
| speech-over-music-ear | 1 | 5.0000 | 5.0000 | 5 | 500.0000 | 500.0000 | 500 |
| speech-over-music-john | 1 | 5.0000 | 5.0000 | 5 | 17120.0000 | 17120.0000 | 17120 |
| short-word-ear | 1 | 5.0000 | 5.0000 | 5 | 437.0000 | 437.0000 | 437 |
| short-name-john | 1 | 5.0000 | 5.0000 | 5 | 360.0000 | 360.0000 | 360 |
| librispeech-sample-1 | 1 | 241.0000 | 241.0000 | 241 | 13580.0000 | 13580.0000 | 13580 |
| librispeech-sample-2 | 1 | 192.0000 | 192.0000 | 192 | 13820.0000 | 13820.0000 | 13820 |
| pace-slow-tuning | 7 | 65.0000 | 93.0000 | 93 | 4640.0000 | 7060.0000 | 7060 |
| pace-normal-tuning | 7 | 93.0000 | 100.0000 | 100 | 5800.0000 | 6400.0000 | 6400 |
| pace-fast-tuning | 11 | 67.0000 | 98.0000 | 98 | 3020.0000 | 5040.0000 | 5040 |
| pace-slow-acceptance | 6 | 91.5000 | 93.0000 | 93 | 5660.0000 | 7280.0000 | 7280 |
| pace-normal-acceptance | 10 | 86.0000 | 145.0000 | 145 | 5300.0000 | 7500.0000 | 7500 |
| pace-fast-acceptance | 5 | 132.0000 | 149.0000 | 149 | 6540.0000 | 6820.0000 | 6820 |
| ami-meeting-30m | 616 | 41.0000 | 88.0000 | 188 | 2080.0000 | 6000.0000 | 11220 |
| ami-meeting-60m | 1126 | 43.0000 | 94.0000 | 169 | 2260.0000 | 7000.0000 | 14000 |

## Gates

Gates evaluate only manifest rows with `split=acceptance`.

| Gate | Rule | Actual | Status |
|---|---:|---:|---|
| `boundary_precision` | min 0.9500 | 0.1267 | DISABLED |
| `boundary_recall` | min 0.9500 | 0.5257 | DISABLED |
| `cer` | max 0.0500 | 0.1503 | DISABLED |
| `false_alarm_rate` | max 0.0500 | 0.2366 | DISABLED |
| `missed_speech_rate` | max 0.0500 | 0.0724 | DISABLED |
| `non_speech_error_cases` | max 0.0000 | 0 | PASS |
| `non_speech_nonempty_cases` | max 0.0000 | 0 | PASS |
| `parity_mismatch_cases` | max 0.0000 | 0 | DISABLED |
| `peak_cuda_mb_max` | max 16384.0000 | — | DISABLED |
| `peak_rss_mb_max` | max 16384.0000 | 3004.1562 | DISABLED |
| `rtf_median` | max 1.0000 | 0.1430 | DISABLED |
| `short_exact_match_rate` | min 1.0000 | 1.0000 | PASS |
| `timestamp_violation_cases` | max 0.0000 | 0 | PASS |
| `wer` | max 0.1000 | 0.2223 | DISABLED |

## Cases

| Case | Split | Strata | Duration (s) | Inference (s) | RTF | Peak RSS (MiB) | Peak CUDA (MiB) | WER | CER | Miss (ms) | Miss rate | False alarm (ms) | False-alarm rate | Timestamp violations | Error |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| silence-5s | acceptance | non_speech, silence, pcm_wav | 5.0000 | 0.1123 | 0.0225 | 414.2188 | — | — | — | 0 | — | 0 | 0.0000 | 0 | — |
| rain-10s | tuning | non_speech, background_noise, mono_44khz_ogg | 10.3310 | 0.1105 | 0.0107 | 427.5156 | — | — | — | 0 | — | 0 | 0.0000 | 0 | — |
| music-jazz-sax-24s | tuning | non_speech, music, saxophone, stereo_44khz_ogg | 24.0000 | 0.8221 | 0.0343 | 552.4844 | — | — | — | 0 | — | 0 | 0.0000 | 0 | — |
| music-36s | acceptance | non_speech, music, stereo_44khz_ogg | 36.4090 | 0.6781 | 0.0186 | 561.9375 | — | — | — | 0 | — | 0 | 0.0000 | 0 | — |
| speech-over-music-ear | tuning | speech, music, speech_over_music, short_utterance, mono_16khz_flac | 24.0000 | 3.2606 | 0.1359 | 7337.9531 | — | 1.0000 | 0.6667 | 27 | 0.0618 | 90 | 0.0038 | 0 | — |
| speech-over-music-john | acceptance | speech, music, speech_over_music, short_utterance, mono_16khz_flac | 36.4090 | 1.5423 | 0.0424 | 1196.1406 | — | 0.0000 | 0.0000 | 0 | 0.0000 | 16699 | 0.4640 | 0 | — |
| short-word-ear | tuning | speech, short_utterance, word, stereo_44khz_ogg | 0.4370 | 2.1157 | 4.8415 | 2058.3438 | — | 1.0000 | 0.6667 | 0 | 0.0000 | 0 | — | 0 | — |
| short-name-john | acceptance | speech, short_utterance, name, stereo_44khz_ogg | 0.8610 | 1.2810 | 1.4878 | 1127.1250 | — | 0.0000 | 0.0000 | 116 | 0.2755 | 55 | 0.1250 | 0 | — |
| librispeech-sample-1 | tuning | speech, read_speech, clean, mono_16khz_flac | 13.6900 | 2.4989 | 0.1825 | 1610.2812 | — | 0.0455 | 0.0041 | 92 | 0.0068 | 190 | 0.9135 | 0 | — |
| librispeech-sample-2 | acceptance | speech, read_speech, clean, mono_16khz_flac | 14.2150 | 2.5151 | 0.1769 | 1634.7344 | — | 0.0000 | 0.0000 | 0 | 0.0000 | 542 | 0.5784 | 0 | — |
| pace-slow-tuning | tuning | speech, read_speech, clean, single_speaker, pace_slow, mono_16khz_flac | 34.9400 | 4.7456 | 0.1358 | 1166.7031 | — | 0.0127 | 0.0049 | — | — | — | — | 0 | — |
| pace-normal-tuning | tuning | speech, read_speech, clean, single_speaker, pace_normal, mono_16khz_flac | 39.7550 | 5.6579 | 0.1423 | 1159.8125 | — | 0.0273 | 0.0265 | — | — | — | — | 0 | — |
| pace-fast-tuning | tuning | speech, read_speech, clean, single_speaker, pace_fast, mono_16khz_flac | 36.4550 | 6.5174 | 0.1788 | 1183.1562 | — | 0.0441 | 0.0088 | — | — | — | — | 0 | — |
| pace-slow-acceptance | acceptance | speech, read_speech, clean, single_speaker, pace_slow, mono_16khz_flac | 35.5900 | 4.9145 | 0.1381 | 1180.2969 | — | 0.0930 | 0.0208 | — | — | — | — | 0 | — |
| pace-normal-acceptance | acceptance | speech, read_speech, clean, single_speaker, pace_normal, mono_16khz_flac | 53.6300 | 7.6683 | 0.1430 | 1183.4219 | — | 0.0201 | 0.0155 | — | — | — | — | 0 | — |
| pace-fast-acceptance | acceptance | speech, read_speech, clean, single_speaker, pace_fast, mono_16khz_flac | 31.7750 | 5.5342 | 0.1742 | 1163.2031 | — | 0.0357 | 0.0100 | — | — | — | — | 0 | — |
| ami-meeting-30m | tuning | speech, meeting, multi_speaker, spontaneous_speech, long_form, 30_minute, mono_16khz_flac | 1800.0000 | 300.6465 | 0.1670 | 2537.3594 | — | 0.2720 | 0.2044 | 229963 | 0.1327 | 4416 | 0.0652 | 0 | — |
| ami-meeting-60m | acceptance | speech, meeting, multi_speaker, spontaneous_speech, long_form, 60_minute, mono_16khz_flac | 3600.0000 | 529.7534 | 0.1472 | 3004.1562 | — | 0.2281 | 0.1554 | 237332 | 0.0727 | 80501 | 0.2406 | 0 | — |

## Interpretation limits

- Boundary and speech-duration metrics compare final TSV intervals with the references. They are end-to-end output metrics, not direct Silero VAD measurements.
- Activity duration metrics exclude cases whose speech interval reference is null. An em dash means no interval reference was scored.
- Short exact match includes recordings whose total annotated speech is within the configured short duration, even when the surrounding recording is longer.
- WER and CER score the retained decoder spans after text normalization for comparison. The application does not rewrite their nonblank text.
- Segment-shape metrics count Unicode code points and measured output duration. They describe compactness, not semantic coherence, and have no enabled gates.
- Audio-classifier identity is declared by the protocol and is not mechanically queried from the backend. Source and evaluator metadata aid auditing but do not hash an uncommitted runtime diff.
- Pipeline settings are defined by the protocol and may differ from application defaults; interpret results only for the recorded configuration.
- CER includes spaces after NFKC, casefolding, punctuation-to-space conversion, and whitespace collapse.
- Peak RSS is sampled for this process and its children, so spikes shorter than the sampling interval may be missed. CUDA memory is PyTorch's peak allocated memory.
- The first case that loads Whisper includes cold model loading; later cases may reuse in-process model caches.
