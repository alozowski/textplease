import gc
import logging
from typing import Any, Dict, List, Optional

import torch
import torchaudio
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline

from textplease.utils.time_utils import format_time


logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
STRIDE_LENGTH = (2, 2)  # in seconds


def transcribe(
    audio_path: str,
    waveform: Optional[torch.Tensor],
    model_name: str,
    device: str,
    chunk_duration_minutes: int = 10,
    pause_threshold: float = 2.0,  # Unused in this backend but kept for API parity
) -> List[Dict[str, Any]]:
    """Transcribe audio using Hugging Face Transformers pipeline with word-level timestamps.

    Args:
        audio_path: Path to the input audio file (.wav format).
        waveform: Optional pre-loaded waveform tensor; if None, the file is loaded via torchaudio.
        model_name: Model identifier on Hugging Face Hub.
        device: Compute device: 'cpu', 'cuda', or 'mps'.
        chunk_duration_minutes: Duration of each processing chunk.
        pause_threshold: Not used; kept for interface consistency.

    Returns:
        A list of segments, each with 'start_time', 'end_time', and 'text'.
    """
    logger.info(f"Loading Transformers model '{model_name}' on device: {device}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name, torch_dtype=torch.float16 if device != "cpu" else torch.float32, low_cpu_mem_usage=True
        ).to(device)

        pipeline_device = 0 if device == "cuda" else -1
        asr_pipe = pipeline(
            task="automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=pipeline_device,
            return_timestamps="word",
        )

        if waveform is None:
            waveform, sample_rate = torchaudio.load(audio_path)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                logger.info("Converted stereo audio to mono")
            if sample_rate != SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)
                waveform = resampler(waveform)
                logger.info(f"Resampled audio from {sample_rate}Hz to {SAMPLE_RATE}Hz")

        chunk_length_s = chunk_duration_minutes * 60
        logger.info(f"Using chunk length: {chunk_length_s}s")

        audio_array = waveform.squeeze(0).numpy()

        result = asr_pipe(
            audio_array, chunk_length_s=chunk_length_s, stride_length_s=STRIDE_LENGTH, return_timestamps="word"
        )

        segments = []
        chunks = result.get("chunks", [])

        if not chunks:
            text = result.get("text", "").strip()
            if text:
                duration_seconds = len(audio_array) / SAMPLE_RATE
                segments.append({
                    "start_time": format_time(0),
                    "end_time": format_time(duration_seconds),
                    "text": text,
                })
        else:
            for chunk in chunks:
                timestamp = chunk.get("timestamp", [0, 0])
                if len(timestamp) >= 2 and chunk.get("text", "").strip():
                    segments.append({
                        "start_time": format_time(timestamp[0]),
                        "end_time": format_time(timestamp[1]),
                        "text": chunk["text"].strip(),
                    })

        logger.info(f"Generated {len(segments)} segments from transformers pipeline")
        return segments

    except Exception as e:
        logger.error(f"Transformers pipeline transcription failed for model '{model_name}': {e}")
        raise

    finally:
        for obj_name in ["model", "processor", "asr_pipe"]:
            if obj_name in locals():
                del locals()[obj_name]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
