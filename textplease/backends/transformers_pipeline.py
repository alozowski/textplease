import gc
import logging
import re
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
    pause_threshold: float = 2.0,
    batch_size: int = 1,
) -> List[Dict[str, Any]]:
    logger.info(f"Loading Transformers model '{model_name}' on device: {device}")
    logger.warning(f"Ignoring batch_size={batch_size} in transformers pipeline")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = None
    processor = None
    asr_pipe = None

    try:
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            low_cpu_mem_usage=True
        ).to(device)

        if waveform is None:
            waveform, _ = torchaudio.load(audio_path)

        audio_array = waveform.squeeze(0).numpy()
        
        # Try to get word-level timestamps first for better segmentation
        segments = _try_word_level_segmentation(model, processor, audio_array, device)
        
        # If word-level fails, try chunk-level with smart splitting
        if not segments:
            logger.warning("Word-level timestamps failed, trying chunk-level with smart splitting")
            segments = _try_chunk_level_segmentation(model, processor, audio_array, device)
        
        # Final fallback: basic transcription with sentence splitting
        if not segments:
            logger.warning("All timestamp methods failed, using sentence-based splitting")
            segments = _fallback_sentence_segmentation(model, processor, audio_array, device)

        logger.info(f"Generated {len(segments)} segments from transformers pipeline")
        return segments

    except Exception as e:
        logger.error(f"Transformers pipeline transcription failed for model '{model_name}': {e}")
        raise

    finally:
        # Clean up resources
        if asr_pipe is not None:
            del asr_pipe
        if model is not None:
            del model
        if processor is not None:
            del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _try_word_level_segmentation(model, processor, audio_array, device):
    """Try word-level timestamps and intelligently group into sentence-like segments"""
    try:
        pipeline_device = 0 if device == "cuda" else -1
        asr_pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=pipeline_device,
            return_timestamps="word",
            chunk_length_s=30,
            batch_size=1,
        )

        logger.info("Attempting word-level timestamp segmentation")
        result = asr_pipe(audio_array)
        
        chunks = result.get("chunks", [])
        if not chunks:
            return []

        logger.debug(f"Received {len(chunks)} word-level chunks")
        
        # Group words into natural segments
        segments = _group_words_into_segments(chunks)
        return segments

    except RuntimeError as e:
        if "expanded size of the tensor" in str(e) or "must match the existing size" in str(e):
            logger.warning(f"Word-level timestamp extraction failed due to tensor shape mismatch: {e}")
            return []
        else:
            raise
    except Exception as e:
        logger.warning(f"Word-level timestamps failed: {e}")
        return []


def _group_words_into_segments(word_chunks, max_segment_duration=12.0, min_segment_duration=3.0):
    """Group word-level chunks into natural sentence-like segments"""
    if not word_chunks:
        return []
    
    segments = []
    current_segment_words = []
    current_start_time = None
    current_end_time = None
    
    for chunk in word_chunks:
        timestamp = chunk.get("timestamp", [0, 0])
        if len(timestamp) != 2:
            continue
            
        start, end = timestamp
        text = chunk.get("text", "").strip()
        
        if not text:
            continue
            
        # Initialize first segment
        if current_start_time is None:
            current_start_time = start
            
        current_end_time = end
        current_segment_words.append(text)
        
        # Check if we should end the current segment
        should_end_segment = False
        current_duration = current_end_time - current_start_time
        
        # End on punctuation if we have reasonable duration
        if current_duration >= min_segment_duration and _ends_with_sentence_boundary(text):
            should_end_segment = True
        
        # Force end if segment is getting too long
        elif current_duration >= max_segment_duration:
            should_end_segment = True
        
        # End if there's a significant pause to the next word
        elif len(word_chunks) > word_chunks.index(chunk) + 1:
            next_chunk = word_chunks[word_chunks.index(chunk) + 1]
            next_timestamp = next_chunk.get("timestamp", [0, 0])
            if len(next_timestamp) == 2:
                pause_duration = next_timestamp[0] - end
                if pause_duration > 1.5 and current_duration >= min_segment_duration:  # 1.5 second pause
                    should_end_segment = True
        
        if should_end_segment and current_segment_words:
            segment_text = " ".join(current_segment_words).strip()
            if segment_text:
                segments.append({
                    "start_time": format_time(current_start_time),
                    "end_time": format_time(current_end_time),
                    "text": segment_text,
                })
            
            # Reset for next segment
            current_segment_words = []
            current_start_time = None
            current_end_time = None
    
    # Add final segment if there are remaining words
    if current_segment_words and current_start_time is not None:
        segment_text = " ".join(current_segment_words).strip()
        if segment_text:
            segments.append({
                "start_time": format_time(current_start_time),
                "end_time": format_time(current_end_time),
                "text": segment_text,
            })
    
    return segments


def _ends_with_sentence_boundary(text):
    """Check if text ends with sentence boundary punctuation"""
    return bool(re.search(r'[.!?,;:]$', text.strip()))


def _try_chunk_level_segmentation(model, processor, audio_array, device):
    """Try chunk-level timestamps and split intelligently"""
    try:
        pipeline_device = 0 if device == "cuda" else -1
        asr_pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=pipeline_device,
            return_timestamps=True,
            chunk_length_s=15,  # Shorter chunks for better segmentation
            stride_length_s=3,
            batch_size=1,
        )

        logger.info("Attempting chunk-level timestamp segmentation")
        result = asr_pipe(audio_array)
        
        chunks = result.get("chunks", [])
        if not chunks:
            return []

        segments = []
        for chunk in chunks:
            timestamp = chunk.get("timestamp", [0, 0])
            if len(timestamp) != 2:
                continue
                
            start, end = timestamp
            text = chunk.get("text", "").strip()
            
            if not text:
                continue
            
            # Split long chunks by sentences
            sub_segments = _split_chunk_by_sentences(text, start, end)
            segments.extend(sub_segments)

        return segments

    except Exception as e:
        logger.warning(f"Chunk-level timestamps failed: {e}")
        return []


def _split_chunk_by_sentences(text, start_time, end_time):
    """Split a text chunk into sentences with estimated timestamps"""
    # Split by sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) <= 1:
        return [{
            "start_time": format_time(start_time),
            "end_time": format_time(end_time),
            "text": text,
        }]
    
    segments = []
    duration = end_time - start_time
    
    # Estimate time per character for proportional timing
    total_chars = sum(len(s) for s in sentences)
    
    current_time = start_time
    for sentence in sentences:
        # Proportional timing based on character count
        sentence_duration = (len(sentence) / total_chars) * duration
        sentence_end_time = current_time + sentence_duration
        
        segments.append({
            "start_time": format_time(current_time),
            "end_time": format_time(sentence_end_time),
            "text": sentence,
        })
        
        current_time = sentence_end_time
    
    return segments


def _fallback_sentence_segmentation(model, processor, audio_array, device):
    """Fallback: basic transcription split by sentences with estimated timing"""
    try:
        pipeline_device = 0 if device == "cuda" else -1
        asr_pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=pipeline_device,
            return_timestamps=False,
            batch_size=1,
        )

        logger.info("Using fallback sentence-based segmentation")
        result = asr_pipe(audio_array)
        
        text = result.get("text", "").strip()
        if not text:
            return []
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        duration_seconds = len(audio_array) / SAMPLE_RATE
        
        # Estimate timing based on character count
        total_chars = sum(len(s) for s in sentences)
        
        segments = []
        current_time = 0.0
        
        for sentence in sentences:
            # Proportional timing
            sentence_duration = (len(sentence) / total_chars) * duration_seconds
            end_time = current_time + sentence_duration
            
            segments.append({
                "start_time": format_time(current_time),
                "end_time": format_time(end_time),
                "text": sentence,
            })
            
            current_time = end_time
        
        return segments

    except Exception as e:
        logger.error(f"Fallback sentence segmentation failed: {e}")
        return []