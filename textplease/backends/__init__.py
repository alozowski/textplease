"""ASR backends for textplease package."""

from .nemo import transcribe as transcribe_with_nemo
from .transformers_pipeline import transcribe as transcribe_with_transformers


__all__ = ["transcribe_with_nemo", "transcribe_with_transformers"]
