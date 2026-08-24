"""Local audio transcription with open-source ASR models."""

import logging


__version__ = "0.1.0"

# Per Python logging HOWTO: libraries must not configure handlers.
# Add NullHandler so log records are silently discarded if the application
# hasn't set up logging, while allowing the application full control.
logging.getLogger(__name__).addHandler(logging.NullHandler())
