"""Transcription provider interfaces and implementations."""

from app.transcription.base import TranscriptionProvider
from app.transcription.faster_whisper_provider import FasterWhisperProvider

__all__ = ["FasterWhisperProvider", "TranscriptionProvider"]
