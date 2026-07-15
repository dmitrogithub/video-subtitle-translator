"""Contract for audio-to-transcript providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from app.models import TranscriptSegment


class TranscriptionProvider(ABC):
    """Replaceable provider that returns timestamped speech segments."""

    @abstractmethod
    def transcribe(
        self,
        audio_path: Path,
        source_language: str | None = None,
    ) -> tuple[list[TranscriptSegment], str]:
        """Return transcript segments and the detected source language code."""
