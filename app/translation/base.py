"""Contract for translated subtitle block providers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.models import TranslationBlock


class TranslationProvider(ABC):
    """Replaceable provider that fills translated text without changing timing."""

    @abstractmethod
    def translate_blocks(
        self,
        blocks: list[TranslationBlock],
        source_language: str,
        target_language: str,
    ) -> list[TranslationBlock]:
        """Return copies of blocks with only ``translated_text`` populated."""
