"""Domain-specific exceptions for subtitle generation."""


class SubtitleTranslatorError(Exception):
    """Base error shown to CLI users without an implementation traceback."""


class ConfigurationError(SubtitleTranslatorError):
    """Raised when a subtitle configuration is invalid."""


class FFmpegNotFoundError(SubtitleTranslatorError):
    """Raised when the FFmpeg executable cannot be located."""


class TranscriptionError(SubtitleTranslatorError):
    """Raised when audio transcription cannot be completed."""


class TranslationError(SubtitleTranslatorError):
    """Raised when a translation provider cannot translate subtitle blocks."""


class UnsupportedLanguageError(SubtitleTranslatorError):
    """Raised when a requested language has no configured provider mapping."""
