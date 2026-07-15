"""Pydantic models shared by the subtitle pipeline."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SubtitleModel(BaseModel):
    """Immutable validated base model for pipeline data."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class TranscriptSegment(SubtitleModel):
    """One timestamped piece of speech returned by a transcriber."""

    id: int = Field(ge=0)
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def has_valid_interval(self) -> "TranscriptSegment":
        """Ensure that a segment has a positive time interval."""
        if self.end <= self.start:
            raise ValueError("end must be greater than start")
        return self


class TranslationBlock(SubtitleModel):
    """A semantic block built from one or more transcript segments."""

    id: int = Field(ge=0)
    segment_ids: list[int] = Field(min_length=1)
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    source_text: str = Field(min_length=1)
    translated_text: str | None = None

    @model_validator(mode="after")
    def has_valid_interval(self) -> "TranslationBlock":
        """Ensure that a block has a positive time interval."""
        if self.end <= self.start:
            raise ValueError("end must be greater than start")
        return self


class SubtitleCue(SubtitleModel):
    """A single SRT subtitle cue."""

    index: int = Field(ge=1)
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def has_valid_interval(self) -> "SubtitleCue":
        """Ensure that a cue never has a zero or negative duration."""
        if self.end <= self.start:
            raise ValueError("end must be greater than start")
        return self


class TranscriptionSettings(SubtitleModel):
    """Settings for the faster-whisper transcriber."""

    engine: Literal["faster-whisper"] = "faster-whisper"
    model: str = "medium"
    device: Literal["auto", "cpu", "cuda"] = "auto"
    compute_type: str = "auto"
    vad_filter: bool = True
    word_timestamps: bool = True
    beam_size: int = Field(default=5, ge=1)
    batch_size: int = Field(default=8, ge=1)


class TranslationSettings(SubtitleModel):
    """Settings for the local NLLB translation provider."""

    provider: Literal["nllb"] = "nllb"
    model: str = "facebook/nllb-200-distilled-600M"
    device: Literal["auto", "cpu", "cuda"] = "auto"
    batch_size: int = Field(default=8, ge=1)
    max_length: int = Field(default=256, ge=1)


class SegmentationSettings(SubtitleModel):
    """Rules for grouping short Whisper segments before translation."""

    max_pause_seconds: float = Field(default=0.8, ge=0)
    max_block_duration: float = Field(default=12.0, gt=0)
    max_block_characters: int = Field(default=300, gt=0)


class SubtitleSettings(SubtitleModel):
    """Rules for turning translated blocks into readable subtitle cues."""

    max_chars_per_line: int = Field(default=42, gt=0)
    max_lines: int = Field(default=2, gt=0)
    min_duration: float = Field(default=1.0, gt=0)
    max_duration: float = Field(default=7.0, gt=0)
    max_chars_per_second: float = Field(default=20.0, gt=0)

    @model_validator(mode="after")
    def durations_are_ordered(self) -> "SubtitleSettings":
        """Reject an impossible minimum/maximum duration combination."""
        if self.min_duration > self.max_duration:
            raise ValueError("min_duration cannot exceed max_duration")
        return self


class SubtitleTranslatorSettings(SubtitleModel):
    """Complete serializable configuration for one subtitle run."""

    transcription: TranscriptionSettings = Field(default_factory=TranscriptionSettings)
    translation: TranslationSettings = Field(default_factory=TranslationSettings)
    segmentation: SegmentationSettings = Field(default_factory=SegmentationSettings)
    subtitles: SubtitleSettings = Field(default_factory=SubtitleSettings)
