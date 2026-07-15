"""Join short transcription segments into translation-friendly blocks."""

from __future__ import annotations

from app.models import SegmentationSettings, TranscriptSegment, TranslationBlock


SENTENCE_ENDINGS = frozenset(".!?…｡！？")


class SegmentMerger:
    """Merge adjacent segments while preserving all original segment IDs."""

    def __init__(self, settings: SegmentationSettings) -> None:
        self.settings = settings

    @staticmethod
    def _ends_sentence(text: str) -> bool:
        return bool(text.rstrip()) and text.rstrip()[-1] in SENTENCE_ENDINGS

    def merge(self, segments: list[TranscriptSegment]) -> list[TranslationBlock]:
        """Return translation blocks subject to pause, duration, and text limits."""
        if not segments:
            return []
        ordered = sorted(segments, key=lambda item: (item.start, item.id))
        blocks: list[TranslationBlock] = []
        current: list[TranscriptSegment] = []

        def flush() -> None:
            if not current:
                return
            blocks.append(
                TranslationBlock(
                    id=len(blocks) + 1,
                    segment_ids=[item.id for item in current],
                    start=current[0].start,
                    end=current[-1].end,
                    source_text=" ".join(item.text.strip() for item in current),
                )
            )
            current.clear()

        for segment in ordered:
            if not current:
                current.append(segment)
                continue
            previous = current[-1]
            current_text = " ".join(item.text.strip() for item in current)
            candidate_text = f"{current_text} {segment.text.strip()}"
            gap = segment.start - previous.end
            candidate_duration = segment.end - current[0].start
            can_merge = (
                gap <= self.settings.max_pause_seconds
                and candidate_duration <= self.settings.max_block_duration
                and len(candidate_text) <= self.settings.max_block_characters
                and not self._ends_sentence(previous.text)
            )
            if can_merge:
                current.append(segment)
            else:
                flush()
                current.append(segment)
        flush()
        return blocks
