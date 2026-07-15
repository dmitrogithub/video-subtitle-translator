from app.models import SegmentationSettings, TranscriptSegment
from app.subtitles.segment_merger import SegmentMerger


def segment(identifier: int, start: float, end: float, text: str) -> TranscriptSegment:
    return TranscriptSegment(id=identifier, start=start, end=end, text=text)


def test_merges_short_adjacent_segments_and_preserves_ids() -> None:
    blocks = SegmentMerger(SegmentationSettings()).merge(
        [segment(4, 0.0, 1.0, "Hello"), segment(9, 1.2, 2.0, "world")]
    )

    assert len(blocks) == 1
    assert blocks[0].segment_ids == [4, 9]
    assert blocks[0].source_text == "Hello world"


def test_does_not_merge_when_pause_is_too_long() -> None:
    blocks = SegmentMerger(SegmentationSettings(max_pause_seconds=0.5)).merge(
        [segment(1, 0.0, 1.0, "Hello"), segment(2, 1.6, 2.0, "again")]
    )

    assert [block.segment_ids for block in blocks] == [[1], [2]]


def test_does_not_merge_after_sentence_ending() -> None:
    blocks = SegmentMerger(SegmentationSettings()).merge(
        [segment(1, 0.0, 1.0, "A completed sentence."), segment(2, 1.1, 2.0, "Next")]
    )

    assert [block.segment_ids for block in blocks] == [[1], [2]]


def test_respects_maximum_block_duration() -> None:
    blocks = SegmentMerger(SegmentationSettings(max_block_duration=2.0)).merge(
        [segment(1, 0.0, 1.0, "One"), segment(2, 1.1, 2.2, "two")]
    )

    assert [block.segment_ids for block in blocks] == [[1], [2]]
