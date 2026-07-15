from app.models import SubtitleSettings, TranslationBlock
from app.subtitles.formatter import SubtitleFormatter


def block(text: str, start: float = 0.0, end: float = 10.0) -> TranslationBlock:
    return TranslationBlock(
        id=1,
        segment_ids=[1],
        start=start,
        end=end,
        source_text="source",
        translated_text=text,
    )


def test_wraps_text_to_two_lines_without_breaking_words() -> None:
    formatter = SubtitleFormatter(SubtitleSettings(max_chars_per_line=12))
    cues = formatter.format_blocks([block("one two three four", end=5.0)])

    assert len(cues) == 1
    assert cues[0].text == "one two\nthree four"
    assert all(len(line) <= 12 for line in cues[0].text.splitlines())


def test_splits_long_blocks_and_distributes_source_time() -> None:
    formatter = SubtitleFormatter(
        SubtitleSettings(max_chars_per_line=12, max_duration=3.0, min_duration=1.0)
    )
    cues = formatter.format_blocks(
        [block("one two three four five six seven eight nine ten", 2.0, 11.0)]
    )

    assert len(cues) >= 3
    assert cues[0].start == 2.0
    assert cues[-1].end == 11.0
    assert all(cue.end > cue.start for cue in cues)
    assert all(left.end <= right.start for left, right in zip(cues, cues[1:]))


def test_skips_empty_translation_without_creating_invalid_cue() -> None:
    formatter = SubtitleFormatter(SubtitleSettings())
    empty = block("", 0.0, 2.0).model_copy(update={"translated_text": ""})

    assert formatter.format_blocks([empty]) == []
