from pathlib import Path

from app.models import (
    SegmentationSettings,
    SubtitleSettings,
    TranscriptSegment,
    TranslationBlock,
)
from app.pipeline import SubtitleTranslationPipeline
from app.subtitles.formatter import SubtitleFormatter
from app.subtitles.segment_merger import SegmentMerger
from app.subtitles.srt_exporter import SRTExporter
from app.transcription.base import TranscriptionProvider
from app.translation.base import TranslationProvider


class FakeExtractor:
    def extract(self, input_video: Path, output_wav: Path) -> Path:
        assert input_video.is_file()
        output_wav.write_bytes(b"not real audio")
        return output_wav


class FakeTranscriber(TranscriptionProvider):
    last_device = "cpu"

    def transcribe(
        self, audio_path: Path, source_language: str | None = None
    ) -> tuple[list[TranscriptSegment], str]:
        assert audio_path.is_file()
        assert source_language is None
        return [TranscriptSegment(id=1, start=0.0, end=3.0, text="Hello")], "en"


class FakeTranslator(TranslationProvider):
    def __init__(self) -> None:
        self.languages: tuple[str, str] | None = None

    def translate_blocks(
        self,
        blocks: list[TranslationBlock],
        source_language: str,
        target_language: str,
    ) -> list[TranslationBlock]:
        self.languages = (source_language, target_language)
        return [
            block.model_copy(update={"translated_text": "Привіт"})
            for block in blocks
        ]


def test_pipeline_uses_provider_contracts_without_loading_real_models(tmp_path: Path) -> None:
    input_video = tmp_path / "video.mp4"
    input_video.write_bytes(b"placeholder")
    translator = FakeTranslator()
    pipeline = SubtitleTranslationPipeline(
        extractor=FakeExtractor(),  # type: ignore[arg-type]
        transcriber=FakeTranscriber(),
        merger=SegmentMerger(SegmentationSettings()),
        translator=translator,
        formatter=SubtitleFormatter(SubtitleSettings()),
        exporter=SRTExporter(),
    )

    result = pipeline.run(input_video, tmp_path / "translated.srt", "uk")

    assert translator.languages == ("en", "uk")
    assert result.read_text(encoding="utf-8") == (
        "1\n00:00:00,000 --> 00:00:03,000\nПривіт\n\n"
    )
    assert pipeline.last_run is not None
    assert pipeline.last_run.transcript_segments == 1
    assert pipeline.last_run.translation_blocks == 1
