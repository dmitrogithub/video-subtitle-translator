"""End-to-end local translated-subtitle pipeline."""

from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from app.audio.extractor import AudioExtractor
from app.components import ComponentLocator
from app.exceptions import ConfigurationError, SubtitleTranslatorError
from app.models import SubtitleTranslatorSettings
from app.subtitles.formatter import SubtitleFormatter
from app.subtitles.segment_merger import SegmentMerger
from app.subtitles.srt_exporter import SRTExporter
from app.transcription.base import TranscriptionProvider
from app.transcription.faster_whisper_provider import FasterWhisperProvider
from app.translation.base import TranslationProvider
from app.translation.nllb_provider import NLLBTranslationProvider


@dataclass(frozen=True)
class SubtitleRunStatistics:
    """Useful run metadata displayed by the CLI or a future GUI."""

    detected_language: str
    transcript_segments: int
    translation_blocks: int
    subtitle_cues: int
    transcription_device: str | None


class SubtitleTranslationPipeline:
    """Coordinate extraction, transcription, translation, formatting, and SRT export."""

    def __init__(
        self,
        extractor: AudioExtractor,
        transcriber: TranscriptionProvider,
        merger: SegmentMerger,
        translator: TranslationProvider,
        formatter: SubtitleFormatter,
        exporter: SRTExporter,
        keep_temp: bool = False,
        logger: logging.Logger | None = None,
    ) -> None:
        self.extractor = extractor
        self.transcriber = transcriber
        self.merger = merger
        self.translator = translator
        self.formatter = formatter
        self.exporter = exporter
        self.keep_temp = keep_temp
        self.logger = logger or logging.getLogger(__name__)
        self.last_run: SubtitleRunStatistics | None = None

    def run(
        self,
        input_video: Path,
        output_srt: Path,
        target_language: str,
        source_language: str | None = None,
    ) -> Path:
        """Generate a translated SRT file and return its resolved path."""
        input_video = input_video.expanduser().resolve()
        output_srt = output_srt.expanduser().resolve()
        if not input_video.is_file():
            raise SubtitleTranslatorError(f"Input video was not found: {input_video}")
        if not target_language.strip():
            raise ConfigurationError("target_language is required.")
        requested_source = source_language if source_language not in {None, "", "auto"} else None
        temporary_root = Path(tempfile.mkdtemp(prefix="video-subtitles-"))
        try:
            audio_path = temporary_root / "audio.wav"
            self.logger.info("Extracting mono 16 kHz WAV audio.")
            self.extractor.extract(input_video, audio_path)
            self.logger.info("Transcribing audio.")
            segments, detected_language = self.transcriber.transcribe(
                audio_path,
                source_language=requested_source,
            )
            actual_source_language = requested_source or detected_language
            self.logger.info("Merging %d transcript segments.", len(segments))
            blocks = self.merger.merge(segments)
            self.logger.info(
                "Translating %d blocks from %s to %s.",
                len(blocks),
                actual_source_language,
                target_language,
            )
            translated = self.translator.translate_blocks(
                blocks,
                actual_source_language,
                target_language,
            )
            cues = self.formatter.format_blocks(translated)
            result = self.exporter.export(cues, output_srt)
            self.last_run = SubtitleRunStatistics(
                detected_language=detected_language,
                transcript_segments=len(segments),
                translation_blocks=len(blocks),
                subtitle_cues=len(cues),
                transcription_device=getattr(self.transcriber, "last_device", None),
            )
            self.logger.info("Created %d SRT cues: %s", len(cues), result)
            return result
        finally:
            if self.keep_temp:
                self.logger.info("Temporary files were kept at: %s", temporary_root)
            else:
                shutil.rmtree(temporary_root, ignore_errors=True)


def build_pipeline(
    settings: SubtitleTranslatorSettings,
    keep_temp: bool = False,
    logger: logging.Logger | None = None,
    ffmpeg_executable: Path | None = None,
    components_dir: Path | None = None,
) -> SubtitleTranslationPipeline:
    """Create the default local faster-whisper and NLLB pipeline."""
    logger = logger or logging.getLogger(__name__)
    locator = ComponentLocator(components_dir) if components_dir else ComponentLocator()
    if settings.transcription.engine != "faster-whisper":
        raise ConfigurationError(
            f"Unsupported transcription engine: {settings.transcription.engine}"
        )
    if settings.translation.provider != "nllb":
        raise ConfigurationError(
            f"Unsupported translation provider: {settings.translation.provider}"
        )
    return SubtitleTranslationPipeline(
        extractor=AudioExtractor(ffmpeg_executable or locator.ffmpeg_path),
        transcriber=FasterWhisperProvider(
            model_name=settings.transcription.model,
            device=settings.transcription.device,
            compute_type=settings.transcription.compute_type,
            vad_filter=settings.transcription.vad_filter,
            word_timestamps=settings.transcription.word_timestamps,
            beam_size=settings.transcription.beam_size,
            batch_size=settings.transcription.batch_size,
            download_root=locator.faster_whisper_models_dir,
            logger=logger,
        ),
        merger=SegmentMerger(settings.segmentation),
        translator=NLLBTranslationProvider(
            model_name=settings.translation.model,
            device=settings.translation.device,
            batch_size=settings.translation.batch_size,
            max_length=settings.translation.max_length,
            cache_dir=locator.nllb_models_dir,
            logger=logger,
        ),
        formatter=SubtitleFormatter(settings.subtitles),
        exporter=SRTExporter(),
        keep_temp=keep_temp,
        logger=logger,
    )
