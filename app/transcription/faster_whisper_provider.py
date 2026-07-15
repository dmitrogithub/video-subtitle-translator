"""faster-whisper adapter for translated subtitle generation."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.cuda import ctranslate2_cuda_report, prepare_nvidia_cuda_dll_directories
from app.exceptions import TranscriptionError
from app.models import TranscriptSegment
from app.transcription.base import TranscriptionProvider


class FasterWhisperProvider(TranscriptionProvider):
    """Transcribe a WAV file with faster-whisper and retain segment timing."""

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        compute_type: str = "auto",
        vad_filter: bool = True,
        word_timestamps: bool = True,
        beam_size: int = 5,
        batch_size: int = 8,
        download_root: Path | None = None,
        model_loader: Callable[[], Any] | None = None,
        cuda_detector: Callable[[], bool] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.vad_filter = vad_filter
        self.word_timestamps = word_timestamps
        self.beam_size = beam_size
        self.batch_size = batch_size
        self.download_root = download_root.expanduser().resolve() if download_root else None
        self.model_loader = model_loader or self._import_module
        self.cuda_detector = cuda_detector or self._cuda_available
        self.logger = logger or logging.getLogger(__name__)
        self.last_device: str | None = None
        self.last_compute_type: str | None = None

    @staticmethod
    def _import_module() -> Any:
        prepare_nvidia_cuda_dll_directories()
        try:
            import faster_whisper  # type: ignore[import-not-found]
        except ImportError as error:
            raise TranscriptionError(
                "faster-whisper is not installed. Run: pip install -r "
                "requirements-local-models.txt"
            ) from error
        return faster_whisper

    @staticmethod
    def _cuda_available() -> bool:
        """Detect whether CTranslate2 exposes a usable CUDA execution path."""
        return bool(ctranslate2_cuda_report()["cuda_available"])

    def _execution(self, device: str | None = None) -> tuple[str, str]:
        requested = device or self.device
        if requested not in {"auto", "cpu", "cuda"}:
            raise TranscriptionError(
                "Unsupported device. Choose one of: auto, cpu, cuda."
            )
        actual_device = "cuda" if requested == "auto" and self.cuda_detector() else requested
        if actual_device == "auto":
            actual_device = "cpu"
        actual_compute = self.compute_type
        if actual_compute == "auto":
            actual_compute = "float16" if actual_device == "cuda" else "int8"
        return actual_device, actual_compute

    def _transcribe_once(
        self,
        module: Any,
        audio_path: Path,
        source_language: str | None,
        device: str,
        compute_type: str,
    ) -> tuple[list[TranscriptSegment], str]:
        model = module.WhisperModel(
            self.model_name,
            device=device,
            compute_type=compute_type,
            download_root=str(self.download_root) if self.download_root else None,
        )
        options = {
            "language": source_language,
            "beam_size": self.beam_size,
            "vad_filter": self.vad_filter,
            "word_timestamps": self.word_timestamps,
        }
        effective_batch_size = self.batch_size if device == "cuda" else 1
        if effective_batch_size > 1:
            inference = module.BatchedInferencePipeline(model=model)
            raw_segments, info = inference.transcribe(
                str(audio_path), batch_size=effective_batch_size, **options
            )
        else:
            raw_segments, info = model.transcribe(str(audio_path), **options)
        segments: list[TranscriptSegment] = []
        for raw_segment in raw_segments:
            text = str(getattr(raw_segment, "text", "")).strip()
            if not text:
                continue
            try:
                segments.append(
                    TranscriptSegment(
                        id=len(segments) + 1,
                        start=float(raw_segment.start),
                        end=float(raw_segment.end),
                        text=text,
                    )
                )
            except (TypeError, ValueError) as error:
                raise TranscriptionError(
                    "faster-whisper returned an invalid timestamped segment."
                ) from error
        if not segments:
            raise TranscriptionError("faster-whisper produced no timestamped text.")
        detected = str(getattr(info, "language", "") or source_language or "").lower()
        if not detected:
            raise TranscriptionError("faster-whisper did not report a source language.")
        return segments, detected

    def transcribe(
        self,
        audio_path: Path,
        source_language: str | None = None,
    ) -> tuple[list[TranscriptSegment], str]:
        """Transcribe audio, falling back to CPU for an automatic CUDA choice."""
        if not audio_path.is_file():
            raise TranscriptionError(f"Audio file was not found: {audio_path}")
        module = self.model_loader()
        selected_device, selected_compute = self._execution()
        self.logger.info(
            "Transcribing with faster-whisper model=%s device=%s compute_type=%s batch_size=%s",
            self.model_name,
            selected_device,
            selected_compute,
            self.batch_size if selected_device == "cuda" else 1,
        )
        try:
            result = self._transcribe_once(
                module,
                audio_path,
                source_language,
                selected_device,
                selected_compute,
            )
        except Exception as error:
            can_retry = self.device == "auto" and selected_device == "cuda"
            if not can_retry:
                if isinstance(error, TranscriptionError):
                    raise
                raise TranscriptionError(f"faster-whisper failed: {error}") from error
            fallback_device, fallback_compute = self._execution("cpu")
            self.logger.warning(
                "CUDA transcription failed; retrying automatically on CPU: %s. "
                "Run `python -m app.cli diagnose`, then install "
                "requirements-cuda.txt to restore GPU execution.",
                error,
            )
            try:
                result = self._transcribe_once(
                    module,
                    audio_path,
                    source_language,
                    fallback_device,
                    fallback_compute,
                )
                selected_device, selected_compute = fallback_device, fallback_compute
            except Exception as fallback_error:
                raise TranscriptionError(
                    "faster-whisper failed on CUDA and CPU fallback. "
                    f"CUDA error: {error}. CPU error: {fallback_error}"
                ) from fallback_error
        self.last_device = selected_device
        self.last_compute_type = selected_compute
        return result
