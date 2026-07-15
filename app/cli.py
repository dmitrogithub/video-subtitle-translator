"""Typer command line interface for local translated subtitles."""

from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import Annotated

import typer

from app.components import ComponentLocator, FFmpegInstaller, require_local_ffmpeg
from app.config import load_settings
from app.cuda import ctranslate2_cuda_report
from app.exceptions import FFmpegNotFoundError, SubtitleTranslatorError
from app.models import SubtitleTranslatorSettings
from app.pipeline import build_pipeline


app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Create translated SRT subtitles locally with faster-whisper and NLLB.",
)


@app.callback()
def subtitle_command() -> None:
    """Subtitle command group; use the ``translate`` subcommand."""


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


@app.command()
def diagnose() -> None:
    """Show FFmpeg, CUDA, and model-cache diagnostics without downloading models."""
    locator = ComponentLocator()
    ffmpeg = locator.inspect()
    cuda = ctranslate2_cuda_report()
    typer.echo(f"Components: {locator.components_dir}")
    typer.echo(f"FFmpeg: {'ready' if ffmpeg.installed else 'missing'} ({ffmpeg.ffmpeg_path})")
    typer.echo(f"Whisper cache: {locator.faster_whisper_models_dir}")
    typer.echo(f"NLLB cache: {locator.nllb_models_dir}")
    typer.echo(
        "CTranslate2 CUDA: "
        f"{'ready' if cuda['cuda_available'] else 'unavailable'} "
        f"(devices: {cuda['cuda_device_count']})"
    )
    if cuda["dll_directories"]:
        typer.echo("CUDA DLL folders:")
        for directory in cuda["dll_directories"]:
            typer.echo(f"  {directory}")
    if cuda["error"]:
        typer.echo(f"CUDA diagnostic: {cuda['error']}")


def _apply_cli_overrides(
    settings_path: Path | None,
    whisper_model: str | None,
    device: str | None,
    compute_type: str | None,
    translator: str | None,
    batch_size: int | None,
    translation_device: str | None,
) -> SubtitleTranslatorSettings:
    """Apply CLI-over-YAML-over-default precedence without mutating settings."""
    settings = load_settings(settings_path)
    transcription_updates = {
        key: value
        for key, value in {
            "model": whisper_model,
            "device": device,
            "compute_type": compute_type,
            "batch_size": batch_size,
        }.items()
        if value is not None
    }
    selected_translation_device = translation_device or device
    translation_updates = {
        key: value
        for key, value in {
            "provider": translator,
            "device": selected_translation_device,
        }.items()
        if value is not None
    }
    if transcription_updates:
        settings = settings.model_copy(
            update={
                "transcription": settings.transcription.model_copy(
                    update=transcription_updates
                )
            }
        )
    if translation_updates:
        settings = settings.model_copy(
            update={
                "translation": settings.translation.model_copy(update=translation_updates)
            }
        )
    return settings


def _ensure_ffmpeg(allow_install_prompt: bool) -> Path:
    """Always resolve FFmpeg in this project's components folder."""
    locator = ComponentLocator()
    status = locator.inspect()
    if status.installed:
        return status.ffmpeg_path
    typer.echo(f"FFmpeg is missing: {status.ffmpeg_path}", err=True)
    if not allow_install_prompt:
        return require_local_ffmpeg(locator)
    try:
        approved = typer.confirm(
            "Download and install FFmpeg into this project's components folder?"
        )
    except (EOFError, typer.Abort) as error:
        raise FFmpegNotFoundError(
            "FFmpeg is required. Run: python -m app.cli setup"
        ) from error
    if not approved:
        raise FFmpegNotFoundError(
            "FFmpeg installation was skipped. Run: python -m app.cli setup"
        )
    typer.echo("Downloading and verifying FFmpeg...")
    path = FFmpegInstaller(locator).install()
    typer.echo(f"FFmpeg installed: {path}")
    return path


@app.command()
def setup(
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Install missing FFmpeg without prompting."),
    ] = False,
) -> None:
    """Check project-local components and offer to install FFmpeg."""
    try:
        locator = ComponentLocator()
        if yes and not locator.inspect().installed:
            typer.echo("Downloading and verifying FFmpeg...")
            path = FFmpegInstaller(locator).install()
        else:
            path = _ensure_ffmpeg(allow_install_prompt=True)
        typer.echo(f"FFmpeg is ready: {path}")
    except SubtitleTranslatorError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error


@app.command()
def translate(
    input_video: Annotated[Path, typer.Argument(help="Source video file.")],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Destination .srt path."),
    ] = None,
    source_language: Annotated[
        str | None,
        typer.Option("--source-language", help="Optional source language override."),
    ] = None,
    target_language: Annotated[
        str,
        typer.Option("--target-language", help="Required target language code."),
    ] = ...,
    whisper_model: Annotated[
        str | None,
        typer.Option("--whisper-model", help="faster-whisper model name."),
    ] = None,
    device: Annotated[
        str | None,
        typer.Option("--device", help="auto, cpu, or cuda for Whisper and NLLB."),
    ] = None,
    compute_type: Annotated[
        str | None,
        typer.Option("--compute-type", help="faster-whisper compute type."),
    ] = None,
    batch_size: Annotated[
        int | None,
        typer.Option("--batch-size", help="GPU faster-whisper batch size."),
    ] = None,
    translation_device: Annotated[
        str | None,
        typer.Option(
            "--translation-device",
            help="auto, cpu, or cuda for NLLB; overrides --device for translation only.",
        ),
    ] = None,
    translator: Annotated[
        str | None,
        typer.Option("--translator", help="Translation provider (currently nllb)."),
    ] = None,
    config: Annotated[
        Path | None,
        typer.Option("--config", help="YAML configuration file."),
    ] = None,
    keep_temp: Annotated[
        bool,
        typer.Option("--keep-temp", help="Keep extracted temporary WAV files."),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show a full traceback on failure."),
    ] = False,
) -> None:
    """Translate one video into a timestamped SRT file."""
    _configure_logging(verbose)
    logger = logging.getLogger(__name__)
    try:
        if not input_video.is_file():
            raise SubtitleTranslatorError(f"Input video was not found: {input_video}")
        ffmpeg_executable = _ensure_ffmpeg(allow_install_prompt=True)
        settings = _apply_cli_overrides(
            config,
            whisper_model,
            device,
            compute_type,
            translator,
            batch_size,
            translation_device,
        )
        output_path = output or input_video.with_suffix(f".{target_language}.srt")
        pipeline = build_pipeline(
            settings,
            keep_temp=keep_temp,
            logger=logger,
            ffmpeg_executable=ffmpeg_executable,
            components_dir=ComponentLocator().components_dir,
        )
        result = pipeline.run(
            input_video=input_video,
            output_srt=output_path,
            target_language=target_language,
            source_language=source_language,
        )
        stats = pipeline.last_run
        typer.echo(f"Whisper model: {settings.transcription.model}")
        typer.echo(f"Selected device: {stats.transcription_device if stats else 'unknown'}")
        if stats is not None:
            typer.echo(f"Detected language: {stats.detected_language}")
            typer.echo(f"Transcript segments: {stats.transcript_segments}")
            typer.echo(f"Translation blocks: {stats.translation_blocks}")
            typer.echo(f"Subtitle cues: {stats.subtitle_cues}")
        typer.echo(f"Result: {result}")
    except SubtitleTranslatorError as error:
        if verbose:
            traceback.print_exc()
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error
    except Exception as error:
        if verbose:
            raise
        typer.echo(f"Unexpected error: {error}. Use --verbose for details.", err=True)
        raise typer.Exit(code=1) from error


def main() -> None:
    """Run the standalone subtitle CLI entry point."""
    app()


if __name__ == "__main__":
    main()
