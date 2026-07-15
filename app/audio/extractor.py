"""Safe FFmpeg audio extraction used by the subtitle pipeline."""

from __future__ import annotations

import subprocess
from pathlib import Path

from app.exceptions import FFmpegNotFoundError, SubtitleTranslatorError


class AudioExtractor:
    """Extract a mono 16 kHz PCM WAV file with FFmpeg."""

    def __init__(self, executable: Path) -> None:
        self.executable = executable.expanduser().resolve()

    def extract(self, input_video: Path, output_wav: Path) -> Path:
        """Extract audio and return the created WAV path."""
        if not input_video.is_file():
            raise SubtitleTranslatorError(f"Input video was not found: {input_video}")
        if not self.executable.is_file():
            raise FFmpegNotFoundError(
                f"FFmpeg was not found in the project components folder: {self.executable}"
            )
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(self.executable),
            "-y",
            "-i",
            str(input_video),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(output_wav),
        ]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as error:
            raise SubtitleTranslatorError(f"Could not start FFmpeg: {error}") from error
        if completed.returncode != 0:
            details = completed.stderr.strip() or "FFmpeg returned no diagnostic output."
            raise SubtitleTranslatorError(f"FFmpeg audio extraction failed: {details}")
        if not output_wav.is_file() or output_wav.stat().st_size == 0:
            raise SubtitleTranslatorError("FFmpeg completed but did not create a WAV file.")
        return output_wav
