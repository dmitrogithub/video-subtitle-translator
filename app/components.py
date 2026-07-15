"""Project-local native component discovery and installation."""

from __future__ import annotations

import hashlib
import shutil
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

from app.exceptions import FFmpegNotFoundError, SubtitleTranslatorError


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_COMPONENTS_DIR = PROJECT_DIR / "components"
FFMPEG_ARCHIVE_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
FFMPEG_CHECKSUM_URL = f"{FFMPEG_ARCHIVE_URL}.sha256"


@dataclass(frozen=True)
class ComponentStatus:
    """Availability of the project-local FFmpeg binary."""

    components_dir: Path
    ffmpeg_path: Path
    installed: bool


class ComponentLocator:
    """Resolve native binaries only from this project's ``components`` folder."""

    def __init__(self, components_dir: Path = DEFAULT_COMPONENTS_DIR) -> None:
        self.components_dir = components_dir.expanduser().resolve()

    @property
    def ffmpeg_path(self) -> Path:
        """Return the expected project-local FFmpeg executable path."""
        executable = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
        return self.components_dir / executable

    @property
    def faster_whisper_models_dir(self) -> Path:
        """Return the project-local cache directory for Whisper models."""
        return self.components_dir / "models" / "faster-whisper"

    @property
    def nllb_models_dir(self) -> Path:
        """Return the project-local cache directory for NLLB models."""
        return self.components_dir / "models" / "nllb"

    def inspect(self) -> ComponentStatus:
        """Return the status without checking or using the system PATH."""
        return ComponentStatus(
            components_dir=self.components_dir,
            ffmpeg_path=self.ffmpeg_path,
            installed=self.ffmpeg_path.is_file(),
        )


class FFmpegInstaller:
    """Download the official Windows essentials archive into ``components``."""

    def __init__(
        self,
        locator: ComponentLocator,
        archive_url: str = FFMPEG_ARCHIVE_URL,
        checksum_url: str = FFMPEG_CHECKSUM_URL,
    ) -> None:
        self.locator = locator
        self.archive_url = archive_url
        self.checksum_url = checksum_url

    @staticmethod
    def _request(url: str):
        return urlopen(Request(url, headers={"User-Agent": "video-subtitle-translator"}))

    def _download(self, url: str, destination: Path) -> None:
        try:
            with self._request(url) as response, destination.open("wb") as output:
                shutil.copyfileobj(response, output)
        except OSError as error:
            raise SubtitleTranslatorError(f"Could not download FFmpeg: {error}") from error

    def _expected_checksum(self) -> str:
        try:
            with self._request(self.checksum_url) as response:
                value = response.read().decode("utf-8").strip().split()[0]
        except (OSError, UnicodeDecodeError, IndexError) as error:
            raise SubtitleTranslatorError(
                f"Could not retrieve the FFmpeg checksum: {error}"
            ) from error
        if len(value) != 64 or any(character not in "0123456789abcdefABCDEF" for character in value):
            raise SubtitleTranslatorError("The downloaded FFmpeg checksum is invalid.")
        return value.lower()

    @staticmethod
    def _sha256(file_path: Path) -> str:
        digest = hashlib.sha256()
        with file_path.open("rb") as archive:
            for chunk in iter(lambda: archive.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _archive_member(archive: zipfile.ZipFile, filename: str) -> zipfile.ZipInfo:
        suffix = f"/bin/{filename}".lower()
        for member in archive.infolist():
            if member.is_dir():
                continue
            if member.filename.replace("\\", "/").lower().endswith(suffix):
                return member
        raise SubtitleTranslatorError(f"The FFmpeg archive does not contain {filename}.")

    def install(self) -> Path:
        """Download, verify, and install FFmpeg without using a shell command."""
        if sys.platform != "win32":
            raise SubtitleTranslatorError(
                "Automatic FFmpeg installation is currently supported on Windows only."
            )
        self.locator.components_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="subtitle-ffmpeg-") as temporary_directory:
            archive_path = Path(temporary_directory) / "ffmpeg.zip"
            self._download(self.archive_url, archive_path)
            expected = self._expected_checksum()
            actual = self._sha256(archive_path)
            if actual != expected:
                raise SubtitleTranslatorError(
                    "FFmpeg archive checksum verification failed; nothing was installed."
                )
            try:
                with zipfile.ZipFile(archive_path) as archive:
                    for executable in ("ffmpeg.exe", "ffprobe.exe"):
                        member = self._archive_member(archive, executable)
                        destination = self.locator.components_dir / executable
                        temporary_output = destination.with_suffix(destination.suffix + ".tmp")
                        with archive.open(member) as source, temporary_output.open("wb") as output:
                            shutil.copyfileobj(source, output)
                        temporary_output.replace(destination)
            except zipfile.BadZipFile as error:
                raise SubtitleTranslatorError("Downloaded FFmpeg archive is not a valid ZIP file.") from error
        if not self.locator.ffmpeg_path.is_file():
            raise SubtitleTranslatorError("FFmpeg installation completed without creating ffmpeg.exe.")
        return self.locator.ffmpeg_path


def require_local_ffmpeg(locator: ComponentLocator | None = None) -> Path:
    """Return local FFmpeg or raise a clear error with the setup command."""
    active_locator = locator or ComponentLocator()
    status = active_locator.inspect()
    if not status.installed:
        raise FFmpegNotFoundError(
            f"FFmpeg was not found in {status.components_dir}. Run: "
            "python -m app.cli setup"
        )
    return status.ffmpeg_path
