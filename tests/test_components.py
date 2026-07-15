from pathlib import Path

from app.components import ComponentLocator
from app.models import SubtitleTranslatorSettings
from app.pipeline import build_pipeline


def test_locator_checks_only_the_project_components_folder(tmp_path: Path) -> None:
    locator = ComponentLocator(tmp_path / "components")

    assert locator.inspect().installed is False

    locator.components_dir.mkdir()
    locator.ffmpeg_path.write_bytes(b"local component")

    status = locator.inspect()
    assert status.installed is True
    assert status.ffmpeg_path.parent == tmp_path / "components"


def test_pipeline_keeps_model_caches_under_components(tmp_path: Path) -> None:
    components_dir = tmp_path / "components"
    locator = ComponentLocator(components_dir)
    pipeline = build_pipeline(
        SubtitleTranslatorSettings(),
        components_dir=components_dir,
        ffmpeg_executable=locator.ffmpeg_path,
    )

    assert pipeline.transcriber.download_root == locator.faster_whisper_models_dir
    assert pipeline.translator.cache_dir == locator.nllb_models_dir
