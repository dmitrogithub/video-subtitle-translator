from app.cli import _apply_cli_overrides


def test_translation_device_overrides_shared_device_flag() -> None:
    settings = _apply_cli_overrides(
        None,
        whisper_model=None,
        device="cuda",
        compute_type=None,
        translator=None,
        batch_size=16,
        translation_device="cpu",
    )

    assert settings.transcription.device == "cuda"
    assert settings.transcription.batch_size == 16
    assert settings.translation.device == "cpu"
