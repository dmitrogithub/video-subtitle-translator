from pathlib import Path
from types import SimpleNamespace

from app.transcription.faster_whisper_provider import FasterWhisperProvider


class FakeModel:
    init_args: dict[str, object] | None = None

    def __init__(self, model_name: str, **kwargs: object) -> None:
        self.model_name = model_name
        self.init_args = kwargs
        FakeModel.init_args = kwargs

    def transcribe(self, audio_path: str, **kwargs: object):
        del audio_path, kwargs
        return [SimpleNamespace(start=0.0, end=1.0, text="CPU")], SimpleNamespace(language="en")


class FakeBatchPipeline:
    batch_size: int | None = None

    def __init__(self, model: FakeModel) -> None:
        self.model = model

    def transcribe(self, audio_path: str, batch_size: int, **kwargs: object):
        del audio_path, kwargs
        FakeBatchPipeline.batch_size = batch_size
        return [SimpleNamespace(start=0.0, end=1.0, text="GPU")], SimpleNamespace(language="en")


class FakeModule:
    WhisperModel = FakeModel
    BatchedInferencePipeline = FakeBatchPipeline


def test_cuda_uses_batched_inference_and_project_model_cache(tmp_path: Path) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"audio")
    model_root = tmp_path / "components" / "models"
    provider = FasterWhisperProvider(
        "small",
        device="cuda",
        compute_type="float16",
        batch_size=16,
        download_root=model_root,
        model_loader=lambda: FakeModule,
        cuda_detector=lambda: True,
    )

    segments, language = provider.transcribe(audio_path)

    assert language == "en"
    assert segments[0].text == "GPU"
    assert FakeBatchPipeline.batch_size == 16
    assert FakeModel.init_args is not None
    assert FakeModel.init_args["download_root"] == str(model_root.resolve())
