from types import SimpleNamespace

from app.models import TranslationBlock
from app.translation.nllb_provider import NLLBTranslationProvider


class FakeTorch:
    class cuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class no_grad:
        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            del args


class FakeBatch(dict):
    def to(self, device: str) -> "FakeBatch":
        assert device == "cpu"
        return self


class FakeTokenizer:
    lang_code_to_id = {"rus_Cyrl": 1, "ukr_Cyrl": 2}

    def __init__(self) -> None:
        self.src_lang: str | None = None

    def __call__(self, texts: list[str], **kwargs: object) -> FakeBatch:
        del kwargs
        return FakeBatch(texts=texts)

    def batch_decode(self, generated: object, **kwargs: object) -> list[str]:
        del generated, kwargs
        return ["переклад"]


class FakeModel:
    generation_kwargs: dict[str, object] | None = None

    def to(self, device: str) -> None:
        assert device == "cpu"

    def eval(self) -> None:
        pass

    def generate(self, **kwargs: object) -> list[int]:
        FakeModel.generation_kwargs = kwargs
        return [1]


def test_nllb_uses_max_length_without_conflicting_generation_options() -> None:
    tokenizer = FakeTokenizer()
    model = FakeModel()
    provider = NLLBTranslationProvider(
        device="cpu",
        max_length=256,
        module_loader=lambda: (
            FakeTorch,
            SimpleNamespace(from_pretrained=lambda *args, **kwargs: tokenizer),
            SimpleNamespace(from_pretrained=lambda *args, **kwargs: model),
        ),
    )
    block = TranslationBlock(
        id=1,
        segment_ids=[1],
        start=0.0,
        end=1.0,
        source_text="текст",
    )

    translated = provider.translate_blocks([block], "ru", "uk")

    assert translated[0].translated_text == "переклад"
    assert FakeModel.generation_kwargs is not None
    assert FakeModel.generation_kwargs["max_length"] == 256
    assert "max_new_tokens" not in FakeModel.generation_kwargs
