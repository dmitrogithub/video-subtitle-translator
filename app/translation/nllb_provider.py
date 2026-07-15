"""Local NLLB translation provider backed by Hugging Face Transformers."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.exceptions import TranslationError, UnsupportedLanguageError
from app.models import TranslationBlock
from app.translation.base import TranslationProvider


LANGUAGE_CODES = {
    "en": "eng_Latn",
    "ru": "rus_Cyrl",
    "uk": "ukr_Cyrl",
    "de": "deu_Latn",
    "fi": "fin_Latn",
    "fr": "fra_Latn",
    "es": "spa_Latn",
    "pl": "pol_Latn",
}


class NLLBTranslationProvider(TranslationProvider):
    """Translate batches locally with ``facebook/nllb-200-distilled-600M``."""

    def __init__(
        self,
        model_name: str = "facebook/nllb-200-distilled-600M",
        device: str = "auto",
        batch_size: int = 8,
        max_length: int = 512,
        cache_dir: Path | None = None,
        module_loader: Callable[[], tuple[Any, Any, Any]] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.cache_dir = cache_dir.expanduser().resolve() if cache_dir else None
        self.module_loader = module_loader or self._import_modules
        self.logger = logger or logging.getLogger(__name__)
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self.last_device: str | None = None

    @staticmethod
    def _import_modules() -> tuple[Any, Any, Any]:
        try:
            import torch  # type: ignore[import-not-found]
            from transformers import (  # type: ignore[import-not-found]
                AutoModelForSeq2SeqLM,
                AutoTokenizer,
            )
        except ImportError as error:
            raise TranslationError(
                "NLLB dependencies are not installed. Run: pip install -r "
                "requirements-local-models.txt"
            ) from error
        return torch, AutoTokenizer, AutoModelForSeq2SeqLM

    @staticmethod
    def _language_code(language: str) -> str:
        normalized = language.lower().replace("_", "-").split("-", 1)[0]
        try:
            return LANGUAGE_CODES[normalized]
        except KeyError as error:
            available = ", ".join(sorted(LANGUAGE_CODES))
            raise UnsupportedLanguageError(
                f"Unsupported language '{language}'. Available languages: {available}."
            ) from error

    def _select_device(self, torch: Any) -> str:
        if self.device not in {"auto", "cpu", "cuda"}:
            raise TranslationError("Unsupported device. Choose one of: auto, cpu, cuda.")
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda" and not torch.cuda.is_available():
            raise TranslationError("CUDA was requested for NLLB but is not available.")
        return self.device

    def _load(self) -> tuple[Any, Any, Any, str]:
        torch, tokenizer_factory, model_factory = self.module_loader()
        selected_device = self._select_device(torch)
        if self._tokenizer is None or self._model is None:
            self.logger.info(
                "Loading NLLB model=%s device=%s", self.model_name, selected_device
            )
            cache_dir = str(self.cache_dir) if self.cache_dir else None
            self._tokenizer = tokenizer_factory.from_pretrained(
                self.model_name, cache_dir=cache_dir, use_safetensors=True
            )
            self._model = model_factory.from_pretrained(
                self.model_name, cache_dir=cache_dir, use_safetensors=True
            )
            self._model.to(selected_device)
            self._model.eval()
        self.last_device = selected_device
        return torch, self._tokenizer, self._model, selected_device

    @staticmethod
    def _target_token_id(tokenizer: Any, target_code: str) -> int:
        mapping = getattr(tokenizer, "lang_code_to_id", None)
        target_id = mapping.get(target_code) if isinstance(mapping, dict) else None
        if target_id is None:
            target_id = tokenizer.convert_tokens_to_ids(target_code)
        if target_id is None or target_id == getattr(tokenizer, "unk_token_id", None):
            raise TranslationError(
                f"NLLB tokenizer does not support language code '{target_code}'."
            )
        return int(target_id)

    def translate_blocks(
        self,
        blocks: list[TranslationBlock],
        source_language: str,
        target_language: str,
    ) -> list[TranslationBlock]:
        """Translate in batches and preserve every non-text block field exactly."""
        if not blocks:
            return []
        source_code = self._language_code(source_language)
        target_code = self._language_code(target_language)
        torch, tokenizer, model, selected_device = self._load()
        tokenizer.src_lang = source_code
        target_token_id = self._target_token_id(tokenizer, target_code)
        translated: list[TranslationBlock] = []
        try:
            total_batches = (len(blocks) + self.batch_size - 1) // self.batch_size
            for offset in range(0, len(blocks), self.batch_size):
                batch_number = offset // self.batch_size + 1
                batch = blocks[offset : offset + self.batch_size]
                self.logger.info(
                    "Translating NLLB batch %d/%d (%d blocks).",
                    batch_number,
                    total_batches,
                    len(batch),
                )
                encoded = tokenizer(
                    [block.source_text for block in batch],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                ).to(selected_device)
                with torch.no_grad():
                    generated = model.generate(
                        **encoded,
                        forced_bos_token_id=target_token_id,
                        max_length=self.max_length,
                    )
                texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
                if len(texts) != len(batch):
                    raise TranslationError("NLLB returned an incomplete translation batch.")
                translated.extend(
                    block.model_copy(update={"translated_text": text.strip()})
                    for block, text in zip(batch, texts, strict=True)
                )
        except TranslationError:
            raise
        except Exception as error:
            raise TranslationError(f"NLLB translation failed: {error}") from error
        return translated
