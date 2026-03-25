from functools import cached_property

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from app.core.config import settings


class TranslationService:
    def __init__(self) -> None:
        self.src_lang = "eng_Latn"
        self.tgt_lang = "kin_Latn"

    @cached_property
    def tokenizer(self):
        return AutoTokenizer.from_pretrained(settings.translation_model_name)

    @cached_property
    def model(self):
        return AutoModelForSeq2SeqLM.from_pretrained(settings.translation_model_name)

    def preload(self) -> None:
        _ = self.tokenizer
        _ = self.model

    def is_loaded(self) -> bool:
        return "tokenizer" in self.__dict__ and "model" in self.__dict__

    def translate(self, text: str, max_length: int = 400) -> str:
        if not text or not text.strip():
            return text

        inputs = self.tokenizer(text, return_tensors="pt")
        translated_tokens = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(self.tgt_lang),
            max_length=max_length,
        )
        return self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
