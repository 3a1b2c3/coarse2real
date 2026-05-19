from ..models.wan_video_text_encoder import WanTextEncoder
from transformers import AutoTokenizer
import torch
import ftfy
import html
import string
import regex as re


class WanPrompter:
    def __init__(self, tokenizer_path=None, text_len=512, clean="whitespace"):
        if clean not in (None, "whitespace", "lower", "canonicalize"):
            raise ValueError(f"Unsupported clean mode: {clean}")
        self.text_len = text_len
        self.clean = clean
        self.refiners = []
        self.text_encoder = None
        self.tokenizer = None
        self.vocab_size = None
        self.fetch_tokenizer(tokenizer_path)

    def _basic_clean(self, text):
        text = ftfy.fix_text(text)
        text = html.unescape(html.unescape(text))
        return text.strip()

    def _whitespace_clean(self, text):
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _canonicalize(self, text, keep_punctuation_exact_string=None):
        text = text.replace("_", " ")
        if keep_punctuation_exact_string:
            text = keep_punctuation_exact_string.join(
                part.translate(str.maketrans("", "", string.punctuation))
                for part in text.split(keep_punctuation_exact_string)
            )
        else:
            text = text.translate(str.maketrans("", "", string.punctuation))
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _clean(self, text):
        if self.clean == "whitespace":
            text = self._whitespace_clean(self._basic_clean(text))
        elif self.clean == "lower":
            text = self._whitespace_clean(self._basic_clean(text)).lower()
        elif self.clean == "canonicalize":
            text = self._canonicalize(self._basic_clean(text))
        return text

    @torch.no_grad()
    def process_prompt(self, prompt, positive=True):
        if isinstance(prompt, list):
            return [self.process_prompt(item, positive=positive) for item in prompt]
        for refiner in self.refiners:
            prompt = refiner(prompt, positive=positive)
        return prompt

    def fetch_tokenizer(self, tokenizer_path=None):
        if tokenizer_path is None:
            return
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.vocab_size = self.tokenizer.vocab_size

    def _tokenize(self, sequence, return_mask=False, **kwargs):
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not loaded. Call `fetch_tokenizer()` first.")

        tokenizer_kwargs = {
            "return_tensors": "pt",
            "padding": "max_length",
            "truncation": True,
            "max_length": self.text_len,
        }
        tokenizer_kwargs.update(kwargs)

        if isinstance(sequence, str):
            sequence = [sequence]
        if self.clean is not None:
            sequence = [self._clean(text) for text in sequence]

        encoded = self.tokenizer(sequence, **tokenizer_kwargs)
        if return_mask:
            return encoded.input_ids, encoded.attention_mask
        return encoded.input_ids

    def fetch_models(self, text_encoder: WanTextEncoder = None):
        self.text_encoder = text_encoder

    def encode_prompt(self, prompt, positive=True, device="cuda"):
        if self.text_encoder is None:
            raise ValueError("Text encoder is not set. Call `fetch_models()` first.")
        prompt = self.process_prompt(prompt, positive=positive)

        ids, mask = self._tokenize(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(device)
        mask = mask.to(device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        prompt_emb = self.text_encoder(ids, mask)
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        return prompt_emb
