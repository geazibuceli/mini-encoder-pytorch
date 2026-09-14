"""Small regex tokenizer and deterministic vocabulary."""

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]"]
TOKEN_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\w\s]", re.UNICODE)


class RegexTokenizer:
    def __init__(self, lowercase: bool = True):
        self.lowercase = lowercase

    def tokenize(self, text: str) -> list[str]:
        text = text.lower() if self.lowercase else text
        return TOKEN_PATTERN.findall(text)


class Vocabulary:
    def __init__(self, token_to_id: dict[str, int]):
        self.token_to_id = dict(token_to_id)
        self.id_to_token = {value: key for key, value in self.token_to_id.items()}
        if not all(token in self.token_to_id for token in SPECIAL_TOKENS):
            raise ValueError("Vocabulary must contain [PAD], [UNK], and [CLS].")
        ids = list(self.token_to_id.values())
        if any(type(value) is not int for value in ids) or sorted(ids) != list(range(len(ids))):
            raise ValueError("Vocabulary IDs must be unique contiguous integers starting at zero.")

    def fingerprint(self):
        payload = json.dumps(self.token_to_id, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def pad_id(self):
        return self.token_to_id["[PAD]"]

    @property
    def unk_id(self):
        return self.token_to_id["[UNK]"]

    @property
    def cls_id(self):
        return self.token_to_id["[CLS]"]

    def encode(self, tokens):
        return [self.token_to_id.get(token, self.unk_id) for token in tokens]

    def decode(self, ids):
        return [self.id_to_token.get(int(index), "[UNK]") for index in ids]

    def save(self, path):
        Path(path).write_text(json.dumps(self.token_to_id, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))


def build_vocabulary(
    texts, tokenizer: RegexTokenizer, min_frequency=1, max_size=None
) -> Vocabulary:
    counts = Counter(token for text in texts for token in tokenizer.tokenize(text))
    words = sorted(
        (word for word, count in counts.items() if count >= min_frequency),
        key=lambda word: (-counts[word], word),
    )
    if max_size is not None:
        words = words[: max(0, max_size - len(SPECIAL_TOKENS))]
    return Vocabulary({token: index for index, token in enumerate(SPECIAL_TOKENS + words)})
