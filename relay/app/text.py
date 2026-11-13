"""Conservative text normalization shared by evidence checks."""

import hashlib
import json
import re
import unicodedata


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\u00ad", "")
    text = re.sub(r"(?<=\w)[-\u2010]\s*\n\s*(?=\w)", "", text)
    return " ".join(text.split())


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False).encode()
    ).hexdigest()
