"""Conservative text normalization shared by evidence checks."""

import re
import unicodedata


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\u00ad", "")
    text = re.sub(r"(?<=\w)[-\u2010]\s*\n\s*(?=\w)", "", text)
    return " ".join(text.split())
