"""Offline o200k_base, with the upstream tokenizer data bundled for cold tests.

Pattern/special tokens follow tiktoken_ext.openai_public.o200k_base (MIT).
The BPE file is verified against tiktoken's published SHA-256 on every load.
"""

from functools import lru_cache
from pathlib import Path

import tiktoken
from tiktoken.load import load_tiktoken_bpe


@lru_cache(maxsize=1)
def get_encoding(name: str = "o200k_base") -> tiktoken.Encoding:
    if name != "o200k_base":
        raise ValueError("P1 supports only the specified offline o200k_base tokenizer")
    path = Path(__file__).with_name("tokenizer_data") / "o200k_base.tiktoken"
    ranks = load_tiktoken_bpe(
        str(path), expected_hash="446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"
    )
    pattern = "|".join(
        [
            r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
            r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
            r"\p{N}{1,3}",
            r" ?[^\s\p{L}\p{N}]+[\r\n/]*",
            r"\s*[\r\n]+",
            r"\s+(?!\S)",
            r"\s+",
        ]
    )
    return tiktoken.Encoding(
        name=name,
        pat_str=pattern,
        mergeable_ranks=ranks,
        special_tokens={"<|endoftext|>": 199999, "<|endofprompt|>": 200018},
    )
