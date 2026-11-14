"""Cross-encoder reranking of the fused shortlist against the user's soft preferences."""

from __future__ import annotations

from functools import lru_cache
from typing import Protocol

import numpy as np


class Reranker(Protocol):
    name: str

    def score(self, query: str, docs: list[str]) -> np.ndarray: ...


class BgeReranker:
    name = "bge-reranker-v2-m3"

    def __init__(self, device: str = "cpu") -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder("BAAI/bge-reranker-v2-m3", device=device, max_length=384)

    def score(self, query: str, docs: list[str]) -> np.ndarray:
        if not docs:
            return np.zeros(0)
        logits = np.asarray(
            self.model.predict([(query, d) for d in docs], batch_size=16, show_progress_bar=False),
            dtype=np.float64,
        )
        return 1.0 / (1.0 + np.exp(-logits))


@lru_cache(maxsize=2)
def get_reranker(name: str, device: str = "cpu") -> Reranker | None:
    return BgeReranker(device) if name == "bge" else None
