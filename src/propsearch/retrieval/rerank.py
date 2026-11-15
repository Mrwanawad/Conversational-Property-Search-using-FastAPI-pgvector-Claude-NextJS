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

        kw = {"model_kwargs": {"dtype": "float16"}} if device.startswith("cuda") else {}
        self.model = CrossEncoder("BAAI/bge-reranker-v2-m3", device=device, max_length=384, **kw)

    def score(self, query: str, docs: list[str]) -> np.ndarray:
        if not docs:
            return np.zeros(0)
        # CrossEncoder already applies a sigmoid to the single-logit head: scores are in [0, 1]
        return np.asarray(
            self.model.predict([(query, d) for d in docs], batch_size=16, show_progress_bar=False),
            dtype=np.float64,
        )


class HfBgeReranker:
    """Same cross-encoder on Hugging Face Inference (text-classification over query/doc pairs)."""

    name = "bge-reranker-v2-m3"

    def score(self, query: str, docs: list[str]) -> np.ndarray:
        from propsearch.retrieval.embed import hf_post

        if not docs:
            return np.zeros(0)
        out = hf_post(
            "BAAI/bge-reranker-v2-m3",
            "text-classification",
            {"inputs": [{"text": query, "text_pair": d} for d in docs], "parameters": {"top_k": 1}},
        )
        # one [{"label", "score"}] per pair; the single-logit head is already passed through a sigmoid
        return np.asarray([(r[0] if isinstance(r, list) else r)["score"] for r in out], dtype=np.float64)


@lru_cache(maxsize=2)
def get_reranker(name: str, device: str = "cpu") -> Reranker | None:
    if name != "bge":
        return None
    return HfBgeReranker() if device == "hf" else BgeReranker(device)
