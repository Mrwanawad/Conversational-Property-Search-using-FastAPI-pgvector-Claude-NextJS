"""Text embedders. bge-m3 (multilingual, handles Arabic + English in one space) or a hashing fallback.

bge-m3 runs locally through sentence-transformers, or remotely on Hugging Face Inference (device "hf",
token in HF_TOKEN) where torch does not fit, such as a serverless function. Both give the same vectors,
so units embedded locally are searched with remote query vectors.
"""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from typing import Protocol

import numpy as np

from propsearch.db import EMBED_DIM
from propsearch.normalize import normalize_text


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str], *, query: bool = False) -> np.ndarray: ...


class HashingEmbedder:
    """Character 3-5 gram feature hashing. No model download; weak semantics, used in tests/CI."""

    name = "hashing"

    def embed(self, texts: list[str], *, query: bool = False) -> np.ndarray:
        out = np.zeros((len(texts), EMBED_DIM), dtype=np.float32)
        for row, t in enumerate(texts):
            s = f" {normalize_text(t)} "
            for n in (3, 4, 5):
                for i in range(len(s) - n + 1):
                    h = int.from_bytes(
                        hashlib.blake2b(s[i : i + n].encode(), digest_size=8).digest(), "little"
                    )
                    out[row, h % EMBED_DIM] += 1.0 if (h >> 63) & 1 else -1.0
            norm = np.linalg.norm(out[row])
            if norm:
                out[row] /= norm
        return out


class BgeM3Embedder:
    name = "bge-m3"

    def __init__(self, device: str = "cpu") -> None:
        from sentence_transformers import SentenceTransformer

        # fp16 on GPU: both bge models fit a 4 GB card; cosine to fp32 vectors is ~0.9999
        kw = {"model_kwargs": {"dtype": "float16"}} if device.startswith("cuda") else {}
        self.model = SentenceTransformer("BAAI/bge-m3", device=device, **kw)

    def embed(self, texts: list[str], *, query: bool = False) -> np.ndarray:
        return np.asarray(
            self.model.encode(texts, batch_size=16, normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )


HF_URL = "https://router.huggingface.co/hf-inference/models/{model}/pipeline/{task}"


def hf_post(model: str, task: str, payload: dict, timeout: float = 8.0):
    import httpx

    r = httpx.post(
        HF_URL.format(model=model, task=task),
        json=payload,
        headers={"Authorization": f"Bearer {os.environ.get('HF_TOKEN', '')}"},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()


class HfBgeM3Embedder:
    name = "bge-m3"

    def embed(self, texts: list[str], *, query: bool = False) -> np.ndarray:
        vecs = np.asarray(hf_post("BAAI/bge-m3", "feature-extraction", {"inputs": texts}), dtype=np.float32)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


@lru_cache(maxsize=2)
def get_embedder(name: str, device: str = "cpu") -> Embedder:
    if name == "bge-m3":
        return HfBgeM3Embedder() if device == "hf" else BgeM3Embedder(device)
    return HashingEmbedder()
