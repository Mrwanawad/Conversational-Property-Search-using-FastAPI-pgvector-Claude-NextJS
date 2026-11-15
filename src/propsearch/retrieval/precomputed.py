"""bge-m3 and bge-reranker-v2-m3 without a model server (EMBED_DEVICE=precomputed).

The rule parser writes soft preferences from a closed set of phrases ("quiet calm area", "close to
schools", ...), joined with "; ". So every model call the ranker makes at query time is known in advance:
one query vector per phrase, and one cross-encoder score per (phrase, unit document) and (phrase,
description sentence). They are computed offline on a machine with the models and stored in Postgres.
A turn then costs one indexed lookup instead of a GPU.

A compound wish ("quiet calm area; family friendly") uses the mean of its phrases. Free text from an LLM
parser is mapped onto the set through the synonym dictionary. What still does not map is skipped; when
nothing maps, the stage raises and the ranker falls back to BM25 for that turn.

    python -m propsearch.retrieval.precomputed     # after `python -m propsearch.seed --embedder bge-m3`
"""

from __future__ import annotations

import hashlib
import time

import numpy as np
from sqlalchemy import delete, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from propsearch.db import SoftScore, SoftVector, Unit, init_db, session_factory


class NotPrecomputed(LookupError):
    pass


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def phrases(soft: str) -> list[str]:
    """Canonical phrases in a soft-preference string. Rule-parser output is already canonical; free text
    from an LLM parser ("near international schools") is mapped through the same synonym dictionary."""
    from propsearch.normalize import Match, tag
    from propsearch.understanding.rules import SOFT_PHRASES

    known = set(SOFT_PHRASES.values())
    out: list[str] = []
    for part in (p.strip() for p in soft.split(";")):
        if part in known:
            out.append(part)
        elif part:
            out += [
                SOFT_PHRASES[c]
                for it in tag(part).items
                if isinstance(it, Match)
                for c in it.concepts
                if c in SOFT_PHRASES
            ]
    return list(dict.fromkeys(out))


class PrecomputedEmbedder:
    name = "bge-m3"

    def __init__(self, Session: sessionmaker[Session]) -> None:
        with Session() as s:
            self.vectors = {
                p: np.asarray(v, dtype=np.float32)
                for p, v in s.execute(select(SoftVector.phrase, SoftVector.embedding))
            }

    def embed(self, texts: list[str], *, query: bool = False) -> np.ndarray:
        out = []
        for t in texts:
            known = [self.vectors[p] for p in phrases(t) if p in self.vectors]
            if not known:
                raise NotPrecomputed(t)
            v = np.mean(known, axis=0)
            out.append(v / np.linalg.norm(v))
        return np.asarray(out, dtype=np.float32)


class PrecomputedReranker:
    name = "bge-reranker-v2-m3"

    def __init__(self, Session: sessionmaker[Session]) -> None:
        self.Session = Session

    def score(self, query: str, docs: list[str]) -> np.ndarray:
        if not docs:
            return np.zeros(0)
        shas = [text_sha(d) for d in docs]
        with self.Session() as s:
            rows = s.execute(
                select(SoftScore.phrase, SoftScore.text_sha, SoftScore.score).where(
                    SoftScore.phrase.in_(phrases(query)), SoftScore.text_sha.in_(set(shas))
                )
            ).all()
        if not rows:
            raise NotPrecomputed(query)
        by_sha: dict[str, list[float]] = {}
        for _, sha, score in rows:
            by_sha.setdefault(sha, []).append(score)
        return np.asarray([np.mean(by_sha.get(sha, [0.0])) for sha in shas], dtype=np.float64)


def precompute(engine: Engine, device: str = "cpu") -> dict:
    """Fill soft_vectors and soft_scores from the local models. Rerun when the inventory changes."""
    from propsearch.respond import SENT_SPLIT
    from propsearch.retrieval.embed import get_embedder
    from propsearch.retrieval.rerank import get_reranker
    from propsearch.seed import CACHE_DIR, unit_doc
    from propsearch.understanding.rules import SOFT_PHRASES

    init_db(engine)
    Session = session_factory(engine)
    with Session() as s:
        units = s.scalars(select(Unit).order_by(Unit.id)).all()
        texts = {unit_doc(u) for u in units}
        texts |= {x for u in units for x in SENT_SPLIT.split(u.description_en)[1:]}
    texts_l = sorted(texts)
    soft = sorted(set(SOFT_PHRASES.values()))

    # cached on disk by content, so the same inventory loads into another database without the models
    cache = CACHE_DIR / f"soft-{text_sha(chr(0).join(soft + texts_l))}.npz"
    if cache.exists():
        z = np.load(cache)
        vecs, scores = z["vecs"], z["scores"]
    else:
        t = time.perf_counter()
        vecs = get_embedder("bge-m3", device).embed(soft, query=True)
        reranker = get_reranker("bge", device)
        scores = np.zeros((len(soft), len(texts_l)), dtype=np.float32)
        for i, p in enumerate(soft):
            scores[i] = reranker.score(p, texts_l)
            print(f"{p}: {len(texts_l)} texts, {time.perf_counter() - t:.0f}s")
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, vecs=vecs, scores=scores)
    shas = [text_sha(x) for x in texts_l]
    rows = [
        {"phrase": p, "text_sha": sha, "score": float(sc)}
        for p, row in zip(soft, scores, strict=True)
        for sha, sc in zip(shas, row, strict=True)
    ]
    with Session() as s:
        s.execute(delete(SoftVector))
        s.execute(delete(SoftScore))
        s.add_all(SoftVector(phrase=p, embedding=v.tolist()) for p, v in zip(soft, vecs, strict=True))
        for i in range(0, len(rows), 5000):
            s.execute(SoftScore.__table__.insert(), rows[i : i + 5000])
        s.commit()
    return {"phrases": len(soft), "texts": len(texts_l), "scores": len(rows)}


if __name__ == "__main__":
    from propsearch.config import get_settings
    from propsearch.db import get_engine

    device = get_settings().embed_device
    print(precompute(get_engine(), "cpu" if device in ("precomputed", "hf") else device))
