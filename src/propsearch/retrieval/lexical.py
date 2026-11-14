"""BM25 over bilingual unit descriptions (rank_bm25), built once per process from the DB."""

from __future__ import annotations

import re
import threading

import numpy as np
from rank_bm25 import BM25Okapi
from sqlalchemy import select
from sqlalchemy.orm import Session

from propsearch.db import Compound, Unit
from propsearch.normalize import normalize_text

_WORD = re.compile(r"[a-z0-9]+|[؀-ۿ]+")
_STOP = {
    "a",
    "an",
    "the",
    "of",
    "in",
    "on",
    "to",
    "and",
    "with",
    "for",
    "is",
    "at",
    "by",
    "from",
    "near",
    "close",
    "في",
    "من",
    "علي",
    "على",
    "و",
    "مع",
    "او",
    "ال",
    "ده",
    "دي",
    "عن",
}


def bm25_tokens(text: str) -> list[str]:
    out = []
    for w in _WORD.findall(normalize_text(text)):
        if w.startswith("ال") and len(w) > 4:
            w = w[2:]
        elif w[:1] in "وبفل" and len(w) > 4 and w[1:3] == "ال":
            w = w[3:]
        elif w.isascii() and len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        if w not in _STOP and len(w) > 1:
            out.append(w)
    return out


class Bm25Index:
    def __init__(self, ids: list[int], docs: list[str], codes: dict[str, int]) -> None:
        self.ids = ids
        self.pos = {uid: i for i, uid in enumerate(ids)}
        self.bm25 = BM25Okapi([bm25_tokens(d) for d in docs])
        self.codes = codes

    def scores(self, query: str, candidates: list[int]) -> dict[int, float]:
        q = bm25_tokens(query)
        if not q or not candidates:
            return {}
        all_scores = self.bm25.get_scores(q)
        idx = np.array([self.pos[c] for c in candidates if c in self.pos])
        return {self.ids[i]: float(all_scores[i]) for i in idx if all_scores[i] > 0}


_lock = threading.Lock()
_index: Bm25Index | None = None


def get_index(session: Session) -> Bm25Index:
    global _index
    with _lock:
        if _index is None:
            rows = session.execute(
                select(
                    Unit.id,
                    Unit.code,
                    Unit.description_en,
                    Unit.description_ar,
                    Compound.name_en,
                    Compound.name_ar,
                )
                .join(Compound, Unit.compound_id == Compound.id)
                .order_by(Unit.id)
            ).all()
            docs = [f"{r.code} {r.name_en} {r.name_ar} {r.description_en} {r.description_ar}" for r in rows]
            _index = Bm25Index([r.id for r in rows], docs, {r.code.upper(): r.id for r in rows})
        return _index


def reset_index() -> None:
    global _index
    with _lock:
        _index = None
