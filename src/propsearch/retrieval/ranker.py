"""Hybrid ranking: SQL candidates -> BM25 + pgvector (RRF) -> cross-encoder -> business rules -> diversity."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from propsearch.data.catalog import LANDMARKS, haversine_km
from propsearch.db import Compound, Developer, Unit
from propsearch.domain import ParsedQuery
from propsearch.retrieval import sql
from propsearch.retrieval.embed import Embedder
from propsearch.retrieval.lexical import get_index, reset_index
from propsearch.retrieval.rerank import Reranker
from propsearch.seed import unit_doc

RRF_K = 60
FUSED_DEPTH = 100
RERANK_DEPTH = 40
REFERENCE_DATE_DAYS = 0  # freshness is measured against the newest listing, so the demo data never "ages"


@dataclass
class Scored:
    unit_id: int
    score: float
    relevance: float = 0.0
    business: float = 0.0
    bm25_rank: int | None = None
    vector_rank: int | None = None
    distance_km: float | None = None
    parts: dict = field(default_factory=dict)


@dataclass
class RankResult:
    total: int
    ranked: list[Scored]
    stage_ms: dict


def rrf(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    out: dict[int, float] = {}
    for ranking in rankings:
        for r, uid in enumerate(ranking):
            out[uid] = out.get(uid, 0.0) + 1.0 / (k + r + 1)
    return out


def vector_ranking(session: Session, embedder: Embedder, text: str, ids: list[int], depth: int) -> list[int]:
    q = embedder.embed([text], query=True)[0].tolist()
    stmt = (
        select(Unit.id)
        .where(Unit.id.in_(ids), Unit.embedding.is_not(None))
        .order_by(Unit.embedding.cosine_distance(q))
        .limit(depth)
    )
    return list(session.scalars(stmt))


def _business(rows: dict[int, dict], pq: ParsedQuery) -> dict[int, tuple[float, float | None, dict]]:
    """Availability, freshness, value vs area median price/m2, developer delivery record, proximity."""
    if not rows:
        return {}
    newest = max(r["updated_at"] for r in rows.values())
    by_area: dict[str, list[float]] = {}
    for r in rows.values():
        by_area.setdefault(r["area"], []).append(r["price"] / r["bua_m2"])
    medians = {a: float(np.median(v)) for a, v in by_area.items()}
    lm = LANDMARKS.get(pq.filters.near_landmark or "")
    out = {}
    for uid, r in rows.items():
        availability = 1.0 if r["status"] == "available" else 0.5
        age_days = (newest - r["updated_at"]).total_seconds() / 86400
        freshness = math.exp(-age_days / 60)
        value = min(1.0, max(0.0, 1.5 - (r["price"] / r["bua_m2"]) / medians[r["area"]]))
        track = r["on_time_rate"]
        parts = {
            "availability": availability,
            "freshness": round(freshness, 3),
            "value": round(value, 3),
            "developer": track,
        }
        score = 0.35 * availability + 0.25 * freshness + 0.25 * value + 0.15 * track
        dist = None
        if lm:
            dist = haversine_km(lm.lat, lm.lng, r["lat"], r["lng"])
            proximity = math.exp(-dist / 5)
            parts["proximity"] = round(proximity, 3)
            score = 0.6 * score + 0.4 * proximity
        out[uid] = (score, dist, parts)
    return out


def rank(
    session: Session,
    pq: ParsedQuery,
    embedder: Embedder,
    reranker: Reranker | None,
    *,
    use_bm25: bool = True,
    use_vector: bool = True,
    diversify: bool = True,
    limit: int | None = None,
) -> RankResult:
    import time

    t0 = time.perf_counter()
    ids = sql.candidate_ids(session, pq.filters)
    stage = {"sql": (time.perf_counter() - t0) * 1000}
    if not ids:
        return RankResult(0, [], stage)

    rows = {
        r.id: r._asdict()
        for r in session.execute(
            select(
                Unit.id,
                Unit.area,
                Unit.price,
                Unit.bua_m2,
                Unit.status,
                Unit.updated_at,
                Unit.compound_id,
                Unit.delivery_year,
                Compound.lat,
                Compound.lng,
                Developer.on_time_rate,
            )
            .join(Compound, Unit.compound_id == Compound.id)
            .join(Developer, Compound.developer_id == Developer.id)
            .where(Unit.id.in_(ids))
        )
    }
    business = _business(rows, pq)
    relevance: dict[int, float] = {}
    bm25_rank: dict[int, int] = {}
    vec_rank: dict[int, int] = {}
    soft = pq.soft_preferences.strip()
    if soft and (use_bm25 or use_vector):
        t = time.perf_counter()
        rankings = []
        if use_bm25:
            scores = get_index(session).scores(soft, ids)
            lex = sorted(scores, key=scores.get, reverse=True)[:FUSED_DEPTH]
            bm25_rank = {u: i for i, u in enumerate(lex)}
            rankings.append(lex)
        stage["bm25"] = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        if use_vector:
            vec = vector_ranking(session, embedder, soft, ids, FUSED_DEPTH)
            vec_rank = {u: i for i, u in enumerate(vec)}
            rankings.append(vec)
        stage["vector"] = (time.perf_counter() - t) * 1000
        fused = rrf(rankings)
        order = sorted(fused, key=fused.get, reverse=True)
        top = max(fused.values()) if fused else 1.0
        relevance = {u: fused[u] / top for u in order}
        if reranker is not None and order:
            t = time.perf_counter()
            head = order[:RERANK_DEPTH]
            docs = {u.id: unit_doc(u) for u in session.scalars(select(Unit).where(Unit.id.in_(head)))}
            ce = reranker.score(soft, [docs[u] for u in head])
            # reranked head stays above the fused tail
            relevance = {u: 0.5 + 0.5 * float(s) for u, s in zip(head, ce, strict=True)}
            for u in order[RERANK_DEPTH:]:
                relevance[u] = 0.5 * fused[u] / top
            stage["rerank"] = (time.perf_counter() - t) * 1000

    scored = []
    for uid in ids:
        b, dist, parts = business[uid]
        rel = relevance.get(uid, 0.0)
        score = 0.7 * rel + 0.3 * b if soft else b
        scored.append(
            Scored(
                uid,
                round(score, 5),
                round(rel, 4),
                round(b, 4),
                bm25_rank.get(uid),
                vec_rank.get(uid),
                round(dist, 2) if dist is not None else None,
                parts,
            )
        )

    if pq.sort == "price_asc":
        scored.sort(key=lambda s: rows[s.unit_id]["price"])
    elif pq.sort == "price_desc":
        scored.sort(key=lambda s: -rows[s.unit_id]["price"])
    elif pq.sort == "price_per_m2_asc":
        scored.sort(key=lambda s: rows[s.unit_id]["price"] / rows[s.unit_id]["bua_m2"])
    elif pq.sort == "delivery_asc":
        scored.sort(key=lambda s: (rows[s.unit_id]["delivery_year"], -s.score))
    else:
        scored.sort(key=lambda s: -s.score)

    if diversify and not pq.filters.compounds and not pq.filters.unit_code and pq.sort == "relevance":
        scored = _diversify(scored, rows, per_compound=2, window=limit or 10)
    return RankResult(len(ids), scored, stage)


def _diversify(scored: list[Scored], rows: dict[int, dict], per_compound: int, window: int) -> list[Scored]:
    """Cap units per compound inside the first page so one big compound does not fill the shortlist."""
    head, tail, seen = [], [], {}
    for s in scored:
        cid = rows[s.unit_id]["compound_id"]
        if len(head) < window and seen.get(cid, 0) < per_compound:
            head.append(s)
            seen[cid] = seen.get(cid, 0) + 1
        else:
            tail.append(s)
    return head + tail


def reset_caches() -> None:
    reset_index()
    sql._COORDS.clear()
