"""Ranking quality on soft wishes ("quiet", "close to schools") against hidden generator labels.

    python scripts/retrieval_eval.py                       # embedded Postgres, all configs
    python scripts/retrieval_eval.py --configs hybrid-hashing hybrid-bge-m3
    DATABASE_URL=postgresql+psycopg://... python scripts/retrieval_eval.py

Each query in evals/retrieval.jsonl is parsed by the rule parser into hard filters + soft preferences,
exactly as a live turn. The SQL candidate set is the pool: a unit is relevant when its hidden soft tags
(written into the description as paraphrases, never stored in the DB) include the query's tags, graded by
how many it has. Recall@20 is therefore measured against every relevant unit in the pool, not only the
ones a user happened to see (click_metrics.py cannot do that).

Metrics through ranx: NDCG@10, Recall@20, MRR, with a paired t-test against the first config.
Ranking is the final pipeline order (relevance + business rules), diversity cap off.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ranx import Qrels, Run, compare  # noqa: E402
from sqlalchemy import select  # noqa: E402

from eval import database_url  # noqa: E402
from propsearch.data.generate import soft_tags_by_code  # noqa: E402
from propsearch.db import Compound, Unit, get_engine, session_factory  # noqa: E402
from propsearch.retrieval.embed import get_embedder  # noqa: E402
from propsearch.retrieval.precomputed import (  # noqa: E402
    PrecomputedEmbedder,
    PrecomputedReranker,
    precompute,
)
from propsearch.retrieval.ranker import rank, reset_caches  # noqa: E402
from propsearch.retrieval.rerank import get_reranker  # noqa: E402
from propsearch.seed import ensure_seeded  # noqa: E402
from propsearch.understanding.entities import CompoundMatcher, apply_entities  # noqa: E402
from propsearch.understanding.rules import RuleParser  # noqa: E402

# name -> (embedder, reranker, use_bm25, use_vector)
CONFIGS = {
    "bm25": ("hashing", "none", True, False),
    "hybrid-hashing": ("hashing", "none", True, True),
    "vector-bge-m3": ("bge-m3", "none", False, True),
    "hybrid-bge-m3": ("bge-m3", "none", True, True),
    "hybrid-bge-m3+rerank": ("bge-m3", "bge", True, True),
    # what the free deployment serves: same models, scores looked up from Postgres (EMBED_DEVICE=precomputed)
    "hybrid-precomputed": ("bge-m3", "none", True, True),
    "hybrid-precomputed+rerank": ("bge-m3", "bge", True, True),
}
METRICS = ["ndcg@10", "recall@20", "mrr"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="evals/retrieval.jsonl")
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="evals/results/retrieval.json")
    args = ap.parse_args()

    engine = get_engine(database_url())
    ensure_seeded(engine, CONFIGS[args.configs[0]][0], args.device)
    Session = session_factory(engine)
    items = [json.loads(line) for line in (ROOT / args.queries).read_text(encoding="utf-8").splitlines()]
    tags = soft_tags_by_code()

    with Session() as db:
        code_of = dict(db.execute(select(Unit.id, Unit.code)).all())
        matcher = CompoundMatcher(
            [(c.slug, c.name_en, c.name_ar, c.area) for c in db.scalars(select(Compound))]
        )
    parser = RuleParser()
    parsed = {}
    for it in items:
        pq = apply_entities(parser.parse(it["query"]), it["query"], matcher)
        parsed[it["id"]] = pq

    qrels: dict[str, dict[str, int]] = {}
    runs: dict[str, dict[str, dict[str, float]]] = {c: {} for c in args.configs}
    pools: dict[str, int] = {}
    for name in args.configs:
        emb_name, rr_name, use_bm25, use_vector = CONFIGS[name]
        ensure_seeded(engine, emb_name, args.device)
        reset_caches()
        if name.startswith("hybrid-precomputed"):
            precompute(engine, args.device)
            embedder = PrecomputedEmbedder(Session)
            reranker = PrecomputedReranker(Session) if rr_name == "bge" else None
        else:
            embedder = get_embedder(emb_name, args.device)
            reranker = get_reranker(rr_name, args.device)
        with Session() as db:
            for it in items:
                pq = parsed[it["id"]]
                res = rank(
                    db, pq, embedder, reranker, use_bm25=use_bm25, use_vector=use_vector, diversify=False
                )
                ids = [s.unit_id for s in res.ranked]
                if it["id"] not in qrels:
                    pools[it["id"]] = len(ids)
                    qrels[it["id"]] = {
                        str(u): g for u in ids if (g := len(set(it["tags"]) & set(tags[code_of[u]])))
                    }
                runs[name][it["id"]] = {str(u): 1.0 / (i + 1) for i, u in enumerate(ids)}
        print(f"{name}: done")

    empty = [q for q, r in qrels.items() if not r]
    kept = [q for q in qrels if q not in empty]
    q = Qrels({k: qrels[k] for k in kept})
    ranx_runs = [Run({k: runs[n][k] for k in kept}, name=n) for n in args.configs]
    report = compare(q, ranx_runs, metrics=METRICS, max_p=0.05, stat_test="student")
    print(report)

    soft = {it["id"]: parsed[it["id"]].soft_preferences for it in items}
    # ranx sorts query ids internally, so read per-query scores back by id rather than by position
    per_query = {r.name: r.scores["ndcg@10"] for r in ranx_runs}
    out = {
        "queries": len(kept),
        "skipped_no_relevant_in_pool": empty,
        "results": {n: {m: round(report.results[n][m], 3) for m in METRICS} for n in args.configs},
        "per_query": [
            {
                "id": k,
                "soft": soft[k],
                "pool": pools[k],
                "relevant": len(qrels[k]),
                "ndcg@10": {n: round(float(per_query[n][k]), 3) for n in args.configs},
            }
            for k in kept
        ],
    }
    dest = ROOT / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out["results"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
