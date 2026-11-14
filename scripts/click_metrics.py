"""Ranking quality from logged clicks and inquiries (implicit relevance labels).

    DATABASE_URL=... python scripts/click_metrics.py [--since 2026-11-14T22:00] [--out evals/results/clicks.json]

Each search turn that showed units is one ranked list (query_log.top_unit_ids). Its graded labels come
from the events on that turn: inquiry = 2, click (Details) = 1, nothing = 0. Only shown units can be
judged, so recall is not computable here; NDCG@10 and MRR are, plus the usual engagement rates.
Turns with no interaction count as abandoned (reciprocal rank 0) and are left out of NDCG.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sqlalchemy import select  # noqa: E402

from propsearch.db import Event, QueryLog, get_engine, session_factory  # noqa: E402

GAIN = {"click": 1, "inquiry": 2}


def dcg(gains: list[int], k: int) -> float:
    return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(gains[:k]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=None, help="ISO timestamp (UTC) of the first turn to include")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    Session = session_factory(get_engine())
    with Session() as db:
        q = select(QueryLog).where(QueryLog.simulated.is_(False)).order_by(QueryLog.id)
        if args.since:
            q = q.where(QueryLog.created_at >= datetime.fromisoformat(args.since))
        turns = [t for t in db.scalars(q) if t.top_unit_ids]
        ids = [t.id for t in turns]
        events = db.scalars(select(Event).where(Event.query_log_id.in_(ids))).all() if ids else []

    labels: dict[int, dict[int, int]] = defaultdict(dict)
    for e in events:
        g = GAIN.get(e.kind, 0)
        labels[e.query_log_id][e.unit_id] = max(labels[e.query_log_id].get(e.unit_id, 0), g)

    rows, ndcgs, rrs = [], [], []
    for t in turns:
        lab = labels.get(t.id, {})
        gains = [lab.get(u, 0) for u in t.top_unit_ids]
        first = next((i + 1 for i, g in enumerate(gains) if g > 0), None)
        rrs.append(1 / first if first else 0.0)
        ndcg = None
        if any(gains):
            ndcg = dcg(gains, args.k) / dcg(sorted(gains, reverse=True), args.k)
            ndcgs.append(ndcg)
        rows.append(
            {
                "turn": t.id,
                "query": t.query,
                "language": t.language,
                "shown": len(gains),
                "gains": gains,
                "ndcg": round(ndcg, 3) if ndcg is not None else None,
            }
        )

    n = len(turns)
    out = {
        "turns_with_results": n,
        "turns_with_interaction": len(ndcgs),
        "clicks": sum(e.kind == "click" for e in events),
        "inquiries": sum(e.kind == "inquiry" for e in events),
        f"ndcg@{args.k}": round(sum(ndcgs) / len(ndcgs), 3) if ndcgs else None,
        "mrr": round(sum(rrs) / n, 3) if n else None,
        "abandonment_rate": round(1 - len(ndcgs) / n, 3) if n else None,
        "inquiry_turn_rate": round(sum(2 in r["gains"] for r in rows) / n, 3) if n else None,
        "turns": rows,
    }
    print(json.dumps({k: v for k, v in out.items() if k != "turns"}, indent=2))
    for r in rows:
        print(f"{r['turn']:>5} ndcg={r['ndcg']} gains={r['gains']}  {r['query'][:70]}")
    if args.out:
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
