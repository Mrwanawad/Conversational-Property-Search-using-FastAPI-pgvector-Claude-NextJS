"""Evaluate query understanding and the full search turn on the labelled query set.

    python scripts/eval.py                         # rules parser, embedded Postgres (pgserver)
    python scripts/eval.py --parser anthropic      # Claude parser (needs ANTHROPIC_API_KEY)
    python scripts/eval.py --parser llm --model openai/gpt-oss-120b   # Groq / OpenRouter (LLM_API_KEY)
    python scripts/eval.py --queries evals/heldout.jsonl               # held-out phrasings
    DATABASE_URL=postgresql+psycopg://... python scripts/eval.py

Metrics:
- filter extraction: exact match per query and per-field precision / recall / F1
- clarification: asked when expected, not asked otherwise
- zero-result rate (after relaxation) and latency p50 / p95 of the whole turn
- groundedness: every unit field in the reply equals its DB row, every plan exists in the DB, and every
  number in the message and the per-unit reasons traces to the DB, the user's filters or the match count.
  Deterministic, so it replaces an LLM-judged faithfulness score (Ragas) for template replies.

Writes evals/results/<set>-<parser>[-<model>].json and prints a summary. Exits 1 if exact match falls under --min-exact.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from propsearch.config import Settings  # noqa: E402
from propsearch.db import Unit, get_engine, session_factory  # noqa: E402
from propsearch.domain import SearchTurnResponse  # noqa: E402
from propsearch.pipeline import SearchEngine  # noqa: E402
from propsearch.respond import _NUM, _allowed_numbers  # noqa: E402
from propsearch.seed import ensure_seeded  # noqa: E402

# Derived defaults, not something the user said: the radius that comes with a landmark.
IGNORED = {"max_distance_km"}


def _norm(v):
    return sorted(v) if isinstance(v, list) else v


def score(pred: dict, gold: dict) -> tuple[bool, dict[str, tuple[int, int, int]]]:
    pred = {k: _norm(v) for k, v in pred.items() if k not in IGNORED}
    gold = {k: _norm(v) for k, v in gold.items()}
    per: dict[str, tuple[int, int, int]] = {}
    for k in set(pred) | set(gold):
        tp = int(k in pred and k in gold and pred[k] == gold[k])
        fp = int(k in pred and not tp)
        fn = int(k in gold and not tp)
        per[k] = (tp, fp, fn)
    return pred == gold, per


FIELDS = ("code", "price", "bua_m2", "bedrooms", "bathrooms", "garden_m2", "floor", "area", "unit_type")


def grounding_errors(db, r: SearchTurnResponse) -> list[str]:
    """Reply facts that do not come from the database or the user's own words."""
    errors = []
    shared = [
        r.parsed.filters.model_dump(mode="json"),
        [x.model_dump() for x in r.relaxations],
        r.total_matches,
        len(r.units),
    ]
    allowed = _allowed_numbers(shared)
    for h in r.units:
        u = db.get(Unit, h.id)
        for f in FIELDS:
            if getattr(h, f) != getattr(u, f):
                errors.append(f"{h.code}.{f}: reply {getattr(h, f)!r} != db {getattr(u, f)!r}")
        plans = [p for p in u.compound.plans]
        if h.best_plan:
            bp = h.best_plan
            if not any(
                (p.name, p.down_payment_pct, p.years) == (bp.name, bp.down_payment_pct, bp.years)
                for p in plans
            ):
                errors.append(f"{h.code}: plan {bp.name} not in db")
            if abs(bp.down_payment - u.price * bp.down_payment_pct / 100) > 0.01 * u.price:
                errors.append(f"{h.code}: down payment {bp.down_payment} off for {bp.down_payment_pct}%")
        row = {c.name: getattr(u, c.name) for c in Unit.__table__.columns if c.name != "embedding"}
        row["updated_at"] = None
        unit_allowed = allowed | _allowed_numbers(
            [
                row,
                [{"pct": p.down_payment_pct, "years": p.years, "disc": p.cash_discount_pct} for p in plans],
                h.best_plan.model_dump() if h.best_plan else None,
                h.price_per_m2,
                # arithmetic on DB and user values: gap to the budget, distance rounded for display
                abs(r.parsed.filters.budget_max - u.price) if r.parsed.filters.budget_max else None,
                round(h.distance_km, 1) if h.distance_km is not None else None,
            ]
        )
        for reason in h.reasons:
            for text in (reason.en, reason.ar):
                bad = [
                    n
                    for n in _NUM.findall(text)
                    if n.replace(",", "") not in unit_allowed and n not in unit_allowed
                ]
                if bad:
                    errors.append(f"{h.code} reason {text!r}: {bad}")
        allowed |= unit_allowed
    bad = [n for n in _NUM.findall(r.message) if n.replace(",", "") not in allowed and n not in allowed]
    if bad:
        errors.append(f"message {r.message!r}: {bad}")
    return errors


def database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    import pgserver

    srv = pgserver.get_server(tempfile.mkdtemp(prefix="propsearch-eval-"), cleanup_mode="stop")
    return srv.get_uri().replace("postgresql://", "postgresql+psycopg://")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parser", choices=["rules", "anthropic", "llm"], default="rules")
    ap.add_argument("--model", default=None, help="LLM model id (overrides LLM_MODEL / ANTHROPIC_MODEL)")
    ap.add_argument("--reasoning-effort", default=None, help="for --parser llm; '' to omit")
    ap.add_argument("--queries", default="evals/queries.jsonl")
    ap.add_argument("--llm-mode", choices=["always", "on_unknown"], default="always")
    ap.add_argument("--min-exact", type=float, default=0.0)
    args = ap.parse_args()

    url = database_url()
    engine = get_engine(url)
    ensure_seeded(engine, "hashing")
    overrides = {}
    if args.model:
        overrides["llm_model" if args.parser == "llm" else "anthropic_model"] = args.model
    if args.reasoning_effort is not None:
        overrides["llm_reasoning_effort"] = args.reasoning_effort
    overrides["llm_mode"] = args.llm_mode
    settings = Settings(
        database_url=url, parser=args.parser, embedder="hashing", reranker="none", **overrides
    )
    search = SearchEngine(settings, session_factory(engine))
    if args.parser == "llm":
        getattr(
            search.parser, "llm", search.parser
        ).max_wait_s = 300  # free tiers rate-limit; an eval should measure the model, not the quota

    items = [json.loads(line) for line in (ROOT / args.queries).read_text(encoding="utf-8").splitlines()]
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    by_lang: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    rows, latencies, zero, clar_ok, misses = [], [], 0, 0, []
    llm_ms, cost = [], 0.0
    grounded, ungrounded, judged = 0, [], 0
    db = session_factory(engine)()
    for it in items:
        r = search.turn(it["query"], simulated=True)
        if r.units:
            judged += 1
            errs = grounding_errors(db, r)
            grounded += not errs
            if errs:
                ungrounded.append({"id": it["id"], "errors": errs})
        pred = r.parsed.filters.model_dump(mode="json", exclude_defaults=True)
        exact, per = score(pred, it["filters"])
        for k, (tp, fp, fn) in per.items():
            t = totals[k]
            t[0] += tp
            t[1] += fp
            t[2] += fn
        by_lang[it["lang"]][0] += exact
        by_lang[it["lang"]][1] += 1
        asked = r.clarification is not None
        clar_ok += asked == bool(it.get("expect_clarification"))
        zero += (not asked) and r.total_matches == 0
        latencies.append(r.timings_ms["total"])
        usage = getattr(search.parser, "last_usage", None)
        if args.parser != "rules" and usage is not None:
            if usage.model:
                llm_ms.append(usage.latency_ms)
            cost += usage.cost_usd
        if not exact:
            misses.append({"id": it["id"], "query": it["query"], "gold": it["filters"], "pred": pred})
        rows.append(
            {
                "id": it["id"],
                "exact": exact,
                "clarified": asked,
                "total": r.total_matches,
                "relaxed": [x.field for x in r.relaxations],
                "parser": r.parsed.parser,
            }
        )

    def f1(tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        return round(p, 3), round(rc, 3), round(2 * p * rc / (p + rc), 3) if p + rc else 0.0

    n = len(items)
    tp, fp, fn = (sum(t[i] for t in totals.values()) for i in range(3))
    lat = sorted(latencies)
    out = {
        "parser": args.parser,
        "queries": n,
        "exact_match": round(sum(r["exact"] for r in rows) / n, 3),
        "exact_match_by_language": {k: round(v[0] / v[1], 3) for k, v in sorted(by_lang.items())},
        "micro_f1": dict(zip(("precision", "recall", "f1"), f1(tp, fp, fn), strict=True)),
        "per_field_f1": {k: f1(*v)[2] for k, v in sorted(totals.items())},
        "clarification_accuracy": round(clar_ok / n, 3),
        "zero_result_rate": round(zero / n, 3),
        "latency_ms": {"p50": round(statistics.median(lat), 1), "p95": round(lat[int(0.95 * (n - 1))], 1)},
        "grounded_turn_rate": round(grounded / judged, 3) if judged else None,
        "ungrounded": ungrounded,
        "misses": misses,
    }
    if args.parser != "rules":
        out["model"] = search.parser.model
        out["fallback_rate"] = round(sum(r["parser"] == "rules_fallback" for r in rows) / n, 3)
        out["llm_mode"] = args.llm_mode if args.parser == "llm" else None
        out["llm_calls"] = len(llm_ms)
        out["llm_latency_ms"] = (
            {
                "p50": round(statistics.median(llm_ms), 1),
                "p95": round(sorted(llm_ms)[int(0.95 * (len(llm_ms) - 1))], 1),
            }
            if llm_ms
            else None
        )
        out["cost_per_query_usd"] = round(cost / n, 6)
    set_name = Path(args.queries).stem
    model_tag = f"-{search.parser.model.split('/')[-1]}" if args.parser != "rules" else ""
    if args.parser == "llm" and args.llm_mode == "on_unknown":
        model_tag += "-gated"
    dest = ROOT / "evals/results" / f"{set_name}-{args.parser}{model_tag}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {k: v for k, v in out.items() if k not in ("misses", "per_field_f1", "ungrounded")}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for u in ungrounded:
        print(f"UNGROUNDED {u['id']}: {u['errors']}")
    for m in misses:
        print(f"MISS {m['id']}: gold={m['gold']} pred={m['pred']}")
    return 0 if out["exact_match"] >= args.min_exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
