"""Search-demand analytics for decision makers (sales, acquisition, pricing, product).

Every search is a statement of demand. This module turns the query log into:
- funnel KPIs (zero-result, clarification, click-through, inquiry conversion, latency, cost),
- demand vs supply per area (what buyers ask for vs what inventory can serve),
- unmet demand clusters with the binding constraint and what relaxing it would unlock,
- vocabulary gaps (unknown terms) that feed synonym-dictionary updates.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from datetime import datetime, timedelta

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from propsearch.data.catalog import AREAS
from propsearch.db import Event, QueryLog, Unit
from propsearch.fmt import type_name

BUDGET_BUCKETS = [
    (0, 3e6, "<3M"),
    (3e6, 5e6, "3-5M"),
    (5e6, 8e6, "5-8M"),
    (8e6, 12e6, "8-12M"),
    (12e6, 20e6, "12-20M"),
    (20e6, float("inf"), "20M+"),
]


def _bucket(v: float | None) -> str:
    if not v:
        return "any"
    return next(label for lo, hi, label in BUDGET_BUCKETS if lo <= v < hi)


def _pct(xs: list[float], q: float) -> float:
    return float(np.percentile(xs, q)) if xs else 0.0


def insights(db: Session, days: int = 30, include_simulated: bool = True) -> dict:
    since = datetime.now() - timedelta(days=days)
    q = select(QueryLog).where(QueryLog.created_at >= since)
    if not include_simulated:
        q = q.where(QueryLog.simulated.is_(False))
    rows = db.scalars(q.order_by(QueryLog.created_at)).all()
    ids = [r.id for r in rows]
    events = (
        db.execute(select(Event.query_log_id, Event.kind).where(Event.query_log_id.in_(ids))).all()
        if ids
        else []
    )
    clicks = {qid for qid, k in events if k == "click"}
    inquiries = {qid for qid, k in events if k == "inquiry"}

    searches = [r for r in rows if not r.clarified]
    with_results = [r for r in searches if r.total_matches > 0]
    lat = [r.latency_ms for r in rows]
    kpis = {
        "turns": len(rows),
        "sessions": len({r.session_id for r in rows}),
        "zero_result_rate": round(sum(r.zero_result for r in searches) / len(searches), 3) if searches else 0,
        "clarification_rate": round(sum(r.clarified for r in rows) / len(rows), 3) if rows else 0,
        "ctr": round(len(clicks & {r.id for r in with_results}) / len(with_results), 3)
        if with_results
        else 0,
        "inquiry_rate": round(len(inquiries & {r.id for r in with_results}) / len(with_results), 3)
        if with_results
        else 0,
        "latency_p50_ms": round(_pct(lat, 50), 1),
        "latency_p95_ms": round(_pct(lat, 95), 1),
        "cost_per_query_usd": round(statistics.fmean(r.cost_usd for r in rows), 5) if rows else 0,
        "llm_fallback_rate": round(sum(r.parser == "rules_fallback" for r in rows) / len(rows), 3)
        if rows
        else 0,
    }

    # supply snapshot
    supply_rows = db.execute(
        select(Unit.area, Unit.unit_type, Unit.price).where(Unit.status == "available")
    ).all()
    supply: dict[str, list[int]] = defaultdict(list)
    supply_type: dict[tuple[str, str], list[int]] = defaultdict(list)
    for area, t, price in supply_rows:
        supply[area].append(price)
        supply_type[(area, t)].append(price)

    demand: dict[str, dict] = {
        a: {"searches": 0, "budgets": [], "zero": 0, "inquiries": 0, "clicks": 0} for a in AREAS
    }
    for r in searches:
        for a in r.filters.get("areas", []) or []:
            if a not in demand:
                continue
            d = demand[a]
            d["searches"] += 1
            d["zero"] += int(r.zero_result)
            d["clicks"] += int(r.id in clicks)
            d["inquiries"] += int(r.id in inquiries)
            if r.filters.get("budget_max"):
                d["budgets"].append(r.filters["budget_max"])
    area_rows = []
    for a, d in demand.items():
        prices = supply.get(a, [])
        p25 = _pct(prices, 25)
        area_rows.append(
            {
                "area": a,
                "name_en": AREAS[a].name_en,
                "name_ar": AREAS[a].name_ar,
                "searches": d["searches"],
                "share_of_demand": round(
                    d["searches"] / max(1, sum(x["searches"] for x in demand.values())), 3
                ),
                "available_units": len(prices),
                "share_of_supply": round(len(prices) / max(1, len(supply_rows)), 3),
                "median_budget": int(statistics.median(d["budgets"])) if d["budgets"] else None,
                "median_price": int(statistics.median(prices)) if prices else None,
                "below_p25_share": round(sum(b < p25 for b in d["budgets"]) / len(d["budgets"]), 3)
                if d["budgets"]
                else None,
                "zero_result_rate": round(d["zero"] / d["searches"], 3) if d["searches"] else None,
                "inquiry_rate": round(d["inquiries"] / d["searches"], 3) if d["searches"] else None,
            }
        )
    area_rows.sort(key=lambda x: -x["searches"])

    # unmet demand: zero-result searches grouped by what was asked
    clusters: dict[tuple, dict] = {}
    for r in searches:
        if not r.zero_result:
            continue
        f = r.filters
        key = (
            ",".join(f.get("areas", []) or ["any"]),
            ",".join(f.get("unit_types", []) or ["any"]),
            f.get("bedrooms_min"),
            _bucket(f.get("budget_max")),
        )
        c = clusters.setdefault(
            key, {"count": 0, "binding": Counter(), "unlocks": defaultdict(list), "examples": []}
        )
        c["count"] += 1
        for b in r.binding_constraints or []:
            c["binding"][b["field"]] += 1
            c["unlocks"][b["field"]].append(b["unlocks"])
        if len(c["examples"]) < 3 and r.query not in c["examples"]:
            c["examples"].append(r.query)
    unmet = []
    for (areas, types, beds, bucket), c in sorted(clusters.items(), key=lambda kv: -kv[1]["count"])[:12]:
        top = c["binding"].most_common(1)
        field = top[0][0] if top else None
        unmet.append(
            {
                "areas": areas,
                "area_names": ", ".join(AREAS[a].name_en for a in areas.split(",") if a in AREAS) or "Any",
                "unit_types": types,
                "type_names": ", ".join(type_name(t) for t in types.split(",") if t != "any") or "Any",
                "bedrooms": beds,
                "budget": bucket,
                "searches": c["count"],
                "binding_constraint": field,
                "median_unlock": int(statistics.median(c["unlocks"][field])) if field else 0,
                "examples": c["examples"],
            }
        )

    binding_total = Counter()
    for r in searches:
        for b in r.binding_constraints or []:
            binding_total[b["field"]] += 1

    soft = Counter(s.strip() for r in searches for s in (r.soft_preferences or "").split(";") if s.strip())
    amen = Counter(a for r in searches for a in r.filters.get("amenities", []) or [])
    unknown = Counter(t for r in rows for t in (r.unknown_terms or []))
    lang_mix = Counter(r.language for r in rows)

    daily: dict[str, dict] = {}
    for r in rows:
        day = r.created_at.strftime("%Y-%m-%d")
        d = daily.setdefault(
            day, {"day": day, "turns": 0, "zero": 0, "searches": 0, "ar": 0, "en": 0, "franco": 0, "mixed": 0}
        )
        d["turns"] += 1
        d[r.language] = d.get(r.language, 0) + 1
        if not r.clarified:
            d["searches"] += 1
            d["zero"] += int(r.zero_result)
    for d in daily.values():
        d["zero_result_rate"] = round(d["zero"] / d["searches"], 3) if d["searches"] else 0

    stage_keys = sorted({k for r in rows for k in (r.timings or {}) if "." not in k})
    stages = {
        k: round(_pct([r.timings[k] for r in rows if k in (r.timings or {})], 95), 1) for k in stage_keys
    }

    low_conf = [
        {
            "query": r.query,
            "confidence": r.confidence,
            "unknown_terms": r.unknown_terms,
            "zero_result": r.zero_result,
        }
        for r in reversed(rows)
        if (r.confidence < 0.6 or r.zero_result) and r.unknown_terms
    ][:15]

    simulated = db.execute(
        select(func.count()).select_from(QueryLog).where(QueryLog.simulated.is_(True))
    ).scalar_one()
    return {
        "window_days": days,
        "includes_simulated": include_simulated,
        "simulated_turns": simulated,
        "kpis": kpis,
        "areas": area_rows,
        "unmet_demand": unmet,
        "binding_constraints": [{"field": k, "searches": v} for k, v in binding_total.most_common()],
        "soft_preferences": [{"label": k, "count": v} for k, v in soft.most_common(12)],
        "amenities": [{"label": k, "count": v} for k, v in amen.most_common(10)],
        "unknown_terms": [{"term": k, "count": v} for k, v in unknown.most_common(20)],
        "language_mix": dict(lang_mix),
        "daily": list(daily.values()),
        "latency_p95_by_stage": stages,
        "review_queue": low_conf,
    }
