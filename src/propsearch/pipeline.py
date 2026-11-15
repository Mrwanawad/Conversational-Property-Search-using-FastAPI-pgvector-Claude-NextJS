"""One conversation turn: normalise -> understand -> (clarify) -> retrieve -> rerank -> relax -> respond -> log."""

from __future__ import annotations

import logging
import statistics
import time
import uuid

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload, sessionmaker

from propsearch import telemetry
from propsearch.config import Settings
from propsearch.db import Compound, QueryLog, SearchSession, Unit
from propsearch.domain import ParsedQuery, SearchTurnResponse
from propsearch.normalize import detect_language
from propsearch.respond import (
    SENT_SPLIT,
    clarification_for,
    llm_narrative,
    suggestions,
    template_message,
    to_hit,
)
from propsearch.retrieval.embed import get_embedder
from propsearch.retrieval.ranker import rank
from propsearch.retrieval.relax import relax
from propsearch.retrieval.rerank import get_reranker
from propsearch.retrieval.sql import min_price
from propsearch.understanding.entities import CompoundMatcher, apply_entities
from propsearch.understanding.rules import RuleParser

log = logging.getLogger(__name__)


def reply_language(lang: str) -> str:
    return "en" if lang == "en" else "ar"


class SearchEngine:
    def __init__(self, settings: Settings, Session: sessionmaker[Session], *, client=None) -> None:
        self.settings = settings
        self.Session = Session
        if settings.embed_device == "precomputed":
            from propsearch.retrieval.precomputed import PrecomputedEmbedder, PrecomputedReranker

            self.embedder = PrecomputedEmbedder(Session)
            self.reranker = PrecomputedReranker(Session) if settings.reranker == "bge" else None
            # sentence scores are stored either way; they pick the quoted evidence sentence
            self.evidence_model = PrecomputedReranker(Session)
        else:
            self.embedder = get_embedder(settings.embedder, settings.embed_device)
            self.reranker = get_reranker(settings.reranker, settings.embed_device)
            self.evidence_model = self.reranker
        with Session() as s:
            comps = [(c.slug, c.name_en, c.name_ar, c.area) for c in s.scalars(select(Compound))]
            self.area_counts = dict(
                s.execute(
                    select(Unit.area, func.count()).where(Unit.status == "available").group_by(Unit.area)
                ).all()
            )
        self.matcher = CompoundMatcher(comps)
        self.rules = RuleParser()
        self.client = client
        if settings.parser == "anthropic":
            from propsearch.understanding.llm import ClaudeParser

            if self.client is None:
                import anthropic

                self.client = anthropic.Anthropic(
                    api_key=settings.anthropic_api_key or None,
                    timeout=settings.anthropic_timeout_s,
                    max_retries=1,
                )
            self.parser = ClaudeParser(
                self.client, settings.anthropic_model, comps, settings.anthropic_effort, self.rules
            )
        elif settings.parser == "llm":
            from propsearch.understanding.llm import GatedParser, OpenAICompatParser

            llm = OpenAICompatParser(
                settings.llm_base_url,
                settings.llm_api_key,
                settings.llm_model,
                comps,
                settings.llm_reasoning_effort,
                settings.llm_timeout_s,
                self.rules,
            )
            self.parser = GatedParser(self.rules, llm) if settings.llm_mode == "on_unknown" else llm
        else:
            self.parser = self.rules

    # ------------------------------------------------------------------ helpers
    def _scorer(self):
        if self.evidence_model is not None:
            return self.evidence_model.score

        def cos(q: str, sents: list[str]) -> np.ndarray:
            vecs = self.embedder.embed([q, *sents])
            return vecs[1:] @ vecs[0]

        return cos

    def _evidence_scorer(self, soft: str, units: list[Unit]):
        """Score every description sentence on the page in one call (one HTTP request on a remote model)."""
        sents = sorted({x for u in units for x in SENT_SPLIT.split(u.description_en)[1:]})
        try:
            scores = dict(zip(sents, self._scorer()(soft, sents), strict=True))
        except Exception as e:
            log.warning("evidence scoring skipped: %r", e)
            return None
        return lambda _q, ss: np.asarray([scores.get(x, 0.0) for x in ss])

    # ------------------------------------------------------------------ turn
    def turn(
        self, message: str, session_id: str | None = None, *, simulated: bool = False
    ) -> SearchTurnResponse:
        t_start = time.perf_counter()
        timings: dict[str, float] = {}
        message = message.strip()[:500]
        with self.Session() as db, telemetry.observe("search_turn", input={"message": message}) as trace:
            sess = db.get(SearchSession, session_id) if session_id else None
            if sess is None:
                sess = SearchSession(id=session_id or str(uuid.uuid4()), state={}, simulated=simulated)
                db.add(sess)
            state = dict(sess.state or {})
            previous = ParsedQuery.model_validate(state["pq"]) if state.get("pq") else None
            context = state.get("context", {})
            language = detect_language(message)
            lang = reply_language(language)

            t = time.perf_counter()
            with telemetry.observe(
                "understand",
                as_type="span" if self.settings.parser == "rules" else "generation",
                input=message,
            ) as obs:
                pq = self.parser.parse(message, previous, context)
                pq = apply_entities(pq, message, self.matcher)
                usage = getattr(self.parser, "last_usage", None)
                obs.update(
                    output=pq.model_dump(mode="json"),
                    **(
                        {
                            "model": usage.model,
                            "usage_details": {"input": usage.input_tokens, "output": usage.output_tokens},
                            "cost_details": {"total": usage.cost_usd},
                        }
                        if usage and usage.model
                        else {}
                    ),
                )
            timings["understand"] = (time.perf_counter() - t) * 1000

            offset = 0
            if pq.intent == "more" and previous:
                pq = previous
                offset = int(state.get("offset", 0)) + self.settings.top_k

            clar = clarification_for(pq, self.area_counts)
            relaxations, binding, total, hits, original = [], [], 0, [], pq.filters
            price_floor = None
            if clar is None:
                t = time.perf_counter()
                with telemetry.observe(
                    "retrieve", as_type="retriever", input=pq.model_dump(mode="json")
                ) as obs:
                    res = rank(db, pq, self.embedder, self.reranker, limit=offset + self.settings.top_k)
                    if res.total == 0:
                        relaxed, relaxations, binding = relax(db, pq.filters)
                        if relaxations:
                            pq_relaxed = pq.model_copy(update={"filters": relaxed})
                            res = rank(
                                db,
                                pq_relaxed,
                                self.embedder,
                                self.reranker,
                                limit=offset + self.settings.top_k,
                            )
                    obs.update(output={"total": res.total, "relaxations": [r.field for r in relaxations]})
                timings.update({f"retrieve.{k}": v for k, v in res.stage_ms.items()})
                timings["retrieve"] = (time.perf_counter() - t) * 1000
                total = res.total
                if total == 0 and pq.filters.budget_max:
                    price_floor = min_price(db, pq.filters)
                page = res.ranked[offset : offset + self.settings.top_k]
                t = time.perf_counter()
                units = {
                    u.id: u
                    for u in db.scalars(
                        select(Unit)
                        .where(Unit.id.in_([s.unit_id for s in page]))
                        .options(
                            selectinload(Unit.compound).selectinload(Compound.plans),
                            selectinload(Unit.compound).selectinload(Compound.developer),
                        )
                    )
                }
                scorer = (
                    self._evidence_scorer(pq.soft_preferences, [units[s.unit_id] for s in page])
                    if pq.soft_preferences and page
                    else None
                )
                eff = pq.model_copy(update={"filters": relaxed}) if relaxations else pq
                hits = [to_hit(units[s.unit_id], s, eff, lang, scorer, pq.filters) for s in page]
                timings["respond"] = (time.perf_counter() - t) * 1000

            input_tokens = getattr(getattr(self.parser, "last_usage", None), "input_tokens", 0) or 0
            output_tokens = getattr(getattr(self.parser, "last_usage", None), "output_tokens", 0) or 0
            cost = getattr(getattr(self.parser, "last_usage", None), "cost_usd", 0.0) or 0.0
            if self.settings.parser == "rules":
                input_tokens = output_tokens = 0
                cost = 0.0

            if clar:
                message_out = clar.question_ar if lang == "ar" else clar.question_en
            else:
                message_out = template_message(pq, total, len(hits), relaxations, lang, original, price_floor)
                if self.settings.llm_summary and self.client is not None and hits:
                    narrative, _ = llm_narrative(
                        self.client, self.settings.anthropic_model, message, hits, lang
                    )
                    if narrative:
                        message_out = f"{message_out}\n\n{narrative}"

            latency = (time.perf_counter() - t_start) * 1000
            ql = QueryLog(
                session_id=sess.id,
                query=message,
                language=language,
                parser=pq.parser,
                intent=pq.intent,
                filters=pq.filters.model_dump(mode="json", exclude_defaults=True),
                soft_preferences=pq.soft_preferences,
                unknown_terms=pq.unknown_terms,
                confidence=pq.confidence,
                clarified=clar is not None,
                total_matches=total,
                zero_result=clar is None and (total == 0 or bool(relaxations)),
                relaxations=[r.model_dump() for r in relaxations],
                binding_constraints=binding,
                top_unit_ids=[h.id for h in hits],
                latency_ms=round(latency, 1),
                timings={k: round(v, 1) for k, v in timings.items()},
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                simulated=simulated,
            )
            db.add(ql)
            prices = [h.price for h in hits]
            sess.state = {
                "pq": pq.model_dump(mode="json"),
                "context": {
                    "median_price": int(statistics.median(prices)) if prices else context.get("median_price"),
                    "median_bua": int(statistics.median(h.bua_m2 for h in hits))
                    if hits
                    else context.get("median_bua"),
                },
                "offset": offset,
                "turns": int(state.get("turns", 0)) + 1,
            }
            db.commit()
            trace.update(
                output={"total": total, "units": [h.code for h in hits], "clarified": clar is not None}
            )

            return SearchTurnResponse(
                session_id=sess.id,
                turn_id=ql.id,
                language=language,
                reply_language=lang,
                parsed=pq,
                message=message_out,
                clarification=clar,
                relaxations=relaxations,
                total_matches=total,
                units=hits,
                suggestions=[] if clar else suggestions(pq, total, offset + len(hits), lang),
                timings_ms={k: round(v, 1) for k, v in {**timings, "total": latency}.items()},
            )
