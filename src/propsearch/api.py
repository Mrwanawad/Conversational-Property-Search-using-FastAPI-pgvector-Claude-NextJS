from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import select, text
from sqlalchemy.orm import selectinload

from propsearch import telemetry
from propsearch.config import Settings, get_settings
from propsearch.data.catalog import AREAS, LANDMARKS
from propsearch.db import Compound, Event, Meta, QueryLog, Unit, get_engine, session_factory
from propsearch.domain import SearchTurnResponse
from propsearch.insights import insights
from propsearch.pipeline import SearchEngine
from propsearch.seed import ensure_seeded

log = logging.getLogger(__name__)

EXAMPLES = [
    "3 غرف في التجمع تحت 8 مليون تقسيط 8 سنين",
    "villa in Zayed, ready to move, garden",
    "3ayez sha2a fel tagamo3 ta7t 7 mlyon w mo2adam 10%",
    "شاليه في الساحل فيو بحر ومكان هادي تحت 12 مليون",
    "apartment near AUC, quiet, good for families, monthly under 70k",
    "twin house in October with a garden, delivery 2027",
]


class SearchRequest(BaseModel):
    message: str = Field(min_length=1, max_length=500)
    session_id: str | None = Field(None, max_length=36)


class EventRequest(BaseModel):
    session_id: str = Field(max_length=36)
    turn_id: int | None = None
    unit_id: int
    kind: Literal["click", "inquiry"]
    position: int | None = None


def create_app(settings: Settings | None = None, *, client=None, seed: bool = True) -> FastAPI:
    settings = settings or get_settings()

    lock = threading.Lock()

    def init(app: FastAPI) -> None:
        with lock:
            if getattr(app.state, "engine", None) is not None:
                return
            engine = get_engine(settings.database_url)
            if seed:
                ensure_seeded(engine, settings.embedder, settings.embed_device)
            app.state.Session = session_factory(engine)
            app.state.engine = SearchEngine(settings, app.state.Session, client=client)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        init(app)
        yield
        telemetry.flush()

    app = FastAPI(title="Conversational Property Search", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def ensure_ready(request: Request, call_next):
        # Serverless runtimes (Vercel) may not send ASGI lifespan events; set up on the first request instead.
        if getattr(request.app.state, "engine", None) is None:
            await run_in_threadpool(init, request.app)
        return await call_next(request)

    app.add_middleware(
        CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"]
    )

    @app.get("/api/health")
    def health(request: Request) -> dict:
        with request.app.state.Session() as db:
            db.execute(text("select 1"))
            emb = db.get(Meta, "embedder")
        return {
            "status": "ok",
            "parser": settings.parser,
            "embedder": emb.value if emb else None,
            "reranker": settings.reranker,
            "langfuse": telemetry.enabled(),
        }

    @app.get("/api/meta")
    def meta() -> dict:
        return {
            "areas": [{"slug": a.slug, "name_en": a.name_en, "name_ar": a.name_ar} for a in AREAS.values()],
            "landmarks": [
                {"slug": lm.slug, "name_en": lm.name_en, "name_ar": lm.name_ar} for lm in LANDMARKS.values()
            ],
            "examples": EXAMPLES,
            "parser": settings.parser,
        }

    @app.post("/api/search", response_model=SearchTurnResponse)
    def search(body: SearchRequest, request: Request) -> SearchTurnResponse:
        return request.app.state.engine.turn(body.message, body.session_id)

    @app.post("/api/events", status_code=201)
    def event(body: EventRequest, request: Request) -> dict:
        with request.app.state.Session() as db:
            if db.get(Unit, body.unit_id) is None:
                raise HTTPException(404, "unit not found")
            if body.turn_id is not None:
                ql = db.get(QueryLog, body.turn_id)
                if ql is None or ql.session_id != body.session_id:
                    raise HTTPException(404, "turn not found")
            db.add(
                Event(
                    query_log_id=body.turn_id,
                    session_id=body.session_id,
                    unit_id=body.unit_id,
                    kind=body.kind,
                    position=body.position,
                )
            )
            db.commit()
        return {"ok": True}

    @app.get("/api/units/{unit_id}")
    def unit(unit_id: int, request: Request) -> dict:
        with request.app.state.Session() as db:
            u = db.scalar(
                select(Unit)
                .where(Unit.id == unit_id)
                .options(
                    selectinload(Unit.compound).selectinload(Compound.plans),
                    selectinload(Unit.compound).selectinload(Compound.developer),
                )
            )
            if u is None:
                raise HTTPException(404, "unit not found")
            c = u.compound
            return {
                "id": u.id,
                "code": u.code,
                "unit_type": u.unit_type,
                "bedrooms": u.bedrooms,
                "bathrooms": u.bathrooms,
                "bua_m2": u.bua_m2,
                "garden_m2": u.garden_m2,
                "floor": u.floor,
                "finishing": u.finishing,
                "price": u.price,
                "delivery_year": u.delivery_year,
                "is_ready": u.is_ready,
                "status": u.status,
                "amenities": u.amenities,
                "description_en": u.description_en,
                "description_ar": u.description_ar,
                "compound": {
                    "name_en": c.name_en,
                    "name_ar": c.name_ar,
                    "area": c.area,
                    "lat": c.lat,
                    "lng": c.lng,
                    "developer": c.developer.name_en,
                    "developer_on_time_rate": c.developer.on_time_rate,
                },
                "plans": [
                    {
                        "name": p.name,
                        "down_payment_pct": p.down_payment_pct,
                        "years": p.years,
                        "cash_discount_pct": p.cash_discount_pct,
                    }
                    for p in c.plans
                ],
            }

    @app.get("/api/insights")
    def get_insights(request: Request, days: int = 30, include_simulated: bool = True) -> dict:
        with request.app.state.Session() as db:
            return insights(db, days=max(1, min(days, 365)), include_simulated=include_simulated)

    return app


app = create_app()
