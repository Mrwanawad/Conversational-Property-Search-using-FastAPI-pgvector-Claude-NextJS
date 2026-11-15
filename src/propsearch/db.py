from __future__ import annotations

from datetime import datetime
from functools import lru_cache

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

from propsearch.config import get_settings

EMBED_DIM = 1024


class Base(DeclarativeBase):
    pass


class Developer(Base):
    __tablename__ = "developers"
    id: Mapped[int] = mapped_column(primary_key=True)
    slug: Mapped[str] = mapped_column(String(64), unique=True)
    name_en: Mapped[str] = mapped_column(String(128))
    name_ar: Mapped[str] = mapped_column(String(128))
    on_time_rate: Mapped[float] = mapped_column(Float)


class Compound(Base):
    __tablename__ = "compounds"
    id: Mapped[int] = mapped_column(primary_key=True)
    slug: Mapped[str] = mapped_column(String(64), unique=True)
    name_en: Mapped[str] = mapped_column(String(128))
    name_ar: Mapped[str] = mapped_column(String(128))
    area: Mapped[str] = mapped_column(String(32), index=True)
    developer_id: Mapped[int] = mapped_column(ForeignKey("developers.id"))
    lat: Mapped[float] = mapped_column(Float)
    lng: Mapped[float] = mapped_column(Float)
    amenities: Mapped[list[str]] = mapped_column(ARRAY(Text))
    delivered: Mapped[bool] = mapped_column(Boolean)
    developer: Mapped[Developer] = relationship()
    plans: Mapped[list[PaymentPlan]] = relationship(back_populates="compound", order_by="PaymentPlan.id")


class PaymentPlan(Base):
    __tablename__ = "payment_plans"
    id: Mapped[int] = mapped_column(primary_key=True)
    compound_id: Mapped[int] = mapped_column(ForeignKey("compounds.id"), index=True)
    name: Mapped[str] = mapped_column(String(64))
    down_payment_pct: Mapped[float] = mapped_column(Float)
    years: Mapped[int] = mapped_column(Integer)
    cash_discount_pct: Mapped[float] = mapped_column(Float, default=0.0)
    compound: Mapped[Compound] = relationship(back_populates="plans")


class Unit(Base):
    __tablename__ = "units"
    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(32), unique=True)
    compound_id: Mapped[int] = mapped_column(ForeignKey("compounds.id"), index=True)
    area: Mapped[str] = mapped_column(String(32), index=True)
    unit_type: Mapped[str] = mapped_column(String(16), index=True)
    bedrooms: Mapped[int] = mapped_column(Integer)
    bathrooms: Mapped[int] = mapped_column(Integer)
    bua_m2: Mapped[int] = mapped_column(Integer)
    garden_m2: Mapped[int] = mapped_column(Integer, default=0)
    floor: Mapped[int | None] = mapped_column(Integer, nullable=True)
    finishing: Mapped[str] = mapped_column(String(16))
    price: Mapped[int] = mapped_column(Integer, index=True)
    delivery_year: Mapped[int] = mapped_column(Integer)
    is_ready: Mapped[bool] = mapped_column(Boolean)
    status: Mapped[str] = mapped_column(String(16), index=True)
    amenities: Mapped[list[str]] = mapped_column(ARRAY(Text))
    description_en: Mapped[str] = mapped_column(Text)
    description_ar: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(DateTime)
    embedding = mapped_column(Vector(EMBED_DIM), nullable=True)
    compound: Mapped[Compound] = relationship()


class SearchSession(Base):
    __tablename__ = "search_sessions"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())
    state: Mapped[dict] = mapped_column(JSONB, default=dict)
    simulated: Mapped[bool] = mapped_column(Boolean, default=False)


class QueryLog(Base):
    """One row per user turn: the raw material for monitoring and the /insights dashboard."""

    __tablename__ = "query_log"
    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[str] = mapped_column(String(36), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), index=True)
    query: Mapped[str] = mapped_column(Text)
    language: Mapped[str] = mapped_column(String(8))
    parser: Mapped[str] = mapped_column(String(32))
    intent: Mapped[str] = mapped_column(String(16))
    filters: Mapped[dict] = mapped_column(JSONB)
    soft_preferences: Mapped[str] = mapped_column(Text, default="")
    unknown_terms: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    clarified: Mapped[bool] = mapped_column(Boolean, default=False)
    total_matches: Mapped[int] = mapped_column(Integer, default=0)
    zero_result: Mapped[bool] = mapped_column(Boolean, default=False)
    relaxations: Mapped[list] = mapped_column(JSONB, default=list)
    binding_constraints: Mapped[list] = mapped_column(JSONB, default=list)
    top_unit_ids: Mapped[list[int]] = mapped_column(ARRAY(Integer), default=list)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    timings: Mapped[dict] = mapped_column(JSONB, default=dict)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    simulated: Mapped[bool] = mapped_column(Boolean, default=False)


class Event(Base):
    """Implicit relevance feedback: impressions, clicks and inquiries per shown unit."""

    __tablename__ = "events"
    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    query_log_id: Mapped[int | None] = mapped_column(ForeignKey("query_log.id"), nullable=True, index=True)
    session_id: Mapped[str] = mapped_column(String(36), index=True)
    unit_id: Mapped[int] = mapped_column(ForeignKey("units.id"))
    kind: Mapped[str] = mapped_column(String(16))  # click | inquiry
    position: Mapped[int | None] = mapped_column(Integer, nullable=True)
    simulated: Mapped[bool] = mapped_column(Boolean, default=False)


class Meta(Base):
    __tablename__ = "meta"
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(Text)


class SoftVector(Base):
    """bge-m3 vector of each soft-preference phrase the parsers emit, computed offline."""

    __tablename__ = "soft_vectors"
    phrase: Mapped[str] = mapped_column(Text, primary_key=True)
    embedding = mapped_column(Vector(EMBED_DIM))


class SoftScore(Base):
    """Cross-encoder score of (soft phrase, text) for every unit document and description sentence."""

    __tablename__ = "soft_scores"
    phrase: Mapped[str] = mapped_column(Text, primary_key=True)
    text_sha: Mapped[str] = mapped_column(String(16), primary_key=True)
    score: Mapped[float] = mapped_column(Float)


@lru_cache(maxsize=4)
def get_engine(url: str | None = None) -> Engine:
    return create_engine(url or get_settings().database_url, pool_pre_ping=True, pool_size=5, max_overflow=5)


def session_factory(engine: Engine | None = None) -> sessionmaker[Session]:
    return sessionmaker(bind=engine or get_engine(), expire_on_commit=False)


def init_db(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)
