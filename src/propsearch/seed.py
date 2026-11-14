"""Create the schema, load the synthetic inventory and embed unit descriptions.

python -m propsearch.seed            # load if empty, (re)embed if the embedder changed
python -m propsearch.seed --reset    # drop everything and reload
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

from sqlalchemy import delete, func, select, text
from sqlalchemy.engine import Engine

from propsearch.config import get_settings
from propsearch.data.generate import generate, manifest
from propsearch.db import (
    Base,
    Compound,
    Developer,
    Meta,
    PaymentPlan,
    Unit,
    get_engine,
    init_db,
    session_factory,
)
from propsearch.retrieval.embed import get_embedder


def unit_doc(u: Unit) -> str:
    """Text that is embedded and BM25-indexed: both languages so either query language can match."""
    return f"{u.description_en}\n{u.description_ar}"


def load_inventory(engine: Engine) -> int:
    developers, compounds, units = generate()
    Session = session_factory(engine)
    with Session() as s:
        dev_ids = {}
        for d in developers:
            row = Developer(**d)
            s.add(row)
            s.flush()
            dev_ids[d["slug"]] = row.id
        comp_ids = {}
        for c in compounds:
            row = Compound(
                slug=c.slug,
                name_en=c.name_en,
                name_ar=c.name_ar,
                area=c.area,
                developer_id=dev_ids[c.developer],
                lat=c.lat,
                lng=c.lng,
                amenities=c.amenities,
                delivered=c.delivered,
            )
            s.add(row)
            s.flush()
            comp_ids[c.slug] = row.id
            for p in c.plans:
                s.add(
                    PaymentPlan(
                        compound_id=row.id,
                        name=p.name,
                        down_payment_pct=p.down_payment_pct,
                        years=p.years,
                        cash_discount_pct=p.cash_discount_pct,
                    )
                )
        for u in units:
            s.add(
                Unit(
                    code=u.code,
                    compound_id=comp_ids[u.compound],
                    area=u.area,
                    unit_type=u.unit_type,
                    bedrooms=u.bedrooms,
                    bathrooms=u.bathrooms,
                    bua_m2=u.bua_m2,
                    garden_m2=u.garden_m2,
                    floor=u.floor,
                    finishing=u.finishing,
                    price=u.price,
                    delivery_year=u.delivery_year,
                    is_ready=u.is_ready,
                    status=u.status,
                    amenities=u.amenities,
                    description_en=u.description_en,
                    description_ar=u.description_ar,
                    updated_at=datetime.fromisoformat(u.updated_at),
                )
            )
        s.merge(Meta(key="inventory_manifest", value=json.dumps(manifest(units))))
        s.commit()
    return len(units)


def embed_units(engine: Engine, embedder_name: str, device: str = "cpu") -> int:
    embedder = get_embedder(embedder_name, device)
    Session = session_factory(engine)
    with Session() as s:
        units = s.scalars(select(Unit).order_by(Unit.id)).all()
        batch = 64
        for i in range(0, len(units), batch):
            chunk = units[i : i + batch]
            vecs = embedder.embed([unit_doc(u) for u in chunk])
            for u, v in zip(chunk, vecs, strict=True):
                u.embedding = v.tolist()
        s.merge(Meta(key="embedder", value=embedder.name))
        s.commit()
        return len(units)


def current_embedder(engine: Engine) -> str | None:
    with session_factory(engine)() as s:
        row = s.get(Meta, "embedder")
        return row.value if row else None


def ensure_seeded(engine: Engine, embedder_name: str, device: str = "cpu", reset: bool = False) -> dict:
    if reset:
        Base.metadata.drop_all(engine)
    init_db(engine)
    with engine.connect() as conn:
        n = conn.execute(select(func.count()).select_from(Unit)).scalar_one()
    loaded = 0
    if n == 0:
        loaded = load_inventory(engine)
    embedded = 0
    if loaded or current_embedder(engine) != embedder_name:
        t = time.perf_counter()
        embedded = embed_units(engine, embedder_name, device)
        print(f"embedded {embedded} units with {embedder_name} in {time.perf_counter() - t:.1f}s")
        with engine.begin() as conn:
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS units_embedding_hnsw ON units "
                    "USING hnsw (embedding vector_cosine_ops)"
                )
            )
    return {"loaded": loaded, "embedded": embedded}


def clear_logs(engine: Engine, simulated_only: bool = True) -> None:
    from propsearch.db import Event, QueryLog, SearchSession

    with session_factory(engine)() as s:
        for model in (Event, QueryLog, SearchSession):
            stmt = delete(model)
            if simulated_only:
                stmt = stmt.where(model.simulated.is_(True))
            s.execute(stmt)
        s.commit()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--embedder", default=None)
    args = ap.parse_args()
    st = get_settings()
    print(ensure_seeded(get_engine(), args.embedder or st.embedder, st.embed_device, reset=args.reset))


if __name__ == "__main__":
    main()
