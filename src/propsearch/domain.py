"""Core schema shared by every stage: filters, parsed queries, search results."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from propsearch.data.catalog import AREAS, LANDMARKS


class UnitType(StrEnum):
    apartment = "apartment"
    duplex = "duplex"
    penthouse = "penthouse"
    studio = "studio"
    townhouse = "townhouse"
    twin_house = "twin_house"
    villa = "villa"
    chalet = "chalet"


class Finishing(StrEnum):
    fully_finished = "fully_finished"
    semi_finished = "semi_finished"
    core_shell = "core_shell"


class Amenity(StrEnum):
    # unit level
    garden = "garden"
    roof = "roof"
    private_pool = "private_pool"
    sea_view = "sea_view"
    maids_room = "maids_room"
    # compound level
    swimming_pool = "swimming_pool"
    clubhouse = "clubhouse"
    gym = "gym"
    kids_area = "kids_area"
    crystal_lagoon = "crystal_lagoon"
    beach_access = "beach_access"
    golf = "golf"
    security = "security"
    parking = "parking"


Language = Literal["ar", "en", "franco", "mixed"]
Sort = Literal["relevance", "price_asc", "price_desc", "price_per_m2_asc", "delivery_asc"]


class SearchFilters(BaseModel):
    """Hard constraints. Every field maps to a SQL predicate; None / [] means unconstrained."""

    areas: list[str] = Field(default_factory=list, description="Area slugs")
    compounds: list[str] = Field(default_factory=list, description="Compound slugs")
    unit_types: list[UnitType] = Field(default_factory=list)
    bedrooms_min: int | None = Field(None, ge=0, le=10)
    bedrooms_max: int | None = Field(None, ge=0, le=10)
    budget_min: int | None = Field(None, ge=0, description="EGP")
    budget_max: int | None = Field(None, ge=0, description="EGP")
    bua_min: int | None = Field(None, ge=0, description="Built-up area, m2")
    bua_max: int | None = Field(None, ge=0)
    down_payment_max_pct: float | None = Field(None, ge=0, le=100)
    down_payment_max_amount: int | None = Field(None, ge=0, description="EGP")
    installment_years_min: int | None = Field(None, ge=0, le=15)
    monthly_installment_max: int | None = Field(None, ge=0, description="EGP per month")
    ready_to_move: bool | None = None
    delivery_year_max: int | None = Field(None, ge=2024, le=2040)
    finishing: list[Finishing] = Field(default_factory=list)
    amenities: list[Amenity] = Field(default_factory=list)
    near_landmark: str | None = None
    max_distance_km: float | None = Field(None, gt=0)
    unit_code: str | None = None

    @field_validator("areas")
    @classmethod
    def _known_areas(cls, v: list[str]) -> list[str]:
        unknown = [a for a in v if a not in AREAS]
        if unknown:
            raise ValueError(f"unknown areas: {unknown}")
        return sorted(set(v))

    @field_validator("near_landmark")
    @classmethod
    def _known_landmark(cls, v: str | None) -> str | None:
        if v is not None and v not in LANDMARKS:
            raise ValueError(f"unknown landmark: {v}")
        return v

    @field_validator("unit_types", "finishing", "amenities", "compounds")
    @classmethod
    def _dedupe(cls, v: list) -> list:
        return sorted(set(v))

    def is_empty(self) -> bool:
        return self == SearchFilters()

    def active_fields(self) -> list[str]:
        return [k for k, v in self.model_dump().items() if v not in (None, [], "")]


class Ambiguity(BaseModel):
    field: str
    surface: str
    options: list[str]


class ParsedQuery(BaseModel):
    """Output of query understanding for one user turn, already merged with session state."""

    filters: SearchFilters = Field(default_factory=SearchFilters)
    soft_preferences: str = Field("", description="Free-text wishes that are not hard filters")
    sort: Sort = "relevance"
    intent: Literal["search", "refine", "reset", "more", "out_of_scope"] = "search"
    ambiguities: list[Ambiguity] = Field(default_factory=list)
    parser: str = "rules"
    confidence: float = 1.0
    unknown_terms: list[str] = Field(default_factory=list)


class Reason(BaseModel):
    kind: Literal["match", "near", "info", "relaxed"]
    en: str
    ar: str


class PlanSummary(BaseModel):
    name: str
    down_payment_pct: float
    down_payment: int
    years: int
    monthly: int
    cash_discount_pct: float = 0.0


class UnitHit(BaseModel):
    id: int
    code: str
    title_en: str
    title_ar: str
    compound: str
    compound_ar: str
    developer: str
    area: str
    area_en: str
    area_ar: str
    unit_type: UnitType
    bedrooms: int
    bathrooms: int
    bua_m2: int
    garden_m2: int
    floor: int | None
    finishing: Finishing
    price: int
    price_per_m2: int
    delivery: str
    status: str
    amenities: list[str]
    best_plan: PlanSummary | None
    distance_km: float | None = None
    score: float
    reasons: list[Reason]
    description_en: str
    description_ar: str


class Clarification(BaseModel):
    question_en: str
    question_ar: str
    options: list[dict]  # {"label_en", "label_ar", "value"}


class Relaxation(BaseModel):
    field: str
    from_value: str
    to_value: str
    en: str
    ar: str
    unlocked: int


class SearchTurnResponse(BaseModel):
    session_id: str
    turn_id: int
    language: Language
    reply_language: Literal["ar", "en"]
    parsed: ParsedQuery
    message: str
    clarification: Clarification | None = None
    relaxations: list[Relaxation] = Field(default_factory=list)
    total_matches: int = 0
    units: list[UnitHit] = Field(default_factory=list)
    suggestions: list[dict] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)
