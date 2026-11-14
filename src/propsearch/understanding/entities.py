"""Inventory-specific entities the static dictionary cannot know: compound names and unit codes."""

from __future__ import annotations

import re

from propsearch.domain import ParsedQuery
from propsearch.normalize import normalize_text

UNIT_CODE = re.compile(r"\b([A-Za-z]{2}-[A-Za-z]{3}-[A-Za-z]\d{3})\b")


class CompoundMatcher:
    def __init__(self, compounds: list[tuple[str, str, str, str]]) -> None:
        """compounds: (slug, name_en, name_ar, area)."""
        self.surfaces: list[tuple[str, str, str]] = []
        for slug, en, ar, area in compounds:
            for s in {en, ar, en.replace(" ", "")}:
                self.surfaces.append((f" {normalize_text(s)} ", slug, area))
        self.surfaces.sort(key=lambda x: -len(x[0]))

    def find(self, text: str) -> list[tuple[str, str]]:
        cleaned = re.sub(r"[^\w\s-]", " ", normalize_text(text))
        t = f" {cleaned} "
        found: dict[str, str] = {}
        for surface, slug, area in self.surfaces:
            if surface in t:
                found[slug] = area
                t = t.replace(surface, " ")
        return list(found.items())


def apply_entities(pq: ParsedQuery, text: str, matcher: CompoundMatcher) -> ParsedQuery:
    f = pq.filters
    updates: dict = {}
    m = UNIT_CODE.search(text)
    if m:
        updates["unit_code"] = m.group(1).upper()
    found = matcher.find(text)
    if found:
        updates["compounds"] = sorted({*f.compounds, *(s for s, _ in found)})
        # a named compound pins the area; drop a conflicting carried-over area
        areas = {a for _, a in found}
        if f.areas and not areas & set(f.areas):
            updates["areas"] = sorted(areas)
    if not updates:
        return pq
    return pq.model_copy(update={"filters": f.model_copy(update=updates)})
