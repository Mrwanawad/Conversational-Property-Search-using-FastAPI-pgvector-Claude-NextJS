"""Query normalisation: Arabic orthography, Franco-Arabic, digits, and synonym concept tagging.

Arabic normalisation uses camel-tools when it is installed (the `nlp` extra) and an equivalent
built-in implementation otherwise, so the API image stays small. tests/unit/test_normalize.py
checks that both paths agree.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

try:  # pragma: no cover - exercised only when the nlp extra is installed
    from camel_tools.utils.dediac import dediac_ar as _camel_dediac
    from camel_tools.utils.normalize import (
        normalize_alef_ar as _camel_alef,
    )
    from camel_tools.utils.normalize import (
        normalize_alef_maksura_ar as _camel_maksura,
    )
    from camel_tools.utils.normalize import (
        normalize_teh_marbuta_ar as _camel_teh,
    )

    HAS_CAMEL = True
except Exception:
    HAS_CAMEL = False

SYNONYMS_PATH = Path(__file__).parent / "data" / "synonyms.yaml"

_AR_DIACRITICS = re.compile("[ؐ-ًؚ-ٰٟۖ-ۭ]")
_TATWEEL = "ـ"
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_PUNCT = str.maketrans({"،": ",", "؛": ";", "؟": "?", "٪": "%", "٫": "."})
_ARABIC_CHAR = re.compile("[؀-ۿ]")
_LATIN_CHAR = re.compile("[a-zA-Z]")
_ORDINAL = re.compile(r"^\d+(st|nd|rd|th)$")
_REPEAT = re.compile(r"(.)\1{2,}")
_LATIN_DOUBLE = re.compile(r"([a-z])\1")
_TOKEN = re.compile(r"[a-z0-9']*[a-z][a-z0-9']*|\d+(?:[.,]\d+)*|[؀-ۿ]+|%|~|-|/|\+")
# letters glued to a number that are really a unit: "8m", "3br", "10%", "120m2"
_NUM_UNIT = re.compile(r"^(\d+(?:\.\d+)?)(m|mn|mil|k|br|bdr|bed|beds|y|yrs|sqm|m2|million|years|yr)$")
_AR_PREFIXES = ("وبال", "فبال", "بال", "فال", "وال", "لل", "ال", "و", "ب", "ف", "ل", "ك")


def normalize_arabic(text: str) -> str:
    if HAS_CAMEL:
        text = _camel_teh(_camel_maksura(_camel_alef(_camel_dediac(text))))
    else:
        text = _AR_DIACRITICS.sub("", text)
        text = re.sub("[إأآٱ]", "ا", text)
        text = text.replace("ى", "ي").replace("ة", "ه")
    text = text.replace(_TATWEEL, "").replace("ؤ", "و").replace("ئ", "ي")
    return text


def normalize_text(text: str) -> str:
    """Script-aware normalisation applied identically to queries and dictionary surfaces."""
    text = text.translate(_DIGITS).translate(_PUNCT).lower()
    text = normalize_arabic(text)
    text = _REPEAT.sub(r"\1", text)  # "هاااادي" -> "هادي", "pleeease" -> "please"
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)  # 8,500,000 -> 8500000
    return text


def _fold_latin(token: str) -> str:
    """Collapse doubled Latin letters so "villa"/"vila", "sha2aa"/"sha2a" match."""
    return _LATIN_DOUBLE.sub(r"\1", token) if _LATIN_CHAR.search(token) else token


def tokenize(text: str) -> list[str]:
    out: list[str] = []
    for tok in _TOKEN.findall(normalize_text(text)):
        m = _NUM_UNIT.match(tok)
        if m:
            out.extend([m.group(1), m.group(2)])
        else:
            out.append(tok)
    return out


def is_franco_token(tok: str) -> bool:
    """Latin word that uses digits for Arabic sounds: 3ayez, sha2a, ta7t, so5na."""
    letters = sum(c.isalpha() for c in tok)
    digits = [c for c in tok if c.isdigit()]
    return (
        letters >= 2
        and 0 < len(digits) <= 2
        and all(d in "2356789" for d in digits)
        and tok.isascii()
        and not _ORDINAL.match(tok)
        and not _NUM_UNIT.match(tok)
    )


def detect_language(text: str) -> str:
    """ar | en | franco | mixed. Franco = Latin script with Arabic-sound digits (3, 7, 2, 5) inside words."""
    has_ar = bool(_ARABIC_CHAR.search(text))
    latin_tokens = re.findall(r"[a-zA-Z0-9']+", text)
    words = [t.lower() for t in latin_tokens if _LATIN_CHAR.search(t)]
    franco_hits = sum(1 for t in words if is_franco_token(t))
    franco_lex = _franco_lexicon()
    franco_hits += sum(1 for t in words if _fold_latin(t) in franco_lex)
    if has_ar and words:
        return "mixed"
    if has_ar:
        return "ar"
    if franco_hits:
        return "franco"
    return "en"


@dataclass
class Match:
    concepts: tuple[str, ...]
    surface: str
    start: int
    end: int


@dataclass
class TaggedQuery:
    original: str
    normalized: str
    tokens: list[str]
    language: str
    items: list[Match | str] = field(default_factory=list)  # concept matches and leftover tokens, in order

    def concepts(self) -> list[str]:
        return [c for it in self.items if isinstance(it, Match) for c in it.concepts]

    def has(self, concept: str) -> bool:
        return concept in self.concepts()

    @property
    def unknown_terms(self) -> list[str]:
        stop = _stopwords()
        return [
            t
            for t in self.items
            if isinstance(t, str)
            and not _is_number(t)
            and len(t) > 1
            and t not in "%~-/"
            and t not in stop
            and not any(s in stop for s in _strip_prefix(t))
        ]


def _is_number(tok: str) -> bool:
    return bool(re.fullmatch(r"\d+(?:\.\d+)?", tok))


@lru_cache(maxsize=1)
def _load() -> tuple[dict[tuple[str, ...], tuple[str, ...]], int, frozenset[str]]:
    raw = yaml.safe_load(SYNONYMS_PATH.read_text(encoding="utf-8"))
    table: dict[tuple[str, ...], list[str]] = {}
    for section, mapping in raw.items():
        if section == "stopwords":
            continue
        for concept, surfaces in mapping.items():
            for surface in surfaces:
                key = tuple(_fold_latin(t) for t in tokenize(str(surface)))
                if not key:
                    continue
                table.setdefault(key, [])
                if concept not in table[key]:
                    table[key].append(concept)
    stop = frozenset(_fold_latin(t) for s in raw["stopwords"] for t in tokenize(str(s)))
    max_len = max(len(k) for k in table)
    return {k: tuple(v) for k, v in table.items()}, max_len, stop


def _stopwords() -> frozenset[str]:
    return _load()[2]


@lru_cache(maxsize=1)
def _franco_lexicon() -> frozenset[str]:
    """Single Latin tokens from the dictionary that are Franco-Arabic (contain an Arabic-sound digit)."""
    table, _, stop = _load()
    single = {k[0] for k in table if len(k) == 1}
    return frozenset(t for t in single | stop if is_franco_token(t))


def _strip_prefix(token: str) -> list[str]:
    if not _ARABIC_CHAR.search(token):
        return []
    out = []
    for p in _AR_PREFIXES:
        if token.startswith(p) and len(token) - len(p) >= 2:
            rest = token[len(p) :]
            out.append(rest)
            if not rest.startswith("ال"):
                out.append("ال" + rest)
    return out


def tag(text: str) -> TaggedQuery:
    """Greedy longest-match of dictionary surfaces over normalised tokens (with Arabic clitic stripping)."""
    table, max_len, _ = _load()
    tokens = tokenize(text)
    folded = [_fold_latin(t) for t in tokens]
    items: list[Match | str] = []
    i = 0
    while i < len(folded):
        hit: Match | None = None
        for n in range(min(max_len, len(folded) - i), 0, -1):
            window = folded[i : i + n]
            variants = [tuple(window)] + [(alt, *window[1:]) for alt in _strip_prefix(window[0])]
            for key in variants:
                if key in table:
                    hit = Match(table[key], " ".join(tokens[i : i + n]), i, i + n)
                    break
            if hit:
                break
        if hit:
            items.append(hit)
            i = hit.end
        else:
            items.append(folded[i])
            i += 1
    return TaggedQuery(text, normalize_text(text), tokens, detect_language(text), items)
