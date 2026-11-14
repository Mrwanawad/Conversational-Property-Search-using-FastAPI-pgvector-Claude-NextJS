# Bilingual Conversational Property Search

Buyers in Egypt search the way they talk: *"3 غرف في التجمع تحت 8 مليون تقسيط 8 سنين"*, *"3ayez sha2a fel tagamo3 ta7t 7 mlyon"*, *"villa in Zayed, ready to move, garden"*. Filter forms fail on Egyptian Arabic, Franco-Arabic and vague wishes.

This service turns one chat message (Arabic, Franco-Arabic, English or mixed) into validated filters plus soft preferences. It retrieves and ranks matching units, explains why each one fits, asks a question when the query is ambiguous, and relaxes filters with an explanation when nothing matches. Follow-ups like "cheaper", "more" or "with a garden" refine the same session.

Every price, size and unit code in a reply comes from the database. No model ever writes a number.

<p align="center"><img src="docs/demo.gif" width="100%" alt="Demo: an Egyptian Arabic query is parsed into filters, ranked units appear with reasons, a one-tap 'cheaper' refinement, payment plans and an inquiry" /></p>

**Live demo:** https://conversational-property-search.vercel.app (synthetic inventory, not real listings). API: https://propsearch-api.vercel.app/docs

## How a turn works

```mermaid
flowchart LR
    Q[User message] --> N[Normalise<br/>Arabic folding, Franco map,<br/>synonym dictionary]
    N --> U{Understand}
    U -->|PARSER=rules| R[Rule parser]
    U -->|PARSER=anthropic / llm| C[LLM tool call<br/>Claude, Groq, OpenRouter]
    C -.any failure.-> R
    R --> M[Merge with session state<br/>refine / more / reset]
    C --> M
    M --> A{Ambiguous?}
    A -->|yes| CL[Clarifying question]
    A -->|no| S[SQL hard filters<br/>Postgres]
    S --> H[Hybrid rank<br/>BM25 + pgvector, RRF,<br/>business rules, diversity]
    H --> Z{0 results?}
    Z -->|yes| RX[Cheapest relaxation<br/>+ price floor]
    Z -->|no| RE[Grounded reply<br/>reasons per unit]
    RX --> RE
    RE --> L[(Query log, events)]
```

| Stage | Where | What it does |
|---|---|---|
| Normalise | `normalize.py`, `data/synonyms.yaml` | Folds Arabic letter variants, maps Franco-Arabic, tags areas, types, amenities, money and numbers |
| Understand | `understanding/rules.py`, `understanding/llm.py` | Message to `SearchFilters` (Pydantic). Claude path uses tool calling with a strict schema and falls back to rules on any error |
| Session | `pipeline.py` | Stores the parsed state per session, so "cheaper" lowers the budget 10% under the last results' median price |
| Retrieve | `retrieval/sql.py` | Each filter is one SQL predicate. Payment-plan filters must hold on the same plan |
| Rank | `retrieval/ranker.py` | Reciprocal rank fusion of BM25 and pgvector over unit descriptions, then availability and freshness rules and a per-compound diversity cap |
| Relax | `retrieval/relax.py` | Finds the cheapest change that unlocks units (budget +10%, nearby area, sibling type, ...). Reports which constraints are binding |
| Respond | `respond.py` | Template reply in the user's language, reasons per unit, follow-up suggestions. Optional Claude narrative, checked against DB fields |
| Monitor | `insights.py`, `/api/insights` | Zero-result rate, binding constraints, unknown terms, language mix, latency and cost |

## Results

Two labelled sets, scored through the full turn on Postgres with `python scripts/eval.py` (results in `evals/results/`):

- **Dev set** (`evals/queries.jsonl`, 32 queries: 10 Arabic, 8 Franco, 10 English, 4 mixed). Written next to the dictionary, so it is a regression gate (CI fails under 0.9).
- **Held-out set** (`evals/heldout.jsonl`, 24 queries). Phrasings the dictionary was not built for: "Fifth Settlement", "keys now", "25 million tops", "no more than 15% upfront", "تلات أوض", "odten". Labelled before any parser was run on it.

| Parser | Dev exact match | Held-out exact match | LLM calls | LLM latency p50 | Cost / query |
|---|---|---|---|---|---|
| Rules only | **1.00** | 0.71 | 0 | — | $0 |
| Groq `gpt-oss-120b`, every turn | 0.94 | **0.92** | 100% | ~0.9-1.3 s | ~$0.0005-0.0006 |
| Groq `gpt-oss-120b`, gated (deployed) | **1.00** | **0.92** | 3% dev, 42% held-out | ~1.0 s | ~$0.00002-0.00025 |
| Groq `gpt-oss-20b`, every turn | — | 0.83 | 100% | ~1.0 s | ~$0.0003 |

**Gated** means rules first, and the LLM only sees a message when the rules leave a term unread (`LLM_MODE=on_unknown`). It keeps the rules' exact handling of what the dictionary knows, and adds the LLM where the dictionary has a gap. On the held-out set it sent 10 of 24 messages to Groq; on the dev set, 1 of 32. Most turns stay at ~10 ms and the free tier's 8k tokens per minute goes much further.

Notes:
- Field micro F1 for the gated parser: 1.00 dev, 0.98 held-out. Clarification accuracy 1.00 on both.
- Remaining held-out misses: "two-bed" (rules drop it silently, so the gate never fires) and "a pool of its own" read as a shared pool. Both are known gaps, kept in the set on purpose rather than patched into the dictionary.
- `qwen/qwen3.8-27b` was tried and dropped: it called a tool that does not exist on 46% of turns (each one fell back to rules).
- The gpt-oss-20b dev run hit the free-tier quota on every call, so it has no dev number. The every-turn gpt-oss-120b dev run also fell back to rules on 16% of turns (cause not logged in that run); its two misses (dropped area, dropped garden on Franco queries) are model misses.
- Rules-only turn latency is 9 ms p50 / 12 ms p95 (hashing embedder, no reranker, laptop). Zero-result rate after relaxation is 0.031 on dev: a villa under 2M has no match at any relaxation, so the reply gives the price floor.
- When an LLM call fails (rate limit, timeout, invalid tool arguments), that turn uses the rule parser. Search never goes down because of a model.

## Run it

You need Python 3.11+ and Postgres with pgvector.

```bash
docker compose up --build
```

The chat UI is on http://localhost:3000 and the API on http://localhost:8000 (Swagger at `/docs`). The synthetic inventory (14 developers, 45 compounds, 1,492 units) is seeded on first start.

Without Docker:

```bash
pip install -e ".[dev,localdb]"
```

```bash
pytest -q
```

Tests start an embedded Postgres with pgvector (`pgserver`) unless `TEST_DATABASE_URL` is set.

```bash
python scripts/eval.py
```

Example request:

```bash
curl -s localhost:8000/api/search -H "content-type: application/json" -d '{"message": "3ayez sha2a fel tagamo3 ta7t 7 mlyon w mo2adam 10%"}'
```

Send the returned `session_id` with the next message to refine ("cheaper", "more", "with a garden", "new search").

Web UI without Docker (API running on :8000):

```bash
cd frontend
npm ci
npm run dev
```

## Web UI

Next.js chat in `frontend/`. The browser only calls same-origin `/api/*`; Next rewrites it to the FastAPI service (`API_URL`), so there is no CORS setup and the backend address never ships to the client.

- Arabic replies render right-to-left, English left-to-right, chosen per turn from `reply_language`.
- Each turn shows what was understood as filter pills, so a wrong parse is visible at once.
- Unit cards show price, price per m², best payment plan and the match reasons from the API. Details opens payment plans and the developer's on-time record.
- Clarification options and follow-up suggestions are one-tap chips.
- Details and Inquire clicks are posted to `/api/events` as implicit relevance labels.
- `/insights` shows the monitoring view: KPIs, demand vs supply by area, unmet demand, synonym review queue, language mix, latency by stage.

## End-to-end tests

Playwright drives the real stack (UI, API, Postgres) on desktop and mobile viewports: Arabic RTL search, English and Franco-Arabic parsing, clarification, follow-up refinement, impossible queries, unit details and inquiry, and the insights page. 16/16 pass.

```bash
docker compose up -d --build --wait
cd frontend && npx playwright install chromium && npx playwright test
```

The README demo is scripted in `docs/demo/scene.py` (Playwright) so it can be re-recorded when the UI changes.

Point it at a deployment with `E2E_BASE_URL=https://... npx playwright test`. CI runs it on every push after the unit tests and the eval gate.

## Deploy

The live demo runs on free tiers with no card: two Vercel projects and a Neon Postgres (pgvector).

- **API:** Vercel Python function. `api/index.py` wraps the FastAPI app, `vercel.json` routes every path to it, and `requirements.txt` holds the runtime deps. The app sets itself up on the first request because serverless runtimes may skip ASGI lifespan.
- **Database:** Neon via the Vercel marketplace (`vercel integration add neon`), which injects `DATABASE_URL`. Seed it once from any machine: `DATABASE_URL=<neon direct url> python -m propsearch.seed`.
- **Web:** Vercel project with root directory `frontend` and `API_URL=https://<api-project>.vercel.app`.
- **Containers:** the same API image runs anywhere Docker does. `render.yaml` is a Render Blueprint for it, but Render requires a card even on the free plan.

The Playwright suite runs against production too: 16/16 pass on https://conversational-property-search.vercel.app.

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/search` | One chat turn: `{message, session_id?}` returns filters, units, reasons, clarification, relaxations, suggestions |
| GET | `/api/units/{id}` | Unit detail with compound, developer on-time rate and payment plans |
| POST | `/api/events` | Click or inquiry on a shown unit (implicit relevance label). The turn must belong to the session |
| GET | `/api/insights` | Monitoring aggregates over the query log |
| GET | `/api/meta`, `/api/health` | Areas, landmarks, examples; health and active models |

## Configuration

See `.env.example`.

| Variable | Default | Options |
|---|---|---|
| `PARSER` | `rules` | `llm` (any OpenAI-compatible API: Groq, OpenRouter; needs `LLM_API_KEY`), `anthropic` (Claude, needs `ANTHROPIC_API_KEY`) |
| `LLM_BASE_URL` | Groq | e.g. `https://openrouter.ai/api/v1` |
| `LLM_MODEL` | `openai/gpt-oss-120b` | any tool-calling model on that endpoint |
| `LLM_MODE` | `on_unknown` | `always` (LLM on every turn) |
| `EMBEDDER` | `hashing` | `bge-m3` (downloads the model; `pip install -e ".[ml]"`) |
| `RERANKER` | `none` | `bge` (bge-reranker-v2-m3 cross-encoder) |
| `LLM_SUMMARY` | `false` | Adds a Claude narrative under the template reply |
| `LANGFUSE_*` | empty | Traces each turn (understand, retrieve) when set |

## Stack

Python, FastAPI, Pydantic, SQLAlchemy, PostgreSQL + pgvector, rank-bm25, sentence-transformers (bge-m3, bge-reranker-v2-m3), Anthropic SDK (tool calling), camel-tools (optional), Langfuse, Next.js, TypeScript, Playwright, pytest, ruff, Docker, GitHub Actions, Render, Vercel.

## Design choices

- **SQL for hard filters, hybrid search for soft wishes.** Budgets, bedrooms and unit codes are exact; "quiet, good for families" is not. Pure vector search would miss numbers and compound names.
- **Rules first, LLM optional.** The rule parser is free, fast and deterministic, and it covers the labelled set. Claude is used when it's enabled and falls back to rules on any failure, so search never goes down because of a model.
- **No generated facts.** Replies are built from DB fields. The optional narrative is checked against those fields before it's shown.
- **Explain, don't hide, empty results.** Relaxation says what was changed and how many units it unlocked; binding constraints are logged as unmet demand.

## Data

All inventory is synthetic (`data/generate.py`, fixed seed). Developer, compound and unit names are fictional.

## Not built yet

- Scored Claude parser run (built and tested for fallback; no Anthropic API key) and a bge-m3 / reranker retrieval comparison (NDCG@10, recall@20, MRR). The click and inquiry events the UI logs are the intended source of those relevance labels.
- The live demo runs the hashing embedder and no reranker to fit a serverless function. bge-m3 and the cross-encoder need a bigger box.

## License

MIT
