# Bilingual Conversational Property Search — Architecture

## Scenario
Buyers search like they talk: "3 غرف في التجمع تحت 8 مليون تقسيط 8 سنين" or "villa in Zayed, ready to move, garden". Filter-based search fails on dialect, Franco-Arabic, and vague needs. The system turns natural language into structured filters plus semantic intent, retrieves and ranks matching units, explains why each fits, and supports follow-up refinement in conversation.

## Input
- User query (Arabic, Franco-Arabic, English, mixed) + conversation history.
- Inventory database: units, compounds, prices, payment plans, delivery, amenities, descriptions.

## Output
- Structured filters (JSON): area, type, bedrooms, budget, payment plan, delivery, amenities.
- Ranked units with match reasons and price/plan summary.
- Clarifying question when the query is ambiguous.
- Zero-result fallback: relaxed filters with explanation.

## Architecture

```
 User (chat UI)
        |
        v
 [Normalization]  camel-tools Arabic normalization, Franco-Arabic mapping,
                  area/compound synonym dictionary
        |
        v
 [Query understanding]  Claude tool calling -> Pydantic filter schema
                        + free-text "soft preferences"
        |
        v
 [Retrieval]
   hard filters -> SQL over inventory
   soft prefs   -> hybrid search (pgvector bge-m3 + BM25) over unit descriptions
        |
        v
 [Rerank]  cross-encoder reranker + business rules (availability, freshness)
        |
        v
 [Response]  Claude: grounded summary per unit (only DB fields), follow-up suggestions
        |
        v
 [Session memory]  conversation state for refinements ("cheaper", "closer to AUC")
        |
        v
 [Observability]  Langfuse traces, zero-result log
```

## Tools Used
- LLM: Claude tool calling, Pydantic structured output
- Arabic NLP: camel-tools, custom synonym/transliteration map
- Retrieval: PostgreSQL + pgvector, BM25 (rank_bm25 / OpenSearch), bge-m3, bge-reranker
- Backend: FastAPI, SQLAlchemy
- Frontend: Next.js chat UI (RTL support)
- Evaluation: Ragas, custom labeled query set, ranx (IR metrics)
- Observability: Langfuse; DevOps: Docker, GitHub Actions, pytest, ruff

## Tools Not Used (and why)
- LLM generating unit facts: answers use only DB fields (no hallucinated prices).
- Pure vector search: compound names, unit codes, numbers need lexical + SQL.
- Fine-tuned LLM: tool calling + schema is enough; cheaper to maintain.

## Methodologies
- Query-to-filter extraction with schema validation; hybrid retrieval; reranking.
- Labeled query set (Arabic / Franco / English / mixed) for regression testing.

## Evaluation Metrics
- Filter extraction exact-match and per-field F1.
- Retrieval: NDCG@10, recall@20, MRR.
- Zero-result rate, clarification rate.
- Latency p95, cost per query.
- Business: search-to-inquiry conversion.

## Production Monitoring
- Zero-result and low-confidence queries → synonym dictionary updates.
- Query language mix drift, latency, token cost.
- Click-through and inquiry feedback as implicit relevance labels.

## Deployment Plan
- Backend: Render (FastAPI Docker + Postgres with pgvector).
- Frontend: Vercel (Next.js chat UI).
- Demo inventory: synthetic sample units.
