# MCP Internet Search / DeepSearch

`searchnx` is a FastAPI-based search service with two production-style pipelines:

- `POST /search`: hybrid product (e-commerce) search
- `POST /web-search`: universal web search with provider aggregation, ranking, and optional deep analysis

It also includes an MCP server (`mcp_server.py`) dedicated to search workflows, including deep and targeted internet search.

## Built for OpenCode

This platform was actively developed and validated for OpenCode agent workflows.

- OpenCode website: https://opencode.ai
- OpenCode GitHub: https://github.com/anomalyco/opencode
- Typical usage: run Memory-MCP as MCP backend for OpenCode sessions and reusable memory.

## Table of Contents

- [What You Get](#what-you-get)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
  - [`GET /health`](#get-health)
  - [`POST /search`](#post-search)
  - [`POST /web-search`](#post-web-search)
- [LLM Modes and Providers](#llm-modes-and-providers)
- [MCP Server](#mcp-server)
- [Configuration](#configuration)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Roadmap](#roadmap)

## What You Get

### Product Search (`POST /search`)

- Rule-based query analysis with optional LLM enrichment.
- Language detection (`ru`, `en`, `mixed`, `unknown`).
- Spelling correction for common query typos.
- Intent detection: `informational`, `navigational`, `transactional`, `unknown`.
- Entity extraction: brand, category, number.
- Filter extraction:
  - price range (`price_min`, `price_max`)
  - brand (`brand` and `brands`)
  - category
  - in-stock flag
- Comparative query handling (`comparative=true`) with multi-brand support.
- Subquery decomposition for better retrieval.
- Hybrid ranking pipeline:
  - keyword score
  - semantic score (synonym expansion)
  - business score (rating, popularity, stock)
- Intent-aware reranking and brand diversification for comparison queries.

### Web Search (`POST /web-search`)

- Rule-based web query planning with optional LLM enhancement.
- Intent detection: `informational`, `navigational`, `research`, `news`, `transactional`, `unknown`.
- Freshness planning: `any` or `recent`.
- Domain hints extraction (including `site:...` patterns).
- Research facets and must-include term extraction.
- Query decomposition into focused subqueries.
- Multi-provider aggregation: `brave`, `serpapi`, `searxng`.
- URL canonicalization and deduplication (including AMP variants).
- Research-oriented reranking with:
  - rank signal
  - topical query match
  - provider agreement
  - subquery coverage
  - freshness
  - source quality
- Hard topical filters (minimum relevance, low-signal rejection, news date guard).
- Domain diversity control (`max_per_domain`).
- Regional filtering and prioritization:
  - `exclude_regions` (default: `['ru']`)
  - `priority_regions` (default: `['ua', 'en']`)
- Optional deep analysis:
  - scrape top pages
  - synthesize a cited answer with an LLM
  - return key points, gaps, follow-ups, and confidence

## Architecture

### Product pipeline

1. Query analysis (`app/analyzer.py`): normalize, classify, extract entities/filters.
2. Optional LLM merge (`rules+llm`) with strict fallback behavior.
3. Retrieval/ranking (`app/retrieval.py`) over `data/products.json`.
4. Intent-aware rerank + comparison diversification.

### Web pipeline

1. Query planning (`app/web_analyzer.py`) with rules and optional LLM planner.
2. Provider fan-out (`app/web_search.py`) across enabled web search engines.
3. Dedup + scoring fusion + topical filtering.
4. Optional scraping (`app/scraper.py`) and synthesis (`app/deep_analysis.py`).

## Quick Start

### 1) Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set at least one LLM provider and one web provider you want to use.

### 3) Run API server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4) Smoke check

```bash
curl -s http://127.0.0.1:8000/health
```

## API Reference

### `GET /health`

Returns service health and product corpus size.

Example response:

```json
{
  "status": "ok",
  "documents": 15
}
```

### `POST /search`

Product search request model:

- `query` (string, required)
- `top_k` (int, default `10`, range `1..50`)
- `include_debug` (bool, default `false`)
- `llm_mode` (`off|auto|force`, default `auto`)

Example:

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "compare iphone and samsung under 80000",
    "top_k": 5,
    "llm_mode": "auto",
    "include_debug": true
  }'
```

Response includes:

- normalized analysis (intent, entities, filters, subqueries)
- ranked product list with score and reasoning
- optional debug stats (`query_tokens`, corpus/filter/retrieval counts)

### `POST /web-search`

Web search request model:

- `query` (string, required)
- `top_k` (int, default `10`, range `1..50`)
- `llm_mode` (`off|auto|force`, default `auto`)
- `include_debug` (bool, default `false`)
- `search_mode` (`research|balanced`, default `research`)
- `max_per_domain` (int, default `2`, range `1..5`)
- `min_query_match` (float, default `0.18`, range `0..1`)
- `require_date_for_news` (bool, default `true`)
- `deep_analysis` (bool, default `false`)
- `max_scrape_pages` (int, default `5`, range `1..10`)
- `exclude_regions` (list[string], default `['ru']`)
- `priority_regions` (list[string], default `['ua', 'en']`)

Example:

```bash
curl -X POST http://127.0.0.1:8000/web-search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "latest artemis mission news",
    "top_k": 10,
    "llm_mode": "auto",
    "search_mode": "research",
    "max_per_domain": 2,
    "min_query_match": 0.2,
    "require_date_for_news": true,
    "deep_analysis": true,
    "max_scrape_pages": 5,
    "include_debug": true
  }'
```

Response includes:

- query analysis and planning metadata
- deduplicated/ranked web results with domain, score, and reasoning
- optional deep synthesis output (`summary`, `answer`, citations, confidence)
- optional debug payload (provider counts, dedup stats, filtered-out count, warnings)

## LLM Modes and Providers

Both search analyzers support:

- `llm_mode=off`: rules only
- `llm_mode=auto`: use LLM when query complexity/uncertainty is high
- `llm_mode=force`: always try LLM, fallback to rules if provider fails

Supported LLM providers:

- `deepseek`
- `openai` (OpenAI-compatible endpoints)
- `ollama`

If a provider is unavailable or returns invalid output, the service falls back to rules safely.

## MCP Server

`mcp_server.py` exposes two tools:

- `web_search`
- `product_search`

This MCP server is used specifically for search tasks: product lookup and deep internet search through aggregated web providers.

Run it directly:

```bash
source .venv/bin/activate
python mcp_server.py
```

Example OpenCode MCP config (`~/.config/opencode/mcp.json`):

```json
{
  "mcpServers": {
    "searchnx": {
      "command": "/absolute/path/to/searchnx/.venv/bin/python",
      "args": ["/absolute/path/to/searchnx/mcp_server.py"],
      "env": {
        "PYTHONPATH": "/absolute/path/to/searchnx"
      }
    }
  }
}
```

Notes:

- MCP `web_search.top_k` is capped to `1..20` in the MCP layer.
- MCP `product_search.top_k` is capped to `1..50`.

## Configuration

Copy `.env.example` and set values based on your stack.

### Core

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `deepseek` | `deepseek`, `openai`, or `ollama` |
| `LLM_TIMEOUT_SEC` | `20` | Shared timeout for provider requests |
| `LLM_DEBUG` | `0` | Enable verbose LLM debug logging |

### DeepSeek

| Variable | Default |
|---|---|
| `DEEPSEEK_API_KEY` | `""` |
| `DEEPSEEK_MODEL` | `deepseek-chat` |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com/v1` |
| `DEEPSEEK_TIMEOUT_CONNECT_SEC` | `5` |
| `DEEPSEEK_TIMEOUT_READ_SEC` | `20` |
| `DEEPSEEK_MAX_TOKENS` | `220` |
| `DEEPSEEK_RETRIES` | `1` |

### OpenAI-compatible

| Variable | Default |
|---|---|
| `OPENAI_API_KEY` | `""` |
| `OPENAI_MODEL` | `gpt-4o-mini` |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` |
| `OPENAI_TIMEOUT_CONNECT_SEC` | `5` |
| `OPENAI_TIMEOUT_READ_SEC` | `20` |
| `OPENAI_MAX_TOKENS` | `220` |

### Ollama

| Variable | Default |
|---|---|
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` |
| `OLLAMA_MODEL` | `qwen2.5:7b-instruct` |

### Web providers

| Variable | Default | Description |
|---|---|---|
| `WEB_SEARCH_PROVIDERS` | `brave,searxng,serpapi` | Comma-separated provider list |
| `WEB_SEARCH_TIMEOUT_SEC` | `8` | Provider HTTP timeout |
| `BRAVE_SEARCH_API_KEY` | `""` | Required for Brave |
| `BRAVE_SEARCH_BASE_URL` | Brave default URL | Brave endpoint override |
| `SERPAPI_API_KEY` | `""` | Required for SerpAPI |
| `SERPAPI_BASE_URL` | SerpAPI default URL | SerpAPI endpoint override |
| `SEARXNG_BASE_URL` | `""` | Base URL for SearXNG instance |

### Scraper and synthesis

| Variable | Default | Description |
|---|---|---|
| `SCRAPER_TIMEOUT_CONNECT_SEC` | `5` | Scraper connect timeout |
| `SCRAPER_TIMEOUT_READ_SEC` | `10` | Scraper read timeout |
| `SCRAPER_MAX_CHARS` | `8000` | Max extracted chars per page |
| `SYNTHESIS_MAX_TOKENS` | `1500` | Token budget for deep synthesis |

## Testing

Run all tests:

```bash
python3 -m pytest
```

Current test coverage validates:

- product query analysis and ranking behavior
- web provider aggregation, deduplication, filtering, and diversity
- date/freshness logic and region filtering
- JSON repair for LLM outputs
- deep analysis fallbacks and scraper guards

## Project Structure

```text
searchnx/
├── app/
│   ├── main.py            # FastAPI routes
│   ├── models.py          # Pydantic request/response models
│   ├── analyzer.py        # Product query analysis + LLM merge
│   ├── retrieval.py       # Product retrieval and ranking
│   ├── web_analyzer.py    # Web query planning + LLM merge
│   ├── web_search.py      # Provider aggregation and web ranking
│   ├── scraper.py         # HTML scraping and cleanup
│   └── deep_analysis.py   # Multi-source synthesis via LLM
├── data/
│   └── products.json      # Demo product catalog
├── tests/
│   ├── test_search.py
│   └── test_web_search.py
├── mcp_server.py          # MCP tool server
├── .env.example
├── requirements.txt
└── README.md
```

## Roadmap

Planned improvements before production deployment:

1. Replace in-memory JSON product index with a scalable backend (OpenSearch + vector store).
2. Add stronger retrieval fusion (e.g., RRF) and cross-encoder reranking.
3. Introduce ranking metrics (`MRR`, `nDCG@10`, `zero_result_rate`) and click feedback loops.
4. Add A/B controls and feature flags for ranking changes.

## Provider Documentation

- DeepSeek API: https://api-docs.deepseek.com/
- Ollama API: https://docs.ollama.com/api/chat
- OpenAI API: https://platform.openai.com/docs/api-reference
- Brave Search API: https://api.search.brave.com/app/documentation/web-search/get-started
- SerpAPI docs: https://serpapi.com/search-api
- SearXNG API: https://docs.searxng.org/dev/search_api.html
