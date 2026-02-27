# searchnx MCP Server

`searchnx` exposes MCP tools so AI clients can run both web search and product search through a standard Model Context Protocol server.

## Features

- `web_search`: multi-provider web search with optional deep analysis.
- `product_search`: hybrid e-commerce search over the product catalog.

## Requirements

- Python 3.11+
- Installed dependencies from `requirements.txt`
- Optional provider credentials in `.env` for LLM/web integrations

## Installation

```bash
cd /mcp/searchnx
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment Setup

```bash
cp .env.example .env
```

Minimal example:

```ini
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your_deepseek_api_key
WEB_SEARCH_PROVIDERS=searxng
SEARXNG_BASE_URL=https://your-searxng-instance.example
```

## Run MCP Server

```bash
source .venv/bin/activate
python mcp_server.py
```

The server runs over stdio and is intended to be launched by an MCP client.

## OpenCode Integration

Create or update `~/.config/opencode/mcp.json`:

```json
{
  "mcpServers": {
    "searchnx": {
      "command": "/mcp/searchnx/.venv/bin/python",
      "args": ["/mcp/searchnx/mcp_server.py"],
      "env": {
        "PYTHONPATH": "/mcp/searchnx"
      }
    }
  }
}
```

Then restart OpenCode.

Quick check:

```bash
opencode mcp list
```

Expected: `searchnx` appears as connected.

## Claude Desktop Integration

Add the same MCP server entry to:

- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

Restart Claude Desktop after editing config.

## Tools

### `web_search`

Searches the web for recent information and can optionally synthesize an answer from scraped pages.

Parameters:

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `query` | string | required | Search query |
| `top_k` | integer | `10` | Clamped to `1..20` in MCP layer |
| `deep_analysis` | boolean | `true` | Scrape pages + synthesize answer |
| `max_scrape_pages` | integer | `5` | Clamped to `1..10` |
| `exclude_regions` | array[string] | `['ru']` | Region filter |
| `priority_regions` | array[string] | `['ua', 'en']` | Ranking priority |

### `product_search`

Searches products with natural language filters.

Parameters:

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `query` | string | required | Product query |
| `top_k` | integer | `10` | Clamped to `1..50` in MCP layer |

## Behavior Notes

- `web_search` MCP handler uses:
  - `search_mode="balanced"`
  - `llm_mode="auto"`
  - `min_query_match=0.05`
  - `require_date_for_news=false`
- `product_search` MCP handler uses `llm_mode="auto"`.
- Tool output is returned as formatted text blocks suitable for chat UIs.

## Validation and Testing

Run backend tests:

```bash
python3 -m pytest
```

Optional API smoke test:

```bash
uvicorn app.main:app --reload --port 8000
```

In another terminal:

```bash
curl -X POST http://127.0.0.1:8000/web-search \
  -H 'Content-Type: application/json' \
  -d '{"query":"latest artemis mission update","deep_analysis":false}'
```

## Troubleshooting

- No MCP connection: verify command/args paths in MCP config.
- Empty web results: verify `WEB_SEARCH_PROVIDERS` and provider credentials.
- LLM fallback behavior: verify `LLM_PROVIDER` and matching API keys.
- Slow/failed scraping: tune `SCRAPER_TIMEOUT_*` and `SCRAPER_MAX_CHARS`.

## Related Documentation

- Main project docs: `README.md`
- Quick setup: `docs/QUICKSTART.md`
