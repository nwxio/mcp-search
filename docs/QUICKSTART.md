# MCP Quickstart

## 1. Installation

```bash
cd /private/tmp/searchnx
source .venv/bin/activate
pip install mcp
```

## 2. .env Configuration

```bash
# Minimal .env
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-your-key
WEB_SEARCH_PROVIDERS=searxng
SEARXNG_BASE_URL=https://s.netwize.work
```

## 3. Connect to OpenCode

```bash
root@vps:/opt/OpenCODE/mcp/searchnx/.config/opencode# cat opencode.json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "searchnx": {
      "type": "local",
      "command": ["/mcp/searchnx/.venv/bin/python", "/mcp/searchnx/mcp_server.py"],
      "enabled": true,
      "environment": {
        "PYTHONPATH": "/mcp/searchnx"
      }
    }
  }
}
root@vps:/opt/OpenCODE/mcp/searchnx/.config/opencode# cat mcp.json
{
  "mcpServers": {
    "searchnx": {
      "command": "/mcp/searchnx/.venv/bin/python",
      "args": ["/mcp/searchnx/mcp_server.py"],
      "env": {
        "PYTHONPATH": "/mcp/searchnx/",
        "LLM_PROVIDER": "deepseek",
        "LLM_TIMEOUT_SEC": "20",
        "LLM_DEBUG": "0",
        "DEEPSEEK_API_KEY": "sk-xxxx",
        "DEEPSEEK_MODEL": "deepseek-chat",
        "DEEPSEEK_BASE_URL": "https://api.deepseek.com/v1",
        "DEEPSEEK_TIMEOUT_CONNECT_SEC": "5",
        "DEEPSEEK_TIMEOUT_READ_SEC": "45",
        "DEEPSEEK_MAX_TOKENS": "180",
        "DEEPSEEK_RETRIES": "2",
        "WEB_SEARCH_PROVIDERS": "searxng",
        "WEB_SEARCH_TIMEOUT_SEC": "8",
        "SEARXNG_BASE_URL": "https://your.searchnx.resource",
        "SCRAPER_TIMEOUT_CONNECT_SEC": "5",
        "SCRAPER_TIMEOUT_READ_SEC": "12",
        "SCRAPER_MAX_CHARS": "8000",
        "SYNTHESIS_MAX_TOKENS": "1500"
      }
    }
  }
}

(.venv) bash-5.2# opencode mcp add

┌  Add MCP server
│
◇  Enter MCP server name
│  searchnx
│
◇  Select MCP server type
│  Local
│
◇  Enter command to run
│  /mcp/searchnx/.venv/bin/python /mcp/searchnx/mcp_server.py
│
◆  MCP server "searchnx" added to /root/.config/opencode/opencode.json
│
└  MCP server added successfully
```

## 4. Restart OpenCode

```bash
# Restart OpenCode
```

## 5. Usage

After connection, OpenCode will expose these tools:

### web_search
```
Find the latest news about the Artemis mission
What is RAG in machine learning?
```

### product_search
```
Find iPhone models under 70000 rubles
Wireless headphones from Sony or JBL
```

## Verification

```bash
# HTTP API test (server must be running)
uvicorn app.main:app --reload --port 8000 &

curl -X POST http://127.0.0.1:8000/web-search \
  -H 'Content-Type: application/json' \
  -d '{"query":"test","deep_analysis":false}'
```
