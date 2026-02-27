from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from app.analyzer import analyze_query
from app.models import SearchRequest, SearchResponse, WebSearchRequest, WebSearchResponse
from app.retrieval import HybridSearchEngine
from app.web_search import web_search

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "products.json"

app = FastAPI(title="searchnx", version="0.1.0")
engine = HybridSearchEngine(DATA_PATH)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "documents": engine.corpus_size}


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    analysis = analyze_query(request.query, llm_mode=request.llm_mode)
    results, debug = engine.search(analysis, top_k=request.top_k)

    return SearchResponse(
        query=request.query,
        analysis=analysis,
        total=len(results),
        results=results,
        debug=debug if request.include_debug else None,
    )


@app.post("/web-search", response_model=WebSearchResponse)
def universal_web_search(request: WebSearchRequest) -> WebSearchResponse:
    return web_search(request)
