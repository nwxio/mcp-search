from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class Intent(str, Enum):
    INFORMATIONAL = "informational"
    NAVIGATIONAL = "navigational"
    TRANSACTIONAL = "transactional"
    UNKNOWN = "unknown"


class Entity(BaseModel):
    type: str
    value: str


class QueryFilters(BaseModel):
    price_min: int | None = None
    price_max: int | None = None
    brand: str | None = None
    brands: list[str] = Field(default_factory=list)
    category: str | None = None
    in_stock: bool | None = None


class QueryAnalysis(BaseModel):
    original_query: str
    normalized_query: str
    corrected_query: str
    rewritten_query: str
    language: str
    intent: Intent
    entities: list[Entity] = Field(default_factory=list)
    filters: QueryFilters = Field(default_factory=QueryFilters)
    comparative: bool = False
    subqueries: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    source: str = "rules"
    llm_used: bool = False


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=50)
    include_debug: bool = False
    llm_mode: Literal["off", "auto", "force"] = "auto"


class SearchResult(BaseModel):
    id: str
    title: str
    description: str
    brand: str
    category: str
    price: int
    rating: float
    in_stock: bool
    score: float
    reasoning: list[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    query: str
    analysis: QueryAnalysis
    total: int
    results: list[SearchResult]
    debug: dict[str, Any] | None = None


class WebIntent(str, Enum):
    INFORMATIONAL = "informational"
    NAVIGATIONAL = "navigational"
    RESEARCH = "research"
    NEWS = "news"
    TRANSACTIONAL = "transactional"
    UNKNOWN = "unknown"


class WebFreshness(str, Enum):
    ANY = "any"
    RECENT = "recent"


class WebQueryAnalysis(BaseModel):
    original_query: str
    normalized_query: str
    rewritten_query: str
    language: str
    intent: WebIntent
    required_freshness: WebFreshness = WebFreshness.ANY
    domain_hints: list[str] = Field(default_factory=list)
    subqueries: list[str] = Field(default_factory=list)
    research_facets: list[str] = Field(default_factory=list)
    must_include_terms: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    source: str = "rules"
    llm_used: bool = False
    llm_error: str | None = None


class WebSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=50)
    llm_mode: Literal["off", "auto", "force"] = "auto"
    include_debug: bool = False
    search_mode: Literal["research", "balanced"] = "research"
    max_per_domain: int = Field(default=2, ge=1, le=5)
    min_query_match: float = Field(default=0.18, ge=0.0, le=1.0)
    require_date_for_news: bool = True
    deep_analysis: bool = False
    max_scrape_pages: int = Field(default=5, ge=1, le=10)
    exclude_regions: list[str] = Field(default_factory=lambda: ["ru"])
    priority_regions: list[str] = Field(default_factory=lambda: ["ua", "en"])


class ScrapedPageInfo(BaseModel):
    url: str
    title: str
    text_preview: str
    error: str | None = None


class DeepAnalysisResult(BaseModel):
    summary: str
    answer: str
    key_points: list[str] = Field(default_factory=list)
    sources_used: list[int] = Field(default_factory=list)
    confidence: float = 0.0
    gaps: list[str] = Field(default_factory=list)
    follow_up: list[str] = Field(default_factory=list)
    scraped_pages: list[ScrapedPageInfo] = Field(default_factory=list)
    error: str | None = None


class WebSearchResult(BaseModel):
    title: str
    url: str
    snippet: str
    domain: str
    published_at: str | None = None
    providers: list[str] = Field(default_factory=list)
    source_quality: float = 0.0
    score: float
    reasoning: list[str] = Field(default_factory=list)


class WebSearchResponse(BaseModel):
    query: str
    analysis: WebQueryAnalysis
    total: int
    results: list[WebSearchResult]
    deep_analysis: DeepAnalysisResult | None = None
    debug: dict[str, Any] | None = None
