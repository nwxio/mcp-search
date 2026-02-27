from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv

from app.deep_analysis import DeepAnalysis, synthesize_answer
from app.models import (
    DeepAnalysisResult,
    ScrapedPageInfo,
    WebFreshness,
    WebIntent,
    WebSearchRequest,
    WebSearchResponse,
    WebSearchResult,
)
from app.scraper import scrape_pages
from app.web_analyzer import analyze_web_query

load_dotenv()

logger = logging.getLogger(__name__)

HIGH_TRUST_DOMAINS = {
    "reuters.com",
    "apnews.com",
    "bbc.com",
    "dw.com",
    "cnn.com",
    "nytimes.com",
    "washingtonpost.com",
    "theguardian.com",
    "un.org",
    "who.int",
    "nasa.gov",
    "europa.eu",
    "gov.ua",
    "gov",
}

LOW_SIGNAL_PATTERNS = {
    "часть речи",
    "какая часть речи",
    "значение слова",
    "словарь",
    "dictionary",
    "definition",
    "grammar",
}

REGION_DOMAINS: dict[str, set[str]] = {
    "ru": {
        "ria.ru", "tass.ru", "rbc.ru", "rt.com", "sputniknews.com",
        "lenta.ru", "gazeta.ru", "kp.ru", "mk.ru", "rg.ru",
        "vz.ru", "news.ru", "iz.ru", "aif.ru", "argumenti.ru",
        "vedomosti.ru", "kommersant.ru", "fontanka.ru", "kp.ru",
        "russian.rt.com", "russianfedora.com", "roscosmos.ru",
        "mil.ru", "mid.ru", "government.ru", "kremlin.ru",
        "ya.ru", "yandex.ru", "yandex.com", "rambler.ru",
        "mail.ru", "vk.com", "ok.ru", "dzen.ru", "dzen.ru",
        "russian.today", "tvzvezda.ru", "radiosputnik.ria.ru",
        "tass.com", "smotrim.ru", "ren.tv", "ntv.ru", "1tv.ru",
        "5-tv.ru", "tvcentre.ru", "mir24.tv", "russia-24.net",
        "russia-1.net", "russiatoday.ru", "otv24.ru",
    },
    "ua": {
        "pravda.com.ua", "unian.net", "ukrinform.ua", "interfax.com.ua",
        "tsn.ua", "censor.net.ua", "lb.ua", "zn.ua", "nv.ua",
        "hromadske.ua", "radiosvoboda.org", "bbcrussian.com",
        "svoboda.org", "dw.com", "currenttime.tv", "meduza.io",
        "suspilne.media", "5.ua", "24tv.ua", "channels.tv",
        "espreso.tv", "apostrophe.ua", "gordonua.com",
        "korrespondent.net", "portaltele.com.ua", "obozrevatel.com",
        "gazeta.ua", "vikna.if.ua", "volynpost.com",
        "ukrainer.net", "hromadske.ua", "public.nazk.gov.ua",
        "uafuture.net", "dovidka.in.ua", "armyinform.com.ua",
        "fakty.com.ua", "vikna.tv", "rbc.ua", "kyiv.tsn.ua",
        "my.ua", "dialog.ua",
    },
    "by": {
        "belta.by", "sputnik.by", "onliner.by", "tut.by",
        "charter97.org", "naviny.by", "bymedia.net", "zerkalo.io",
    },
    "en": {
        "bbc.com", "cnn.com", "reuters.com", "apnews.com",
        "nytimes.com", "washingtonpost.com", "theguardian.com",
        "wsj.com", "bloomberg.com", "economist.com", "time.com",
        "newsweek.com", "nbcnews.com", "abcnews.go.com", "cbsnews.com",
        "usatoday.com", "latimes.com", "chicagotribune.com",
        "npr.org", "politico.com", "axios.com", "thehill.com",
        "foreignpolicy.com", "foreignaffairs.com", "aljazeera.com",
        "dw.com", "france24.com", "kyivindependent.com",
        "kyivpost.com", "un.org", "nato.int", "europa.eu",
        "state.gov", "whitehouse.gov", "defense.gov",
        "hrw.org", "amnesty.org", "unhcr.org", "who.int",
    },
}


def _get_domain_region(domain: str) -> str | None:
    base = domain.lower().removeprefix("www.")
    
    for region, domains in REGION_DOMAINS.items():
        if base in domains:
            return region
        for d in domains:
            if base.endswith(f".{d}") or d.endswith(f".{base}"):
                return region
    
    if base.endswith(".ru") or base.endswith(".su") or base.endswith(".рф"):
        return "ru"
    if base.endswith(".ua") or base.endswith(".укр"):
        return "ua"
    if base.endswith(".by"):
        return "by"
    
    return None


@dataclass(frozen=True)
class ProviderResult:
    provider: str
    subquery: str
    rank: int
    title: str
    url: str
    snippet: str
    published_at: str | None = None


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[\w\-]+", text.lower())


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _domain(url: str) -> str:
    parsed = urlparse(url)
    return parsed.netloc.lower().removeprefix("www.")


def _canonical_url(url: str) -> str:
    parsed = urlparse(url)
    scheme = parsed.scheme.lower() if parsed.scheme else "https"
    netloc = parsed.netloc.lower().removeprefix("www.")
    netloc = netloc.removeprefix("amp.")
    path = parsed.path or "/"
    if path.startswith("/amp/"):
        path = path[4:]
    if path.endswith("/amp"):
        path = path[:-4] or "/"
    path = path.rstrip("/") or "/"
    return f"{scheme}://{netloc}{path}"


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    value = value.strip()
    if not value:
        return None

    iso_like = value.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(iso_like)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)
    except ValueError:
        pass

    now = datetime.now(UTC)
    lower = value.lower()
    if any(token in lower for token in ["hour", "час", "мин", "minute"]):
        return now
    if any(token in lower for token in ["day", "дн", "сут"]):
        return now
    if any(token in lower for token in ["week", "нед"]):
        return now - timedelta(days=7)
    return None


def _freshness_score(published_at: str | None) -> float:
    dt = _parse_date(published_at)
    if dt is None:
        return 0.35

    age_days = max(0.0, (datetime.now(UTC) - dt).total_seconds() / 86400)
    if age_days <= 2:
        return 1.0
    if age_days <= 7:
        return 0.85
    if age_days <= 30:
        return 0.65
    if age_days <= 180:
        return 0.4
    return 0.2


def _freshness_age_days(published_at: str | None) -> float | None:
    dt = _parse_date(published_at)
    if dt is None:
        return None
    return max(0.0, (datetime.now(UTC) - dt).total_seconds() / 86400)


def _query_match_score(query_tokens: set[str], title: str, snippet: str) -> float:
    if not query_tokens:
        return 0.0
    doc_tokens = set(_tokenize(f"{title} {snippet}"))
    if not doc_tokens:
        return 0.0
    return len(query_tokens.intersection(doc_tokens)) / max(1, len(query_tokens))


def _source_quality(domain: str, title: str, snippet: str) -> float:
    score = 0.35
    base_domain = domain
    if base_domain.startswith("www."):
        base_domain = base_domain[4:]

    if base_domain in HIGH_TRUST_DOMAINS:
        score += 0.4
    if base_domain.endswith(".gov") or base_domain.endswith(".edu"):
        score += 0.25
    if base_domain.endswith(".org"):
        score += 0.12

    text = f"{title} {snippet}".lower()
    if "official" in text or "официаль" in text:
        score += 0.08
    if any(pattern in text for pattern in LOW_SIGNAL_PATTERNS):
        score -= 0.25

    return max(0.0, min(1.0, score))


def _passes_topicality(
    query_match: float,
    intent: str,
    query_token_count: int,
    title: str,
    snippet: str,
    source_quality: float,
    min_query_match: float,
    must_include_terms: list[str],
    search_mode: str,
    require_date_for_news: bool,
    published_at: str | None,
) -> bool:
    if query_match < min_query_match:
        return False
    if query_token_count >= 5 and query_match < max(0.16, min_query_match):
        return False

    text = f"{title} {snippet}".lower()
    if must_include_terms:
        matched = any(term in text for term in must_include_terms)
        if search_mode == "research" and not matched and query_match < 0.35:
            return False

    if any(pattern in text for pattern in LOW_SIGNAL_PATTERNS) and query_match < 0.45:
        return False

    if intent == "news":
        news_markers = {"удар", "атака", "attack", "strike", "drone", "missile", "news", "новости"}
        if query_match < 0.2 and not any(marker in text for marker in news_markers):
            return False
        if require_date_for_news and published_at is None:
            return False

    if search_mode == "research" and source_quality < 0.2 and query_match < 0.45:
        return False
    return True


def _select_diverse_top(
    rows: list[WebSearchResult],
    top_k: int,
    max_per_domain: int,
) -> list[WebSearchResult]:
    if not rows:
        return rows

    selected: list[WebSearchResult] = []
    domain_counts: dict[str, int] = {}

    for row in rows:
        if len(selected) >= top_k:
            break
        count = domain_counts.get(row.domain, 0)
        if count >= max_per_domain:
            continue
        selected.append(row)
        domain_counts[row.domain] = count + 1

    if len(selected) >= top_k:
        return selected

    for row in rows:
        if len(selected) >= top_k:
            break
        if row in selected:
            continue
        selected.append(row)

    return selected[:top_k]


def _enabled_web_providers() -> list[str]:
    raw = os.getenv("WEB_SEARCH_PROVIDERS", "brave,searxng,serpapi")
    names: list[str] = []
    for item in raw.split(","):
        name = item.strip().lower()
        if not name:
            continue
        if name not in names:
            names.append(name)
    logger.debug("Enabled web providers: %s", names)
    return names


def _search_brave(query: str, top_k: int, freshness: WebFreshness) -> list[dict[str, Any]]:
    api_key = os.getenv("BRAVE_SEARCH_API_KEY")
    if not api_key:
        return []

    endpoint = os.getenv("BRAVE_SEARCH_BASE_URL", "https://api.search.brave.com/res/v1/web/search")
    params: dict[str, Any] = {
        "q": query,
        "count": top_k,
    }
    if freshness == WebFreshness.RECENT:
        params["freshness"] = "pw"

    response = httpx.get(
        endpoint,
        params=params,
        headers={
            "X-Subscription-Token": api_key,
            "Accept": "application/json",
        },
        timeout=float(os.getenv("WEB_SEARCH_TIMEOUT_SEC", "8")),
    )
    response.raise_for_status()
    payload = response.json()

    items = payload.get("web", {}).get("results", [])
    if not isinstance(items, list):
        return []
    return items


def _search_serpapi(query: str, top_k: int, freshness: WebFreshness) -> list[dict[str, Any]]:
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        return []

    endpoint = os.getenv("SERPAPI_BASE_URL", "https://serpapi.com/search.json")
    params: dict[str, Any] = {
        "engine": "google",
        "q": query,
        "num": top_k,
        "api_key": api_key,
    }
    if freshness == WebFreshness.RECENT:
        params["tbs"] = "qdr:w"

    response = httpx.get(
        endpoint,
        params=params,
        timeout=float(os.getenv("WEB_SEARCH_TIMEOUT_SEC", "8")),
    )
    response.raise_for_status()
    payload = response.json()
    items = payload.get("organic_results", [])
    if not isinstance(items, list):
        return []
    return items


def _search_searxng(query: str, top_k: int, freshness: WebFreshness, intent: WebIntent = WebIntent.INFORMATIONAL) -> list[dict[str, Any]]:
    base_url = os.getenv("SEARXNG_BASE_URL")
    if not base_url:
        logger.warning("SEARXNG_BASE_URL not set")
        return []

    endpoint = f"{base_url.rstrip('/')}/search"

    if intent == WebIntent.NEWS:
        categories = "news"
    elif intent == WebIntent.RESEARCH:
        categories = "general,news"
    else:
        categories = "general"

    params: dict[str, Any] = {
        "q": query,
        "format": "json",
        "language": "auto",
        "safesearch": 0,
        "categories": categories,
    }
    if freshness == WebFreshness.RECENT:
        params["time_range"] = "month"

    logger.debug("SearXNG request: %s params=%s", endpoint, params)
    response = httpx.get(
        endpoint,
        params=params,
        timeout=float(os.getenv("WEB_SEARCH_TIMEOUT_SEC", "8")),
    )
    response.raise_for_status()
    payload = response.json()
    items = payload.get("results", [])
    logger.debug("SearXNG returned %d items", len(items))
    if not isinstance(items, list):
        return []
    return items[:top_k]


def _normalize_provider_items(provider: str, subquery: str, rows: list[dict[str, Any]]) -> list[ProviderResult]:
    normalized: list[ProviderResult] = []
    for idx, item in enumerate(rows, start=1):
        if not isinstance(item, dict):
            continue

        title = item.get("title") or item.get("name")
        url = item.get("url") or item.get("link")
        snippet = item.get("description") or item.get("snippet") or item.get("content") or ""
        published_at = (
            item.get("publishedDate")
            or item.get("pubdate")
            or item.get("age")
            or item.get("date")
            or item.get("published_at")
        )

        if not isinstance(title, str) or not isinstance(url, str):
            continue
        title = _normalize_whitespace(title)
        url = url.strip()
        snippet = _normalize_whitespace(snippet) if isinstance(snippet, str) else ""
        if not title or not url:
            continue

        normalized.append(
            ProviderResult(
                provider=provider,
                subquery=subquery,
                rank=idx,
                title=title,
                url=url,
                snippet=snippet,
                published_at=published_at if isinstance(published_at, str) else None,
            )
        )
    return normalized


def _query_provider(provider: str, subquery: str, per_query_k: int, freshness: WebFreshness, intent: WebIntent = WebIntent.INFORMATIONAL) -> list[ProviderResult]:
    try:
        if provider == "brave":
            rows = _search_brave(subquery, per_query_k, freshness)
        elif provider == "serpapi":
            rows = _search_serpapi(subquery, per_query_k, freshness)
        elif provider == "searxng":
            rows = _search_searxng(subquery, per_query_k, freshness, intent)
        else:
            return []
        logger.debug("Provider %s returned %d rows for query '%s'", provider, len(rows), subquery)
        return _normalize_provider_items(provider, subquery, rows)
    except Exception as exc:
        logger.warning("Provider %s failed for query '%s': %s", provider, subquery, exc)
        return []


def web_search(request: WebSearchRequest) -> WebSearchResponse:
    analysis = analyze_web_query(request.query, llm_mode=request.llm_mode)
    max_subqueries = 6 if request.search_mode == "research" else 4
    subqueries = (
        analysis.subqueries[:max_subqueries]
        if analysis.subqueries
        else [analysis.rewritten_query or analysis.normalized_query]
    )

    providers = _enabled_web_providers()
    per_query_k = max(10, min(30, request.top_k * 3 if request.search_mode == "research" else request.top_k * 2))

    all_rows: list[ProviderResult] = []
    provider_counts: dict[str, int] = {provider: 0 for provider in providers}

    for subquery in subqueries:
        for provider in providers:
            rows = _query_provider(provider, subquery, per_query_k, analysis.required_freshness, analysis.intent)
            provider_counts[provider] += len(rows)
            all_rows.extend(rows)

    query_tokens = set(_tokenize(analysis.rewritten_query or analysis.normalized_query))

    grouped: dict[str, dict[str, Any]] = {}
    for row in all_rows:
        canonical = _canonical_url(row.url)
        item = grouped.get(canonical)
        if item is None:
            item = {
                "title": row.title,
                "url": row.url,
                "snippet": row.snippet,
                "domain": _domain(row.url),
                "published_at": row.published_at,
                "providers": set(),
                "subqueries": set(),
                "rank_signals": [],
            }
            grouped[canonical] = item

        if len(row.snippet) > len(item["snippet"]):
            item["snippet"] = row.snippet
        if item["published_at"] is None and row.published_at:
            item["published_at"] = row.published_at

        item["providers"].add(row.provider)
        item["subqueries"].add(row.subquery)
        item["rank_signals"].append(1.0 / (row.rank + 3.0))

    results: list[WebSearchResult] = []
    total_providers = max(1, len(providers))
    total_subqueries = max(1, len(subqueries))
    filtered_out = 0

    for _, item in grouped.items():
        rank_signal = sum(item["rank_signals"]) / max(1, len(item["rank_signals"]))
        query_match = _query_match_score(query_tokens, item["title"], item["snippet"])
        source_quality = _source_quality(item["domain"], item["title"], item["snippet"])
        if total_providers == 1:
            provider_agreement = 0.5
        else:
            provider_agreement = len(item["providers"]) / total_providers
        subquery_coverage = len(item["subqueries"]) / total_subqueries
        freshness = _freshness_score(item["published_at"])
        freshness_age = _freshness_age_days(item["published_at"])

        if not _passes_topicality(
            query_match=query_match,
            intent=analysis.intent.value,
            query_token_count=len(query_tokens),
            title=item["title"],
            snippet=item["snippet"],
            source_quality=source_quality,
            min_query_match=request.min_query_match,
            must_include_terms=analysis.must_include_terms,
            search_mode=request.search_mode,
            require_date_for_news=request.require_date_for_news,
            published_at=item["published_at"],
        ):
            filtered_out += 1
            continue

        if request.exclude_regions:
            domain_region = _get_domain_region(item["domain"])
            if domain_region and domain_region in request.exclude_regions:
                filtered_out += 1
                continue

        domain_boost = 0.0
        if analysis.domain_hints and item["domain"] in analysis.domain_hints:
            domain_boost = 0.08

        priority_boost = 0.0
        if request.priority_regions:
            domain_region = _get_domain_region(item["domain"])
            if domain_region and domain_region in request.priority_regions:
                priority_boost = 0.15

        if analysis.required_freshness == WebFreshness.RECENT:
            if request.search_mode == "research":
                score = (
                    0.25 * rank_signal
                    + 0.24 * query_match
                    + 0.08 * provider_agreement
                    + 0.12 * subquery_coverage
                    + 0.17 * freshness
                    + 0.14 * source_quality
                    + domain_boost
                    + priority_boost
                )
            else:
                score = (
                    0.32 * rank_signal
                    + 0.30 * query_match
                    + 0.08 * provider_agreement
                    + 0.12 * subquery_coverage
                    + 0.18 * freshness
                    + domain_boost
                    + priority_boost
                )
            if freshness_age is not None and freshness_age > 180:
                score -= 0.15
        else:
            if request.search_mode == "research":
                score = (
                    0.29 * rank_signal
                    + 0.25 * query_match
                    + 0.08 * provider_agreement
                    + 0.16 * subquery_coverage
                    + 0.07 * freshness
                    + 0.15 * source_quality
                    + domain_boost
                    + priority_boost
                )
            else:
                score = (
                    0.42 * rank_signal
                    + 0.28 * query_match
                    + 0.10 * provider_agreement
                    + 0.15 * subquery_coverage
                    + 0.05 * freshness
                    + domain_boost
                    + priority_boost
                )

        reasoning = [
            f"rank_signal={rank_signal:.2f}",
            f"query_match={query_match:.2f}",
            f"provider_agreement={provider_agreement:.2f}",
            f"subquery_coverage={subquery_coverage:.2f}",
            f"freshness={freshness:.2f}",
            f"source_quality={source_quality:.2f}",
        ]
        if analysis.required_freshness == WebFreshness.RECENT:
            reasoning.append("recent_mode")
        if request.search_mode == "research":
            reasoning.append("research_mode")
        if freshness_age is not None:
            reasoning.append(f"age_days={freshness_age:.0f}")
        if domain_boost > 0:
            reasoning.append("domain_hint_boost")

        results.append(
            WebSearchResult(
                title=item["title"],
                url=item["url"],
                snippet=item["snippet"],
                domain=item["domain"],
                published_at=item["published_at"],
                providers=sorted(item["providers"]),
                source_quality=round(source_quality, 4),
                score=round(score, 4),
                reasoning=reasoning,
            )
        )

    results.sort(key=lambda row: row.score, reverse=True)
    results = _select_diverse_top(results, request.top_k, request.max_per_domain)

    deep_analysis_result: DeepAnalysisResult | None = None
    if request.deep_analysis and results:
        logger.info("Starting deep analysis for %d results", len(results))
        urls_to_scrape = [r.url for r in results[:request.max_scrape_pages]]
        scraped_pages = scrape_pages(urls_to_scrape, max_concurrent=3)

        analysis_obj = synthesize_answer(
            query=request.query,
            pages=scraped_pages,
            intent=analysis.intent,
            max_pages=request.max_scrape_pages,
        )

        page_infos = [
            ScrapedPageInfo(
                url=p.url,
                title=p.title,
                text_preview=p.text[:300] + "..." if len(p.text) > 300 else p.text,
                error=p.error,
            )
            for p in analysis_obj.scraped_pages
        ]

        deep_analysis_result = DeepAnalysisResult(
            summary=analysis_obj.summary,
            answer=analysis_obj.answer,
            key_points=analysis_obj.key_points,
            sources_used=analysis_obj.sources_used,
            confidence=analysis_obj.confidence,
            gaps=analysis_obj.gaps,
            follow_up=analysis_obj.follow_up,
            scraped_pages=page_infos,
            error=analysis_obj.error,
        )
        logger.info("Deep analysis completed with confidence %.2f", analysis_obj.confidence)

    debug = None
    if request.include_debug:
        warnings: list[str] = []
        if len(providers) < 2:
            warnings.append("single_provider_mode: add brave/serpapi for deeper recall")
        if request.llm_mode == "force" and not analysis.llm_used:
            warnings.append("llm_force_failed: analysis fell back to rules")
            if analysis.llm_error:
                warnings.append(f"llm_error:{analysis.llm_error}")
        debug = {
            "providers": providers,
            "provider_result_counts": provider_counts,
            "subqueries": subqueries,
            "raw_count": len(all_rows),
            "dedup_count": len(grouped),
            "filtered_out": filtered_out,
            "search_mode": request.search_mode,
            "max_per_domain": request.max_per_domain,
            "min_query_match": request.min_query_match,
            "warnings": warnings,
        }

    return WebSearchResponse(
        query=request.query,
        analysis=analysis,
        total=len(results),
        results=results,
        deep_analysis=deep_analysis_result,
        debug=debug,
    )
