from datetime import UTC, datetime

from fastapi.testclient import TestClient

from app.main import app
from app.web_analyzer import _extract_json_from_text, _extract_payload_from_freeform_text, analyze_web_query
from app.web_search import ProviderResult


client = TestClient(app)


def test_web_search_aggregates_and_deduplicates(monkeypatch) -> None:
    monkeypatch.setenv("WEB_SEARCH_PROVIDERS", "brave,serpapi")

    def fake_query_provider(provider: str, subquery: str, per_query_k: int, freshness, intent=None):
        if provider == "brave":
            return [
                ProviderResult(
                    provider="brave",
                    subquery=subquery,
                    rank=1,
                    title="NASA Artemis mission update",
                    url="https://www.nasa.gov/artemis/update",
                    snippet="Latest update from NASA on Artemis.",
                    published_at="2026-02-12T10:00:00Z",
                ),
                ProviderResult(
                    provider="brave",
                    subquery=subquery,
                    rank=2,
                    title="ESA mission overview",
                    url="https://www.esa.int/Space/Overview",
                    snippet="Overview of current mission plans.",
                    published_at="2026-01-21",
                ),
            ]

        if provider == "serpapi":
            return [
                ProviderResult(
                    provider="serpapi",
                    subquery=subquery,
                    rank=1,
                    title="Artemis program update | NASA",
                    url="https://nasa.gov/artemis/update?src=google",
                    snippet="NASA program status and milestones.",
                    published_at="2026-02-12",
                )
            ]

        return []

    monkeypatch.setattr("app.web_search._query_provider", fake_query_provider)

    response = client.post(
        "/web-search",
        json={
            "query": "latest artemis mission news",
            "top_k": 5,
            "llm_mode": "off",
            "include_debug": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()

    assert payload["analysis"]["intent"] in {"news", "research"}
    assert payload["analysis"]["required_freshness"] == "recent"
    assert payload["total"] >= 2

    top = payload["results"][0]
    assert top["domain"] == "nasa.gov"
    assert "brave" in top["providers"]
    assert "serpapi" in top["providers"]

    assert payload["debug"]["raw_count"] > payload["debug"]["dedup_count"]


def test_web_search_handles_missing_provider_keys(monkeypatch) -> None:
    monkeypatch.setenv("WEB_SEARCH_PROVIDERS", "brave")
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)

    response = client.post(
        "/web-search",
        json={
            "query": "python asyncio tutorial",
            "top_k": 5,
            "llm_mode": "off",
            "include_debug": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 0
    assert payload["debug"]["provider_result_counts"]["brave"] == 0


def test_web_query_recent_guardrail_kept_when_llm_downgrades(monkeypatch) -> None:
    llm_payload = {
        "intent": "research",
        "rewritten_query": "когда россия последний раз атаковала украину",
        "required_freshness": "any",
        "domain_hints": ["news", "military"],
        "subqueries": ["когда россия последний раз атаковала украину"],
        "confidence": 0.87,
    }
    monkeypatch.setattr("app.web_analyzer._request_llm", lambda _: (llm_payload, None))

    analysis = analyze_web_query(
        "Когда Россия последний аттаковала украину и куда попала",
        llm_mode="force",
    )
    assert analysis.required_freshness.value == "recent"
    assert analysis.intent.value == "news"
    assert analysis.domain_hints == []


def test_web_search_filters_irrelevant_low_match(monkeypatch) -> None:
    monkeypatch.setenv("WEB_SEARCH_PROVIDERS", "searxng")

    def fake_query_provider(provider: str, subquery: str, per_query_k: int, freshness, intent=None):
        return [
            ProviderResult(
                provider=provider,
                subquery=subquery,
                rank=1,
                title="Россия ночью атаковала Украину",
                url="https://example-news.org/attack",
                snippet="Последние новости об ударе по Украине и куда попали дроны.",
                published_at="2026-02-10T00:00:00Z",
            ),
            ProviderResult(
                provider=provider,
                subquery=subquery,
                rank=2,
                title="Слово когда какая часть речи",
                url="https://russkiiyazyk.ru/kogda",
                snippet="Разбор слова когда как части речи.",
                published_at=None,
            ),
        ]

    monkeypatch.setattr("app.web_search._query_provider", fake_query_provider)

    response = client.post(
        "/web-search",
        json={
            "query": "Когда Россия последний аттаковала украину и куда попала",
            "top_k": 5,
            "llm_mode": "off",
            "include_debug": True,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["analysis"]["required_freshness"] == "recent"
    urls = [row["url"] for row in payload["results"]]
    assert "https://russkiiyazyk.ru/kogda" not in urls
    assert payload["debug"]["filtered_out"] >= 1


def test_web_search_respects_max_per_domain(monkeypatch) -> None:
    monkeypatch.setenv("WEB_SEARCH_PROVIDERS", "searxng")

    def fake_query_provider(provider: str, subquery: str, per_query_k: int, freshness, intent=None):
        return [
            ProviderResult(
                provider=provider,
                subquery=subquery,
                rank=1,
                title="Update A",
                url="https://same.example/a",
                snippet="Latest attack update with location details.",
                published_at="2026-02-12T00:00:00Z",
            ),
            ProviderResult(
                provider=provider,
                subquery=subquery,
                rank=2,
                title="Update B",
                url="https://same.example/b",
                snippet="Second report with timeline details.",
                published_at="2026-02-12T05:00:00Z",
            ),
            ProviderResult(
                provider=provider,
                subquery=subquery,
                rank=3,
                title="Update C other source",
                url="https://other.example/c",
                snippet="Independent report with confirmed strike locations.",
                published_at="2026-02-12T03:00:00Z",
            ),
        ]

    monkeypatch.setattr("app.web_search._query_provider", fake_query_provider)

    response = client.post(
        "/web-search",
        json={
            "query": "latest strike locations in ukraine",
            "top_k": 3,
            "llm_mode": "off",
            "max_per_domain": 1,
            "search_mode": "research",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    domains = [row["domain"] for row in payload["results"]]
    assert domains.count("same.example") <= 1


def test_web_search_require_date_for_news(monkeypatch) -> None:
    monkeypatch.setenv("WEB_SEARCH_PROVIDERS", "searxng")

    def fake_query_provider(provider: str, subquery: str, per_query_k: int, freshness, intent=None):
        return [
            ProviderResult(
                provider=provider,
                subquery=subquery,
                rank=1,
                title="Russia attacked Ukraine overnight",
                url="https://dated.example/1",
                snippet="Confirmed strike details with areas hit.",
                published_at="2026-02-11T00:00:00Z",
            ),
            ProviderResult(
                provider=provider,
                subquery=subquery,
                rank=2,
                title="Russia attacked Ukraine latest",
                url="https://nodate.example/2",
                snippet="General overview of attacks and locations.",
                published_at=None,
            ),
        ]

    monkeypatch.setattr("app.web_search._query_provider", fake_query_provider)

    response = client.post(
        "/web-search",
        json={
            "query": "latest russia attack ukraine where hit",
            "top_k": 5,
            "llm_mode": "off",
            "require_date_for_news": True,
            "search_mode": "research",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    urls = [row["url"] for row in payload["results"]]
    assert "https://dated.example/1" in urls
    assert "https://nodate.example/2" not in urls


def test_web_search_dedups_amp_versions(monkeypatch) -> None:
    monkeypatch.setenv("WEB_SEARCH_PROVIDERS", "searxng")

    def fake_query_provider(provider: str, subquery: str, per_query_k: int, freshness, intent=None):
        return [
            ProviderResult(
                provider=provider,
                subquery=subquery,
                rank=1,
                title="Same report",
                url="https://zn.ua/war/sample-report.html",
                snippet="Main article with latest strike data.",
                published_at="2026-02-12T00:00:00Z",
            ),
            ProviderResult(
                provider=provider,
                subquery=subquery,
                rank=2,
                title="Same report AMP",
                url="https://zn.ua/amp/war/sample-report.html",
                snippet="AMP mirror of the same article.",
                published_at="2026-02-12T00:00:00Z",
            ),
        ]

    monkeypatch.setattr("app.web_search._query_provider", fake_query_provider)

    response = client.post(
        "/web-search",
        json={
            "query": "latest strike data",
            "top_k": 5,
            "llm_mode": "off",
            "include_debug": True,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["debug"]["raw_count"] > payload["debug"]["dedup_count"]
    assert payload["total"] == 1


def test_extract_json_from_text_repairs_common_llm_json_issues() -> None:
    raw = """
    Here is result:
    {intent: 'news', rewritten_query: 'abc', required_freshness: 'recent', domain_hints: [],
    research_facets: ['timeline'], must_include_terms: ['abc'], subqueries: ['abc'], confidence: 0.9,}
    """
    parsed = _extract_json_from_text(raw)
    assert parsed is not None
    assert parsed["intent"] == "news"
    assert parsed["required_freshness"] == "recent"


def test_extract_payload_from_freeform_text() -> None:
    raw = """
    intent: news
    required_freshness: recent
    rewritten_query: "russia latest strike ukraine location"
    subqueries: ["russia latest strike ukraine location", "russia strike ukraine where hit"]
    confidence: 0.81
    """
    payload = _extract_payload_from_freeform_text(raw)
    assert payload is not None
    assert payload["intent"] == "news"
    assert payload["required_freshness"] == "recent"
    assert len(payload["subqueries"]) == 2


def test_openai_provider_resolved(monkeypatch) -> None:
    from app.deep_analysis import _get_llm_provider
    from app.web_analyzer import _resolve_provider

    monkeypatch.setenv("LLM_PROVIDER", "openai")

    assert _get_llm_provider() == "openai"
    assert _resolve_provider() == "openai"


def test_deep_analysis_openai_fallback_on_missing_key(monkeypatch) -> None:
    from app.deep_analysis import synthesize_answer
    from app.scraper import ScrapedPage

    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    pages = [ScrapedPage(url="https://example.com", title="Test", text="Some content")]
    result = synthesize_answer("test query", pages)

    assert result.error == "synthesis_failed"


def test_parse_date_week_handles_month_boundary(monkeypatch) -> None:
    from app import web_search as web_search_module

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 3, 2, 10, 30, 0, tzinfo=tz or UTC)

    monkeypatch.setattr(web_search_module, "datetime", FixedDateTime)

    parsed = web_search_module._parse_date("1 week ago")
    assert parsed == datetime(2026, 2, 23, 10, 30, 0, tzinfo=UTC)


def test_deep_analysis_returns_no_sources_on_empty_pages() -> None:
    from app.deep_analysis import synthesize_answer

    result = synthesize_answer("test query", [])
    assert result.error == "no_sources"
    assert result.confidence == 0.0


def test_scrape_page_skips_non_html_extensions() -> None:
    from app.scraper import scrape_page

    result = scrape_page("https://example.com/file.pdf")
    assert result.error == "skipped:unsupported_url"


def test_exclude_regions_filters_russian_sources(monkeypatch) -> None:
    from app.web_search import _get_domain_region, web_search
    from app.models import WebSearchRequest

    assert _get_domain_region("lenta.ru") == "ru"
    assert _get_domain_region("ria.ru") == "ru"
    assert _get_domain_region("pravda.com.ua") == "ua"
    assert _get_domain_region("bbc.com") == "en"

    def fake_query_provider(provider: str, subquery: str, per_query_k: int, freshness, intent=None):
        return [
            ProviderResult(
                provider=provider,
                subquery=subquery,
                rank=1,
                title="Test query Russian source",
                url="https://lenta.ru/news/2026/02/12/test",
                snippet="Test query news from Russian source.",
                published_at="2026-02-12T00:00:00Z",
            ),
            ProviderResult(
                provider=provider,
                subquery=subquery,
                rank=2,
                title="Test query International source",
                url="https://bbc.com/news/world-123",
                snippet="Test query news from international source.",
                published_at="2026-02-12T00:00:00Z",
            ),
            ProviderResult(
                provider=provider,
                subquery=subquery,
                rank=3,
                title="Test query Ukrainian source",
                url="https://pravda.com.ua/news/2026/02/12/test",
                snippet="Test query news from Ukrainian source.",
                published_at="2026-02-12T00:00:00Z",
            ),
        ]

    monkeypatch.setattr("app.web_search._query_provider", fake_query_provider)
    monkeypatch.setenv("WEB_SEARCH_PROVIDERS", "searxng")

    request = WebSearchRequest(
        query="test query",
        top_k=10,
        exclude_regions=["ru"],
        llm_mode="off",
        min_query_match=0.0,
    )

    result = web_search(request)

    domains = [r.domain for r in result.results]
    assert "lenta.ru" not in domains
    assert "bbc.com" in domains
    assert "pravda.com.ua" in domains
