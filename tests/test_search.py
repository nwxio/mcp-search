from fastapi.testclient import TestClient

from app import analyzer
from app.main import app


client = TestClient(app)


def test_health_ok() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["documents"] >= 1


def test_transactional_query_extracts_price_filter() -> None:
    response = client.post(
        "/search",
        json={"query": "купить айфон до 70000", "top_k": 5, "include_debug": True},
    )
    assert response.status_code == 200

    payload = response.json()
    assert payload["analysis"]["intent"] == "transactional"
    assert payload["analysis"]["filters"]["price_max"] == 70000
    assert payload["analysis"]["filters"]["brand"] == "apple"
    assert payload["analysis"]["filters"]["brands"] == ["apple"]
    assert payload["total"] >= 1

    for row in payload["results"]:
        assert row["price"] <= 70000
        assert row["brand"] == "apple"


def test_category_request_returns_headphones() -> None:
    response = client.post("/search", json={"query": "лучшие bluetooth наушники", "top_k": 5})
    assert response.status_code == 200

    payload = response.json()
    assert payload["analysis"]["filters"]["category"] == "headphones"
    assert payload["total"] >= 1
    assert any(item["category"] == "headphones" for item in payload["results"])


def test_llm_mode_off_uses_rules_only() -> None:
    response = client.post(
        "/search",
        json={"query": "apple iphone", "top_k": 3, "llm_mode": "off"},
    )
    assert response.status_code == 200

    payload = response.json()
    assert payload["analysis"]["source"] == "rules"
    assert payload["analysis"]["llm_used"] is False


def test_llm_mode_force_merges_payload(monkeypatch) -> None:
    fake_payload = {
        "intent": "transactional",
        "rewritten_query": "iphone 14",
        "entities": [{"type": "brand", "value": "apple"}],
        "filters": {
            "price_min": None,
            "price_max": 65000,
            "brand": "apple",
            "brands": ["apple"],
            "category": "smartphone",
            "in_stock": True,
        },
        "confidence": 0.93,
        "comparative": False,
        "subqueries": ["iphone 14", "apple smartphone"],
    }

    monkeypatch.setattr(analyzer, "_request_llm_json", lambda _: fake_payload)

    response = client.post(
        "/search",
        json={"query": "что взять iphone 14 с доставкой", "top_k": 5, "llm_mode": "force"},
    )
    assert response.status_code == 200
    payload = response.json()

    assert payload["analysis"]["source"] == "rules+llm"
    assert payload["analysis"]["llm_used"] is True
    assert payload["analysis"]["filters"]["price_max"] == 65000
    assert payload["analysis"]["rewritten_query"] == "iphone 14"


def test_provider_default_is_deepseek(monkeypatch) -> None:
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setattr(analyzer, "_request_deepseek_json", lambda _: {"intent": "unknown", "rewritten_query": "", "entities": [], "filters": {"price_min": None, "price_max": None, "brand": None, "brands": [], "category": None, "in_stock": None}, "confidence": 0.5, "comparative": False, "subqueries": []})
    monkeypatch.setattr(analyzer, "_request_ollama_json", lambda _: None)

    payload = analyzer._request_llm_json(analyzer._analyze_with_rules("apple iphone"))
    assert payload is not None


def test_provider_ollama_selected(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setattr(analyzer, "_request_deepseek_json", lambda _: None)
    monkeypatch.setattr(analyzer, "_request_ollama_json", lambda _: {"intent": "unknown", "rewritten_query": "", "entities": [], "filters": {"price_min": None, "price_max": None, "brand": None, "brands": [], "category": None, "in_stock": None}, "confidence": 0.5, "comparative": False, "subqueries": []})

    payload = analyzer._request_llm_json(analyzer._analyze_with_rules("apple iphone"))
    assert payload is not None


def test_provider_openai_selected(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setattr(analyzer, "_request_deepseek_json", lambda _: None)
    monkeypatch.setattr(analyzer, "_request_ollama_json", lambda _: None)

    resolved = analyzer._resolve_provider()
    assert resolved == "openai"


def test_compare_query_does_not_pin_single_brand() -> None:
    response = client.post(
        "/search",
        json={"query": "сравни iphone и samsung до 80000", "top_k": 5, "llm_mode": "off"},
    )
    assert response.status_code == 200
    payload = response.json()

    assert payload["analysis"]["comparative"] is True
    assert payload["analysis"]["filters"]["brand"] is None
    brands = payload["analysis"]["filters"]["brands"]
    assert "apple" in brands
    assert "samsung" in brands

    result_brands = [item["brand"] for item in payload["results"]]
    assert "apple" in result_brands
    assert "samsung" in result_brands


def test_compare_query_force_llm_still_keeps_multi_brand(monkeypatch) -> None:
    fake_payload = {
        "intent": "transactional",
        "rewritten_query": "iphone samsung сравнение",
        "entities": [{"type": "brand", "value": "apple"}],
        "filters": {
            "price_min": None,
            "price_max": 80000,
            "brand": "apple",
            "brands": ["apple"],
            "category": "smartphone",
            "in_stock": None,
        },
        "comparative": True,
        "subqueries": ["apple smartphone", "samsung smartphone"],
        "confidence": 0.91,
    }
    monkeypatch.setattr(analyzer, "_request_llm_json", lambda _: fake_payload)

    response = client.post(
        "/search",
        json={"query": "сравни iphone и samsung до 80000", "top_k": 5, "llm_mode": "force"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["analysis"]["comparative"] is True
    assert payload["analysis"]["filters"]["brand"] is None
    assert "samsung" in payload["analysis"]["filters"]["brands"]
    result_brands = [item["brand"] for item in payload["results"]]
    assert "apple" in result_brands
    assert "samsung" in result_brands
