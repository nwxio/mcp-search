from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

import httpx
from dotenv import load_dotenv

from app.models import Entity, Intent, QueryAnalysis, QueryFilters

load_dotenv()


RU_STOPWORDS = {
    "и",
    "в",
    "на",
    "для",
    "с",
    "по",
    "или",
    "а",
    "же",
    "у",
    "о",
    "про",
    "как",
    "что",
    "какой",
    "какие",
    "где",
    "все",
    "самый",
    "самые",
}

EN_STOPWORDS = {
    "and",
    "for",
    "with",
    "about",
    "the",
    "a",
    "to",
    "in",
    "on",
    "of",
}

SPELLING_FIXES = {
    "айфон": "iphone",
    "айфоны": "iphone",
    "самсунг": "samsung",
    "сяоми": "xiaomi",
    "блютуз": "bluetooth",
    "наушникии": "наушники",
    "noutbuk": "ноутбук",
}

BRAND_ALIASES = {
    "apple": {"apple", "iphone", "ipad", "макбук", "macbook", "айфон"},
    "samsung": {"samsung", "самсунг", "galaxy"},
    "xiaomi": {"xiaomi", "сяоми", "redmi", "mi"},
    "sony": {"sony", "сони", "playstation", "ps5"},
    "dyson": {"dyson"},
}

CATEGORY_ALIASES = {
    "smartphone": {"smartphone", "phone", "смартфон", "телефон", "iphone", "galaxy"},
    "laptop": {"laptop", "notebook", "ноутбук", "macbook"},
    "headphones": {"headphones", "наушники", "гарнитура", "bluetooth", "buds"},
    "console": {"console", "консоль", "ps5", "playstation", "xbox"},
    "vacuum": {"vacuum", "пылесос", "dyson"},
}

TRANSACTIONAL_HINTS = {
    "купить",
    "заказать",
    "цена",
    "стоимость",
    "доставка",
    "дешево",
    "скидка",
    "распродажа",
    "buy",
    "price",
    "order",
    "sale",
}

INFORMATIONAL_HINTS = {
    "как",
    "почему",
    "обзор",
    "сравнение",
    "отзывы",
    "лучший",
    "best",
    "review",
    "compare",
    "vs",
}

NAVIGATIONAL_HINTS = {
    "официальный",
    "оригинал",
    "exact",
    "model",
    "артикул",
    "sku",
}

COMPLEXITY_MARKERS = {
    "или",
    "либо",
    "против",
    "vs",
    "compare",
    "сравни",
    "лучше",
    "what",
    "which",
}

NUMBER_PATTERN = re.compile(r"\d+[\s_]?\d*")

LLM_ANALYSIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "intent": {
            "type": "string",
            "enum": ["informational", "navigational", "transactional", "unknown"],
        },
        "rewritten_query": {"type": "string"},
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["type", "value"],
                "additionalProperties": False,
            },
        },
        "filters": {
            "type": "object",
            "properties": {
                "price_min": {"type": ["integer", "null"]},
                "price_max": {"type": ["integer", "null"]},
                "brand": {"type": ["string", "null"]},
                "brands": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "category": {"type": ["string", "null"]},
                "in_stock": {"type": ["boolean", "null"]},
            },
            "required": ["price_min", "price_max", "brand", "brands", "category", "in_stock"],
            "additionalProperties": False,
        },
        "comparative": {"type": "boolean"},
        "subqueries": {
            "type": "array",
            "items": {"type": "string"},
        },
        "confidence": {"type": "number"},
    },
    "required": [
        "intent",
        "rewritten_query",
        "entities",
        "filters",
        "comparative",
        "subqueries",
        "confidence",
    ],
    "additionalProperties": False,
}

LLM_SYSTEM_PROMPT = (
    "You are a query analysis engine for e-commerce search. "
    "Return strict JSON only, matching the provided schema. "
    "Infer user intent, normalized rewrite, entities and filters. "
    "If query compares multiple brands/models, set comparative=true, put all compared brands in filters.brands, "
    "and keep filters.brand as null. Build 2-4 short subqueries for decomposition. "
    "Do not hallucinate unsupported values."
)

SUPPORTED_LLM_PROVIDERS = {"deepseek", "ollama", "openai"}


@dataclass(frozen=True)
class ParsedNumeric:
    value: int
    raw: str


def _detect_language(query: str) -> str:
    cyrillic = sum(1 for ch in query if "а" <= ch.lower() <= "я" or ch.lower() == "ё")
    latin = sum(1 for ch in query if "a" <= ch.lower() <= "z")
    if cyrillic > 0 and latin > 0:
        return "mixed"
    if cyrillic > latin:
        return "ru"
    if latin > 0:
        return "en"
    return "unknown"


def _normalize(query: str) -> str:
    query = query.strip().lower()
    query = re.sub(r"\s+", " ", query)
    return query


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[\w\-]+", text.lower())


def _correct_spelling(tokens: list[str]) -> list[str]:
    return [SPELLING_FIXES.get(token, token) for token in tokens]


def _extract_number(num_text: str) -> ParsedNumeric | None:
    cleaned = num_text.replace(" ", "").replace("_", "")
    if not cleaned.isdigit():
        return None
    value = int(cleaned)
    return ParsedNumeric(value=value, raw=num_text)


def _extract_price_filters(query: str) -> QueryFilters:
    filters = QueryFilters()

    max_patterns = [
        r"до\s+(\d+[\s_]?\d*)",
        r"дешевле\s+(\d+[\s_]?\d*)",
        r"не\s+дороже\s+(\d+[\s_]?\d*)",
        r"under\s+(\d+[\s_]?\d*)",
        r"below\s+(\d+[\s_]?\d*)",
    ]
    min_patterns = [
        r"от\s+(\d+[\s_]?\d*)",
        r"дороже\s+(\d+[\s_]?\d*)",
        r"above\s+(\d+[\s_]?\d*)",
        r"over\s+(\d+[\s_]?\d*)",
    ]

    for pattern in max_patterns:
        match = re.search(pattern, query)
        if not match:
            continue
        num = _extract_number(match.group(1))
        if num:
            filters.price_max = num.value
            break

    for pattern in min_patterns:
        match = re.search(pattern, query)
        if not match:
            continue
        num = _extract_number(match.group(1))
        if num:
            filters.price_min = num.value
            break

    if "в наличии" in query or "in stock" in query:
        filters.in_stock = True

    return filters


def _extract_brands(tokens: list[str]) -> list[str]:
    token_set = set(tokens)
    brands: list[str] = []
    for brand, aliases in BRAND_ALIASES.items():
        if token_set.intersection(aliases):
            brands.append(brand)
    return brands


def _extract_category(tokens: list[str]) -> str | None:
    token_set = set(tokens)
    for category, aliases in CATEGORY_ALIASES.items():
        if token_set.intersection(aliases):
            return category
    return None


def _is_comparative_query(corrected_query: str, brands: list[str]) -> bool:
    if len(brands) >= 2:
        return True
    return any(marker in corrected_query for marker in COMPLEXITY_MARKERS)


def _build_subqueries(corrected_query: str, brands: list[str], category: str | None) -> list[str]:
    subqueries: list[str] = []
    if len(brands) >= 2:
        for brand in brands[:3]:
            parts = [brand]
            if category:
                parts.append(category)
            subqueries.append(" ".join(parts))
        subqueries.append(f"{' vs '.join(brands[:2])} сравнение")
        return subqueries[:4]

    tokens = _tokenize(corrected_query)
    if len(tokens) > 4:
        subqueries.append(" ".join(tokens[:3]))
        subqueries.append(" ".join(tokens[-3:]))
    return [query for query in subqueries if query]


def _classify_intent(tokens: list[str], filters: QueryFilters) -> Intent:
    token_set = set(tokens)
    if filters.price_min is not None or filters.price_max is not None:
        return Intent.TRANSACTIONAL

    if token_set.intersection(TRANSACTIONAL_HINTS):
        return Intent.TRANSACTIONAL
    if token_set.intersection(INFORMATIONAL_HINTS):
        return Intent.INFORMATIONAL
    if token_set.intersection(NAVIGATIONAL_HINTS):
        return Intent.NAVIGATIONAL

    if len(tokens) <= 2:
        return Intent.NAVIGATIONAL
    return Intent.UNKNOWN


def _build_rewrite(tokens: list[str], language: str) -> str:
    stopwords = RU_STOPWORDS if language == "ru" else EN_STOPWORDS
    filtered = [t for t in tokens if t not in stopwords and not t.isdigit()]
    if not filtered:
        return " ".join(tokens)
    return " ".join(filtered)


def _confidence(intent: Intent, entities: list[Entity], rewritten_query: str) -> float:
    score = 0.35
    if intent != Intent.UNKNOWN:
        score += 0.25
    if entities:
        score += min(0.25, 0.08 * len(entities))
    if rewritten_query:
        score += 0.10
    return round(min(0.99, score), 2)


def _analyze_with_rules(query: str) -> QueryAnalysis:
    normalized = _normalize(query)
    language = _detect_language(normalized)
    raw_tokens = _tokenize(normalized)
    corrected_tokens = _correct_spelling(raw_tokens)
    corrected_query = " ".join(corrected_tokens)

    filters = _extract_price_filters(corrected_query)

    brands = _extract_brands(corrected_tokens)
    category = _extract_category(corrected_tokens)
    if len(brands) == 1:
        filters.brand = brands[0]
    filters.brands = brands
    if category:
        filters.category = category

    entities: list[Entity] = []
    for brand in brands:
        entities.append(Entity(type="brand", value=brand))
    if category:
        entities.append(Entity(type="category", value=category))

    price_numbers = [_extract_number(match.group(0)) for match in NUMBER_PATTERN.finditer(corrected_query)]
    for parsed in price_numbers:
        if parsed is None:
            continue
        if parsed.value >= 1000:
            entities.append(Entity(type="number", value=str(parsed.value)))

    intent = _classify_intent(corrected_tokens, filters)
    rewritten = _build_rewrite(corrected_tokens, language)
    comparative = _is_comparative_query(corrected_query, brands)
    if comparative and intent == Intent.UNKNOWN:
        intent = Intent.INFORMATIONAL
    subqueries = _build_subqueries(corrected_query, brands, category)

    return QueryAnalysis(
        original_query=query,
        normalized_query=normalized,
        corrected_query=corrected_query,
        rewritten_query=rewritten,
        language=language,
        intent=intent,
        entities=entities,
        filters=filters,
        comparative=comparative,
        subqueries=subqueries,
        confidence=_confidence(intent, entities, rewritten),
        source="rules",
        llm_used=False,
    )


def _should_use_llm(analysis: QueryAnalysis) -> bool:
    tokens = _tokenize(analysis.corrected_query)
    if analysis.intent == Intent.UNKNOWN:
        return True
    if analysis.confidence < 0.72:
        return True
    if len(tokens) >= 7:
        return True
    if any(marker in analysis.corrected_query for marker in COMPLEXITY_MARKERS):
        return True
    return False


def _extract_output_text(payload: dict[str, Any]) -> str | None:
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    output = payload.get("output")
    if not isinstance(output, list):
        return None

    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                return text
    return None


def _extract_json_from_text(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        left = text.find("{")
        right = text.rfind("}")
        if left < 0 or right < 0 or right <= left:
            return None
        try:
            parsed = json.loads(text[left : right + 1])
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None


def _build_llm_user_payload(analysis: QueryAnalysis) -> dict[str, Any]:
    return {
        "query": analysis.original_query,
        "normalized_query": analysis.normalized_query,
        "corrected_query": analysis.corrected_query,
        "language": analysis.language,
        "rules_hint": {
            "intent": analysis.intent.value,
            "entities": [entity.model_dump() for entity in analysis.entities],
            "filters": analysis.filters.model_dump(),
            "comparative": analysis.comparative,
            "subqueries": analysis.subqueries,
        },
    }


def _resolve_provider() -> str:
    provider = os.getenv("LLM_PROVIDER", "deepseek").strip().lower()
    if provider not in SUPPORTED_LLM_PROVIDERS:
        return "deepseek"
    return provider


def _request_deepseek_json(analysis: QueryAnalysis) -> dict[str, Any] | None:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return None

    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    timeout_sec = float(os.getenv("LLM_TIMEOUT_SEC", "8"))
    user_payload = _build_llm_user_payload(analysis)

    request_body: dict[str, Any] = {
        "model": model,
        "temperature": 0,
        "max_tokens": 350,
        "messages": [
            {
                "role": "system",
                "content": LLM_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": json.dumps(user_payload, ensure_ascii=False),
            },
        ],
        "response_format": {"type": "json_object"},
    }

    try:
        response = httpx.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=request_body,
            timeout=timeout_sec,
        )
        response.raise_for_status()
        payload = response.json()
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        message = choices[0].get("message")
        if not isinstance(message, dict):
            return None
        text = message.get("content")
        if not isinstance(text, str):
            return None
        parsed = _extract_json_from_text(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _request_ollama_json(analysis: QueryAnalysis) -> dict[str, Any] | None:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
    timeout_sec = float(os.getenv("LLM_TIMEOUT_SEC", "8"))
    user_payload = _build_llm_user_payload(analysis)

    request_body = {
        "model": model,
        "stream": False,
        "format": "json",
        "messages": [
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        "options": {
            "temperature": 0,
        },
    }

    try:
        response = httpx.post(
            f"{base_url}/api/chat",
            headers={"Content-Type": "application/json"},
            json=request_body,
            timeout=timeout_sec,
        )
        response.raise_for_status()
        payload = response.json()
        message = payload.get("message")
        if not isinstance(message, dict):
            return None
        text = message.get("content")
        if not isinstance(text, str):
            return None
        parsed = _extract_json_from_text(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _request_openai_json(analysis: QueryAnalysis) -> dict[str, Any] | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    timeout_sec = float(os.getenv("LLM_TIMEOUT_SEC", "20"))
    user_payload = _build_llm_user_payload(analysis)

    request_body: dict[str, Any] = {
        "model": model,
        "temperature": 0,
        "max_tokens": 350,
        "stream": False,
        "messages": [
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
    }

    try:
        response = httpx.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=request_body,
            timeout=timeout_sec,
        )
        response.raise_for_status()
        payload = response.json()
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        message = choices[0].get("message")
        if not isinstance(message, dict):
            return None
        text = message.get("content")
        if not isinstance(text, str):
            return None
        parsed = _extract_json_from_text(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _request_llm_json(analysis: QueryAnalysis) -> dict[str, Any] | None:
    provider = _resolve_provider()
    if provider == "ollama":
        return _request_ollama_json(analysis)
    if provider == "openai":
        return _request_openai_json(analysis)
    return _request_deepseek_json(analysis)


def _parse_intent(value: Any) -> Intent | None:
    if not isinstance(value, str):
        return None
    try:
        return Intent(value)
    except ValueError:
        return None


def _normalize_entity_payload(raw_entities: Any) -> list[Entity]:
    if not isinstance(raw_entities, list):
        return []

    entities: list[Entity] = []
    for item in raw_entities:
        if not isinstance(item, dict):
            continue
        entity_type = item.get("type")
        entity_value = item.get("value")
        if not isinstance(entity_type, str) or not isinstance(entity_value, str):
            continue
        entity_type = entity_type.strip().lower()
        entity_value = entity_value.strip().lower()
        if not entity_type or not entity_value:
            continue
        entities.append(Entity(type=entity_type, value=entity_value))
    return entities


def _merge_entities(*entity_lists: list[Entity]) -> list[Entity]:
    merged: list[Entity] = []
    seen: set[tuple[str, str]] = set()
    for entities in entity_lists:
        for entity in entities:
            key = (entity.type, entity.value)
            if key in seen:
                continue
            merged.append(entity)
            seen.add(key)
    return merged


def _merge_analysis(rule_analysis: QueryAnalysis, llm_payload: dict[str, Any]) -> QueryAnalysis:
    merged = rule_analysis.model_copy(deep=True)

    intent = _parse_intent(llm_payload.get("intent"))
    if intent is not None:
        merged.intent = intent

    llm_rewrite = llm_payload.get("rewritten_query")
    if isinstance(llm_rewrite, str) and llm_rewrite.strip():
        merged.rewritten_query = _normalize(llm_rewrite)

    raw_filters = llm_payload.get("filters")
    if isinstance(raw_filters, dict):
        price_min = raw_filters.get("price_min")
        if isinstance(price_min, int) and price_min >= 0:
            merged.filters.price_min = price_min

        price_max = raw_filters.get("price_max")
        if isinstance(price_max, int) and price_max >= 0:
            merged.filters.price_max = price_max

        brand = raw_filters.get("brand")
        if isinstance(brand, str) and brand.strip():
            merged.filters.brand = brand.strip().lower()
        elif brand is None:
            merged.filters.brand = None

        brands = raw_filters.get("brands")
        if isinstance(brands, list):
            normalized_brands: list[str] = []
            for item in brands:
                if not isinstance(item, str):
                    continue
                val = item.strip().lower()
                if not val or val in normalized_brands:
                    continue
                normalized_brands.append(val)
            if normalized_brands:
                merged.filters.brands = normalized_brands

        category = raw_filters.get("category")
        if isinstance(category, str) and category.strip():
            merged.filters.category = category.strip().lower()

        in_stock = raw_filters.get("in_stock")
        if isinstance(in_stock, bool):
            merged.filters.in_stock = in_stock

    llm_entities = _normalize_entity_payload(llm_payload.get("entities"))
    filter_entities: list[Entity] = []
    if merged.filters.brand:
        filter_entities.append(Entity(type="brand", value=merged.filters.brand))
    for brand in merged.filters.brands:
        filter_entities.append(Entity(type="brand", value=brand))
    if merged.filters.category:
        filter_entities.append(Entity(type="category", value=merged.filters.category))

    merged.entities = _merge_entities(llm_entities, rule_analysis.entities, filter_entities)

    # If query is comparative across brands, avoid narrowing results to a single brand.
    merged_brand_entities = [
        entity.value
        for entity in merged.entities
        if entity.type == "brand"
    ]
    unique_brand_entities = list(dict.fromkeys(merged_brand_entities))
    if len(unique_brand_entities) >= 2:
        merged.comparative = True
        merged.filters.brands = unique_brand_entities
        merged.filters.brand = None

    llm_comparative = llm_payload.get("comparative")
    if isinstance(llm_comparative, bool):
        merged.comparative = llm_comparative or merged.comparative

    llm_subqueries = llm_payload.get("subqueries")
    if isinstance(llm_subqueries, list):
        parsed_subqueries: list[str] = []
        for item in llm_subqueries:
            if not isinstance(item, str):
                continue
            text = _normalize(item)
            if text and text not in parsed_subqueries:
                parsed_subqueries.append(text)
        if parsed_subqueries:
            merged.subqueries = parsed_subqueries[:4]

    llm_confidence = llm_payload.get("confidence")
    if isinstance(llm_confidence, (int, float)):
        merged.confidence = round(max(0.0, min(0.99, float(llm_confidence))), 2)
    else:
        merged.confidence = round(min(0.99, max(rule_analysis.confidence, rule_analysis.confidence + 0.08)), 2)

    merged.source = "rules+llm"
    merged.llm_used = True
    return merged


def analyze_query(query: str, llm_mode: str = "auto") -> QueryAnalysis:
    rule_analysis = _analyze_with_rules(query)

    if llm_mode == "off":
        return rule_analysis

    should_use_llm = llm_mode == "force" or (llm_mode == "auto" and _should_use_llm(rule_analysis))
    if not should_use_llm:
        return rule_analysis

    llm_payload = _request_llm_json(rule_analysis)
    if not llm_payload:
        if llm_mode == "force":
            fallback = rule_analysis.model_copy(deep=True)
            fallback.source = "rules_fallback"
            return fallback
        return rule_analysis

    return _merge_analysis(rule_analysis, llm_payload)
