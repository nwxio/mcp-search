from __future__ import annotations

import json
import logging
import os
import re
from ast import literal_eval
from typing import Any

import httpx
from dotenv import load_dotenv

from app.models import WebFreshness, WebIntent, WebQueryAnalysis

load_dotenv()

if os.getenv("LLM_DEBUG", "").lower() in ("1", "true", "yes"):
    logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)

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
    "это",
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
    "is",
    "are",
    "what",
    "how",
}

SPELLING_FIXES = {
    "аттаковала": "атаковала",
    "аттаковали": "атаковали",
    "аттаковал": "атаковал",
}

NEWS_HINTS = {
    "news",
    "latest",
    "today",
    "breaking",
    "recent",
    "новости",
    "сегодня",
    "последние",
    "срочно",
}

RECENCY_HINTS = {
    "последний",
    "последняя",
    "последнее",
    "последние",
    "последнего",
    "last",
    "latest",
    "today",
    "вчера",
}

TRANSACTIONAL_HINTS = {
    "buy",
    "price",
    "order",
    "купить",
    "цена",
    "заказать",
    "доставка",
}

RESEARCH_HINTS = {
    "best",
    "review",
    "compare",
    "vs",
    "guide",
    "analysis",
    "лучший",
    "обзор",
    "сравни",
    "сравнение",
    "анализ",
}

QUESTION_HINTS = {
    "what",
    "why",
    "how",
    "кто",
    "что",
    "почему",
    "как",
    "зачем",
}

DOMAIN_RE = re.compile(r"(?:site:)?([a-z0-9.-]+\.[a-z]{2,})", re.IGNORECASE)

LLM_SYSTEM_PROMPT = (
    "You are a web search query planner. "
    "Return strict json only for universal internet search planning: intent, rewrite, freshness, domain hints, "
    "research facets, must-include terms, subqueries. "
    "Generate 4-8 focused subqueries that improve recall and precision and are useful for research. "
    "Output only one JSON object with keys: intent, rewritten_query, required_freshness, domain_hints, "
    "research_facets, must_include_terms, subqueries, confidence."
)

LLM_WEB_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "intent": {
            "type": "string",
            "enum": ["informational", "navigational", "research", "news", "transactional", "unknown"],
        },
        "rewritten_query": {"type": "string"},
        "required_freshness": {"type": "string", "enum": ["any", "recent"]},
        "domain_hints": {"type": "array", "items": {"type": "string"}},
        "research_facets": {"type": "array", "items": {"type": "string"}},
        "must_include_terms": {"type": "array", "items": {"type": "string"}},
        "subqueries": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"},
    },
    "required": [
        "intent",
        "rewritten_query",
        "required_freshness",
        "domain_hints",
        "research_facets",
        "must_include_terms",
        "subqueries",
        "confidence",
    ],
    "additionalProperties": False,
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[\w\-.:/]+", text.lower())


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


def _clean_domain(raw: str) -> str:
    domain = raw.lower().strip(" .,/:")
    domain = domain.removeprefix("http://").removeprefix("https://")
    domain = domain.split("/")[0]
    return domain


def _extract_domain_hints(query: str) -> list[str]:
    domains: list[str] = []
    for match in DOMAIN_RE.finditer(query):
        domain = _clean_domain(match.group(1))
        if "." not in domain:
            continue
        if domain not in domains:
            domains.append(domain)
    return domains


def _correct_tokens(tokens: list[str]) -> list[str]:
    return [SPELLING_FIXES.get(token, token) for token in tokens]


def _extract_freshness(tokens: set[str], query: str) -> WebFreshness:
    if tokens.intersection(NEWS_HINTS) or tokens.intersection(RECENCY_HINTS):
        return WebFreshness.RECENT
    if re.search(r"\bпоследн\w*\b", query):
        return WebFreshness.RECENT
    return WebFreshness.ANY


def _extract_research_facets(tokens: set[str], query: str, intent: WebIntent) -> list[str]:
    facets: list[str] = []

    def add(name: str) -> None:
        if name not in facets:
            facets.append(name)

    if any(token in tokens for token in {"когда", "последний", "последняя", "latest", "last", "today"}):
        add("timeline")
    if any(token in tokens for token in {"куда", "где", "where", "location", "район"}):
        add("location")
    if any(token in tokens for token in {"кто", "who", "какие", "which"}):
        add("actors")
    if any(token in tokens for token in {"попала", "удар", "атака", "атаковала", "hit", "strike"}):
        add("impact")
    if any(token in tokens for token in {"почему", "как", "why", "how"}):
        add("context")
    if intent in {WebIntent.NEWS, WebIntent.RESEARCH}:
        add("verification")

    if "timeline" in facets and "verification" not in facets:
        add("verification")
    return facets


def _extract_must_include_terms(tokens: list[str], language: str) -> list[str]:
    stopwords = RU_STOPWORDS if language == "ru" else EN_STOPWORDS
    blocked = stopwords.union(RECENCY_HINTS).union({"site", "http", "https"})
    terms: list[str] = []
    for token in tokens:
        if token in blocked:
            continue
        if len(token) < 4:
            continue
        if token.isdigit():
            continue
        if "." in token:
            continue
        if token not in terms:
            terms.append(token)
    return terms[:6]


def _classify_intent(tokens: set[str], query: str, domain_hints: list[str], freshness: WebFreshness) -> WebIntent:
    if freshness == WebFreshness.RECENT:
        return WebIntent.NEWS
    if "site:" in query or domain_hints:
        return WebIntent.NAVIGATIONAL
    if tokens.intersection(TRANSACTIONAL_HINTS):
        return WebIntent.TRANSACTIONAL
    if tokens.intersection(RESEARCH_HINTS):
        return WebIntent.RESEARCH
    if tokens.intersection(QUESTION_HINTS) or query.endswith("?"):
        return WebIntent.INFORMATIONAL
    if len(tokens) <= 2:
        return WebIntent.NAVIGATIONAL
    return WebIntent.RESEARCH


def _build_rewrite(tokens: list[str], language: str) -> str:
    stopwords = RU_STOPWORDS if language == "ru" else EN_STOPWORDS
    cleaned = []
    for token in tokens:
        if token in stopwords:
            continue
        if token.startswith("site:"):
            continue
        if DOMAIN_RE.fullmatch(token):
            continue
        cleaned.append(token)
    return " ".join(cleaned) if cleaned else " ".join(tokens)


def _build_subqueries(
    normalized_query: str,
    rewritten_query: str,
    domain_hints: list[str],
    intent: WebIntent,
    language: str,
    research_facets: list[str],
    must_terms: list[str],
) -> list[str]:
    subqueries: list[str] = []
    if rewritten_query:
        subqueries.append(rewritten_query)

    if intent in {WebIntent.RESEARCH, WebIntent.INFORMATIONAL}:
        subqueries.append(f"{rewritten_query} explained")
        subqueries.append(f"{rewritten_query} key facts")

    if intent == WebIntent.NEWS:
        if language == "ru":
            subqueries.append(f"{rewritten_query} последние новости")
            subqueries.append(f"{rewritten_query} сводка")
        else:
            subqueries.append(f"{rewritten_query} latest updates")
            subqueries.append(f"{rewritten_query} latest report")

    for domain in domain_hints[:2]:
        subqueries.append(f"{rewritten_query} site:{domain}")

    if "timeline" in research_facets:
        if language == "ru":
            subqueries.append(f"{rewritten_query} дата и время")
        else:
            subqueries.append(f"{rewritten_query} date and time")
    if "location" in research_facets:
        if language == "ru":
            subqueries.append(f"{rewritten_query} куда попали")
        else:
            subqueries.append(f"{rewritten_query} exact locations")
    if "verification" in research_facets:
        if language == "ru":
            subqueries.append(f"{rewritten_query} подтвержденные источники")
        else:
            subqueries.append(f"{rewritten_query} verified sources")

    if must_terms:
        focused = " ".join(must_terms[:3])
        if focused:
            subqueries.append(f"{rewritten_query} {focused}")

    if not subqueries:
        subqueries.append(normalized_query)

    dedup: list[str] = []
    for subquery in subqueries:
        item = _normalize(subquery)
        if item and item not in dedup:
            dedup.append(item)
    return dedup[:8]


def _confidence(intent: WebIntent, subqueries: list[str], domain_hints: list[str]) -> float:
    score = 0.4
    if intent != WebIntent.UNKNOWN:
        score += 0.2
    if len(subqueries) >= 2:
        score += 0.2
    if domain_hints:
        score += 0.1
    return round(min(0.99, score), 2)


def _rules_analysis(query: str) -> WebQueryAnalysis:
    normalized = _normalize(query)
    language = _detect_language(normalized)
    tokens = _correct_tokens(_tokenize(normalized))
    normalized = " ".join(tokens)
    token_set = set(tokens)
    domain_hints = _extract_domain_hints(normalized)
    freshness = _extract_freshness(token_set, normalized)
    intent = _classify_intent(token_set, normalized, domain_hints, freshness)
    research_facets = _extract_research_facets(token_set, normalized, intent)
    must_terms = _extract_must_include_terms(tokens, language)
    rewritten = _build_rewrite(tokens, language)
    subqueries = _build_subqueries(
        normalized,
        rewritten,
        domain_hints,
        intent,
        language,
        research_facets,
        must_terms,
    )

    return WebQueryAnalysis(
        original_query=query,
        normalized_query=normalized,
        rewritten_query=rewritten,
        language=language,
        intent=intent,
        required_freshness=freshness,
        domain_hints=domain_hints,
        subqueries=subqueries,
        research_facets=research_facets,
        must_include_terms=must_terms,
        confidence=_confidence(intent, subqueries, domain_hints),
        source="rules",
        llm_used=False,
    )


def _should_use_llm(analysis: WebQueryAnalysis) -> bool:
    if analysis.intent in {WebIntent.UNKNOWN, WebIntent.RESEARCH, WebIntent.NEWS}:
        return True
    if analysis.confidence < 0.72:
        return True
    if len(analysis.subqueries) <= 1:
        return True
    if len(_tokenize(analysis.normalized_query)) >= 8:
        return True
    return False


def _extract_json_from_text(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    candidates: list[str] = [text]

    left = text.find("{")
    right = text.rfind("}")
    if left >= 0 and right >= 0 and right > left:
        candidates.append(text[left : right + 1])

    brace_depth = 0
    start_idx = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if brace_depth == 0:
                start_idx = i
            brace_depth += 1
        elif ch == "}":
            brace_depth -= 1
            if brace_depth == 0 and start_idx >= 0:
                candidates.append(text[start_idx : i + 1])
    if brace_depth == 0 and start_idx >= 0 and len(text) > start_idx:
        end_idx = text.rfind("}")
        if end_idx > start_idx:
            candidates.append(text[start_idx : end_idx + 1])

    def _repair_json(candidate: str) -> str:
        repaired = candidate
        repaired = repaired.replace("\u201c", '"').replace("\u201d", '"')
        repaired = repaired.replace("\u2018", "'").replace("\u2019", "'")
        repaired = repaired.replace("\u00a0", " ")
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
        repaired = re.sub(r",\s*,", ",", repaired)
        repaired = re.sub(r"^\s*,", "", repaired)
        repaired = re.sub(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)', r'\1"\2"\3', repaired)
        repaired = re.sub(r":\s*'([^']*?)'", r': "\1"', repaired)
        repaired = re.sub(r"\btrue\b", "true", repaired, flags=re.IGNORECASE)
        repaired = re.sub(r"\bfalse\b", "false", repaired, flags=re.IGNORECASE)
        repaired = re.sub(r"\bnull\b", "null", repaired, flags=re.IGNORECASE)
        repaired = re.sub(r'"\s*\n\s*"', " ", repaired)
        return repaired

    def _load_json(candidate: str) -> dict[str, Any] | None:
        try:
            payload = json.loads(candidate)
            return payload if isinstance(payload, dict) else None
        except json.JSONDecodeError:
            return None

    for candidate in candidates:
        parsed = _load_json(candidate)
        if parsed is not None:
            return parsed

    for candidate in candidates:
        repaired = _repair_json(candidate)
        if repaired != candidate:
            parsed = _load_json(repaired)
            if parsed is not None:
                logger.debug("JSON repaired successfully")
                return parsed

    for candidate in candidates:
        repaired = _repair_json(candidate)
        repaired = re.sub(r"\btrue\b", "True", repaired, flags=re.IGNORECASE)
        repaired = re.sub(r"\bfalse\b", "False", repaired, flags=re.IGNORECASE)
        repaired = re.sub(r"\bnull\b", "None", repaired, flags=re.IGNORECASE)

        try:
            payload = literal_eval(repaired)
            if isinstance(payload, dict):
                logger.debug("JSON parsed via literal_eval")
                return payload
        except Exception:
            continue

    return None


def _extract_list_field(text: str, field: str) -> list[str] | None:
    patterns = [
        rf"{field}\s*[:=]\s*(\[[^\]]*\])",
        rf"\"{field}\"\s*:\s*(\[[^\]]*\])",
        rf"'{field}'\s*:\s*(\[[^\]]*\])",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            continue
        raw = match.group(1).strip()
        try:
            parsed = literal_eval(raw)
            if isinstance(parsed, list):
                values = []
                for item in parsed:
                    if isinstance(item, str):
                        value = _normalize(item)
                        if value and value not in values:
                            values.append(value)
                return values
        except Exception:
            continue
    return None


def _extract_string_field(text: str, field: str) -> str | None:
    patterns = [
        rf"{field}\s*[:=]\s*\"([^\"]+)\"",
        rf"{field}\s*[:=]\s*'([^']+)'",
        rf"\"{field}\"\s*:\s*\"([^\"]+)\"",
        rf"'{field}'\s*:\s*'([^']+)'",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            value = _normalize(match.group(1))
            if value:
                return value
    return None


def _extract_number_field(text: str, field: str) -> float | None:
    patterns = [
        rf"{field}\s*[:=]\s*([0-9]*\.?[0-9]+)",
        rf"\"{field}\"\s*:\s*([0-9]*\.?[0-9]+)",
        rf"'{field}'\s*:\s*([0-9]*\.?[0-9]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        try:
            return float(match.group(1))
        except ValueError:
            continue
    return None


def _extract_payload_from_freeform_text(text: str) -> dict[str, Any] | None:
    lowered = text.lower()
    payload: dict[str, Any] = {}

    intent_candidates = ["informational", "navigational", "research", "news", "transactional", "unknown"]
    for intent in intent_candidates:
        if re.search(rf"\b{intent}\b", lowered):
            payload["intent"] = intent
            break

    freshness_candidates = ["recent", "any"]
    for freshness in freshness_candidates:
        if re.search(rf"\b{freshness}\b", lowered):
            payload["required_freshness"] = freshness
            break

    rewritten = _extract_string_field(text, "rewritten_query")
    if rewritten:
        payload["rewritten_query"] = rewritten

    domain_hints = _extract_list_field(text, "domain_hints")
    if domain_hints is not None:
        payload["domain_hints"] = domain_hints

    facets = _extract_list_field(text, "research_facets")
    if facets is not None:
        payload["research_facets"] = facets

    must_terms = _extract_list_field(text, "must_include_terms")
    if must_terms is not None:
        payload["must_include_terms"] = must_terms

    subqueries = _extract_list_field(text, "subqueries")
    if subqueries is not None:
        payload["subqueries"] = subqueries

    confidence = _extract_number_field(text, "confidence")
    if confidence is not None:
        payload["confidence"] = confidence

    return payload if payload else None


def _resolve_provider() -> str:
    provider = os.getenv("LLM_PROVIDER", "deepseek").strip().lower()
    valid_providers = {"deepseek", "ollama", "openai"}
    return provider if provider in valid_providers else "deepseek"


def _llm_input_payload(rules: WebQueryAnalysis) -> dict[str, Any]:
    compact_terms = rules.must_include_terms[:4]
    compact_facets = rules.research_facets[:4]
    compact_subqueries = rules.subqueries[:3]
    return {
        "query": rules.original_query[:350],
        "normalized_query": rules.normalized_query[:350],
        "rules_hint": {
            "intent": rules.intent.value,
            "required_freshness": rules.required_freshness.value,
            "domain_hints": rules.domain_hints[:2],
            "research_facets": compact_facets,
            "must_include_terms": compact_terms,
            "subqueries": compact_subqueries,
        },
    }


def _deepseek_timeout(multiplier: float = 1.0) -> httpx.Timeout:
    connect = float(os.getenv("DEEPSEEK_TIMEOUT_CONNECT_SEC", "5"))
    read = float(os.getenv("DEEPSEEK_TIMEOUT_READ_SEC", os.getenv("LLM_TIMEOUT_SEC", "20")))
    read = max(5.0, read * multiplier)
    return httpx.Timeout(connect=connect, read=read, write=read, pool=connect)


def _request_deepseek(rules: WebQueryAnalysis) -> tuple[dict[str, Any] | None, str | None]:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return None, "deepseek:missing_api_key"

    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
    max_tokens = int(os.getenv("DEEPSEEK_MAX_TOKENS", "220"))
    retries = max(0, int(os.getenv("DEEPSEEK_RETRIES", "1")))
    last_error = "deepseek:unknown"

    for attempt in range(retries + 1):
        use_json_mode = attempt == 0
        attempt_max_tokens = max(120, max_tokens - (attempt * 40))
        timeout = _deepseek_timeout(multiplier=1.0 + 0.6 * attempt)

        request_body: dict[str, Any] = {
            "model": model,
            "temperature": 0,
            "max_tokens": attempt_max_tokens,
            "stream": False,
            "messages": [
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(_llm_input_payload(rules), ensure_ascii=False),
                },
            ],
        }
        if use_json_mode:
            request_body["response_format"] = {"type": "json_object"}

        try:
            response = httpx.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=request_body,
                timeout=timeout,
            )
            response.raise_for_status()
            payload = response.json()
            choices = payload.get("choices")
            if not isinstance(choices, list) or not choices:
                last_error = "deepseek:empty_choices"
                continue
            message = choices[0].get("message")
            if not isinstance(message, dict):
                last_error = "deepseek:empty_message"
                continue
            content = message.get("content")
            if not isinstance(content, str):
                last_error = "deepseek:empty_content"
                continue
            logger.debug("DeepSeek response (attempt %d): %s", attempt, content[:500])
            parsed = _extract_json_from_text(content)
            if isinstance(parsed, dict):
                return parsed, None
            parsed_freeform = _extract_payload_from_freeform_text(content)
            if isinstance(parsed_freeform, dict):
                logger.debug("DeepSeek: used freeform extraction")
                return parsed_freeform, None
            logger.warning("DeepSeek: failed to parse JSON from response: %s", content[:300])
            last_error = "deepseek:invalid_json"
        except Exception as exc:
            last_error = f"deepseek:{exc.__class__.__name__}"

    return None, last_error


def _request_ollama(rules: WebQueryAnalysis) -> tuple[dict[str, Any] | None, str | None]:
    model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    timeout_sec = float(os.getenv("LLM_TIMEOUT_SEC", "8"))

    request_body = {
        "model": model,
        "stream": False,
        "format": "json",
        "messages": [
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(_llm_input_payload(rules), ensure_ascii=False),
            },
        ],
        "options": {"temperature": 0},
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
            return None, "ollama:empty_message"
        content = message.get("content")
        if not isinstance(content, str):
            return None, "ollama:empty_content"
        parsed = _extract_json_from_text(content)
        if isinstance(parsed, dict):
            return parsed, None
        parsed_freeform = _extract_payload_from_freeform_text(content)
        if isinstance(parsed_freeform, dict):
            return parsed_freeform, None
        return None, "ollama:invalid_json"
    except Exception as exc:
        return None, f"ollama:{exc.__class__.__name__}"


def _openai_timeout(multiplier: float = 1.0) -> httpx.Timeout:
    connect = float(os.getenv("OPENAI_TIMEOUT_CONNECT_SEC", "5"))
    read = float(os.getenv("OPENAI_TIMEOUT_READ_SEC", os.getenv("LLM_TIMEOUT_SEC", "20")))
    read = max(5.0, read * multiplier)
    return httpx.Timeout(connect=connect, read=read, write=read, pool=connect)


def _request_openai(rules: WebQueryAnalysis) -> tuple[dict[str, Any] | None, str | None]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None, "openai:missing_api_key"

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "220"))

    request_body: dict[str, Any] = {
        "model": model,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
        "messages": [
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(_llm_input_payload(rules), ensure_ascii=False)},
        ],
    }

    try:
        timeout = _openai_timeout(multiplier=1.0)
        response = httpx.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=request_body,
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return None, "openai:empty_choices"
        message = choices[0].get("message")
        if not isinstance(message, dict):
            return None, "openai:empty_message"
        content = message.get("content")
        if not isinstance(content, str):
            return None, "openai:empty_content"
        logger.debug("OpenAI response: %s", content[:500])
        parsed = _extract_json_from_text(content)
        if isinstance(parsed, dict):
            return parsed, None
        parsed_freeform = _extract_payload_from_freeform_text(content)
        if isinstance(parsed_freeform, dict):
            logger.debug("OpenAI: used freeform extraction")
            return parsed_freeform, None
        return None, "openai:invalid_json"
    except Exception as exc:
        return None, f"openai:{exc.__class__.__name__}"


def _request_llm(rules: WebQueryAnalysis) -> tuple[dict[str, Any] | None, str | None]:
    provider = _resolve_provider()
    if provider == "ollama":
        return _request_ollama(rules)
    if provider == "openai":
        return _request_openai(rules)
    return _request_deepseek(rules)


def _parse_intent(value: Any) -> WebIntent | None:
    if not isinstance(value, str):
        return None
    try:
        return WebIntent(value)
    except ValueError:
        return None


def _parse_freshness(value: Any) -> WebFreshness | None:
    if not isinstance(value, str):
        return None
    try:
        return WebFreshness(value)
    except ValueError:
        return None


def _merge_analysis(rules: WebQueryAnalysis, llm_payload: dict[str, Any]) -> WebQueryAnalysis:
    merged = rules.model_copy(deep=True)

    intent = _parse_intent(llm_payload.get("intent"))
    if intent is not None:
        merged.intent = intent

    rewritten = llm_payload.get("rewritten_query")
    if isinstance(rewritten, str) and rewritten.strip():
        merged.rewritten_query = _normalize(rewritten)

    freshness = _parse_freshness(llm_payload.get("required_freshness"))
    if freshness is not None:
        merged.required_freshness = freshness

    domain_hints = llm_payload.get("domain_hints")
    if isinstance(domain_hints, list):
        clean_domains: list[str] = []
        for item in domain_hints:
            if not isinstance(item, str):
                continue
            domain = _clean_domain(item)
            if "." not in domain:
                continue
            if domain not in clean_domains:
                clean_domains.append(domain)
        if clean_domains:
            merged.domain_hints = clean_domains

    subqueries = llm_payload.get("subqueries")
    if isinstance(subqueries, list):
        clean_subqueries: list[str] = []
        for item in subqueries:
            if not isinstance(item, str):
                continue
            subquery = _normalize(item)
            if subquery and subquery not in clean_subqueries:
                clean_subqueries.append(subquery)
        if clean_subqueries:
            merged.subqueries = clean_subqueries[:8]

    facets = llm_payload.get("research_facets")
    if isinstance(facets, list):
        clean_facets: list[str] = []
        for item in facets:
            if not isinstance(item, str):
                continue
            facet = _normalize(item).replace(" ", "_")
            if not facet or facet in clean_facets:
                continue
            clean_facets.append(facet)
        if clean_facets:
            merged.research_facets = clean_facets[:6]

    must_terms = llm_payload.get("must_include_terms")
    if isinstance(must_terms, list):
        clean_terms: list[str] = []
        for item in must_terms:
            if not isinstance(item, str):
                continue
            term = _normalize(item)
            if not term or term in clean_terms:
                continue
            clean_terms.append(term)
        if clean_terms:
            merged.must_include_terms = clean_terms[:6]

    conf = llm_payload.get("confidence")
    if isinstance(conf, (float, int)):
        merged.confidence = round(max(0.0, min(0.99, float(conf))), 2)
    else:
        merged.confidence = round(min(0.99, max(rules.confidence, rules.confidence + 0.08)), 2)

    if merged.rewritten_query and merged.rewritten_query not in merged.subqueries:
        merged.subqueries = [merged.rewritten_query, *merged.subqueries][:8]

    # Guardrails: do not let LLM weaken recency/news intent for "latest/последний" style queries.
    if rules.required_freshness == WebFreshness.RECENT:
        merged.required_freshness = WebFreshness.RECENT
    if merged.required_freshness == WebFreshness.RECENT and merged.intent in {
        WebIntent.UNKNOWN,
        WebIntent.RESEARCH,
        WebIntent.INFORMATIONAL,
    }:
        merged.intent = WebIntent.NEWS

    if len(merged.subqueries) < 2:
        merged.subqueries = rules.subqueries
    if not merged.research_facets:
        merged.research_facets = rules.research_facets
    if not merged.must_include_terms:
        merged.must_include_terms = rules.must_include_terms

    merged.source = "rules+llm"
    merged.llm_used = True
    return merged


def analyze_web_query(query: str, llm_mode: str = "auto") -> WebQueryAnalysis:
    rules = _rules_analysis(query)

    if llm_mode == "off":
        return rules

    should_use_llm = llm_mode == "force" or (llm_mode == "auto" and _should_use_llm(rules))
    if not should_use_llm:
        return rules

    llm_payload, llm_error = _request_llm(rules)
    if not llm_payload:
        if llm_mode == "force":
            fallback = rules.model_copy(deep=True)
            fallback.source = "rules_fallback"
            fallback.llm_error = llm_error
            return fallback
        return rules

    merged = _merge_analysis(rules, llm_payload)
    merged.llm_error = None
    return merged
