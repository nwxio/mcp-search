from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import httpx
from dotenv import load_dotenv

from app.models import WebFreshness, WebIntent
from app.scraper import ScrapedPage

load_dotenv()

if os.getenv("LLM_DEBUG", "").lower() in ("1", "true", "yes"):
    logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)

SYNTHESIS_SYSTEM_PROMPT = """You are a research analyst. Your task is to synthesize information from multiple sources to answer a user's query.

Guidelines:
1. Analyze all provided sources carefully
2. Synthesize a comprehensive, accurate answer
3. Cite sources using [1], [2], etc. notation
4. If sources conflict, note the disagreement
5. If information is insufficient, clearly state what's missing
6. Structure your answer with clear sections if appropriate
7. Be objective and factual
8. Respond in the same language as the user's query

Output a JSON object with these fields:
- "summary": a 1-2 sentence executive summary
- "answer": the main synthesized answer (can be several paragraphs)
- "key_points": array of 3-5 key findings
- "sources_used": array of source indices that were most useful [1, 2, ...]
- "confidence": number 0-1 indicating confidence in the answer
- "gaps": array of aspects that couldn't be answered from sources
- "follow_up": array of suggested follow-up questions"""

FACT_CHECK_SYSTEM_PROMPT = """You are a fact-checker. Analyze claims from sources and verify accuracy.

Output a JSON object with:
- "verified_facts": array of {claim, verdict: "supported|refuted|unclear", evidence}
- "conflicts": array of conflicting information found
- "confidence": overall confidence 0-1"""


@dataclass(frozen=True)
class DeepAnalysis:
    summary: str
    answer: str
    key_points: list[str]
    sources_used: list[int]
    confidence: float
    gaps: list[str]
    follow_up: list[str]
    scraped_pages: list[ScrapedPage]
    error: str | None = None


def _get_llm_provider() -> str:
    provider = os.getenv("LLM_PROVIDER", "deepseek").strip().lower()
    valid_providers = {"deepseek", "ollama", "openai"}
    return provider if provider in valid_providers else "deepseek"


def _get_timeout(multiplier: float = 1.0) -> httpx.Timeout:
    connect = float(os.getenv("DEEPSEEK_TIMEOUT_CONNECT_SEC", "5"))
    read = float(os.getenv("DEEPSEEK_TIMEOUT_READ_SEC", os.getenv("LLM_TIMEOUT_SEC", "30")))
    read = max(10.0, read * multiplier)
    return httpx.Timeout(connect=connect, read=read, write=read, pool=connect)


def _build_source_context(pages: list[ScrapedPage], max_pages: int = 5, max_chars_per_page: int = 2000) -> str:
    context_parts = []
    for i, page in enumerate(pages[:max_pages]):
        if page.error or not page.text:
            continue
        text = page.text[:max_chars_per_page]
        context_parts.append(f"[{i+1}] Source: {page.url}\nTitle: {page.title}\nContent:\n{text}")
    return "\n\n---\n\n".join(context_parts)


def _extract_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None

    if text.startswith("```"):
        import re
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    left = text.find("{")
    right = text.rfind("}")
    if left >= 0 and right > left:
        text = text[left:right+1]

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    try:
        from ast import literal_eval
        return literal_eval(text)
    except Exception:
        return None


def _synthesize_with_deepseek(query: str, source_context: str) -> dict[str, Any] | None:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        logger.warning("DEEPSEEK_API_KEY not set")
        return None

    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
    max_tokens = int(os.getenv("SYNTHESIS_MAX_TOKENS", "1500"))

    user_message = f"""User Query: {query}

Available Sources:
{source_context}

Please synthesize an answer based on these sources. Remember to cite sources using [1], [2], etc."""

    request_body = {
        "model": model,
        "temperature": 0.1,
        "max_tokens": max_tokens,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    }

    try:
        timeout = _get_timeout(multiplier=1.5)
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

        choices = payload.get("choices", [])
        if not choices:
            return None

        content = choices[0].get("message", {}).get("content", "")
        logger.debug("Synthesis response: %s", content[:500])

        return _extract_json(content)

    except Exception as exc:
        logger.error("DeepSeek synthesis error: %s", exc)
        return None


def _synthesize_with_ollama(query: str, source_context: str) -> dict[str, Any] | None:
    model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")

    user_message = f"""User Query: {query}

Available Sources:
{source_context}

Please synthesize an answer based on these sources. Remember to cite sources using [1], [2], etc."""

    request_body = {
        "model": model,
        "stream": False,
        "format": "json",
        "messages": [
            {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "options": {"temperature": 0.1},
    }

    try:
        timeout = float(os.getenv("LLM_TIMEOUT_SEC", "60"))
        response = httpx.post(
            f"{base_url}/api/chat",
            headers={"Content-Type": "application/json"},
            json=request_body,
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()

        content = payload.get("message", {}).get("content", "")
        return _extract_json(content)

    except Exception as exc:
        logger.error("Ollama synthesis error: %s", exc)
        return None


def _synthesize_with_openai(query: str, source_context: str) -> dict[str, Any] | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set")
        return None

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    max_tokens = int(os.getenv("SYNTHESIS_MAX_TOKENS", "1500"))

    user_message = f"""User Query: {query}

Available Sources:
{source_context}

Please synthesize an answer based on these sources. Remember to cite sources using [1], [2], etc."""

    request_body = {
        "model": model,
        "temperature": 0.1,
        "max_tokens": max_tokens,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    }

    try:
        timeout = _get_timeout(multiplier=1.5)
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

        choices = payload.get("choices", [])
        if not choices:
            return None

        content = choices[0].get("message", {}).get("content", "")
        logger.debug("OpenAI synthesis response: %s", content[:500])

        return _extract_json(content)

    except Exception as exc:
        logger.error("OpenAI synthesis error: %s", exc)
        return None


def synthesize_answer(
    query: str,
    pages: list[ScrapedPage],
    intent: WebIntent = WebIntent.INFORMATIONAL,
    max_pages: int = 5,
) -> DeepAnalysis:
    valid_pages = [p for p in pages if not p.error and p.text]
    logger.info("Synthesizing answer from %d valid pages (of %d total)", len(valid_pages), len(pages))

    if not valid_pages:
        return DeepAnalysis(
            summary="",
            answer="Could not retrieve content from any sources.",
            key_points=[],
            sources_used=[],
            confidence=0.0,
            gaps=["No accessible sources"],
            follow_up=["Try different search terms"],
            scraped_pages=pages,
            error="no_sources",
        )

    source_context = _build_source_context(valid_pages, max_pages=max_pages)
    if not source_context:
        return DeepAnalysis(
            summary="",
            answer="Sources contained no usable content.",
            key_points=[],
            sources_used=[],
            confidence=0.0,
            gaps=["Empty source content"],
            follow_up=[],
            scraped_pages=pages,
            error="empty_sources",
        )

    provider = _get_llm_provider()

    if provider == "deepseek":
        result = _synthesize_with_deepseek(query, source_context)
    elif provider == "openai":
        result = _synthesize_with_openai(query, source_context)
    else:
        result = _synthesize_with_ollama(query, source_context)

    if not result:
        return DeepAnalysis(
            summary="",
            answer="Failed to synthesize answer from sources.",
            key_points=[],
            sources_used=[],
            confidence=0.0,
            gaps=["LLM synthesis failed"],
            follow_up=[],
            scraped_pages=pages,
            error="synthesis_failed",
        )

    return DeepAnalysis(
        summary=result.get("summary", ""),
        answer=result.get("answer", ""),
        key_points=result.get("key_points", []),
        sources_used=result.get("sources_used", []),
        confidence=float(result.get("confidence", 0.5)),
        gaps=result.get("gaps", []),
        follow_up=result.get("follow_up", []),
        scraped_pages=pages,
    )
