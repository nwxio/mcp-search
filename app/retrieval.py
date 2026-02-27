from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from app.models import Intent, QueryAnalysis, SearchResult


SYNONYM_MAP = {
    "iphone": {"iphone", "айфон", "apple", "смартфон"},
    "smartphone": {"smartphone", "phone", "телефон", "смартфон"},
    "headphones": {"headphones", "наушники", "buds", "гарнитура"},
    "laptop": {"laptop", "notebook", "ноутбук", "ultrabook"},
    "console": {"console", "консоль", "playstation", "xbox", "ps5"},
    "cheap": {"дешево", "дешевый", "cheap", "budget"},
}


@dataclass(frozen=True)
class Product:
    id: str
    title: str
    description: str
    brand: str
    category: str
    price: int
    rating: float
    in_stock: bool
    popularity: int


def _tokenize(text: str) -> list[str]:
    import re

    return re.findall(r"[\w\-]+", text.lower())


def _expand(tokens: list[str]) -> set[str]:
    expanded = set(tokens)
    for token in tokens:
        expanded.update(SYNONYM_MAP.get(token, set()))
    return expanded


def _keyword_score(query_tokens: list[str], doc_tokens: list[str]) -> float:
    if not query_tokens:
        return 0.0

    query_counts = Counter(query_tokens)
    doc_counts = Counter(doc_tokens)
    overlap = 0.0
    for token, q_count in query_counts.items():
        overlap += min(q_count, doc_counts.get(token, 0))

    density = overlap / max(len(set(query_tokens)), 1)
    phrase_bonus = 0.25 if " ".join(query_tokens[:2]) in " ".join(doc_tokens) else 0.0
    return min(1.0, density + phrase_bonus)


def _semantic_score(query_tokens: list[str], doc_tokens: list[str]) -> float:
    q = _expand(query_tokens)
    d = _expand(doc_tokens)
    if not q or not d:
        return 0.0
    return len(q.intersection(d)) / len(q.union(d))


def _business_score(product: Product, intent: Intent) -> float:
    stock_bonus = 0.2 if product.in_stock else -0.2
    popularity_score = min(1.0, product.popularity / 1000)
    rating_score = product.rating / 5.0

    score = 0.4 * popularity_score + 0.4 * rating_score + 0.2 * stock_bonus
    if intent == Intent.TRANSACTIONAL and product.in_stock:
        score += 0.1

    return max(0.0, min(1.0, score))


def _price_fit(price: int, min_price: int | None, max_price: int | None) -> bool:
    if min_price is not None and price < min_price:
        return False
    if max_price is not None and price > max_price:
        return False
    return True


def _apply_filters(products: list[Product], analysis: QueryAnalysis) -> list[Product]:
    filters = analysis.filters
    result: list[Product] = []
    allowed_brands = set(filters.brands)

    for product in products:
        if filters.brand and not analysis.comparative and product.brand != filters.brand:
            continue
        if analysis.comparative and allowed_brands and product.brand not in allowed_brands:
            continue
        if filters.category and product.category != filters.category:
            continue
        if filters.in_stock is True and not product.in_stock:
            continue
        if not _price_fit(product.price, filters.price_min, filters.price_max):
            continue
        result.append(product)

    return result


def _rerank(
    rows: list[tuple[Product, float, float, float]],
    analysis: QueryAnalysis,
) -> list[tuple[Product, float, float, float, float]]:
    reranked: list[tuple[Product, float, float, float, float]] = []
    mentioned_brands = {
        entity.value for entity in analysis.entities if entity.type == "brand"
    }

    for product, kw, sem, biz in rows:
        score = 0.55 * kw + 0.35 * sem + 0.10 * biz

        if analysis.intent == Intent.TRANSACTIONAL:
            if analysis.filters.price_max is not None:
                normalized_price = product.price / max(analysis.filters.price_max, 1)
                score += max(0.0, 0.2 * (1 - normalized_price))
            if product.in_stock:
                score += 0.05

        if analysis.intent == Intent.NAVIGATIONAL:
            if analysis.rewritten_query in product.title.lower():
                score += 0.15
        if mentioned_brands and product.brand in mentioned_brands:
            score += 0.07

        reranked.append((product, kw, sem, biz, round(score, 4)))

    reranked.sort(key=lambda item: item[4], reverse=True)
    return reranked


def _diversify_for_comparison(
    rows: list[tuple[Product, float, float, float, float]],
    analysis: QueryAnalysis,
) -> list[tuple[Product, float, float, float, float]]:
    if not analysis.comparative:
        return rows

    mentioned_brands = [entity.value for entity in analysis.entities if entity.type == "brand"]
    mentioned_brands = list(dict.fromkeys(mentioned_brands))
    if len(mentioned_brands) < 2:
        return rows

    by_brand: dict[str, list[tuple[Product, float, float, float, float]]] = {}
    for row in rows:
        by_brand.setdefault(row[0].brand, []).append(row)

    selected: list[tuple[Product, float, float, float, float]] = []
    used_ids: set[str] = set()
    for brand in mentioned_brands:
        candidates = by_brand.get(brand, [])
        if not candidates:
            continue
        top = candidates[0]
        selected.append(top)
        used_ids.add(top[0].id)

    for row in rows:
        if row[0].id in used_ids:
            continue
        selected.append(row)

    return selected


class HybridSearchEngine:
    def __init__(self, data_path: Path):
        self._data_path = data_path
        self._products = self._load_products()

    def _load_products(self) -> list[Product]:
        raw = json.loads(self._data_path.read_text(encoding="utf-8"))
        return [Product(**row) for row in raw]

    @property
    def corpus_size(self) -> int:
        return len(self._products)

    def search(self, analysis: QueryAnalysis, top_k: int = 10) -> tuple[list[SearchResult], dict]:
        query_tokens = _tokenize(analysis.rewritten_query or analysis.corrected_query)
        filtered = _apply_filters(self._products, analysis)

        scored: list[tuple[Product, float, float, float]] = []
        for product in filtered:
            doc_tokens = _tokenize(f"{product.title} {product.description} {product.brand} {product.category}")
            kw = _keyword_score(query_tokens, doc_tokens)
            sem = _semantic_score(query_tokens, doc_tokens)
            biz = _business_score(product, analysis.intent)
            if kw <= 0.0 and sem <= 0.0:
                continue
            scored.append((product, kw, sem, biz))

        reranked = _rerank(scored, analysis)
        reranked = _diversify_for_comparison(reranked, analysis)[:top_k]

        results: list[SearchResult] = []
        for product, kw, sem, biz, total in reranked:
            reasoning = [
                f"keyword={kw:.2f}",
                f"semantic={sem:.2f}",
                f"business={biz:.2f}",
            ]
            if analysis.filters.price_max is not None:
                reasoning.append(f"price<={analysis.filters.price_max}")
            if analysis.filters.brand:
                reasoning.append(f"brand={analysis.filters.brand}")
            if analysis.comparative:
                reasoning.append("comparative_query")

            results.append(
                SearchResult(
                    id=product.id,
                    title=product.title,
                    description=product.description,
                    brand=product.brand,
                    category=product.category,
                    price=product.price,
                    rating=product.rating,
                    in_stock=product.in_stock,
                    score=total,
                    reasoning=reasoning,
                )
            )

        debug = {
            "query_tokens": query_tokens,
            "corpus_size": len(self._products),
            "filtered_size": len(filtered),
            "retrieved_size": len(scored),
        }
        return results, debug
