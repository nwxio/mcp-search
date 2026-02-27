from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
from dotenv import load_dotenv

load_dotenv()

if os.getenv("LLM_DEBUG", "").lower() in ("1", "true", "yes"):
    logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

SKIP_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".zip", ".rar", ".7z", ".tar", ".gz",
    ".mp3", ".mp4", ".avi", ".mov", ".wmv", ".flv",
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico",
    ".exe", ".dmg", ".apk", ".ipa",
}

SKIP_DOMAINS = {
    "youtube.com", "youtu.be", "vimeo.com",
    "facebook.com", "instagram.com", "twitter.com", "x.com",
    "tiktok.com", "linkedin.com", "pinterest.com",
    "amazon.com", "ebay.com", "aliexpress.com",
}

BOILERPLATE_PATTERNS = [
    r"cookie\s*(policy|notice|consent)",
    r"subscribe\s*(to|for|now)",
    r"sign\s*(up|in)",
    r"follow\s*(us|me)",
    r"share\s*(this|on)",
    r"related\s*(articles|posts)",
    r"you\s*may\s*also\s*like",
    r"recommended\s*(for|articles)",
    r"advertisement",
    r"sponsored",
    r"privacy\s*policy",
    r"terms\s*of\s*(service|use)",
]


@dataclass(frozen=True)
class ScrapedPage:
    url: str
    title: str
    text: str
    error: str | None = None


def _get_timeout() -> httpx.Timeout:
    connect = float(os.getenv("SCRAPER_TIMEOUT_CONNECT_SEC", "5"))
    read = float(os.getenv("SCRAPER_TIMEOUT_READ_SEC", "10"))
    return httpx.Timeout(connect=connect, read=read, write=read, pool=connect)


def _should_skip_url(url: str) -> bool:
    parsed = urlparse(url)
    path_lower = parsed.path.lower()

    for ext in SKIP_EXTENSIONS:
        if path_lower.endswith(ext):
            return True

    domain = parsed.netloc.lower().removeprefix("www.")
    for skip_domain in SKIP_DOMAINS:
        if domain == skip_domain or domain.endswith(f".{skip_domain}"):
            return True

    return False


def _extract_title(html: str) -> str:
    match = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE | re.DOTALL)
    if match:
        title = match.group(1).strip()
        title = re.sub(r"\s+", " ", title)
        parts = re.split(r"\s*[-|–—]\s*", title)
        if parts:
            return parts[0].strip()
    return ""


def _clean_html(html: str) -> str:
    html = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)
    html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<nav[^>]*>.*?</nav>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<footer[^>]*>.*?</footer>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<header[^>]*>.*?</header>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<aside[^>]*>.*?</aside>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<form[^>]*>.*?</form>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<noscript[^>]*>.*?</noscript>", "", html, flags=re.DOTALL | re.IGNORECASE)
    return html


def _get_meta_content(html: str) -> str:
    content_parts = []

    meta_desc = re.search(
        r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']+)["\']',
        html, re.IGNORECASE
    )
    if not meta_desc:
        meta_desc = re.search(
            r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']description["\']',
            html, re.IGNORECASE
        )
    if meta_desc:
        content_parts.append(meta_desc.group(1).strip())

    for match in re.finditer(
        r'<meta[^>]+property=["\']article:([^"\']+)["\'][^>]+content=["\']([^"\']+)["\']',
        html, re.IGNORECASE
    ):
        prop = match.group(1)
        if prop in ("tag", "section"):
            content_parts.append(match.group(2).strip())

    return " ".join(content_parts)


def _html_to_text(html: str) -> str:
    html = _clean_html(html)

    for tag in ["<br>", "<br/>", "<br />"]:
        html = html.replace(tag, "\n")

    for tag in ["</p>", "</div>", "</article>", "</section>", "</li>"]:
        html = html.replace(tag, "\n")

    html = re.sub(r"<h[1-6][^>]*>", "\n\n", html)
    html = re.sub(r"</h[1-6]>", "\n\n", html)

    text = re.sub(r"<[^>]+>", " ", html)

    text = text.replace("&nbsp;", " ")
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&quot;", '"')
    text = text.replace("&#39;", "'")
    text = re.sub(r"&#(\d+);", lambda m: chr(int(m.group(1))), text)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            cleaned_lines.append("")
            continue

        lowered = line.lower()
        is_boilerplate = any(re.search(pattern, lowered) for pattern in BOILERPLATE_PATTERNS)
        if is_boilerplate:
            continue

        if len(line) < 10 and not re.search(r"\d", line):
            continue

        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _extract_main_content(html: str, min_paragraphs: int = 2) -> str:
    text = _html_to_text(html)

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    good_paragraphs = []
    for p in paragraphs:
        if len(p) < 50:
            continue
        words = len(p.split())
        if words < 8:
            continue
        good_paragraphs.append(p)

    if len(good_paragraphs) >= min_paragraphs:
        return "\n\n".join(good_paragraphs)

    return text


def scrape_page(url: str) -> ScrapedPage:
    if _should_skip_url(url):
        return ScrapedPage(url=url, title="", text="", error="skipped:unsupported_url")

    try:
        timeout = _get_timeout()
        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9,ru;q=0.8",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }

        response = httpx.get(url, headers=headers, timeout=timeout, follow_redirects=True)

        if response.status_code >= 400:
            return ScrapedPage(url=url, title="", text="", error=f"http:{response.status_code}")

        content_type = response.headers.get("content-type", "")
        if "text/html" not in content_type and "application/xhtml" not in content_type:
            return ScrapedPage(url=url, title="", text="", error="skipped:not_html")

        html = response.text
        if len(html) < 200:
            return ScrapedPage(url=url, title="", text="", error="skipped:empty_page")

        title = _extract_title(html)
        meta = _get_meta_content(html)
        main_text = _extract_main_content(html)

        if not main_text and meta:
            main_text = meta

        max_chars = int(os.getenv("SCRAPER_MAX_CHARS", "8000"))
        if len(main_text) > max_chars:
            main_text = main_text[:max_chars] + "..."

        logger.debug("Scraped %s: title='%s', text_len=%d", url, title[:50], len(main_text))

        return ScrapedPage(url=url, title=title, text=main_text)

    except httpx.TimeoutException:
        return ScrapedPage(url=url, title="", text="", error="timeout")
    except Exception as exc:
        logger.debug("Scrape error for %s: %s", url, exc)
        return ScrapedPage(url=url, title="", text="", error=f"error:{exc.__class__.__name__}")


async def scrape_page_async(url: str, client: httpx.AsyncClient) -> ScrapedPage:
    if _should_skip_url(url):
        return ScrapedPage(url=url, title="", text="", error="skipped:unsupported_url")

    try:
        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9,ru;q=0.8",
        }

        response = await client.get(url, headers=headers, follow_redirects=True)

        if response.status_code >= 400:
            return ScrapedPage(url=url, title="", text="", error=f"http:{response.status_code}")

        content_type = response.headers.get("content-type", "")
        if "text/html" not in content_type and "application/xhtml" not in content_type:
            return ScrapedPage(url=url, title="", text="", error="skipped:not_html")

        html = response.text
        if len(html) < 200:
            return ScrapedPage(url=url, title="", text="", error="skipped:empty_page")

        title = _extract_title(html)
        meta = _get_meta_content(html)
        main_text = _extract_main_content(html)

        if not main_text and meta:
            main_text = meta

        max_chars = int(os.getenv("SCRAPER_MAX_CHARS", "8000"))
        if len(main_text) > max_chars:
            main_text = main_text[:max_chars] + "..."

        return ScrapedPage(url=url, title=title, text=main_text)

    except httpx.TimeoutException:
        return ScrapedPage(url=url, title="", text="", error="timeout")
    except Exception as exc:
        logger.debug("Scrape error for %s: %s", url, exc)
        return ScrapedPage(url=url, title="", text="", error=f"error:{exc.__class__.__name__}")


async def scrape_pages_async(urls: list[str], max_concurrent: int = 5) -> list[ScrapedPage]:
    if not urls:
        return []

    timeout = _get_timeout()
    limits = httpx.Limits(max_connections=max_concurrent)

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        tasks = [scrape_page_async(url, client) for url in urls]
        results = await asyncio.gather(*tasks)

    return list(results)


def scrape_pages(urls: list[str], max_concurrent: int = 5) -> list[ScrapedPage]:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    
    if loop is not None:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, scrape_pages_async(urls, max_concurrent))
            return future.result()
    else:
        return asyncio.run(scrape_pages_async(urls, max_concurrent))
