"""
publish/html_extraction.py

Extracts title, excerpt, and body from the legacy flat-HTML digest format.
Used as a fallback when digest_json is not available.
"""

from __future__ import annotations
import re


def _extract_title(digest_html: str, fallback_topic: str) -> str:
    match = re.search(r"<h1[^>]*>(.*?)</h1>", digest_html, re.IGNORECASE | re.DOTALL)
    if match:
        title = re.sub(r"<[^>]+>", "", match.group(1)).strip()
        if title:
            return title
    return fallback_topic.replace("-", " ").title()


def _extract_excerpt(digest_html: str) -> str:
    match = re.search(r"<p[^>]*>(.*?)</p>", digest_html, re.IGNORECASE | re.DOTALL)
    if match:
        text = re.sub(r"<[^>]+>", "", match.group(1)).strip()
        parts = re.split(r'(?<=\.)\s+', text)
        if len(parts) >= 2:
            return " ".join(parts[:2])
        return text[:160]
    return ""


def _extract_body(digest_html: str) -> str:
    """Remove the leading <h1> — the article template renders the title separately."""
    return re.sub(
        r"<h1[^>]*>.*?</h1>\s*", "", digest_html, count=1,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()
