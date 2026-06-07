"""
nodes/synthesis/html.py

HTML generation and sanitization for the synthesis digest.

Public interface
────────────────
  digest_json_to_html(digest_json)  DigestStructured → HTML string
  sanitize_digest_html(html)        Strips dangerous tags and attributes
"""

from __future__ import annotations
import html as html_lib

from bs4 import BeautifulSoup

from contracts.nodes import DigestStructured


_BLOCKED_HTML_TAGS: frozenset[str] = frozenset({
    "script", "style", "iframe", "object", "embed", "applet",
    "form", "input", "button", "select", "textarea",
    "noscript", "template", "svg", "math",
    "link", "meta", "base",
})


def digest_json_to_html(digest_json: DigestStructured) -> str:
    """
    Converts a DigestStructured object to minimal HTML for the output supervisor.

    The output supervisor checks for <h1>, <h2>, <em> structural markers and
    minimum content length. This produces a readable HTML representation that
    satisfies those checks without requiring a full template render.
    """
    blocks_html = ""
    for i, block in enumerate(digest_json.content_blocks):
        bullets = "\n".join(
            f"<li>{html_lib.escape(b)}</li>" for b in block.tier_2_bullets
        )
        blocks_html += (
            f"<h2>{i+1}. {html_lib.escape(block.section_title)}</h2>\n"
            f"<em>{html_lib.escape(block.tier_1_hook)}</em>\n"
            f"<ul>{bullets}</ul>\n"
            f"<p>{html_lib.escape(block.tier_3_deep_dive)}</p>\n"
        )

    return (
        f"<html>\n"
        f"<h1>{html_lib.escape(digest_json.metadata.title)}</h1>\n"
        f"<p>{html_lib.escape(digest_json.metadata.summary_hook)}</p>\n"
        f"<p><em>Trend: {html_lib.escape(digest_json.metadata.overall_trend_context)}</em></p>\n"
        f"{blocks_html}"
        f"</html>"
    )


def sanitize_digest_html(html: str) -> str:
    """
    Removes dangerous tags and attributes from synthesis-generated HTML.

    Applied as a defense-in-depth pass after generation. The LLM is
    instructed not to emit these tags, but external article content woven
    into the prompt could theoretically carry injected markup through.

    Blocked tags are fully decomposed (tag + content removed).
    On-event attributes and javascript: hrefs are stripped from safe tags.
    """
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup.find_all(_BLOCKED_HTML_TAGS):
        tag.decompose()

    for tag in soup.find_all(True):
        for attr in list(tag.attrs):
            if attr.lower().startswith("on"):
                del tag.attrs[attr]
            elif attr in ("href", "src", "action", "formaction"):
                val = tag.attrs[attr]
                if isinstance(val, str) and val.strip().lower().startswith("javascript:"):
                    del tag.attrs[attr]

    return str(soup)
