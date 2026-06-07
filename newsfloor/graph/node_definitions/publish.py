"""
node_definitions/publish.py

Publishes the finished digest as a static HTML article to the personal site S3 bucket.

Runs after delivery_node — the digest content is approved, so we publish regardless
of whether the email succeeded. Failure here returns published=False and the graph
continues to trend_node unchanged.

What this node does:
  1. Extracts title, excerpt, and body from synthesis digest_html
  2. Renders the article using newsfloor/templates/article.html
  3. Uploads articles/YYYY-MM-DD.html to S3
  4. Reads articles/manifest.json (or starts fresh) and prepends the new entry
  5. Uploads updated manifest.json — written daily so it never ages past 1 day,
     keeping it safe from the 90-day S3 lifecycle expiration on articles/*
  6. Regenerates articles/index.html from the manifest using
     newsfloor/templates/articles_index.html and uploads it
  7. Regenerates sitemap.xml with all active articles and uploads it
  8. Creates a CloudFront invalidation for the three updated paths

Skips gracefully (published=False, skipped=True) when personal_site_bucket
is empty — lets the Lambda run before the personal site is configured.
"""

from __future__ import annotations
import html as html_lib
import json
import logging
import re
from datetime import date, timedelta
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from config import settings
from contracts.nodes import DigestStructured, PublishTaskInput, PublishTaskResult

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).parent.parent.parent / "templates"

_UNICODE_TO_ASCII: dict[str, str] = {
    "—": "--",   # em dash
    "–": "-",    # en dash
    "‘": "'",    # left single quote
    "’": "'",    # right single quote
    "“": '"',    # left double quote
    "”": '"',    # right double quote
    "…": "...",  # ellipsis
    " ": " ",    # non-breaking space
}
_SANITIZE_TABLE = str.maketrans(_UNICODE_TO_ASCII)


def _sanitize_ascii(text: str) -> str:
    """Replace common Unicode punctuation then strip any remaining non-ASCII.

    S3 metadata values must be pure ASCII; LLM output frequently contains
    em dashes and smart quotes that would otherwise fail validation.
    """
    return text.translate(_SANITIZE_TABLE).encode("ascii", "ignore").decode("ascii")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(task_input: PublishTaskInput) -> PublishTaskResult:
    """
    Publishes the digest article to the personal site S3 bucket.
    Never raises — returns published=False with error details on failure.
    """
    if not task_input.bucket:
        logger.info({
            "node":   "publish",
            "run_id": task_input.run_id,
            "status": "skipped — PERSONAL_SITE_BUCKET not configured",
        })
        return PublishTaskResult(run_id=task_input.run_id, published=False, skipped=True)

    logger.info({
        "node":   "publish",
        "run_id": task_input.run_id,
        "bucket": task_input.bucket,
        "domain": task_input.domain,
    })

    try:
        dj = task_input.digest_json

        if dj is not None:
            title          = dj.metadata.title or _extract_title(task_input.digest_html, task_input.topic)
            excerpt        = dj.metadata.summary_hook or _extract_excerpt(task_input.digest_html)
            content_blocks = _render_content_blocks_html(dj)
        else:
            title          = _extract_title(task_input.digest_html, task_input.topic)
            excerpt        = _extract_excerpt(task_input.digest_html)
            content_blocks = _extract_body(task_input.digest_html)

        date_str     = task_input.run_id   # YYYY-MM-DD
        pub_date     = date.fromisoformat(date_str)
        date_display = pub_date.strftime("%-d %B %Y")  # e.g. "1 June 2026"

        s3 = boto3.client("s3", region_name=settings.aws_region)

        # 1. Render and upload the article page
        article_html = _render_article(
            title, excerpt, content_blocks, date_str, date_display,
            task_input.author_name, task_input.domain,
        )
        s3.put_object(
            Bucket=task_input.bucket,
            Key=f"articles/{date_str}.html",
            Body=article_html.encode("utf-8"),
            ContentType="text/html; charset=utf-8",
            CacheControl="max-age=300",
            Metadata={"title": _sanitize_ascii(title)[:255], "excerpt": _sanitize_ascii(excerpt)[:511]},
        )
        logger.info({"node": "publish", "run_id": task_input.run_id, "step": "article uploaded"})

        # 2. Update and upload the manifest
        manifest = _load_manifest(s3, task_input.bucket)
        manifest = _update_manifest(manifest, date_str, title, excerpt)
        s3.put_object(
            Bucket=task_input.bucket,
            Key="articles/manifest.json",
            Body=json.dumps(manifest, ensure_ascii=False).encode("utf-8"),
            ContentType="application/json",
            CacheControl="no-cache, no-store",
        )
        logger.info({"node": "publish", "run_id": task_input.run_id, "step": "manifest updated", "total_entries": len(manifest)})

        # 3. Regenerate and upload articles/index.html
        index_html = _render_index(manifest, pub_date)
        s3.put_object(
            Bucket=task_input.bucket,
            Key="articles/index.html",
            Body=index_html.encode("utf-8"),
            ContentType="text/html; charset=utf-8",
            CacheControl="max-age=300",
        )

        # 4. Regenerate and upload sitemap.xml
        sitemap_xml = _render_sitemap(manifest, task_input.domain, pub_date)
        s3.put_object(
            Bucket=task_input.bucket,
            Key="sitemap.xml",
            Body=sitemap_xml.encode("utf-8"),
            ContentType="application/xml",
            CacheControl="max-age=86400",
        )

        # 5. CloudFront invalidation — only for the paths we actually changed
        cf = boto3.client("cloudfront", region_name="us-east-1")
        cf.create_invalidation(
            DistributionId=task_input.cf_dist_id,
            InvalidationBatch={
                "Paths": {
                    "Quantity": 3,
                    "Items": [
                        f"/articles/{date_str}.html",
                        "/articles/index.html",
                        "/sitemap.xml",
                    ],
                },
                "CallerReference": f"publish-{date_str}",
            },
        )

        article_url = f"https://{task_input.domain}/articles/{date_str}.html"
        logger.info({
            "node":        "publish",
            "run_id":      task_input.run_id,
            "article_url": article_url,
            "status":      "published",
        })
        return PublishTaskResult(
            run_id      = task_input.run_id,
            published   = True,
            article_url = article_url,
        )

    except Exception as e:
        logger.error({
            "node":    "publish",
            "run_id":  task_input.run_id,
            "error":   str(e),
            "status":  "failed",
        })
        return PublishTaskResult(
            run_id    = task_input.run_id,
            published = False,
            error     = str(e),
        )


# ---------------------------------------------------------------------------
# HTML extraction
# ---------------------------------------------------------------------------

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
        # Take the first two sentences if possible, otherwise truncate to 160 chars
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


# ---------------------------------------------------------------------------
# Manifest management
# ---------------------------------------------------------------------------

def _load_manifest(s3, bucket: str) -> list[dict]:
    try:
        resp = s3.get_object(Bucket=bucket, Key="articles/manifest.json")
        return json.loads(resp["Body"].read().decode("utf-8"))
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("NoSuchKey", "404", "AccessDenied"):
            if code == "AccessDenied":
                logger.warning({
                    "node": "publish",
                    "message": "AccessDenied reading manifest — check s3:GetObject on articles/* in IAM policy",
                })
            return []
        raise
    except Exception as e:
        logger.warning({"node": "publish", "message": f"Manifest load failed: {e} — starting fresh"})
        return []


def _update_manifest(manifest: list[dict], date_str: str, title: str, excerpt: str) -> list[dict]:
    manifest = [e for e in manifest if e.get("date") != date_str]
    manifest.insert(0, {
        "date":    date_str,
        "title":   title,
        "excerpt": excerpt,
        "url":     f"/articles/{date_str}.html",
    })
    manifest.sort(key=lambda e: e.get("date", ""), reverse=True)
    return manifest


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------

def _render_article(
    title: str,
    excerpt: str,
    content_blocks: str,
    date_str: str,
    date_display: str,
    author_name: str,
    domain: str,
) -> str:
    template = (_TEMPLATES_DIR / "article.html").read_text(encoding="utf-8")
    replacements = {
        "{{ARTICLE_TITLE}}":        html_lib.escape(title),
        "{{ARTICLE_DATE}}":         date_str,
        "{{ARTICLE_DATE_DISPLAY}}": date_display,
        "{{ARTICLE_EXCERPT}}":      html_lib.escape(excerpt),
        "{{CONTENT_BLOCKS}}":       content_blocks,
        "{{SITE_DOMAIN}}":          domain,
        "{{AUTHOR_NAME}}":          html_lib.escape(author_name),
    }
    result = template
    for placeholder, value in replacements.items():
        result = result.replace(placeholder, value)
    return result


_SOURCE_RE = re.compile(
    r"(?:^|\n)\s*Source:\s*\[([^\]]+)\][^(]*\(\s*(https?://[^\s)]+)\s*\)",
    re.IGNORECASE,
)


def _extract_sources(text: str) -> tuple[list[tuple[str, str]], str]:
    """
    Pulls Source: [Title] (URL) citations out of tier_3_deep_dive text.
    Returns (sources, cleaned_text) where sources is a list of (title, url) pairs
    and cleaned_text has the source lines removed.
    """
    sources: list[tuple[str, str]] = []
    for m in _SOURCE_RE.finditer(text):
        sources.append((m.group(1).strip(), m.group(2).strip()))
    cleaned = _SOURCE_RE.sub("", text).strip()
    return sources, cleaned


def _render_sources_html(sources: list[tuple[str, str]]) -> str:
    if not sources:
        return ""
    links = " ".join(
        f'<a href="{html_lib.escape(url)}" target="_blank" rel="noopener noreferrer" '
        f'class="text-[#2dd4bf] hover:text-[#5eead4] underline underline-offset-2 transition-colors">'
        f"{html_lib.escape(title)}</a>"
        for title, url in sources
    )
    return (
        f'<div class="flex flex-wrap items-baseline gap-x-2 gap-y-1 mt-2 mb-4 text-xs text-[#8b949e]">'
        f'<span class="font-mono uppercase tracking-wider text-[10px] shrink-0">Sources:</span>'
        f"{links}"
        f"</div>"
    )


def _render_content_blocks_html(digest_json: DigestStructured) -> str:
    """
    Generates progressive-disclosure HTML from the structured digest JSON.
    Uses the existing site's dark aesthetic (#0d1117, #2dd4bf teal, Inter/DM Serif fonts)
    with native <details>/<summary> elements for zero-JS toggleable disclosure.

    Sources are parsed from tier_3_deep_dive and rendered below the section heading
    so they are immediately visible without expanding the deep-dive panel.
    """
    parts: list[str] = []

    for i, block in enumerate(digest_json.content_blocks):
        bullets_html = "\n".join(
            f"<li>{_md_bold_to_html(html_lib.escape(b))}</li>"
            for b in block.tier_2_bullets
        )

        sources, cleaned_deep_dive = _extract_sources(block.tier_3_deep_dive)
        sources_html   = _render_sources_html(sources)
        deep_dive_html = _tier3_to_html(cleaned_deep_dive)

        code_html = ""
        if block.visual_assets.code_block:
            code_escaped = html_lib.escape(block.visual_assets.code_block)
            code_html = (
                '<div class="space-y-2 mt-4">'
                '<span class="text-[10px] font-mono text-[#8b949e] block uppercase tracking-wider">Reference Architecture</span>'
                f'<pre class="rounded-lg overflow-x-auto bg-[#0d1117] p-4 font-mono text-xs text-[#c9d1d9] border border-[#21262d]"><code>{code_escaped}</code></pre>'
                "</div>"
            )

        mermaid_html = ""
        if block.visual_assets.mermaid_diagram:
            mermaid_html = (
                '<div class="space-y-2 mt-4">'
                '<span class="text-[10px] font-mono text-[#8b949e] block uppercase tracking-wider">State Interaction Chart</span>'
                f'<div class="mermaid bg-[#0d1117] p-4 rounded-lg border border-[#21262d]">'
                f"{block.visual_assets.mermaid_diagram}"
                "</div>"
                "</div>"
            )

        section_title_e = html_lib.escape(block.section_title)
        tier1_e         = html_lib.escape(block.tier_1_hook)

        parts.append(f"""
<section class="border border-[#21262d] rounded-xl p-5 md:p-6 bg-[#161b22] hover:border-[#30363d] transition-all duration-200 mb-6">
  <div class="mb-4">
    <h2 class="font-serif text-xl text-[#e6edf3] mb-1">{i + 1}. {section_title_e}</h2>
    <p class="text-sm font-semibold text-[#2dd4bf] tracking-wide">{tier1_e}</p>
    {sources_html}
  </div>

  <ul class="space-y-3 text-[#c9d1d9] text-sm list-disc pl-5 mb-5">
    {bullets_html}
  </ul>

  <details class="group border-t border-[#21262d] pt-4 cursor-pointer">
    <summary class="list-none flex items-center justify-between text-xs font-mono font-bold text-[#8b949e] hover:text-[#c9d1d9] select-none">
      <span class="flex items-center gap-2">
        <svg class="w-3.5 h-3.5 transform group-open-rotate transition-transform duration-200" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="3" d="M9 5l7 7-7 7"/>
        </svg>
        TECHNICAL DEEP DIVE &amp; CODE ARTIFACTS
      </span>
      <span class="group-open-hide text-[#2dd4bf]">Expand [+]</span>
      <span class="group-open-show" style="display:none;color:#8b949e">Collapse [-]</span>
    </summary>
    <div class="mt-4 text-[#8b949e] text-sm leading-relaxed space-y-4 cursor-default" onclick="event.stopPropagation();">
      {deep_dive_html}
      {code_html}
      {mermaid_html}
    </div>
  </details>
</section>""")

    return "\n".join(parts)


def _md_bold_to_html(text: str) -> str:
    """Converts **bold** markdown anchors to <strong> HTML."""
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)


def _tier3_to_html(text: str) -> str:
    """Converts tier_3_deep_dive multi-paragraph text to <p> tags."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()] if text.strip() else []
    return "\n".join(f'<p class="mb-3">{html_lib.escape(p)}</p>' for p in paragraphs)


def _render_index(manifest: list[dict], today: date) -> str:
    cutoff = today - timedelta(days=89)
    active = [e for e in manifest if date.fromisoformat(e["date"]) >= cutoff]
    article_list_html = _generate_article_list_html(active, today)
    template = (_TEMPLATES_DIR / "articles_index.html").read_text(encoding="utf-8")
    return template.replace("{{ARTICLE_LIST}}", article_list_html)


def _generate_article_list_html(active: list[dict], today: date) -> str:
    if not active:
        return _empty_state_html()

    today_str   = today.isoformat()
    today_entry = next((e for e in active if e["date"] == today_str), None)
    older       = [e for e in active if e["date"] != today_str]

    parts = []
    if today_entry:
        parts.append(_featured_article_html(today_entry))
    if older:
        parts.append(_older_articles_html(older, open_by_default=today_entry is None))

    return "\n".join(parts)


def _empty_state_html() -> str:
    return """\
      <div class="flex flex-col items-center justify-center py-20 text-center">
        <div class="w-12 h-12 rounded-full bg-[#161b22] border border-[#21262d] flex items-center justify-center mb-5">
          <svg class="w-5 h-5 text-[#30363d]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 20H5a2 2 0 01-2-2V6a2 2 0 012-2h10a2 2 0 012 2v1m2 13a2 2 0 01-2-2V7m2 13a2 2 0 002-2V9.5a2.5 2.5 0 00-2.5-2.5H15"/>
          </svg>
        </div>
        <p class="text-[#8b949e] text-sm">No articles yet. The first digest will appear here once the agent runs.</p>
      </div>"""


def _featured_article_html(entry: dict) -> str:
    pub_date     = date.fromisoformat(entry["date"])
    date_display = pub_date.strftime("%-d %B %Y")
    title_e      = html_lib.escape(entry["title"])
    excerpt_e    = html_lib.escape(entry["excerpt"])
    url_e        = html_lib.escape(entry["url"])
    date_str     = entry["date"]

    return f"""\
      <div class="mb-8">
        <div class="rounded-lg border border-[#2dd4bf]/30 bg-[#161b22] p-6">
          <div class="flex items-center gap-3 mb-3">
            <time datetime="{date_str}" class="text-xs font-semibold tracking-widest uppercase text-[#2dd4bf]">Today &middot; {date_display}</time>
            <span class="text-xs text-[#30363d] bg-[#0d1117] px-2 py-0.5 rounded-full border border-[#21262d]">Latest</span>
          </div>
          <h3 class="font-serif text-xl text-white mb-2 leading-snug">{title_e}</h3>
          <p class="text-[#8b949e] text-sm leading-relaxed mb-5">{excerpt_e}</p>
          <a href="{url_e}"
             data-article-url="{url_e}"
             data-title="{title_e}"
             data-excerpt="{excerpt_e}"
             data-date="{date_str}"
             class="inline-flex items-center gap-2 text-sm font-medium text-[#2dd4bf] hover:text-[#5eead4] transition-colors group">
            Read today&apos;s digest
            <svg class="w-4 h-4 group-hover:translate-x-0.5 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7m0 0l-7 7m7-7H3"/>
            </svg>
          </a>
        </div>
      </div>"""


def _older_articles_html(older: list[dict], open_by_default: bool) -> str:
    count     = len(older)
    label     = f"{count} previous article{'s' if count != 1 else ''}"
    open_attr = " open" if open_by_default else ""

    rows = []
    for entry in older:
        pub_date   = date.fromisoformat(entry["date"])
        short_date = pub_date.strftime("%b %-d")
        title_e    = html_lib.escape(entry["title"])
        excerpt_e  = html_lib.escape(entry["excerpt"])
        url_e      = html_lib.escape(entry["url"])
        date_str   = entry["date"]

        rows.append(f"""\
          <li>
            <a href="{url_e}"
               data-article-url="{url_e}"
               data-title="{title_e}"
               data-excerpt="{excerpt_e}"
               data-date="{date_str}"
               class="flex items-center justify-between px-4 py-2.5 rounded-md text-sm text-[#8b949e] hover:text-white hover:bg-[#21262d] transition-all group/item">
              <span>
                <time class="text-[#2dd4bf] text-xs mr-2.5">{short_date}</time>{title_e}
              </span>
              <svg class="w-3.5 h-3.5 shrink-0 opacity-0 group-hover/item:opacity-100 transition-opacity" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"/>
              </svg>
            </a>
          </li>""")

    rows_html = "\n".join(rows)
    return f"""\
      <details class="group"{open_attr}>
        <summary class="cursor-pointer list-none flex items-center justify-between px-4 py-3 rounded-md bg-[#161b22] border border-[#21262d] text-[#8b949e] hover:text-white hover:border-[#30363d] transition-all">
          <span class="text-sm font-medium">{label}</span>
          <svg class="w-4 h-4 transition-transform group-open:rotate-180" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/>
          </svg>
        </summary>
        <ul class="mt-2 space-y-0.5">
{rows_html}
        </ul>
      </details>"""


# ---------------------------------------------------------------------------
# Sitemap rendering
# ---------------------------------------------------------------------------

def _render_sitemap(manifest: list[dict], domain: str, today: date) -> str:
    cutoff  = today - timedelta(days=89)
    active  = [e for e in manifest if date.fromisoformat(e["date"]) >= cutoff]

    urls = [
        f"https://{domain}/",
        f"https://{domain}/articles/index.html",
    ] + [f"https://{domain}{e['url']}" for e in active]

    entries = "\n".join(f"  <url><loc>{u}</loc></url>" for u in urls)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
{entries}
</urlset>"""
