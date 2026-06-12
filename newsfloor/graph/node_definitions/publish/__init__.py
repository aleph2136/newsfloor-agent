"""
node_definitions/publish/__init__.py

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

Internal structure
──────────────────
html_extraction.py  Extracts title, excerpt, and body from legacy digest HTML
manifest.py         Loads and updates the articles/manifest.json
content_blocks.py   Renders structured digest JSON to progressive-disclosure HTML
page_renderers.py   Renders full article, index, and sitemap pages from templates
"""

from __future__ import annotations
import json
import logging
from datetime import date

import boto3

from config import settings
from contracts.nodes import PublishTaskInput, PublishTaskResult

from .content_blocks import (
    _render_content_blocks_html,
)
from .html_extraction import _extract_body, _extract_excerpt, _extract_title
from .manifest import _load_manifest, _update_manifest
from .page_renderers import (
    _render_article,
    _render_index,
    _render_sitemap,
)

logger = logging.getLogger(__name__)

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
