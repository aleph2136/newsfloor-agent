"""
nodes/fetch.py

Fetches raw articles from curated RSS feeds and web sources.

Implementation design
─────────────────────
RSS feeds are parsed directly via feedparser rather than through a CrewAI agent.
This is faster, cheaper (no LLM tokens), and more reliable for structured feed
data. feedparser handles the XML parsing; httpx controls the timeout so a slow
feed cannot hang the Lambda indefinitely.

Articles from all sources are collected into per-source buckets and then
interleaved round-robin before trimming to max_articles. This prevents any
single high-volume feed from consuming the entire article budget.

Thin article enrichment via CrewAI
────────────────────────────────────
Articles whose RSS summary is under 200 characters get a scrape pass via a
lightweight CrewAI Article Enricher agent. The agent uses ScrapeWebsiteTool
to pull richer article content. Enrichment is best-effort — the original
article list is returned unchanged if parsing or scraping fails.

Sources
───────
Curated list of high-quality RSS feeds focused on AI engineering,
agentic systems, and adjacent infrastructure topics. Loaded from
newsfloor/config_data/sources.json — edit that file to add, remove, or
swap feed URLs without changing Python code.

Rework behavior
───────────────
  SOURCE_FETCH_FAILURE   → retry with a subset of most reliable sources
  INSUFFICIENT_ARTICLES  → lower the min_articles threshold for this pass
"""

from __future__ import annotations
import hashlib
import html as html_stdlib
import logging
from datetime import datetime, timezone
from itertools import zip_longest
from urllib.parse import urlparse

import feedparser
import httpx
from bs4 import BeautifulSoup
from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM
from crewai.tools import BaseTool
from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings
from config_loader import load_sources
from node_definitions.crew_utils import kickoff_crew
from contracts.nodes import FetchTaskInput, FetchTaskResult
from contracts.primitives import ArticleRaw, RetryReasonCode

logger = logging.getLogger(__name__)

# Maximum articles taken from any single feed before moving to the next source.
# Prevents a high-volume feed (e.g. simonwillison.net/atom/everything/) from
# consuming the entire max_articles budget before other sources are fetched.
_PER_SOURCE_LIMIT = 2


class ScrapeWebsiteTool(BaseTool):
    name: str = "Scrape Website"
    description: str = (
        "Fetches the text content of a web page. "
        "Input: the full URL of the page to scrape."
    )

    def _run(self, website_url: str) -> str:
        try:
            response = httpx.get(
                website_url,
                timeout=15.0,
                follow_redirects=True,
                headers={"User-Agent": "NewsFloorBot/1.0 (RSS reader for AI engineering news)"},
            )
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            return soup.get_text(separator="\n", strip=True)[:3000]
        except Exception as exc:
            return f"Failed to scrape {website_url}: {exc}"


# RSS/Atom source list — loaded from newsfloor/config_data/sources.json.
# Edit that file to add, remove, or swap feed URLs without changing Python code.
DEFAULT_SOURCES = load_sources()


def run(task_input: FetchTaskInput) -> FetchTaskResult:
    """
    Runs the fetch crew and returns a FetchTaskResult.
    Falls back to direct RSS parsing if the crew cannot complete.
    """
    logger.info({
        "node":         "fetch",
        "topic":        task_input.topic,
        "sources":      len(task_input.sources),
        "max_articles": task_input.max_articles,
        "has_retry":    task_input.retry_instruction is not None,
    })

    sources, min_articles = _apply_retry_adjustments(task_input)

    # --- Direct RSS fetch (primary path) ---
    # We use feedparser directly rather than a CrewAI tool for RSS because
    # it is faster, cheaper (no LLM token cost), and more reliable for
    # structured feed data. The crew is used for scraping full article
    # content when the RSS summary is too short to score meaningfully.
    articles, fetch_errors = _fetch_feeds(
        sources     = sources,
        topic       = task_input.topic,
        focus_angle = task_input.focus_angle,
        max_articles= task_input.max_articles,
    )

    # --- Enrich thin summaries via scraping crew ---
    # Articles whose summary is under 200 chars get a scrape pass to
    # pull a richer excerpt. Keeps token cost low by only scraping when needed.
    thin_articles = [a for a in articles if len(a.summary) < 200 and not a.fetch_error]
    if thin_articles:
        articles = _enrich_thin_articles(articles, thin_articles, task_input)

    logger.info({
        "node":          "fetch",
        "articles_found": len(articles),
        "fetch_errors":   len(fetch_errors),
        "thin_enriched":  len(thin_articles),
    })

    return FetchTaskResult(
        run_id        = task_input.run_id,
        articles      = articles,
        fetch_errors  = fetch_errors,
        article_count = len(articles),
    )


# ---------------------------------------------------------------------------
# RSS feed fetching
# ---------------------------------------------------------------------------

def _fetch_feeds(
    sources:      list[str],
    topic:        str,
    focus_angle:  str,
    max_articles: int,
) -> tuple[list[ArticleRaw], list[str]]:
    """
    Parses each RSS/Atom feed and returns a flat deduplicated list of
    ArticleRaw objects, capped at max_articles.

    Articles are collected from ALL sources (up to _PER_SOURCE_LIMIT each)
    and then interleaved round-robin so no single source dominates the final
    list when it is trimmed to max_articles.
    """
    fetch_errors: list[str]              = []
    seen_ids:     set[str]               = set()
    buckets:      list[list[ArticleRaw]] = []

    for source_url in sources:
        try:
            feed_articles, error = _parse_feed(source_url)
            if error:
                fetch_errors.append(error)
            bucket: list[ArticleRaw] = []
            for article in feed_articles:
                if len(bucket) >= _PER_SOURCE_LIMIT:
                    break
                if article.article_id not in seen_ids:
                    seen_ids.add(article.article_id)
                    bucket.append(article)
            if bucket:
                buckets.append(bucket)
        except Exception as e:
            fetch_errors.append(f"{source_url}: {str(e)}")
            logger.warning({"node": "fetch", "source": source_url, "error": str(e)})

    # Interleave round-robin: take one article at a time from each source bucket
    # so the trimmed list contains articles from as many different sources as
    # possible rather than front-loading whichever source appears first.
    articles: list[ArticleRaw] = []
    for round_items in zip_longest(*buckets):
        for article in round_items:
            if article is not None:
                articles.append(article)
        if len(articles) >= max_articles:
            break

    return articles[:max_articles], fetch_errors


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=4))
def _parse_feed(source_url: str) -> tuple[list[ArticleRaw], str]:
    """
    Parses a single RSS/Atom feed URL.
    Retried once with backoff on failure via tenacity.
    Returns a tuple of (articles, error_string).
    error_string is empty on success.
    """
    try:
        # Fetch the feed content vai https so we control the timeout.
        # Then pass the raw content to feedparser rather than the URL
        # feedparser.parse() can accept a string of XML directly.
        # This prevents a slow or hung feed from blocking the Lambda indefinitely

        response = httpx.get(
            source_url,
            timeout=10.0,
            follow_redirects=True,
            headers={"User-Agent": "NewsFloorBot/1.0 (RSS reader for AI engineering news)"},
        )
        response.raise_for_status()
        feed = feedparser.parse(response.text)

        if feed.bozo and not feed.entries:
            return [], f"{source_url}: malformed feed ({feed.bozo_exception})"

        domain   = urlparse(source_url).netloc
        articles = []

        for entry in feed.entries:
            url   = entry.get("link", "")
            title = _strip_html(entry.get("title", ""))

            if not url or not title:
                continue

            article_id = hashlib.sha256(url.encode()).hexdigest()[:16]

            # Use content > summary > description, in that order of preference
            summary = (
                entry.get("content", [{}])[0].get("value", "")
                or entry.get("summary", "")
                or entry.get("description", "")
            )
            # Strip to plain text approximation — remove HTML tags crudely
            summary = _strip_html(summary)[:500]

            published_at = ""
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                try:
                    published_at = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc).isoformat()
                except Exception:
                    pass

            articles.append(ArticleRaw(
                article_id    = article_id,
                url           = url,
                title         = title,
                source_domain = domain,
                published_at  = published_at,
                summary       = summary,
            ))

        # Sort newest-first so _PER_SOURCE_LIMIT always takes the most recent articles.
        # ISO 8601 strings sort correctly lexicographically; undated articles sort last.
        articles.sort(key=lambda a: a.published_at if a.published_at else "", reverse=True)
        return articles, ""

    except Exception as e:
        return [], f"{source_url}: {str(e)}"


# ---------------------------------------------------------------------------
# Thin article enrichment via scrape crew
# ---------------------------------------------------------------------------

def _enrich_thin_articles(
    all_articles:   list[ArticleRaw],
    thin_articles:  list[ArticleRaw],
    task_input:     FetchTaskInput,
) -> list[ArticleRaw]:
    """
    Uses a lightweight CrewAI scraping agent to pull richer summaries
    for articles whose RSS description was too short to score well.
    Returns the full article list with thin summaries replaced.
    """
    llm = LLM(model=settings.bedrock_model_fetch)
    scrape_tool = ScrapeWebsiteTool()

    enricher = Agent(
        role="Article Enricher",
        goal=(
            "Extract a concise 300-500 character summary from each article URL "
            "that captures the core engineering insight relevant to the topic."
        ),
        backstory=(
            "You are a precise technical reader who can quickly identify the key "
            "engineering insight in an article and summarize it without losing meaning."
        ),
        tools=[scrape_tool],
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    url_list = "\n".join(
        f"- {a.url} (title: {a.title})" for a in thin_articles
    )

    enrich_task = Task(
        description=f"""
For each URL below, scrape the page and extract a 300-500 character
plain text summary of the core engineering insight.
Focus on content relevant to: {task_input.topic} — {task_input.focus_angle}

URLs to enrich:
{url_list}

Return a JSON array where each item has:
  "article_id": "<id>",
  "summary": "<enriched summary>"

Use these article IDs:
{chr(10).join(f"- {a.article_id}: {a.url}" for a in thin_articles)}
        """,
        expected_output="A JSON array of objects with article_id and summary fields.",
        agent=enricher,
    )

    crew = Crew(
        agents  = [enricher],
        tasks   = [enrich_task],
        process = Process.sequential,
        verbose = False,
    )

    try:
        kickoff_crew(crew, "fetch", task_input.run_id, [settings.bedrock_model_fetch])
        raw_output = enrich_task.output.raw

        import json, re as _re
        # Strip markdown fences before parsing — LLMs frequently wrap JSON output.
        cleaned = _re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_output.strip(), flags=_re.IGNORECASE)

        enriched_map: dict[str, str] = {}
        try:
            enrichments = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.warning({
                "node":                 "fetch",
                "enrichment_parse_error": str(e),
                "raw":                  raw_output[:500],
            })
            return all_articles

        for item in enrichments:
            # Validate required fields — a malformed item should not silently
            # insert an empty summary or corrupt an existing one.
            if "article_id" not in item or "summary" not in item:
                logger.warning({
                    "node":    "fetch",
                    "warning": "Enrichment item missing article_id or summary — skipping",
                    "item":    str(item)[:200],
                })
                continue
            enriched_map[item["article_id"]] = _strip_html(item["summary"])

        # Replace summaries in the full article list
        updated = []
        for article in all_articles:
            if article.article_id in enriched_map:
                updated.append(article.model_copy(
                    update={"summary": enriched_map[article.article_id]}
                ))
            else:
                updated.append(article)
        return updated

    except Exception as e:
        # Enrichment is best-effort — return original articles on failure
        logger.warning({"node": "fetch", "enrichment_error": str(e)})
        return all_articles


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_html(text: str) -> str:
    """
    Strips all HTML from text, returning plain content suitable for LLM consumption.

    Unescapes HTML entities first (e.g. &lt;script&gt; → <script>) so that
    entity-encoded tags are also removed rather than passed through as literals.
    BeautifulSoup is used as a full parser rather than a regex/char-scan so
    malformed or nested tags are handled correctly.
    """
    unescaped = html_stdlib.unescape(text)
    return BeautifulSoup(unescaped, "html.parser").get_text(separator=" ", strip=True)


def _apply_retry_adjustments(
    task_input: FetchTaskInput,
) -> tuple[list[str], int]:
    """
    Reads the retry_instruction and returns adjusted sources and min_articles.
    """
    sources      = list(task_input.sources or DEFAULT_SOURCES)
    min_articles = task_input.min_articles

    instruction = task_input.retry_instruction
    if instruction is None:
        return sources, min_articles

    reason = instruction.reason_code
    params = instruction.parameter_adjustment

    if reason == RetryReasonCode.SOURCE_FETCH_FAILURE:
        # Fall back to the three most reliable sources only
        reliable = params.get("reliable_sources", sources[:3])
        sources  = reliable

    elif reason == RetryReasonCode.INSUFFICIENT_ARTICLES:
        # Accept fewer articles on retry rather than failing the gate again
        min_articles = params.get("min_articles", max(1, min_articles - 1))

    return sources, min_articles