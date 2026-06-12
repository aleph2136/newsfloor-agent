"""
publish/page_renderers.py

Renders the article HTML page, articles/index.html, and sitemap.xml from
Jinja-free string-replacement templates stored in newsfloor/templates/.
"""

from __future__ import annotations
import html as html_lib
from datetime import date, timedelta
from pathlib import Path

# Four levels up: publish/ → node_definitions/ → graph/ → newsfloor/ → templates/
_TEMPLATES_DIR = Path(__file__).parent.parent.parent.parent / "templates"


# ---------------------------------------------------------------------------
# Article page
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


# ---------------------------------------------------------------------------
# Index page
# ---------------------------------------------------------------------------

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
# Sitemap
# ---------------------------------------------------------------------------

def _render_sitemap(manifest: list[dict], domain: str, today: date) -> str:
    cutoff = today - timedelta(days=89)
    active = [e for e in manifest if date.fromisoformat(e["date"]) >= cutoff]

    urls = [
        f"https://{domain}/",
        f"https://{domain}/articles/index.html",
    ] + [f"https://{domain}{e['url']}" for e in active]

    entries = "\n".join(f"  <url><loc>{u}</loc></url>" for u in urls)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
{entries}
</urlset>"""
