"""
publish/content_blocks.py

Renders structured digest JSON into progressive-disclosure HTML content blocks.

Each block uses native <details>/<summary> elements for zero-JS toggleable
disclosure. Mermaid diagrams and code snippets are nested inside the details
panel; the section title, hook, and source links are always visible.

[AI Synthesis] paragraphs in tier_3_deep_dive are rendered with a teal badge
to distinguish model-derived observations from source-attributed claims.
"""

from __future__ import annotations
import html as html_lib
import re

from contracts.nodes import DigestStructured

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


def _md_bold_to_html(text: str) -> str:
    """Converts **bold** markdown anchors to <strong> HTML."""
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)


def _tier3_to_html(text: str) -> str:
    """Converts tier_3_deep_dive multi-paragraph text to <p> tags.

    Paragraphs starting with [AI Synthesis] are rendered with a teal badge to
    distinguish model-synthesized observations from source-attributed claims.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()] if text.strip() else []
    parts = []
    for p in paragraphs:
        if p.startswith("[AI Synthesis]"):
            content = html_lib.escape(p[len("[AI Synthesis]"):].lstrip(" :"))
            parts.append(
                f'<p class="mb-3 flex gap-2 items-start">'
                f'<span class="shrink-0 text-[10px] font-mono font-bold text-[#0d1117] '
                f'bg-[#2dd4bf] px-1.5 py-0.5 rounded mt-0.5">AI</span>'
                f'<span class="italic">{content}</span>'
                f'</p>'
            )
        else:
            parts.append(f'<p class="mb-3">{html_lib.escape(p)}</p>')
    return "\n".join(parts)


def _render_trend_roundup_html(roundup: list[str]) -> str:
    if not roundup:
        return ""
    bullets_html = "\n".join(
        f'<li class="flex gap-2 items-start">'
        f'<span class="text-[#2dd4bf] shrink-0 mt-0.5"></span>'
        f'<span>{html_lib.escape(b)}</span>'
        f'</li>'
        for b in roundup
    )
    return f"""
<section class="border border-[#2dd4bf]/25 rounded-xl p-5 md:p-6 bg-[#0d1117] mb-6">
  <div class="flex items-center gap-2.5 mb-4">
    <span class="text-[10px] font-mono font-bold text-[#0d1117] bg-[#2dd4bf] px-1.5 py-0.5 rounded shrink-0">TREND SIGNALS</span>
    <h2 class="font-serif text-lg text-[#e6edf3] leading-snug">Today's Trend Roundup</h2>
  </div>
  <ul class="space-y-2 text-sm text-[#8b949e]">
    {bullets_html}
  </ul>
</section>"""


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
    <div class="prose-content mt-4 text-[#8b949e] text-sm leading-relaxed space-y-4 cursor-default" onclick="event.stopPropagation();">
      {deep_dive_html}
      {code_html}
      {mermaid_html}
    </div>
  </details>
</section>""")

    trend_roundup = _render_trend_roundup_html(digest_json.metadata.trend_signals_roundup)
    return "\n".join(parts) + ("\n" + trend_roundup if trend_roundup else "")
