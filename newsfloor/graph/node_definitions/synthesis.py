"""
nodes/synthesis.py

Writes the digest and extracts trend signals from today's scored articles.

Crew design
───────────
Three agents with distinct responsibilities:

  TrendContextualizer   Reviews active trends and recent signals alongside
                        the passed articles. Identifies which trends today's
                        content confirms, challenges, or extends. Produces
                        a trend analysis brief the writer uses as context.

  DigestWriter          Writes the structured JSON digest using the passed
                        articles and the trend context brief. Personalizes to
                        the engineer profile. Uses Gemini for near-zero cost.
                        Produces a DigestStructured JSON document with tiered
                        content blocks (hook → bullets → deep dive + visuals).

  SignalExtractor       Reads the finished digest and extracts discrete trend
                        signals — specific phrases, concepts, or patterns that
                        should be tracked over time. Produces new_signals and
                        trend_confirmations for the Trend node.

Why three agents
────────────────
Each agent has a different relationship to the material:
- The contextualizer reasons across time (what does this mean given history?)
- The writer reasons about communication (how do I make this useful to Sam?)
- The extractor reasons about patterns (what should the system remember?)

Rework behavior
───────────────
  DIGEST_INSUFFICIENT   → rewrite with stricter depth requirements
  MISSING_REQUIRED_FIELD → regenerate with explicit field checklist
"""

from __future__ import annotations
import json
import logging
import re

from bs4 import BeautifulSoup
from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM

from config import settings
from contracts.nodes import (
    DigestContentBlock,
    DigestMetadata,
    DigestStructured,
    SynthesisTaskInput,
    SynthesisTaskResult,
    VisualAssets,
)
from contracts.primitives import RetryReasonCode
from node_definitions.crew_utils import kickoff_crew

logger = logging.getLogger(__name__)


def run(task_input: SynthesisTaskInput) -> SynthesisTaskResult:
    """
    Runs the synthesis crew and returns a SynthesisTaskResult.
    """
    logger.info({
        "node":              "synthesis",
        "topic":             task_input.topic,
        "articles_count":    len(task_input.passed_articles),
        "active_trends":     len(task_input.active_trends),
        "has_retry":         task_input.retry_instruction is not None,
    })

    depth_instruction = _apply_retry_adjustments(task_input)

    # Gemini for the writer — near-zero cost at equal quality for JSON synthesis.
    # Bedrock Maverick for support agents (contextualizer + signal extractor).
    llm_writer  = LLM(model=settings.gemini_model_synthesis, max_retries=1)
    llm_support = LLM(model=settings.bedrock_model_synthesis_support, max_retries=1)

    # -------------------------------------------------------------------------
    # Build shared context strings used across multiple task prompts
    # -------------------------------------------------------------------------
    article_context = "\n\n".join(
        f"ARTICLE {i+1}\n"
        f"Title:  {a.title}\n"
        f"Source: {a.source_domain}\n"
        f"URL:    {a.url}\n"
        f"Score:  {a.combined_score:.2f}\n"
        f"Summary: {a.summary}"
        for i, a in enumerate(task_input.passed_articles)
    )

    trend_context = "\n".join(
        f"- {t.name} (strength: {t.strength:.2f}, band: {t.strength_band.value}): "
        f"{', '.join(t.key_signals[:3])}"
        for t in task_input.active_trends
    ) or "No active trends yet — this is an early run."

    recent_signals_context = "\n".join(
        f"- {s}" for s in task_input.recent_run_signals[:15]
    ) or "No recent signals yet."

    weekly_narrative_context = (
        task_input.recent_weekly_narrative or "No weekly narrative yet — early run."
    )

    profile = task_input.engineer_profile

    # -------------------------------------------------------------------------
    # Agent 1 — Trend Contextualizer
    # -------------------------------------------------------------------------
    contextualizer = Agent(
        role="Trend Contextualizer",
        goal=(
            "Analyze today's articles in the context of known trends and recent signals. "
            "Identify which trends today's content confirms, challenges, or newly extends. "
            "Produce a concise trend context brief for the digest writer."
        ),
        backstory=(
            "You are an AI engineering trend analyst with a deep memory for patterns "
            "in the field. You can see connections between today's articles and longer-term "
            "movements in AI agentic architecture and engineering practice. You are precise "
            "and avoid overstating trend significance."
        ),
        llm=llm_support,
        verbose=False,
        allow_delegation=False,
    )

    # -------------------------------------------------------------------------
    # Agent 2 — Digest Writer
    # -------------------------------------------------------------------------
    writer = Agent(
        role="Digest Writer",
        goal=(
            "Write a high-quality, personalized structured JSON digest that gives a senior AI "
            "agentic engineer genuine insight they can apply to their work. "
            "Be direct, substantive, and specific. Never be generic."
        ),
        backstory=(
            f"You are writing for {profile.name}, a {profile.experience_level}. "
            f"Their focus areas are: {', '.join(profile.focus_areas)}. "
            f"Background: {profile.background_summary} "
            "You know this person values depth over breadth, engineering precision over "
            "hype, and practical applicability over theoretical interest. "
            "You write in a conversational, down-to-earth tone — like a knowledgeable "
            "colleague explaining something over coffee, not a textbook or a press release. "
            "You use plain language by default and only reach for technical terms when "
            "they genuinely add precision. You never stack jargon. You keep things "
            "readable without dumbing them down. You connect the dots between articles "
            "and trends where real connections exist. "
            "Your job is to produce original analysis and commentary, not summaries. "
            "Never paraphrase a source's structure, argument sequence, or phrasing — "
            "even loosely. Draw on sources for facts, announcements, and signals, "
            "then build your own perspective around them."
        ),
        llm=llm_writer,
        verbose=False,
        allow_delegation=False,
    )

    # -------------------------------------------------------------------------
    # Agent 3 — Signal Extractor
    # -------------------------------------------------------------------------
    extractor = Agent(
        role="Signal Extractor",
        goal=(
            "Extract discrete trend signals from today's digest content — specific "
            "concepts, patterns, or practices that appear to be gaining momentum in "
            "AI agentic architecture and engineering. These signals feed a trend "
            "tracking system that learns over time."
        ),
        backstory=(
            "You are a pattern recognition specialist for technical trends. You have "
            "a precise vocabulary for agentic systems and can identify when a concept "
            "is genuinely emerging versus being used as passing jargon. You express "
            "signals as concise noun phrases, not sentences."
        ),
        llm=llm_support,
        verbose=False,
        allow_delegation=False,
    )

    # -------------------------------------------------------------------------
    # Task 1 — Trend contextualization
    # -------------------------------------------------------------------------
    contextualize_task = Task(
        description=f"""
Analyze today's articles in the context of known trends and recent signals.

TODAY'S TOPIC: {task_input.topic}
FOCUS ANGLE:   {task_input.focus_angle}

ACTIVE TRENDS (name — strength — key signals):
{trend_context}

RECENT SIGNALS FROM PAST RUNS:
{recent_signals_context}

LAST WEEK'S PATTERN (use this to judge whether today's content accelerates, extends, or breaks from recent momentum):
{weekly_narrative_context}

TODAY'S ARTICLES:
{article_context}

Produce a trend context brief covering:
1. Which active trends does today's content confirm or reinforce?
   List each by name.
2. Which active trends does today's content challenge or contradict?
   List each by name with a brief explanation.
3. Are any new concepts emerging that are NOT in the active trends list?
   List each as a short noun phrase.
4. What is the broader pattern signal from today's content — one paragraph.

Keep the brief concise — the writer will use it as context, not quote it.
        """,
        expected_output=(
            "A structured trend context brief covering confirmations, contradictions, "
            "emerging concepts, and a pattern signal paragraph."
        ),
        agent=contextualizer,
    )

    # -------------------------------------------------------------------------
    # Task 2 — Digest writing (structured JSON output)
    # -------------------------------------------------------------------------
    depth_note = f"\n\nSPECIAL INSTRUCTION: {depth_instruction}" if depth_instruction else ""

    write_task = Task(
        description=f"""
Write today's AI agentic engineering digest as a structured JSON document.

TODAY'S TOPIC: {task_input.topic}
FOCUS ANGLE:   {task_input.focus_angle}

ENGINEER PROFILE:
  Name:             {profile.name}
  Focus areas:      {', '.join(profile.focus_areas)}
  Experience level: {profile.experience_level}

TODAY'S ARTICLES (use all of these):
{article_context}

TREND CONTEXT BRIEF (from Trend Contextualizer — injected from previous task):
Use the contextualizer's confirmed trends and pattern signal paragraph to:
- Frame the overall_trend_context in the metadata
- Connect content blocks to each other where the contextualizer identified shared themes

LAST WEEK'S PATTERN (longitudinal context):
{weekly_narrative_context}

OUTPUT JSON SCHEMA (return ONLY this JSON, nothing else):
{{
  "article_id": "{task_input.run_id}-<kebab-slug>",
  "metadata": {{
    "title": "<Specific, compelling headline — not generic>",
    "date": "<YYYY-MM-DD>",
    "summary_hook": "<1 sentence: the key question or tension this digest addresses>",
    "overall_trend_context": "<1 sentence: the broader industry movement today's content reflects>"
  }},
  "content_blocks": [
    {{
      "section_id": "block_<N>",
      "section_title": "<The specific technical concept or pattern>",
      "tier_1_hook": "<1 sentence: main takeaway for a senior engineer>",
      "tier_2_bullets": [
        "**<First 2-4 words as visual anchor>** <rest of bullet>",
        "**<First 2-4 words as visual anchor>** <rest of bullet>"
      ],
      "tier_3_deep_dive": "<Dense technical elaboration; 1-2 paragraphs, 3 sentences max each. Cite sources as: Source: [Title] — [Author] (URL)>",
      "visual_assets": {{
        "mermaid_diagram": "<Valid Mermaid.js flowchart TD or sequenceDiagram — concise node labels>",
        "code_block": "<Illustrative Python or TypeScript snippet>"
      }}
    }}
  ]
}}

CONTENT REQUIREMENTS PER BLOCK:
1. section_title: Name the specific pattern, technique, or architectural decision.
2. tier_1_hook: One sentence capturing the most actionable insight for {profile.name}.
   Frame it in terms of agentic architecture, governance, or observability.
3. tier_2_bullets: Exactly 2-3 bullets. CRITICAL: The first 2-4 words of EVERY bullet
   MUST be wrapped in **double asterisks** — e.g., "**State drift accumulates** silently
   inside long-running workflows before runtime failures surface." These bold anchors
   are mandatory visual anchors — do not skip them.
4. tier_3_deep_dive: Dense technical elaboration, 1-2 paragraphs, 3 sentences max each.
   Connect to active trends where real connections exist.
5. mermaid_diagram: Valid, clean Mermaid.js syntax. Use "flowchart TD" or "sequenceDiagram".
   Keep node labels short. No special characters that break rendering.
6. code_block: Illustrative Python or TypeScript snippet. Modern syntax. No boilerplate.

Create one content block per major article or concept covered. Use all provided articles.
Include a final "Trend Signals" block summarizing what today's content reveals about
where agentic engineering is moving.

WRITING STANDARDS (apply to all tier text):
- Assume deep familiarity with LangGraph, CrewAI, Pydantic, supervisor nodes.
  Never explain foundational concepts. No "LangGraph is a framework for..."
- Use plain language first; reach for technical terms only when they add precision
- Avoid jargon stacking — if three technical words land in a row, rewrite the sentence
- Connect content blocks to each other where genuine connections exist
- Each tier_1_hook must give a concrete takeaway, not a general observation about the field
- Never reproduce verbatim language from source articles
- tier_3_deep_dive must cite sources as: 'Source: [Title] — [Author] (URL)'{depth_note}

Return ONLY the JSON object. No prose before or after. No markdown fences.
        """,
        expected_output=(
            "A valid JSON object matching the DigestStructured schema. "
            "Starts with '{' and ends with '}'. No surrounding markdown."
        ),
        agent=writer,
    )

    # -------------------------------------------------------------------------
    # Task 3 — Signal extraction
    # -------------------------------------------------------------------------
    extract_task = Task(
        description=f"""
Extract trend signals from today's digest content.

TODAY'S TOPIC: {task_input.topic}
ACTIVE TREND NAMES (for confirmation matching):
{chr(10).join(f"- {t.name}" for t in task_input.active_trends) or "None yet"}

Extract trend signals from two sources:
1. The JSON digest written in the previous task (what the writer emphasized)
2. The original articles listed below (what the content actually contained)

Signals present in the articles but absent from the digest are still valid
signals — the writer's editorial choices should not suppress trend tracking.

From both sources, extract:

1. NEW SIGNALS: Specific, observable patterns or practices from today's content
   that are NOT already in the active trends list. Each signal must describe
   something concrete enough to track across multiple sources over time.

   Good signals (specific, observable, trackable):
     "supervisor node pattern used to cap rework loops in production LangGraph pipelines"
     "teams migrating from free-text LLM outputs to Pydantic-validated task contracts"
     "hierarchical manager LLM delegating to worker agents via CrewAI process types"
     "tool call retry budgets used to prevent runaway agent cost loops"

   Bad signals (too generic — do not return these):
     "AI agents are improving"
     "multi-agent systems are important"

2. TREND CONFIRMATIONS: Names of active trends (from the list above) that
   today's content directly reinforces. Use the exact trend names.

ORIGINAL ARTICLES (use alongside the digest to catch signals the writer may have omitted):
{article_context}

Return a JSON object with exactly these fields:
{{
  "new_signals": ["<signal>", "<signal>", ...],
  "trend_confirmations": ["<trend name>", ...],
  "digest_summary": "<plain text 3-5 sentence summary of today's digest for storage>"
}}
        """,
        expected_output=(
            "A JSON object with new_signals (list), trend_confirmations (list), "
            "and digest_summary (string) fields."
        ),
        agent=extractor,
    )

    # -------------------------------------------------------------------------
    # Crew — sequential, outputs flow forward through context
    # -------------------------------------------------------------------------
    crew = Crew(
        agents  = [contextualizer, writer, extractor],
        tasks   = [contextualize_task, write_task, extract_task],
        process = Process.sequential,
        verbose = False,
    )

    kickoff_crew(
        crew, "synthesis", task_input.run_id,
        [settings.gemini_model_synthesis, settings.bedrock_model_synthesis_support],
    )

    # -------------------------------------------------------------------------
    # Parse results — guard against partial crew failure
    # -------------------------------------------------------------------------
    if not write_task.output or not write_task.output.raw:
        raise RuntimeError("Synthesis crew write task produced no output.")

    raw_json = _strip_markdown_fences(write_task.output.raw.strip())
    digest_json = _parse_digest_json(raw_json, task_input.run_id)
    digest_html = _digest_json_to_html(digest_json)
    digest_html = _sanitize_digest_html(digest_html)

    if not extract_task.output or not extract_task.output.raw:
        logger.warning({"node": "synthesis", "warning": "Signal extractor produced no output — using empty signals"})
        signals_output = {"new_signals": [], "trend_confirmations": [], "digest_summary": ""}
    else:
        signals_output = _parse_signals_output(extract_task.output.raw)

    logger.info({
        "node":               "synthesis",
        "digest_html_length": len(digest_html),
        "content_blocks":     len(digest_json.content_blocks) if digest_json else 0,
        "new_signals":        len(signals_output["new_signals"]),
        "trend_confirmations": len(signals_output["trend_confirmations"]),
    })

    return SynthesisTaskResult(
        run_id              = task_input.run_id,
        digest_html         = digest_html,
        digest_json         = digest_json,
        digest_summary      = signals_output["digest_summary"],
        new_signals         = signals_output["new_signals"],
        trend_confirmations = signals_output["trend_confirmations"],
    )


# ---------------------------------------------------------------------------
# JSON parsing and HTML generation
# ---------------------------------------------------------------------------

def _parse_digest_json(raw: str, run_id: str) -> DigestStructured:
    """
    Parses the writer's JSON output into a DigestStructured object.
    Returns a minimal fallback on failure so the run never crashes here.
    """
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in writer output")
        data = json.loads(match.group())

        metadata = DigestMetadata(
            title=data.get("metadata", {}).get("title", "Today's AI Digest"),
            date=data.get("metadata", {}).get("date", run_id),
            summary_hook=data.get("metadata", {}).get("summary_hook", ""),
            overall_trend_context=data.get("metadata", {}).get("overall_trend_context", ""),
        )

        blocks = []
        for i, b in enumerate(data.get("content_blocks", [])):
            visual_raw = b.get("visual_assets", {})
            blocks.append(DigestContentBlock(
                section_id       = b.get("section_id", f"block_{i+1}"),
                section_title    = b.get("section_title", ""),
                tier_1_hook      = b.get("tier_1_hook", ""),
                tier_2_bullets   = b.get("tier_2_bullets", []),
                tier_3_deep_dive = b.get("tier_3_deep_dive", ""),
                visual_assets    = VisualAssets(
                    mermaid_diagram = visual_raw.get("mermaid_diagram", ""),
                    code_block      = visual_raw.get("code_block", ""),
                ),
            ))

        return DigestStructured(
            article_id     = data.get("article_id", run_id),
            metadata       = metadata,
            content_blocks = blocks,
        )

    except Exception as exc:
        logger.warning({"node": "synthesis", "warning": f"Could not parse digest JSON: {exc} — using minimal fallback"})
        return DigestStructured(
            article_id = run_id,
            metadata   = DigestMetadata(
                title="Today's AI Engineering Digest",
                date=run_id,
                summary_hook="",
                overall_trend_context="",
            ),
            content_blocks=[DigestContentBlock(
                section_id="block_1",
                section_title="Digest",
                tier_1_hook="",
                tier_2_bullets=[f"**Raw output** {raw[:300]}"],
                tier_3_deep_dive="",
            )],
        )


def _digest_json_to_html(digest_json: DigestStructured) -> str:
    """
    Converts a DigestStructured object to minimal HTML for the output supervisor.

    The output supervisor checks for <h1>, <h2>, <em> structural markers and
    minimum content length. This produces a readable HTML representation that
    satisfies those checks without requiring a full template render.
    """
    import html as html_lib

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


# ---------------------------------------------------------------------------
# HTML sanitization (applied to supervisor HTML)
# ---------------------------------------------------------------------------

_BLOCKED_HTML_TAGS: frozenset[str] = frozenset({
    "script", "style", "iframe", "object", "embed", "applet",
    "form", "input", "button", "select", "textarea",
    "noscript", "template", "svg", "math",
    "link", "meta", "base",
})


def _sanitize_digest_html(html: str) -> str:
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


def _strip_markdown_fences(text: str) -> str:
    """Removes ```json/```html/``` fences that LLMs sometimes wrap around output."""
    return re.sub(
        r"^```(?:json|html)?\s*\n?(.*?)\n?```\s*$",
        r"\1",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()


def _parse_signals_output(raw_output: str) -> dict:
    """
    Parses the signal extractor's JSON output.
    Returns safe defaults on parse failure so the run never crashes here.

    Fences are stripped first — LLMs frequently wrap JSON in markdown blocks.
    """
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_output.strip(), flags=re.IGNORECASE)
    try:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        data  = json.loads(match.group()) if match else {}
        return {
            "new_signals":         data.get("new_signals", []),
            "trend_confirmations": data.get("trend_confirmations", []),
            "digest_summary":      data.get("digest_summary", ""),
        }
    except (json.JSONDecodeError, AttributeError):
        logger.warning({"node": "synthesis", "warning": "Could not parse signal extractor output"})
        return {
            "new_signals":         [],
            "trend_confirmations": [],
            "digest_summary":      "",
        }


def _apply_retry_adjustments(task_input: SynthesisTaskInput) -> str:
    """
    Returns an additional instruction string for the writer task on rework.
    Empty string on first pass — only adds instructions when retrying.
    """
    instruction = task_input.retry_instruction
    if instruction is None:
        return ""

    reason = instruction.reason_code
    params = instruction.parameter_adjustment

    if reason == RetryReasonCode.DIGEST_INSUFFICIENT:
        base = (
            "The previous digest was rejected for insufficient depth. "
            "Each content block must have a tier_3_deep_dive of at least 3 sentences. "
            "The tier_1_hook must connect explicitly to agentic architecture or "
            "engineering governance — not a general observation."
        )

        failed = params.get("failed_criteria", [])
        criterion_notes: list[str] = []

        if "PERSONALIZED" in failed:
            criterion_notes.append(
                "The previous digest was not sufficiently personalized — "
                "every section must connect directly to agentic architecture, "
                "governance, or observability as they apply to the reader's work."
            )
        if "ACTIONABLE" in failed:
            criterion_notes.append(
                "The previous closing takeaway was too generic — "
                "name a specific pattern, tradeoff, or decision the reader can act on."
            )
        if "CONNECTED" in failed:
            criterion_notes.append(
                "The previous digest treated articles in isolation — "
                "explicitly connect at least two content blocks to each other or to an active trend."
            )

        if criterion_notes:
            return base + " Additionally: " + " ".join(criterion_notes)
        return base

    if reason == RetryReasonCode.MISSING_REQUIRED_FIELD:
        missing = params.get("missing_fields", [])
        return (
            f"The previous digest was missing required fields: {', '.join(missing)}. "
            "Ensure every required section is present and complete."
        )

    return ""
