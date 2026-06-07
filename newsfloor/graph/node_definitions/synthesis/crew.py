"""
nodes/synthesis/crew.py

Agent definitions, task prompts, and crew assembly for the synthesis node.

Three agents with distinct responsibilities:

  TrendContextualizer   Reviews active trends and recent signals alongside
                        the passed articles. Identifies which trends today's
                        content confirms, challenges, or extends.

  DigestWriter          Writes the structured JSON digest. Uses Gemini for
                        near-zero cost at near-identical quality.

  SignalExtractor       Reads the finished digest and extracts discrete trend
                        signals for the trend tracking system.

Public interface
────────────────
  build_synthesis_crew()  Returns (crew, write_task, extract_task)
"""

from __future__ import annotations

from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM

from contracts.nodes import SynthesisTaskInput


def build_synthesis_crew(
    task_input:        SynthesisTaskInput,
    depth_instruction: str,
    llm_writer:        LLM,
    llm_support:       LLM,
) -> tuple[Crew, Task, Task]:
    """
    Assembles the three-agent synthesis crew.
    Returns (crew, write_task, extract_task) so the caller can read task outputs.
    """
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
    # Agents
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
    # Tasks
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
            "Starts with '{{' and ends with '}}'. No surrounding markdown."
        ),
        agent=writer,
    )

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

    crew = Crew(
        agents  = [contextualizer, writer, extractor],
        tasks   = [contextualize_task, write_task, extract_task],
        process = Process.sequential,
        verbose = False,
    )

    return crew, write_task, extract_task
