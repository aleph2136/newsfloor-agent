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

  DigestWriter          Writes the HTML digest using the passed articles and
                        the trend context brief. Personalizes to the engineer
                        profile. Produces the formatted digest_html.

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

Collapsing these into one agent produces output that optimizes poorly for
all three — the writing gets muddled with trend analysis, or signal
extraction gets influenced by writing quality rather than content substance.

Rework behavior
───────────────
  DIGEST_INSUFFICIENT   → rewrite with stricter depth requirements
  MISSING_REQUIRED_FIELD → regenerate with explicit field checklist
"""

from __future__ import annotations
import logging

from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM

from config import settings
from contracts.nodes import SynthesisTaskInput, SynthesisTaskResult
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

    # Sonnet for synthesis — this is the highest-value LLM call in the pipeline.
    # The digest is what the user actually reads. Worth the extra token cost.
    # max_retries=1: one retry on transient failure, then fail fast — the graph's
    # degraded mode handles repeated failures better than hammering a throttled API.
    llm_writer       = LLM(model=settings.bedrock_model_sonnet, max_retries=1)
    llm_support      = LLM(model=settings.bedrock_model_haiku,  max_retries=1)

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

    profile = task_input.engineer_profile

    # -------------------------------------------------------------------------
    # Agent 1 — Trend Contextualizer
    # -------------------------------------------------------------------------
    contextualizer = Agent(
        role="Trend Contextualizer",
        goal=(
            "Analyze today's articles in the context of known trends and recent signals. "
            "Identify which trends are confirmed, challenged, or newly emerging based on "
            "today's content. Produce a concise trend context brief for the digest writer."
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
            "Write a high-quality, personalized HTML digest that gives a senior AI "
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
            "then build your own perspective around them. "
            "Your output is editorial commentary for a technically sophisticated reader. "
            "A reader should come away with your analysis — and then go read the "
            "sources if they want the full picture. "
            "Sources are always cited with URLs using the format: "
            "'Source: [Title] — [Author] ([URL])'"
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
    # Task 2 — Digest writing
    # -------------------------------------------------------------------------
    depth_note = f"\n\nSPECIAL INSTRUCTION: {depth_instruction}" if depth_instruction else ""

    write_task = Task(
        description=f"""
Write today's AI agentic engineering digest as a complete HTML document.

TODAY'S TOPIC: {task_input.topic}
FOCUS ANGLE:   {task_input.focus_angle}

ENGINEER PROFILE:
  Name:             {profile.name}
  Focus areas:      {', '.join(profile.focus_areas)}
  Experience level: {profile.experience_level}

TODAY'S ARTICLES (use all of these):
{article_context}

TREND CONTEXT BRIEF (from contextualizer — use this to frame the digest):
(This will be provided by the previous task output)

HTML STRUCTURE REQUIREMENTS:
  - Subject line as <h1>: make it specific and compelling, not generic
  - Brief intro paragraph (2-3 sentences) connecting topic to focus angle
    and why it matters now given trend context
  - One section per article using <h2> for article title (linked to URL)
  - Per article: 3-5 sentence summary tailored to {profile.name}'s focus areas,
    followed by a "Why this matters" sentence in <em> tags
  - Trend signals section: <h2>Trend Signals</h2> with a bulleted list of
    what today's content suggests about where the field is moving
  - Closing paragraph: one practical takeaway {profile.name} can act on

WRITING STANDARDS:
  - Write conversationally — like a knowledgeable colleague, not a technical paper
  - Assume deep familiarity with LangGraph, CrewAI, Pydantic, supervisor nodes, and agentic
    design patterns. Never explain foundational concepts. No "LangGraph is a library for...",
    no "agents are autonomous systems...", no introductory framing of any kind.
  - Use plain language first; reach for technical terms only when they add precision
  - Avoid jargon stacking — if three technical words land in a row, rewrite the sentence
  - Every sentence must earn its place, but it should also flow naturally when read aloud
  - Specificity over generality — name the pattern, technique, or tradeoff
  - Connect articles to each other where genuine connections exist
  - Do not summarize what is already in the article title
  - Each article section must close with a concrete takeaway: a specific pattern, tradeoff,
    or design decision the reader can apply to their own agent system — not a general
    observation about the field{depth_note}

Return the complete HTML as a string. Start with <html> and end with </html>.
        """,
        expected_output=(
            "A complete HTML digest document starting with <html> and ending with </html>."
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

Review the digest written in the previous task and extract:

1. NEW SIGNALS: Specific, observable patterns or practices from today's content
   that are NOT already in the active trends list. Each signal must describe
   something concrete enough to track across multiple sources over time.
   Express as a short phrase or sentence — specific enough that a reader could
   identify an article as confirming or contradicting this signal.

   Good signals (specific, observable, trackable):
     "supervisor node pattern used to cap rework loops in production LangGraph pipelines"
     "teams migrating from free-text LLM outputs to Pydantic-validated task contracts"
     "hierarchical manager LLM delegating to worker agents via CrewAI process types"
     "tool call retry budgets used to prevent runaway agent cost loops"

   Bad signals (too generic — do not return these):
     "AI agents are improving"
     "multi-agent systems are important"
     "better tooling is emerging for agents"
     "LLM orchestration is a growing area"

2. TREND CONFIRMATIONS: Names of active trends (from the list above) that
   today's content directly reinforces. Use the exact trend names.

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

    kickoff_crew(crew, "synthesis", task_input.run_id, [settings.bedrock_model_sonnet, settings.bedrock_model_haiku])

    # -------------------------------------------------------------------------
    # Parse results — guard against partial crew failure
    # -------------------------------------------------------------------------
    if not write_task.output or not write_task.output.raw:
        raise RuntimeError("Synthesis crew write task produced no output.")

    digest_html = write_task.output.raw.strip()

    if not extract_task.output or not extract_task.output.raw:
        logger.warning({"node": "synthesis", "warning": "Signal extractor produced no output — using empty signals"})
        signals_output = {"new_signals": [], "trend_confirmations": [], "digest_summary": ""}
    else:
        signals_output = _parse_signals_output(extract_task.output.raw)

    logger.info({
        "node":               "synthesis",
        "digest_length":      len(digest_html),
        "new_signals":        len(signals_output["new_signals"]),
        "trend_confirmations": len(signals_output["trend_confirmations"]),
    })

    return SynthesisTaskResult(
        run_id              = task_input.run_id,
        digest_html         = digest_html,
        digest_summary      = signals_output["digest_summary"],
        new_signals         = signals_output["new_signals"],
        trend_confirmations = signals_output["trend_confirmations"],
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_signals_output(raw_output: str) -> dict:
    """
    Parses the signal extractor's JSON output.
    Returns safe defaults on parse failure so the run never crashes here.
    """
    import json, re

    try:
        match = re.search(r"\{.*\}", raw_output, re.DOTALL)
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
        return (
            "The previous digest was rejected for insufficient depth. "
            "Each article section must be at least 4 sentences. "
            "The 'Why this matters' sentence must connect explicitly to "
            "agentic architecture or engineering governance."
        )

    if reason == RetryReasonCode.MISSING_REQUIRED_FIELD:
        missing = params.get("missing_fields", [])
        return (
            f"The previous digest was missing required fields: {', '.join(missing)}. "
            "Ensure every required section is present and complete."
        )

    return ""