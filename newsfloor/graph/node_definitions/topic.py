"""
nodes/topic.py

Selects today's topic and focus angle using a two-agent CrewAI crew.

Crew design
───────────
Two agents with distinct responsibilities:

  TopicStrategist   Decides which topic to cover today given recent history,
                    active trends, and the available rotation list. Produces
                    a candidate selection with rationale and confidence.

  FocusRefiner      Takes the strategist's selection and sharpens the focus
                    angle — ensuring it's specific to agentic architecture and
                    engineering rather than generic. Produces the final output.

Why two agents instead of one
──────────────────────────────
A single agent asked to both select and refine tends to anchor on its first
choice and rationalize it. Separating selection from refinement means the
refiner can push back on a vague focus angle without being invested in the
topic choice itself. In practice this produces sharper, more actionable angles.

Rework behavior
───────────────
If the input supervisor routes back here with a RetryInstruction, the node
reads the reason_code and adjusts via TopicPromptContext:

  WEAK_TOPIC_SELECTION   → exclude the previous topic from the available list
  LOW_CONFIDENCE         → clear recent_topics so previously-covered topics
                           are not excluded; gives the strategist more options
                           when confidence was low on the first pass
  LOW_QUALITY_ARTICLES   → treated like WEAK_TOPIC_SELECTION; the articles
                           scored poorly against this topic, so try a different one

All prompt values are read exclusively from a TopicPromptContext object
returned by _apply_retry_adjustments. run() never reads from task_input
directly when building task description strings — this guarantees that retry
adjustments (e.g. clearing recent_topics on LOW_CONFIDENCE) actually take effect.
"""

from __future__ import annotations
import json
import logging
import re
from dataclasses import dataclass

from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM

from config import settings
from config_loader import load_topics
from contracts.nodes import (
    TopicTaskInput,
    TopicTaskResult,
)
from contracts.primitives import RetryReasonCode
from node_definitions.crew_utils import kickoff_crew

logger = logging.getLogger(__name__)

# Topic rotation list — loaded from newsfloor/config_data/topics.json.
# Edit that file to add, remove, or reorder topics without changing Python code.
AVAILABLE_TOPICS = load_topics()


@dataclass
class TopicPromptContext:
    """
    All values injected into topic selection prompt strings.

    Assembled by _apply_retry_adjustments. run() reads exclusively from this
    object when building task descriptions so retry adjustments cannot be
    accidentally bypassed by reading task_input fields directly.
    """
    available_topics:   list[str]
    recent_topics:      list[str]
    exclusions:         list[str]
    active_trend_names: list[str]
    recent_signals:     list[str]
    recent_weekly_narrative: str


def run(task_input: TopicTaskInput) -> TopicTaskResult:
    """
    Runs the topic selection crew and returns a TopicTaskResult.

    Adjusts behavior if a retry_instruction is present in the input.
    """
    logger.info({
        "node":             "topic",
        "recent_topics":    task_input.recent_topics[:5],
        "active_trends":    task_input.active_trend_names[:5],
        "has_retry":        task_input.retry_instruction is not None,
    })

    ctx = _apply_retry_adjustments(task_input)

    llm = LLM(model=settings.bedrock_model_topic)

    # -------------------------------------------------------------------------
    # Agent 1 — Topic Strategist
    # Selects the best topic from the available list given context.
    # -------------------------------------------------------------------------
    strategist = Agent(
        role="Topic Strategist",
        goal=(
            "Select the most valuable topic for today's AI agentic engineering digest "
            "given recent coverage history, active trends, and the engineer's focus areas."
            "Ensure that the topic is timely, relevant to current trends, and has not been covered recently."
        ),
        backstory=(
            "You are an expert in AI agentic architecture and engineering with a deep "
            "understanding of what practitioners need to stay current. You have a talent "
            "for identifying which topics are most timely and most relevant given what "
            "has already been covered recently."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )
 
    # -------------------------------------------------------------------------
    # Agent 2 — Focus Refiner
    # Sharpens the selected topic into a specific, actionable focus angle.
    # -------------------------------------------------------------------------
    refiner = Agent(
        role="Focus Refiner",
        goal=(
            "Take a selected topic and sharpen it into a specific focus angle "
            "that is directly relevant to agentic architecture, engineering discipline, "
            "governance, observability, and building reliable systems that create "
            "real value for people."
        ),
        backstory=(
            "You are a senior AI systems engineer who specializes in translating broad "
            "topics into precise, actionable engineering questions. You push back on "
            "vague framing and insist on angles that a practitioner can actually apply."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )
 
    # -------------------------------------------------------------------------
    # Task 1 — Select a topic
    # -------------------------------------------------------------------------
    select_task = Task(
        description=f"""
Select the single best topic for today's AI agentic engineering digest.

AVAILABLE TOPICS (select exactly one):
{chr(10).join(f"- {t}" for t in ctx.available_topics)}

TOPICS COVERED RECENTLY (avoid these):
{chr(10).join(f"- {t}" for t in ctx.recent_topics) or "None yet"}

ACTIVE TRENDS (favour topics that intersect with these):
{chr(10).join(f"- {t}" for t in ctx.active_trend_names) or "None yet"}

RECENT SIGNALS (context for what is moving in the field):
{chr(10).join(f"- {s}" for s in ctx.recent_signals[:10]) or "None yet"}

LAST WEEK'S PATTERN (use this to time your selection against recent momentum):
{ctx.recent_weekly_narrative or "No weekly narrative yet — early run."}

{f"EXCLUSIONS (do not select these): {', '.join(ctx.exclusions)}" if ctx.exclusions else ""}

Select the topic that is most timely, most relevant to current trends,
and has not been covered recently. Return only the topic name and a
two to three sentence rationale explaining why this topic now.
        """,
        expected_output=(
            "The selected topic name followed by a two to three sentence rationale "
            "explaining why this topic is the right choice today."
        ),
        agent=strategist,
    )
 
    # -------------------------------------------------------------------------
    # Task 2 — Refine the focus angle
    # -------------------------------------------------------------------------
    refine_task = Task(
        description=f"""
The Topic Strategist has selected a topic. Your job is to sharpen it into
a specific focus angle for an engineer whose profile is:
 
  Focus areas: {', '.join([
      'AI agentic architecture',
      'AI agentic engineering',
      'agent observability and governance',
      'reliable and safe agentic systems',
      'building agentic tools that create real value for people',
  ])}
 
The focus angle must:
- Name a concrete team type, production scenario, or application context
  (e.g., "platform teams operating LLM pipelines", "teams shipping agent-based products to users")
- Add a lens or constraint that the topic alone does not provide
- Connect directly to engineering practice — a practitioner should be able to act on it
- Lean toward practical, real-world engineering concerns and ideas rather than abstract or strictly academic questions
- Be a single sentence starting with an action verb

The focus angle must NOT:
- Restate or paraphrase the topic using different words
- Be semantically equivalent to the topic (a reader must be able to tell them apart)

Example — topic: "multi-agent orchestration patterns"
  BAD:  "Design orchestration patterns to coordinate multi-agent workflows effectively"
        (restates the topic, adds nothing)
  GOOD: "Identify which orchestration topologies let platform teams add agents incrementally
        without destabilizing existing production pipelines"
        (names a team type and adds a concrete engineering constraint)
 
Return a JSON object with exactly these fields:
{{
  "topic": "<the selected topic name unchanged>",
  "focus_angle": "<your refined single-sentence focus angle>",
  "rationale": "<two to three sentences explaining why this topic and angle now>",
  "confidence": <float between 0.0 and 1.0>
}}
        """,
        expected_output=(
            "A JSON object with fields: topic, focus_angle, rationale, confidence."
        ),
        agent=refiner,
        output_json=True,
    )
 
    # -------------------------------------------------------------------------
    # Crew — sequential, strategist runs first then refiner
    # -------------------------------------------------------------------------
    crew = Crew(
        agents  = [strategist, refiner],
        tasks   = [select_task, refine_task],
        process = Process.sequential,
        verbose = False,
    )
 
    kickoff_crew(crew, "topic", task_input.run_id, [settings.bedrock_model_topic])

    if not refine_task.output or not refine_task.output.raw:
        raise RuntimeError("Topic crew produced no output")

    task_result: TopicTaskResult = _parse_topic_result(refine_task.output.raw)
 
    logger.info({
        "node":        "topic",
        "topic":       task_result.topic,
        "focus_angle": task_result.focus_angle,
        "confidence":  task_result.confidence,
    })
 
    return task_result
 
 
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
 
def _parse_topic_result(raw: str) -> TopicTaskResult:
    """
    Parse TopicTaskResult from the raw LLM text output.

    Strips markdown code fences if present before parsing JSON.
    Raises RuntimeError with the raw text if parsing fails.
    """
    text = raw.strip()
    # Strip ```json ... ``` or ``` ... ``` fences
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return TopicTaskResult.model_validate(json.loads(text))
    except Exception as exc:
        raise RuntimeError(
            f"Topic crew returned unparseable output: {exc}\nRaw: {raw!r}"
        ) from exc


def _apply_retry_adjustments(task_input: TopicTaskInput) -> TopicPromptContext:
    """
    Reads the retry_instruction and returns a fully resolved TopicPromptContext.

    On first pass (no retry) all values come straight from task_input.
    On retry, adjustments are applied based on reason_code:

      WEAK_TOPIC_SELECTION  — previous topic removed from available list and
                              added to exclusions so the strategist picks something new
      LOW_CONFIDENCE        — recent_topics cleared so no recency penalty applies;
                              the strategist can revisit any topic on the rotation list
      LOW_QUALITY_ARTICLES  — same as WEAK_TOPIC_SELECTION; articles scored poorly
                              against this topic, so a different topic is warranted

    Hard recency filter
    ────────────────────
    The "TOPICS COVERED RECENTLY (avoid these)" prompt text is only a soft
    instruction — the strategist is free to ignore it, and in practice it
    competes against "favour topics that intersect active trends," which
    pulls toward whatever was just covered (since covering a topic creates
    or reinforces a trend in that same area). That combination is what let
    the same topic repeat for several days in a row.

    To make recency avoidance actually binding, recent_topics (and any
    retry exclusions) are removed from available_topics here, in code,
    before the prompt is ever built. If that empties the rotation list —
    e.g. the lookback window covers the whole list — fall back to the
    full list rather than leaving the strategist with nothing to choose.
    """
    available  = list(task_input.available_topics or AVAILABLE_TOPICS)
    recent     = list(task_input.recent_topics)
    exclusions: list[str] = []

    instruction = task_input.retry_instruction
    if instruction is not None:
        reason = instruction.reason_code
        params = instruction.parameter_adjustment

        if reason in (RetryReasonCode.WEAK_TOPIC_SELECTION, RetryReasonCode.LOW_QUALITY_ARTICLES):
            previous_topic = params.get("previous_topic", "")
            if previous_topic:
                exclusions.append(previous_topic)

        elif reason == RetryReasonCode.LOW_CONFIDENCE:
            # Clear the recency list so previously-covered topics are not excluded.
            # The strategist had low confidence — broadening the candidate space
            # is more useful than avoiding repeats on this pass.
            recent = []

    excluded = set(recent) | set(exclusions)
    available_after_filter = [t for t in available if t not in excluded]
    if available_after_filter:
        available = available_after_filter
    # else: recency window exhausted the rotation list — keep the full list
    # so the crew still has candidates to choose from.

    return TopicPromptContext(
        available_topics        = available,
        recent_topics           = recent,
        exclusions              = exclusions,
        active_trend_names      = list(task_input.active_trend_names),
        recent_signals          = list(task_input.recent_signals),
        recent_weekly_narrative = getattr(task_input, "recent_weekly_narrative", ""),
    )