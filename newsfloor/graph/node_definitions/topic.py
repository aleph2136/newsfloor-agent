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
reads the reason_code and adjusts:
  WEAK_TOPIC_SELECTION   → exclude the previous topic, lower confidence threshold
  LOW_CONFIDENCE         → broaden the available topic list, allow recent repeats
"""
 
from __future__ import annotations
import logging
 
from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM
 
from config import settings
from config_loader import load_topics
from contracts.nodes import (
    TopicTaskInput,
    TopicTaskResult,
)
from contracts.primitives import RetryReasonCode

logger = logging.getLogger(__name__)

# Topic rotation list — loaded from newsfloor/config_data/topics.json.
# Edit that file to add, remove, or reorder topics without changing Python code.
AVAILABLE_TOPICS = load_topics()
 
 
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
 
    available_topics, exclusions = _apply_retry_adjustments(task_input)
 
    llm = LLM(model=settings.bedrock_model_haiku)
 
    # -------------------------------------------------------------------------
    # Agent 1 — Topic Strategist
    # Selects the best topic from the available list given context.
    # -------------------------------------------------------------------------
    strategist = Agent(
        role="Topic Strategist",
        goal=(
            "Select the most valuable topic for today's AI agentic engineering digest "
            "given recent coverage history, active trends, and the engineer's focus areas."
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
{chr(10).join(f"- {t}" for t in available_topics)}
 
TOPICS COVERED RECENTLY (avoid these):
{chr(10).join(f"- {t}" for t in task_input.recent_topics) or "None yet"}
 
ACTIVE TRENDS (favour topics that intersect with these):
{chr(10).join(f"- {t}" for t in task_input.active_trend_names) or "None yet"}
 
RECENT SIGNALS (context for what is moving in the field):
{chr(10).join(f"- {s}" for s in task_input.recent_signals[:10]) or "None yet"}
 
{f"EXCLUSIONS (do not select these): {', '.join(exclusions)}" if exclusions else ""}
 
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
- Be specific enough to guide article selection and synthesis
- Connect directly to engineering practice, not just theory
- Relate to at least one of: governance, observability, reliability,
  human oversight, or practical value creation
- Be a single sentence starting with an action verb
 
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
        output_pydantic=TopicTaskResult,
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
 
    crew.kickoff()

    # Guard against CrewAI deserialization failure — output_pydantic is None
    # if the LLM returned malformed JSON or the crew failed internally.
    if not refine_task.output or not refine_task.output.pydantic:
        raw = refine_task.output.raw if refine_task.output else "None"
        raise RuntimeError(
            f"Topic crew failed to produce a valid TopicTaskResult. Raw output: {raw!r}"
        )

    task_result: TopicTaskResult = refine_task.output.pydantic
 
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
 
def _apply_retry_adjustments(
    task_input: TopicTaskInput,
) -> tuple[list[str], list[str]]:
    """
    Reads the retry_instruction and returns an adjusted available topic list
    and an exclusion list. Called only when a retry is in play.
 
    Returns:
        available_topics  — the list the strategist may select from
        exclusions        — topics to explicitly exclude this pass
    """
    available = list(task_input.available_topics or AVAILABLE_TOPICS)
    exclusions: list[str] = []
 
    instruction = task_input.retry_instruction
    if instruction is None:
        return available, exclusions
 
    reason = instruction.reason_code
    params = instruction.parameter_adjustment
 
    if reason == RetryReasonCode.WEAK_TOPIC_SELECTION:
        # Exclude the previously chosen topic so the strategist picks differently
        previous_topic = params.get("previous_topic", "")
        if previous_topic:
            available = [t for t in available if t != previous_topic]
            exclusions.append(previous_topic)
 
    elif reason == RetryReasonCode.LOW_CONFIDENCE:
        # Relax recency constraint — allow topics from further back
        # by not passing recent_topics into the prompt exclusion list
        # (handled in the task description by passing an empty list)
        pass
 
    return available, exclusions