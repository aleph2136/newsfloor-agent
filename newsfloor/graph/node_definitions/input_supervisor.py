"""
nodes/input_supervisor.py

Evaluates the input stage as a unit before synthesis begins.

This supervisor sees the combined output of topic, fetch, and scoring
and makes a single holistic decision: is the input stage good enough
to produce a meaningful digest, or does it need another pass?

Why holistic evaluation matters
────────────────────────────────
Evaluating each node in isolation misses the interactions between them.
A topic with a confidence of 0.6 might be fine if 8 high-quality articles
were found. The same topic with only 2 low-scoring articles tells a
different story — the topic may have been too niche or too recently covered
to have good source material right now. Only by seeing all three results
together can the supervisor make a well-reasoned gate decision.

Rework behavior
───────────────
When rework is requested, the supervisor writes a RetryInstruction
targeting topic_node. Routing back to topic means all three input nodes
rerun — a fresh topic selection naturally leads to a fresh fetch and
fresh scoring. The retry instruction carries the specific reason so
topic_node can adjust its selection strategy accordingly.

Max reworks
───────────
After 2 reworks the supervisor is forced to proceed regardless of quality.
It sets the run_status signal in its rationale so the trend node can
record this run as degraded. A thin digest is always better than no digest.
"""

from __future__ import annotations
import logging

from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM

from config import settings
from contracts.nodes import InputSupervisorInput
from contracts.primitives import (
    NodeName,
    RetryInstruction,
    RetryReasonCode,
    SupervisorDecision,
    SupervisorRoute,
)

logger = logging.getLogger(__name__)

# Minimum number of articles that must pass scoring for the gate to pass.
# Below this the digest will be too thin to be useful.
MIN_PASSED_ARTICLES = 2

# Minimum combined confidence we expect from the input stage overall.
# Derived from topic confidence + average article score.
MIN_STAGE_CONFIDENCE = 0.5


def run(supervisor_input: InputSupervisorInput) -> SupervisorDecision:
    """
    Evaluates the input stage and returns a SupervisorDecision.
    Forces proceed in degraded mode if max_reworks has been reached.
    """
    rework_count = supervisor_input.rework_count

    logger.info({
        "node":                "input_supervisor",
        "rework_count":        rework_count,
        "max_reworks":         supervisor_input.max_reworks,
        "topic":               supervisor_input.topic_result.topic,
        "articles_fetched":    supervisor_input.fetch_result.article_count,
        "articles_passed":     supervisor_input.scoring_result.high_quality_count,
        "fetch_errors":        len(supervisor_input.fetch_result.fetch_errors),
    })

    # --- Hard gate: force proceed after max reworks ---
    if rework_count >= supervisor_input.max_reworks:
        logger.warning({
            "node":    "input_supervisor",
            "message": "Max reworks reached — proceeding in degraded mode",
        })
        return SupervisorDecision(
            supervisor   = NodeName.INPUT_SUPERVISOR,
            route        = SupervisorRoute.PROCEED,
            rework_count = rework_count,
            rationale    = (
                f"Max reworks ({supervisor_input.max_reworks}) reached. "
                "Proceeding in degraded mode with available input."
            ),
        )

    # --- Structural checks before calling the LLM ---
    # These are deterministic failures that don't need LLM reasoning.
    structural_issue = _check_structural_gates(supervisor_input)
    if structural_issue:
        return _request_rework(
            reason_code  = structural_issue["reason_code"],
            params       = structural_issue["params"],
            rationale    = structural_issue["rationale"],
            rework_count = rework_count,
        )

    # --- LLM-backed holistic evaluation ---
    return _evaluate_with_llm(supervisor_input, rework_count)


# ---------------------------------------------------------------------------
# Structural gate checks — no LLM needed
# ---------------------------------------------------------------------------

def _check_structural_gates(
    supervisor_input: InputSupervisorInput,
) -> dict | None:
    """
    Checks hard structural criteria that don't require LLM reasoning.
    Returns a dict describing the issue if a gate fails, None if all pass.
    Checked in order of severity — returns on the first failure found.
    """
    scoring = supervisor_input.scoring_result
    fetch   = supervisor_input.fetch_result
    topic   = supervisor_input.topic_result

    # Not enough articles passed scoring to support a meaningful synthesis
    if scoring.high_quality_count < MIN_PASSED_ARTICLES:
        return {
            "reason_code": RetryReasonCode.INSUFFICIENT_ARTICLES,
            "params": {
                "previous_topic":  topic.topic,
                "min_articles":    MIN_PASSED_ARTICLES,
                "articles_passed": scoring.high_quality_count,
            },
            "rationale": (
                f"Only {scoring.high_quality_count} article(s) passed scoring "
                f"(minimum {MIN_PASSED_ARTICLES}). Topic may be too niche or "
                "fetch sources may not cover it well."
            ),
        }

    # Topic confidence is too low — strategist wasn't sure about this selection
    if topic.confidence < 0.4:
        return {
            "reason_code": RetryReasonCode.WEAK_TOPIC_SELECTION,
            "params": {
                "previous_topic": topic.topic,
                "min_confidence": 0.4,
            },
            "rationale": (
                f"Topic confidence {topic.confidence:.2f} is below threshold (0.4). "
                "Requesting a new topic selection."
            ),
        }

    # All fetch sources failed
    if fetch.article_count == 0:
        return {
            "reason_code": RetryReasonCode.SOURCE_FETCH_FAILURE,
            "params": {
                "reliable_sources": None,   # topic node will use defaults
            },
            "rationale": "No articles were fetched — all sources may have failed.",
        }

    return None


# ---------------------------------------------------------------------------
# LLM-backed holistic evaluation
# ---------------------------------------------------------------------------

def _evaluate_with_llm(
    supervisor_input: InputSupervisorInput,
    rework_count:     int,
) -> SupervisorDecision:
    """
    Uses a single LLM-backed agent to evaluate the input stage holistically.
    Called only when structural checks pass — the LLM reasons about quality,
    not about hard criteria that can be checked deterministically.
    """
    llm = LLM(model=settings.bedrock_model_input_supervisor)

    evaluator = Agent(
        role="Input Stage Evaluator",
        goal=(
            "Evaluate whether the combined output of topic selection, article fetch, "
            "and article scoring is strong enough to produce a meaningful, high-quality "
            "digest for an AI agentic engineering practitioner."
        ),
        backstory=(
            "You are a senior editorial director for a technical AI engineering digest. "
            "You have high standards — you know that a digest built on weak input "
            "material wastes the reader's time. You evaluate input quality holistically, "
            "not just by the numbers."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    # Build a compact summary of passed articles for the evaluator
    passed_summaries = "\n".join(
        f"- [{a.combined_score:.2f}] {a.title} ({a.source_domain}): {a.summary[:150]}"
        for a in supervisor_input.scoring_result.passed_articles
    )

    evaluate_task = Task(
        description=f"""
Evaluate the input stage for today's AI agentic engineering digest.

TOPIC SELECTED: {supervisor_input.topic_result.topic}
FOCUS ANGLE: {supervisor_input.topic_result.focus_angle}
TOPIC CONFIDENCE: {supervisor_input.topic_result.confidence:.2f}
TOPIC RATIONALE: {supervisor_input.topic_result.rationale}

FETCH RESULTS:
  Articles fetched:  {supervisor_input.fetch_result.article_count}
  Fetch errors:      {len(supervisor_input.fetch_result.fetch_errors)}

SCORING RESULTS:
  Articles passed:   {supervisor_input.scoring_result.high_quality_count}
  Articles filtered: {supervisor_input.scoring_result.low_quality_count}

PASSED ARTICLES (score — title — source — summary excerpt):
{passed_summaries or "None"}

EVALUATION CRITERIA:
1. Is the topic timely and well-chosen given current agentic engineering trends?
2. Do the passed articles provide enough substance and diversity to support
   a meaningful digest on this topic and focus angle?
3. Is there a coherent thread across the articles that synthesis can work with?
4. Would a senior AI agentic engineer find genuine value in this material?

If the input is strong enough, return PROCEED.
If not, return REWORK with a specific reason code from this list:
  WEAK_TOPIC_SELECTION      — topic is poorly chosen or redundant
  INSUFFICIENT_ARTICLES     — not enough quality articles for the topic
  LOW_QUALITY_ARTICLES      — articles passed threshold but lack real substance

Return a JSON object with exactly these fields:
{{
  "decision": "PROCEED" or "REWORK",
  "reason_code": "<one of the codes above, or null if PROCEED>",
  "rationale": "<two to three sentences explaining your decision>"
}}
        """,
        expected_output=(
            "A JSON object with fields: decision, reason_code, rationale."
        ),
        agent=evaluator,
    )

    crew = Crew(
        agents  = [evaluator],
        tasks   = [evaluate_task],
        process = Process.sequential,
        verbose = False,
    )

    crew.kickoff()

    if not evaluate_task.output or not evaluate_task.output.raw:
        logger.warning({
            "node":    "input_supervisor",
            "warning": "Evaluator crew produced no output — defaulting to PROCEED",
        })
        return SupervisorDecision(
            supervisor   = NodeName.INPUT_SUPERVISOR,
            route        = SupervisorRoute.PROCEED,
            rework_count = rework_count,
            rationale    = "Evaluator produced no output — proceeding by default.",
        )

    return _parse_llm_decision(
        raw_output   = evaluate_task.output.raw,
        rework_count = rework_count,
        topic        = supervisor_input.topic_result.topic,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_llm_decision(
    raw_output:  str,
    rework_count: int,
    topic:        str,
) -> SupervisorDecision:
    """
    Parses the LLM evaluator's JSON output into a SupervisorDecision.
    Defaults to PROCEED on parse failure — a supervisor that can't
    evaluate should not block the pipeline.
    """
    import json, re

    try:
        match = re.search(r"\{.*\}", raw_output, re.DOTALL)
        data  = json.loads(match.group()) if match else {}
    except (json.JSONDecodeError, AttributeError):
        logger.warning({
            "node":    "input_supervisor",
            "warning": "Could not parse LLM decision — defaulting to PROCEED",
        })
        return SupervisorDecision(
            supervisor   = NodeName.INPUT_SUPERVISOR,
            route        = SupervisorRoute.PROCEED,
            rework_count = rework_count,
            rationale    = "LLM output could not be parsed — proceeding by default.",
        )

    decision_str = data.get("decision", "PROCEED").upper()
    route        = SupervisorRoute.REWORK if decision_str == "REWORK" else SupervisorRoute.PROCEED
    rationale    = data.get("rationale", "")
    reason_str   = data.get("reason_code")

    if route == SupervisorRoute.REWORK and reason_str:
        try:
            reason_code = RetryReasonCode(reason_str.lower())
        except ValueError:
            reason_code = RetryReasonCode.WEAK_TOPIC_SELECTION

        return _request_rework(
            reason_code  = reason_code,
            params       = {"previous_topic": topic},
            rationale    = rationale,
            rework_count = rework_count,
        )

    return SupervisorDecision(
        supervisor   = NodeName.INPUT_SUPERVISOR,
        route        = SupervisorRoute.PROCEED,
        rework_count = rework_count,
        rationale    = rationale,
    )


def _request_rework(
    reason_code:  RetryReasonCode,
    params:       dict,
    rationale:    str,
    rework_count: int,
) -> SupervisorDecision:
    """Constructs a REWORK SupervisorDecision with a typed RetryInstruction."""
    return SupervisorDecision(
        supervisor   = NodeName.INPUT_SUPERVISOR,
        route        = SupervisorRoute.REWORK,
        rework_count = rework_count,
        rationale    = rationale,
        retry_instruction = RetryInstruction(
            target_node          = NodeName.TOPIC,
            reason_code          = reason_code,
            parameter_adjustment = params,
        ),
    )