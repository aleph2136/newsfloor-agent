"""
nodes/output_supervisor.py

Evaluates the synthesis output before it goes to delivery.

This supervisor has one question: is this digest worth sending?

It evaluates the finished digest against the engineer profile and the
topic/focus angle it was generated from. If the digest is thin, generic,
off-topic, or missing required structure it requests a rework of synthesis
only — the articles are not re-fetched, the topic is not re-selected.

Why this supervisor is different from the input supervisor
──────────────────────────────────────────────────────────
The input supervisor evaluated raw material. This supervisor evaluates
a finished artifact. The failure modes are different:

  Input failures:   wrong topic, not enough articles, low scores
  Output failures:  generic writing, missing sections, poor personalization,
                    digest doesn't reflect the focus angle, signals not extracted

The LLM evaluator here reads the actual digest HTML and checks it against
the engineer profile — something no structural check can do.

Structural checks are still run first for the same reason as before:
deterministic failures don't need LLM reasoning.

Rework behavior
───────────────
Rework always targets synthesis_node only. The articles and topic are
reused as-is — only the writing and extraction reruns.

  DIGEST_INSUFFICIENT    → rewrite with stricter depth requirements
  MISSING_REQUIRED_FIELD → regenerate with explicit field checklist
"""

from __future__ import annotations
import logging
import re

from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM

from config import settings
from contracts.nodes import OutputSupervisorInput
from contracts.primitives import (
    NodeName,
    RetryInstruction,
    RetryReasonCode,
    SupervisorDecision,
    SupervisorRoute,
)
from node_definitions.crew_utils import kickoff_crew

logger = logging.getLogger(__name__)

# Minimum digest length in characters. Below this it's almost certainly
# too thin to be useful regardless of content quality.
MIN_DIGEST_LENGTH = 800

# Required HTML structural markers. If any are missing the digest is
# structurally incomplete regardless of writing quality.
REQUIRED_HTML_MARKERS = ["<h1>", "<h2>", "<em>"]


def run(supervisor_input: OutputSupervisorInput) -> SupervisorDecision:
    """
    Evaluates the synthesis output and returns a SupervisorDecision.
    Forces proceed in degraded mode if max_reworks has been reached.
    """
    rework_count = supervisor_input.rework_count

    logger.info({
        "node":           "output_supervisor",
        "rework_count":   rework_count,
        "max_reworks":    supervisor_input.max_reworks,
        "digest_length":  len(supervisor_input.synthesis_result.digest_html),
        "new_signals":    len(supervisor_input.synthesis_result.new_signals),
        "confirmations":  len(supervisor_input.synthesis_result.trend_confirmations),
    })

    # --- Hard gate: force proceed after max reworks ---
    if rework_count >= supervisor_input.max_reworks:
        logger.warning({
            "node":    "output_supervisor",
            "message": "Max reworks reached — proceeding in degraded mode",
        })
        return SupervisorDecision(
            supervisor   = NodeName.OUTPUT_SUPERVISOR,
            route        = SupervisorRoute.PROCEED,
            rework_count = rework_count,
            rationale    = (
                f"Max reworks ({supervisor_input.max_reworks}) reached. "
                "Proceeding in degraded mode with current digest."
            ),
        )

    # --- Structural checks first ---
    structural_issue = _check_structural_gates(supervisor_input)
    if structural_issue:
        return _request_rework(
            reason_code  = structural_issue["reason_code"],
            params       = structural_issue["params"],
            rationale    = structural_issue["rationale"],
            rework_count = rework_count,
        )

    # --- LLM-backed quality evaluation ---
    return _evaluate_with_llm(supervisor_input, rework_count)


# ---------------------------------------------------------------------------
# Structural gate checks
# ---------------------------------------------------------------------------

def _check_structural_gates(
    supervisor_input: OutputSupervisorInput,
) -> dict | None:
    """
    Checks hard structural criteria that don't require LLM reasoning.
    Returns a dict describing the issue if a gate fails, None if all pass.
    """
    digest_html = supervisor_input.synthesis_result.digest_html

    # Digest is too short to contain meaningful content
    if len(digest_html) < MIN_DIGEST_LENGTH:
        return {
            "reason_code": RetryReasonCode.DIGEST_INSUFFICIENT,
            "params": {
                "digest_length":    len(digest_html),
                "min_length":       MIN_DIGEST_LENGTH,
            },
            "rationale": (
                f"Digest is only {len(digest_html)} characters "
                f"(minimum {MIN_DIGEST_LENGTH}). Content is too thin to be useful."
            ),
        }

    # Required HTML structural markers are missing
    missing_markers = [
        marker for marker in REQUIRED_HTML_MARKERS
        if marker not in digest_html.lower()
    ]
    if missing_markers:
        return {
            "reason_code": RetryReasonCode.MISSING_REQUIRED_FIELD,
            "params": {
                "missing_fields": missing_markers,
            },
            "rationale": (
                f"Digest is missing required HTML structure: "
                f"{', '.join(missing_markers)}. "
                "The digest must include h1, h2, and em tags."
            ),
        }

    # No signals were extracted — synthesis crew likely failed silently
    if not supervisor_input.synthesis_result.new_signals and \
       not supervisor_input.synthesis_result.trend_confirmations:
        return {
            "reason_code": RetryReasonCode.MISSING_REQUIRED_FIELD,
            "params": {
                "missing_fields": ["new_signals or trend_confirmations"],
            },
            "rationale": (
                "No trend signals or confirmations were extracted. "
                "Signal extraction likely failed — reworking synthesis."
            ),
        }

    return None


# ---------------------------------------------------------------------------
# LLM-backed quality evaluation
# ---------------------------------------------------------------------------

def _evaluate_with_llm(
    supervisor_input: OutputSupervisorInput,
    rework_count:     int,
) -> SupervisorDecision:
    """
    Uses a single LLM-backed agent to evaluate digest quality holistically.
    Reads the actual digest HTML and checks it against the engineer profile
    and the topic/focus angle it was generated from.
    """
    llm = LLM(model=settings.bedrock_model_output_supervisor, max_retries=1)

    profile = supervisor_input.engineer_profile

    evaluator = Agent(
        role="Digest Quality Evaluator",
        goal=(
            "Evaluate whether today's digest is genuinely worth sending to a "
            "senior AI agentic engineer. Apply high standards — generic content "
            "that could have been written without reading the articles should fail. "
            "Content that gives the engineer real, applicable insight should pass."
        ),
        backstory=(
            f"You are evaluating a digest written for {profile.name}, "
            f"a {profile.experience_level} whose focus areas include: "
            f"{', '.join(profile.focus_areas)}. "
            "You have read thousands of technical digests and can immediately "
            "tell the difference between genuine insight and padded summaries. "
            "You apply the same standard you would want applied to your own inbox."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    # Trim the digest for the prompt — we don't need the full HTML,
    # just enough to evaluate quality. 3000 chars covers most digests.
    digest_excerpt = supervisor_input.synthesis_result.digest_html[:3000]
    if len(supervisor_input.synthesis_result.digest_html) > 3000:
        digest_excerpt += "\n... [truncated for evaluation]"

    evaluate_task = Task(
        description=f"""
Evaluate the quality of today's AI agentic engineering digest.

EXPECTED TOPIC:      {supervisor_input.topic}
EXPECTED FOCUS ANGLE: {supervisor_input.focus_angle}

ENGINEER PROFILE:
  Name:         {profile.name}
  Focus areas:  {', '.join(profile.focus_areas)}
  Level:        {profile.experience_level}

SIGNALS EXTRACTED:
  New signals:          {', '.join(supervisor_input.synthesis_result.new_signals) or 'None'}
  Trend confirmations:  {', '.join(supervisor_input.synthesis_result.trend_confirmations) or 'None'}

DIGEST (excerpt):
{digest_excerpt}

EVALUATION CRITERIA — the digest must pass ALL of these:

1. ON TOPIC: Does the digest actually address the expected topic and focus angle,
   or does it drift into generic AI content?

2. PERSONALIZED: Is the content tailored to {profile.name}'s focus areas
   (agentic architecture, engineering governance, observability, reliability)?
   Or could it have been written for any developer?

3. SUBSTANTIVE: Does each article section contain genuine engineering insight?
   Are the "Why this matters" callouts specific and applicable?

4. CONNECTED: Does the digest connect articles to each other or to trends
   where real connections exist? Or are they treated as isolated summaries?

5. ACTIONABLE: Does the closing takeaway give {profile.name} something
   concrete to think about or act on?

If the digest passes all criteria, return PROCEED.
If it fails on one or more criteria, return REWORK.

Return a JSON object with exactly these fields:
{{
  "decision": "PROCEED" or "REWORK",
  "reason_code": "DIGEST_INSUFFICIENT" or "MISSING_REQUIRED_FIELD" or null,
  "failed_criteria": ["<criterion name>", ...],
  "rationale": "<two to three sentences explaining your decision>"
}}
        """,
        expected_output=(
            "A JSON object with fields: decision, reason_code, "
            "failed_criteria, rationale."
        ),
        agent=evaluator,
    )

    crew = Crew(
        agents  = [evaluator],
        tasks   = [evaluate_task],
        process = Process.sequential,
        verbose = False,
    )

    kickoff_crew(crew, "output_supervisor", supervisor_input.run_id, [settings.bedrock_model_output_supervisor])

    if not evaluate_task.output or not evaluate_task.output.raw:
        logger.warning({
            "node":    "output_supervisor",
            "warning": "Evaluator crew produced no output — defaulting to PROCEED",
        })
        return SupervisorDecision(
            supervisor   = NodeName.OUTPUT_SUPERVISOR,
            route        = SupervisorRoute.PROCEED,
            rework_count = rework_count,
            rationale    = "Evaluator produced no output — proceeding by default.",
        )

    return _parse_llm_decision(
        raw_output   = evaluate_task.output.raw,
        rework_count = rework_count,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_llm_decision(
    raw_output:   str,
    rework_count: int,
) -> SupervisorDecision:
    """
    Parses the LLM evaluator's JSON output into a SupervisorDecision.
    Defaults to PROCEED on parse failure.
    """
    import json

    try:
        match = re.search(r"\{.*\}", raw_output, re.DOTALL)
        data  = json.loads(match.group()) if match else {}
    except (json.JSONDecodeError, AttributeError):
        logger.warning({
            "node":    "output_supervisor",
            "warning": "Could not parse LLM decision — defaulting to PROCEED",
        })
        return SupervisorDecision(
            supervisor   = NodeName.OUTPUT_SUPERVISOR,
            route        = SupervisorRoute.PROCEED,
            rework_count = rework_count,
            rationale    = "LLM output could not be parsed — proceeding by default.",
        )

    decision_str    = data.get("decision", "PROCEED").upper()
    route           = SupervisorRoute.REWORK if decision_str == "REWORK" else SupervisorRoute.PROCEED
    rationale       = data.get("rationale", "")
    reason_str      = data.get("reason_code")
    failed_criteria = data.get("failed_criteria", [])

    if route == SupervisorRoute.REWORK and reason_str:
        try:
            reason_code = RetryReasonCode(reason_str.lower())
        except ValueError:
            reason_code = RetryReasonCode.DIGEST_INSUFFICIENT

        return _request_rework(
            reason_code  = reason_code,
            params       = {"failed_criteria": failed_criteria},
            rationale    = rationale,
            rework_count = rework_count,
        )

    logger.info({
        "node":     "output_supervisor",
        "decision": decision_str,
        "rationale": rationale,
    })

    return SupervisorDecision(
        supervisor   = NodeName.OUTPUT_SUPERVISOR,
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
    """Constructs a REWORK SupervisorDecision targeting synthesis_node."""
    logger.info({
        "node":        "output_supervisor",
        "decision":    "REWORK",
        "reason_code": reason_code.value,
        "rationale":   rationale,
    })

    return SupervisorDecision(
        supervisor   = NodeName.OUTPUT_SUPERVISOR,
        route        = SupervisorRoute.REWORK,
        rework_count = rework_count,
        rationale    = rationale,
        retry_instruction = RetryInstruction(
            target_node          = NodeName.SYNTHESIS,
            reason_code          = reason_code,
            parameter_adjustment = params,
        ),
    )