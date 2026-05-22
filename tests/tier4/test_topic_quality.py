# tests/tier4/test_topic_quality.py
#
# LLM-as-judge tests for the Topic node.
#
# What we're checking
# ────────────────────
# The topic node selects a topic and focus angle for the daily digest.
# Structural checks (confidence in [0,1], topic is a non-empty string) are
# already enforced by the Pydantic contract. These tests catch regressions
# in *reasoning quality*:
#
#   1. topic_from_rotation      — topic must be one of the available options,
#                                  not a hallucinated or paraphrased variant
#   2. topic_not_repeated       — must not duplicate a topic from the last 14 days
#   3. focus_angle_is_specific  — focus angle names a concrete scenario or team,
#                                  not a generic restatement of the topic
#   4. rationale_is_contextual  — rationale references something from the input
#                                  (active trends, recent coverage, or profile)
#   5. confidence_is_calibrated — confidence sits in the plausible range [0.4, 0.95];
#                                  always-high scores (0.95+) suggest prompt anchoring
#
# Tests 1 and 2 are deterministic and run without the judge.
# Tests 3–5 delegate to the LLM judge because they require semantic reasoning.

import pytest
from conftest import requires_llm, judge_output

from config_loader import load_topics
from contracts.nodes import TopicTaskInput, TopicTaskResult
from node_definitions.topic import run as run_topic


# ---------------------------------------------------------------------------
# Canonical test input
# Fixed context so results are as deterministic as the LLM allows.
# ---------------------------------------------------------------------------

RECENT_TOPICS = [
    "agent observability and tracing",
    "structured outputs and contract-driven agents",
    "cost and latency optimization in agentic pipelines",
    "LLM tool use patterns",
    "vector databases for agent memory",
    "prompt caching strategies",
    "evaluation frameworks for LLM systems",
    "multi-modal agents",
    "fine-tuning vs prompting trade-offs",
    "LLM security and prompt injection",
]

ACTIVE_TREND_NAMES = [
    "Supervisor Patterns in Multi-Agent Systems",
    "Structured Outputs and Contract-Driven Agents",
]

RECENT_SIGNALS = [
    "LangGraph's supervisor node pattern is being adopted for production orchestration",
    "Teams are shifting toward Pydantic-validated agent outputs to reduce parsing failures",
]


@pytest.fixture(scope="module")
def topic_input() -> TopicTaskInput:
    return TopicTaskInput(
        run_id="tier4-topic-test",
        recent_topics=RECENT_TOPICS,
        active_trend_names=ACTIVE_TREND_NAMES,
        recent_signals=RECENT_SIGNALS,
        available_topics=load_topics(),
    )


@pytest.fixture(scope="module")
def topic_result(topic_input) -> TopicTaskResult:
    return run_topic(topic_input)


# ---------------------------------------------------------------------------
# Deterministic assertions (no judge needed)
# ---------------------------------------------------------------------------

@pytest.mark.tier4
@requires_llm
def test_topic_is_from_rotation(topic_result, topic_input):
    """Topic must be selected from available_topics — never hallucinated."""
    assert topic_result.topic in topic_input.available_topics, (
        f"Topic '{topic_result.topic}' is not in the available rotation."
    )


@pytest.mark.tier4
@requires_llm
def test_topic_not_in_recent_coverage(topic_result):
    """Topic must not duplicate recent coverage within the last 14 days."""
    assert topic_result.topic not in RECENT_TOPICS, (
        f"Topic '{topic_result.topic}' was covered recently and should have been avoided."
    )


@pytest.mark.tier4
@requires_llm
def test_confidence_in_plausible_range(topic_result):
    """Confidence must sit in [0.4, 0.95] — extremes suggest prompt anchoring."""
    assert 0.4 <= topic_result.confidence <= 0.95, (
        f"Confidence {topic_result.confidence} is outside the plausible range [0.4, 0.95]. "
        "Always-high scores suggest the model is not genuinely calibrating."
    )


# ---------------------------------------------------------------------------
# Semantic quality — evaluated by the judge
# ---------------------------------------------------------------------------

@pytest.mark.tier4
@requires_llm
def test_focus_angle_is_specific(topic_result):
    """Focus angle should name a concrete use case, not restate the topic."""
    verdict = judge_output(
        output={
            "topic": topic_result.topic,
            "focus_angle": topic_result.focus_angle,
        },
        criteria=[
            {
                "name": "focus_angle_is_specific",
                "description": (
                    "The focus_angle names a concrete scenario, team type, use case, or "
                    "application context. It is NOT a generic restatement of the topic. "
                    "For example, 'apply to platform engineering teams building LLM workflows' "
                    "is specific. 'Think about multi-agent systems' is not."
                ),
            },
            {
                "name": "focus_angle_distinct_from_topic",
                "description": (
                    "The focus_angle adds a lens or constraint beyond what the topic itself says. "
                    "It should not be semantically equivalent to the topic with different wording."
                ),
            },
        ],
    )
    assert verdict["criteria_results"]["focus_angle_is_specific"], (
        f"Focus angle '{topic_result.focus_angle}' failed specificity check. "
        f"Judge rationale: {verdict['rationale']}"
    )
    assert verdict["criteria_results"]["focus_angle_distinct_from_topic"], (
        f"Focus angle '{topic_result.focus_angle}' is too similar to topic '{topic_result.topic}'. "
        f"Judge rationale: {verdict['rationale']}"
    )


@pytest.mark.tier4
@requires_llm
def test_rationale_references_input_context(topic_result):
    """Rationale must reference something from the input — trends, recency, or profile."""
    verdict = judge_output(
        output={
            "topic": topic_result.topic,
            "rationale": topic_result.rationale,
        },
        criteria=[
            {
                "name": "rationale_is_contextual",
                "description": (
                    "The rationale references at least one concrete contextual factor: "
                    "an active trend by name or theme, a note about recent coverage, "
                    "or a connection to the engineer's profile (platform engineering, AWS, agents). "
                    "It must not be a generic explanation like 'this is an important topic'."
                ),
            },
        ],
    )
    assert verdict["criteria_results"]["rationale_is_contextual"], (
        f"Rationale '{topic_result.rationale}' lacks contextual grounding. "
        f"Judge rationale: {verdict['rationale']}"
    )


@pytest.mark.tier4
@requires_llm
def test_topic_selection_overall_quality(topic_result):
    """End-to-end quality check: is this a sensible selection for Sam today?"""
    verdict = judge_output(
        output={
            "topic": topic_result.topic,
            "focus_angle": topic_result.focus_angle,
            "rationale": topic_result.rationale,
            "confidence": topic_result.confidence,
        },
        criteria=[
            {
                "name": "selection_is_coherent",
                "description": (
                    "The topic, focus_angle, and rationale form a coherent unit — "
                    "the focus_angle sharpens the topic and the rationale explains "
                    "why this combination was chosen today."
                ),
            },
            {
                "name": "relevant_to_ai_engineer",
                "description": (
                    "The selection is relevant to a senior engineer working on AI agentic "
                    "architecture, multi-agent systems, platform engineering, and AWS. "
                    "It would produce a digest worth reading for someone at that level."
                ),
            },
        ],
    )
    assert verdict["passed"], (
        f"Topic selection failed overall quality check. "
        f"Results: {verdict['criteria_results']}. "
        f"Judge rationale: {verdict['rationale']}"
    )
