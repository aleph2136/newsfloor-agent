# tests/tier4/test_synthesis_quality.py
#
# LLM-as-judge tests for the Synthesis node.
#
# What we're checking
# ────────────────────
# The synthesis node receives scored articles and produces:
#   - digest_html:         full HTML digest for SES delivery
#   - digest_summary:      3-5 sentence plain-text summary for DynamoDB
#   - new_signals:         trend signals extracted from today's articles
#   - trend_confirmations: names of active trends this run reinforces
#
# Structural invariants (valid HTML tags present, non-empty strings) are
# checked deterministically first. Semantic quality is delegated to the judge.
#
# Judge criteria
# ───────────────
#   html_has_required_structure  — contains <h1>, ≥2 <h2>, and body content
#   topic_and_angle_addressed    — digest references the topic and focus angle
#   appropriate_technical_depth  — content suits a senior engineer; not introductory
#   no_phantom_urls              — every URL in the digest appears in the input articles
#   signals_are_specific         — new_signals describe observable patterns, not platitudes
#   summary_is_coherent          — plain-text summary reads as complete sentences

import re
import pytest
from conftest import requires_llm, judge_output

from contracts.nodes import SynthesisTaskInput, SynthesisTaskResult
from node_definitions.synthesis import run as run_synthesis

TOPIC       = "multi-agent orchestration patterns"
FOCUS_ANGLE = "apply to platform engineering teams building LLM workflows"

RECENT_RUN_SIGNALS = [
    "LangGraph supervisor pattern adoption growing in production deployments",
    "Pydantic-validated outputs reducing downstream parsing failures",
    "Teams investing in rework loop caps to prevent runaway agent costs",
]


# ---------------------------------------------------------------------------
# Run synthesis once per module — LLM calls are expensive
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthesis_input(sam_profile, active_trends, canonical_passed_articles) -> SynthesisTaskInput:
    return SynthesisTaskInput(
        run_id="tier4-synthesis-test",
        topic=TOPIC,
        focus_angle=FOCUS_ANGLE,
        passed_articles=canonical_passed_articles,
        active_trends=active_trends,
        recent_run_signals=RECENT_RUN_SIGNALS,
        engineer_profile=sam_profile,
    )


@pytest.fixture(scope="module")
def synthesis_result(synthesis_input) -> SynthesisTaskResult:
    return run_synthesis(synthesis_input)


# ---------------------------------------------------------------------------
# Deterministic structural assertions
# ---------------------------------------------------------------------------

@pytest.mark.tier4
@requires_llm
def test_digest_html_contains_h1(synthesis_result):
    assert "<h1" in synthesis_result.digest_html.lower(), (
        "digest_html must contain an <h1> tag."
    )


@pytest.mark.tier4
@requires_llm
def test_digest_html_contains_at_least_two_h2(synthesis_result):
    h2_count = len(re.findall(r"<h2", synthesis_result.digest_html, re.IGNORECASE))
    assert h2_count >= 2, (
        f"digest_html contains only {h2_count} <h2> tag(s) — expected at least 2 sections."
    )


@pytest.mark.tier4
@requires_llm
def test_digest_html_is_non_empty(synthesis_result):
    assert len(synthesis_result.digest_html.strip()) > 200, (
        "digest_html is suspiciously short — the digest may not have been generated."
    )


@pytest.mark.tier4
@requires_llm
def test_digest_summary_is_non_empty(synthesis_result):
    assert len(synthesis_result.digest_summary.strip()) > 50, (
        "digest_summary is too short to be a 3-5 sentence summary."
    )


@pytest.mark.tier4
@requires_llm
def test_new_signals_extracted(synthesis_result):
    assert len(synthesis_result.new_signals) >= 2, (
        f"Expected at least 2 new_signals; got {len(synthesis_result.new_signals)}. "
        "The synthesis prompt may not be extracting signals correctly."
    )


@pytest.mark.tier4
@requires_llm
def test_new_signals_are_strings(synthesis_result):
    for signal in synthesis_result.new_signals:
        assert isinstance(signal, str) and len(signal) > 0, (
            f"Signal is empty or not a string: {signal!r}"
        )


# ---------------------------------------------------------------------------
# No phantom URL check — deterministic
# ---------------------------------------------------------------------------

@pytest.mark.tier4
@requires_llm
def test_no_phantom_article_urls(synthesis_result, canonical_passed_articles):
    """Every URL embedded in digest_html must come from the input articles."""
    input_urls = {a.url for a in canonical_passed_articles}

    # Extract http/https URLs from the digest
    found_urls = set(re.findall(r'https?://[^\s"<>]+', synthesis_result.digest_html))

    phantom = found_urls - input_urls
    assert not phantom, (
        f"digest_html contains URLs not present in the input articles: {phantom}. "
        "This indicates hallucinated sources."
    )


# ---------------------------------------------------------------------------
# Semantic quality — evaluated by the judge
# ---------------------------------------------------------------------------

@pytest.mark.tier4
@requires_llm
def test_digest_addresses_topic_and_focus_angle(synthesis_result):
    verdict = judge_output(
        output={
            "topic":        TOPIC,
            "focus_angle":  FOCUS_ANGLE,
            "digest_html":  synthesis_result.digest_html[:3000],  # truncate for judge context
        },
        criteria=[
            {
                "name": "topic_addressed",
                "description": (
                    f"The digest content is clearly about '{TOPIC}'. "
                    "It discusses concepts directly related to this topic, not something adjacent."
                ),
            },
            {
                "name": "focus_angle_addressed",
                "description": (
                    f"The digest explicitly applies the content to '{FOCUS_ANGLE}'. "
                    "There should be at least one concrete reference to platform engineering, "
                    "LLM workflows, or team-level application of the topic."
                ),
            },
        ],
    )
    assert verdict["criteria_results"]["topic_addressed"], (
        f"Digest does not address the topic '{TOPIC}'. "
        f"Judge rationale: {verdict['rationale']}"
    )
    assert verdict["criteria_results"]["focus_angle_addressed"], (
        f"Digest does not address the focus angle '{FOCUS_ANGLE}'. "
        f"Judge rationale: {verdict['rationale']}"
    )


@pytest.mark.tier4
@requires_llm
def test_digest_has_appropriate_technical_depth(synthesis_result, sam_profile):
    verdict = judge_output(
        output={
            "engineer_profile": {
                "experience_level": sam_profile.experience_level,
                "focus_areas": sam_profile.focus_areas,
            },
            "digest_html": synthesis_result.digest_html[:3000],
        },
        criteria=[
            {
                "name": "appropriate_technical_depth",
                "description": (
                    "The digest uses technical language appropriate for a senior engineer "
                    "working on AI agentic architecture. It assumes familiarity with concepts "
                    "like LangGraph, CrewAI, supervisor nodes, and Pydantic. "
                    "It does NOT over-explain basic LLM concepts or use introductory framing "
                    "like 'LLMs are AI systems that...'."
                ),
            },
            {
                "name": "actionable_for_engineer",
                "description": (
                    "The digest contains at least one concrete takeaway, pattern, or insight "
                    "that a senior engineer could apply directly — not just a high-level summary."
                ),
            },
        ],
    )
    assert verdict["passed"], (
        f"Synthesis failed technical depth check. "
        f"Results: {verdict['criteria_results']}. "
        f"Judge rationale: {verdict['rationale']}"
    )


@pytest.mark.tier4
@requires_llm
def test_new_signals_are_specific_not_generic(synthesis_result):
    verdict = judge_output(
        output={"new_signals": synthesis_result.new_signals},
        criteria=[
            {
                "name": "signals_are_specific",
                "description": (
                    "Each signal describes a specific, observable pattern or development — "
                    "something that could be tracked as a trend. "
                    "Generic statements like 'AI is advancing rapidly' or "
                    "'multi-agent systems are important' do NOT pass. "
                    "Specific examples that pass: "
                    "'Supervisor node pattern used to cap rework loops in production LangGraph pipelines', "
                    "'Teams migrating from free-text LLM outputs to Pydantic-validated task contracts'."
                ),
            },
            {
                "name": "signals_not_duplicates",
                "description": (
                    "The signals are distinct from each other — they each describe a "
                    "different pattern or development, not variations of the same point."
                ),
            },
        ],
    )
    assert verdict["criteria_results"]["signals_are_specific"], (
        f"new_signals contain generic statements. "
        f"Signals: {synthesis_result.new_signals}. "
        f"Judge rationale: {verdict['rationale']}"
    )
    assert verdict["criteria_results"]["signals_not_duplicates"], (
        f"new_signals contain duplicate or near-duplicate entries. "
        f"Signals: {synthesis_result.new_signals}. "
        f"Judge rationale: {verdict['rationale']}"
    )


@pytest.mark.tier4
@requires_llm
def test_digest_summary_is_coherent(synthesis_result):
    verdict = judge_output(
        output={"digest_summary": synthesis_result.digest_summary},
        criteria=[
            {
                "name": "summary_is_coherent",
                "description": (
                    "The digest_summary is written in complete sentences and reads as a "
                    "coherent 3-5 sentence paragraph. It captures the main themes of the "
                    "digest without being a bullet list or fragment."
                ),
            },
            {
                "name": "summary_references_topic",
                "description": (
                    f"The summary references the topic '{TOPIC}' or concepts closely related to it."
                ),
            },
        ],
    )
    assert verdict["passed"], (
        f"digest_summary failed coherence check. "
        f"Summary: {synthesis_result.digest_summary!r}. "
        f"Judge rationale: {verdict['rationale']}"
    )
