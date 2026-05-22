# tests/tier4/test_scoring_quality.py
#
# LLM-as-judge tests for the Scoring node.
#
# What we're checking
# ────────────────────
# The scoring node calls an LLM (RelevanceAnalyst) to assign relevance scores,
# then combines them deterministically with source reputation. These tests verify
# the LLM is actually discriminating — not assigning uniform or random scores.
#
# Deterministic checks (run first, no judge needed):
#   scores_in_valid_range        — all scores are floats in [0.0, 1.0]
#   counts_match_article_input   — total scored == total input, no silent drops
#   passed_and_filtered_disjoint — no article appears in both lists
#
# Semantic checks (delegated to the judge):
#   clearly_relevant_score_above_threshold   — the two "obviously on-topic" articles
#                                              should score > 0.5 combined
#   clearly_irrelevant_score_below_threshold — sport/finance articles should score < 0.4
#   discrimination_exists                    — gap between highest and lowest > 0.3
#   rationales_match_scores                  — high-scoring articles have rationales
#                                              saying they're relevant (not contradictions)
#
# Input design
# ─────────────
# articles_for_scoring (from conftest) provides three groups:
#   clearly_relevant:   about LangGraph supervisors and multi-agent coordination
#   clearly_irrelevant: sports results and financial news
#   borderline:         AI coding assistants (mentions agents tangentially)
# Topic: "multi-agent orchestration patterns"

import pytest
from conftest import requires_llm, judge_output

from contracts.nodes import ScoringTaskInput, ScoringTaskResult
from contracts.primitives import ArticleRaw
from node_definitions.scoring import run as run_scoring

TOPIC       = "multi-agent orchestration patterns"
FOCUS_ANGLE = "apply to platform engineering teams building LLM workflows"
ACTIVE_TRENDS = [
    "Supervisor Patterns in Multi-Agent Systems",
    "Structured Outputs and Contract-Driven Agents",
]
REPUTATION_MAP = {
    "blog.langchain.dev":    0.85,
    "lilianweng.github.io":  0.92,
    "bbc.com":               0.70,
    "reuters.com":           0.75,
    "techcrunch.com":        0.65,
}


# ---------------------------------------------------------------------------
# Run scoring once per module
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def all_articles(articles_for_scoring) -> list[ArticleRaw]:
    return (
        articles_for_scoring["clearly_relevant"]
        + articles_for_scoring["clearly_irrelevant"]
        + articles_for_scoring["borderline"]
    )


@pytest.fixture(scope="module")
def scoring_input(all_articles) -> ScoringTaskInput:
    return ScoringTaskInput(
        run_id="tier4-scoring-test",
        topic=TOPIC,
        focus_angle=FOCUS_ANGLE,
        articles=all_articles,
        source_reputation_map=REPUTATION_MAP,
        active_trend_names=ACTIVE_TRENDS,
        score_threshold=0.5,
    )


@pytest.fixture(scope="module")
def scoring_result(scoring_input) -> ScoringTaskResult:
    return run_scoring(scoring_input)


# ---------------------------------------------------------------------------
# Deterministic assertions
# ---------------------------------------------------------------------------

@pytest.mark.tier4
@requires_llm
def test_all_input_articles_are_scored(scoring_result, all_articles):
    """No article should be silently dropped — every input must appear in scored_articles."""
    assert len(scoring_result.scored_articles) == len(all_articles), (
        f"Expected {len(all_articles)} scored articles; got {len(scoring_result.scored_articles)}."
    )


@pytest.mark.tier4
@requires_llm
def test_counts_match_passed_plus_filtered(scoring_result):
    total = len(scoring_result.passed_articles) + len(scoring_result.filtered_articles)
    assert total == len(scoring_result.scored_articles), (
        "passed_articles + filtered_articles must equal scored_articles."
    )


@pytest.mark.tier4
@requires_llm
def test_high_quality_count_matches_passed_len(scoring_result):
    assert scoring_result.high_quality_count == len(scoring_result.passed_articles)


@pytest.mark.tier4
@requires_llm
def test_low_quality_count_matches_filtered_len(scoring_result):
    assert scoring_result.low_quality_count == len(scoring_result.filtered_articles)


@pytest.mark.tier4
@requires_llm
def test_all_scores_in_valid_range(scoring_result):
    for article in scoring_result.scored_articles:
        assert 0.0 <= article.relevance_score <= 1.0, (
            f"relevance_score {article.relevance_score} out of range for {article.article_id}"
        )
        assert 0.0 <= article.reputation_score <= 1.0, (
            f"reputation_score {article.reputation_score} out of range for {article.article_id}"
        )
        assert 0.0 <= article.combined_score <= 1.0, (
            f"combined_score {article.combined_score} out of range for {article.article_id}"
        )


@pytest.mark.tier4
@requires_llm
def test_passed_and_filtered_are_disjoint(scoring_result):
    passed_ids   = {a.article_id for a in scoring_result.passed_articles}
    filtered_ids = {a.article_id for a in scoring_result.filtered_articles}
    overlap = passed_ids & filtered_ids
    assert not overlap, (
        f"Articles appear in both passed and filtered lists: {overlap}"
    )


@pytest.mark.tier4
@requires_llm
def test_passed_threshold_flag_consistent_with_threshold(scoring_input, scoring_result):
    """passed_threshold must be True iff combined_score >= score_threshold."""
    threshold = scoring_input.score_threshold
    for article in scoring_result.scored_articles:
        expected = article.combined_score >= threshold
        assert article.passed_threshold == expected, (
            f"Article {article.article_id}: combined_score={article.combined_score:.4f}, "
            f"threshold={threshold}, but passed_threshold={article.passed_threshold}"
        )


# ---------------------------------------------------------------------------
# Score discrimination — clearly relevant vs clearly irrelevant
# ---------------------------------------------------------------------------

@pytest.mark.tier4
@requires_llm
def test_clearly_relevant_articles_score_above_irrelevant(scoring_result, articles_for_scoring):
    """Relevant articles must have higher combined scores than irrelevant ones."""
    relevant_ids   = {a.article_id for a in articles_for_scoring["clearly_relevant"]}
    irrelevant_ids = {a.article_id for a in articles_for_scoring["clearly_irrelevant"]}

    scored_map = {a.article_id: a for a in scoring_result.scored_articles}

    avg_relevant   = sum(scored_map[i].combined_score for i in relevant_ids)   / len(relevant_ids)
    avg_irrelevant = sum(scored_map[i].combined_score for i in irrelevant_ids) / len(irrelevant_ids)

    assert avg_relevant > avg_irrelevant, (
        f"Clearly relevant articles (avg={avg_relevant:.3f}) should score higher than "
        f"clearly irrelevant articles (avg={avg_irrelevant:.3f}). "
        "The RelevanceAnalyst prompt may not be discriminating correctly."
    )


@pytest.mark.tier4
@requires_llm
def test_discrimination_gap_is_meaningful(scoring_result):
    """max combined_score - min combined_score > 0.3 — scores should not be uniform."""
    scores = [a.combined_score for a in scoring_result.scored_articles]
    gap = max(scores) - min(scores)
    assert gap > 0.3, (
        f"Score range is only {gap:.3f}. Scores appear uniform — the LLM may not be "
        "discriminating relevance (all high, all low, or random noise)."
    )


# ---------------------------------------------------------------------------
# Semantic quality — evaluated by the judge
# ---------------------------------------------------------------------------

@pytest.mark.tier4
@requires_llm
def test_clearly_relevant_articles_pass_threshold(scoring_result, articles_for_scoring):
    """Clearly on-topic articles should pass the 0.5 threshold after scoring."""
    relevant_ids = {a.article_id for a in articles_for_scoring["clearly_relevant"]}
    scored_map   = {a.article_id: a for a in scoring_result.scored_articles}

    failed = [aid for aid in relevant_ids if not scored_map[aid].passed_threshold]
    assert not failed, (
        f"Clearly relevant articles failed the threshold: {failed}. "
        "Scores: " + str({aid: scored_map[aid].combined_score for aid in failed})
    )


@pytest.mark.tier4
@requires_llm
def test_clearly_irrelevant_articles_fail_threshold(scoring_result, articles_for_scoring):
    """Sports and finance articles should not pass the relevance threshold."""
    irrelevant_ids = {a.article_id for a in articles_for_scoring["clearly_irrelevant"]}
    scored_map     = {a.article_id: a for a in scoring_result.scored_articles}

    passed = [aid for aid in irrelevant_ids if scored_map[aid].passed_threshold]
    assert not passed, (
        f"Clearly irrelevant articles passed the threshold: {passed}. "
        "Scores: " + str({aid: scored_map[aid].combined_score for aid in passed}) +
        " — the RelevanceAnalyst may be scoring too permissively."
    )


@pytest.mark.tier4
@requires_llm
def test_score_rationales_match_scores(scoring_result, articles_for_scoring):
    """High-scoring articles should have rationales confirming relevance, not contradicting it."""
    relevant_ids = {a.article_id for a in articles_for_scoring["clearly_relevant"]}
    scored_map   = {a.article_id: a for a in scoring_result.scored_articles}

    high_scoring = [scored_map[aid] for aid in relevant_ids]

    verdict = judge_output(
        output=[
            {
                "article_id":       a.article_id,
                "title":            a.title,
                "combined_score":   a.combined_score,
                "score_rationale":  a.score_rationale,
            }
            for a in high_scoring
        ],
        criteria=[
            {
                "name": "rationale_consistent_with_score",
                "description": (
                    "For each article, the score_rationale's explanation of relevance is "
                    "consistent with a high combined_score. The rationale should NOT say "
                    "the article is irrelevant or only tangentially related while the score "
                    "is high. Contradictions between rationale text and score indicate "
                    "a parsing or prompt alignment failure."
                ),
            },
        ],
    )
    assert verdict["criteria_results"]["rationale_consistent_with_score"], (
        f"Score rationales contradict the numeric scores for relevant articles. "
        f"Judge rationale: {verdict['rationale']}"
    )
