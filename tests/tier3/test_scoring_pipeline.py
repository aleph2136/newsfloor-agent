# tests/tier3/test_scoring_pipeline.py
#
# Validates the deterministic scoring math, JSON parsing, and retry threshold
# logic in node_definitions/scoring.py without invoking any LLM.
# All _score_relevance calls are patched out — only the pure functions run.

import json
import pytest
from unittest.mock import patch

from contracts.primitives import ArticleRaw, ArticleScored, NodeName, RetryReasonCode, RetryInstruction
from contracts.nodes import ScoringTaskInput, ScoringTaskResult
from node_definitions.scoring import (
    _combine_scores,
    _parse_relevance_output,
    _apply_retry_adjustments,
    run,
    DEFAULT_REPUTATION,
    RELEVANCE_WEIGHT,
    REPUTATION_WEIGHT,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _article(article_id: str, source_domain: str = "example.com") -> ArticleRaw:
    return ArticleRaw(
        article_id=article_id,
        url=f"https://{source_domain}/{article_id}",
        title=f"Title for {article_id}",
        source_domain=source_domain,
        published_at="2026-01-01T00:00:00",
        summary=f"Summary for {article_id}.",
    )


def _scoring_input(articles=None, reputation_map=None, threshold=0.5, instruction=None) -> ScoringTaskInput:
    return ScoringTaskInput(
        run_id="test-run-001",
        topic="multi-agent orchestration patterns",
        focus_angle="apply to platform engineering teams",
        articles=articles or [],
        source_reputation_map=reputation_map or {},
        active_trend_names=["LLM Routing"],
        score_threshold=threshold,
        retry_instruction=instruction,
    )


# ---------------------------------------------------------------------------
# _combine_scores — arithmetic
# ---------------------------------------------------------------------------

class TestCombineScores:

    def test_high_relevance_unknown_source(self):
        # Docstring example: relevance 0.9, rep 0.5 (unknown) → 0.760
        a = _article("a1", "newsite.io")
        scores = {"a1": {"relevance_score": 0.9, "relevance_rationale": "Strong match."}}
        result = _combine_scores([a], scores, reputation_map={}, score_threshold=0.5)
        # (0.9 × 0.65) + (0.5 × 0.35) = 0.585 + 0.175 = 0.760
        assert result[0].combined_score == pytest.approx(0.760, abs=1e-4)

    def test_low_relevance_high_reputation(self):
        # Docstring example: relevance 0.3, rep 0.9 → 0.510 (marginal pass)
        a = _article("a2", "trusted.com")
        scores = {"a2": {"relevance_score": 0.3, "relevance_rationale": "Tangential."}}
        result = _combine_scores([a], scores, reputation_map={"trusted.com": 0.9}, score_threshold=0.5)
        # (0.3 × 0.65) + (0.9 × 0.35) = 0.195 + 0.315 = 0.510
        assert result[0].combined_score == pytest.approx(0.510, abs=1e-4)
        assert result[0].passed_threshold is True

    def test_all_zeros_produce_zero_combined(self):
        a = _article("a3", "x.com")
        scores = {"a3": {"relevance_score": 0.0, "relevance_rationale": "Irrelevant."}}
        result = _combine_scores([a], scores, reputation_map={"x.com": 0.0}, score_threshold=0.5)
        assert result[0].combined_score == pytest.approx(0.0)

    def test_all_ones_produce_one_combined(self):
        a = _article("a4", "x.com")
        scores = {"a4": {"relevance_score": 1.0, "relevance_rationale": "Perfect."}}
        result = _combine_scores([a], scores, reputation_map={"x.com": 1.0}, score_threshold=0.5)
        assert result[0].combined_score == pytest.approx(1.0)

    def test_missing_from_relevance_scores_uses_default_relevance(self):
        # Article not returned by LLM falls back to DEFAULT_REPUTATION for relevance
        a = _article("a5", "x.com")
        result = _combine_scores([a], relevance_scores={}, reputation_map={}, score_threshold=0.5)
        expected = (DEFAULT_REPUTATION * RELEVANCE_WEIGHT) + (DEFAULT_REPUTATION * REPUTATION_WEIGHT)
        assert result[0].combined_score == pytest.approx(expected, abs=1e-4)
        assert result[0].relevance_score == DEFAULT_REPUTATION

    def test_unknown_domain_uses_default_reputation(self):
        a = _article("a6", "noreq.com")
        scores = {"a6": {"relevance_score": 0.7, "relevance_rationale": "Relevant."}}
        result = _combine_scores([a], scores, reputation_map={}, score_threshold=0.5)
        assert result[0].reputation_score == DEFAULT_REPUTATION

    def test_passes_at_exactly_threshold(self):
        # combined_score == threshold → passes (>=)
        a = _article("a7", "x.com")
        # (0.5 × 0.65) + (0.5 × 0.35) = 0.5 exactly at default weights
        scores = {"a7": {"relevance_score": 0.5, "relevance_rationale": "Moderate."}}
        result = _combine_scores([a], scores, reputation_map={"x.com": 0.5}, score_threshold=0.5)
        assert result[0].passed_threshold is True

    def test_fails_just_below_threshold(self):
        a = _article("a8", "x.com")
        scores = {"a8": {"relevance_score": 0.4, "relevance_rationale": "Low."}}
        # (0.4 × 0.65) + (0.5 × 0.35) = 0.26 + 0.175 = 0.435 < 0.5
        result = _combine_scores([a], scores, reputation_map={"x.com": 0.5}, score_threshold=0.5)
        assert result[0].passed_threshold is False

    def test_combined_score_rounded_to_4_decimals(self):
        a = _article("a9", "x.com")
        scores = {"a9": {"relevance_score": 0.333, "relevance_rationale": "Approx."}}
        result = _combine_scores([a], scores, reputation_map={"x.com": 0.333}, score_threshold=0.5)
        assert result[0].combined_score == round(result[0].combined_score, 4)

    def test_empty_article_list_returns_empty(self):
        result = _combine_scores([], {}, {}, score_threshold=0.5)
        assert result == []

    def test_score_rationale_contains_both_numeric_scores(self):
        a = _article("a10", "x.com")
        scores = {"a10": {"relevance_score": 0.8, "relevance_rationale": "Very good."}}
        result = _combine_scores([a], scores, reputation_map={"x.com": 0.7}, score_threshold=0.5)
        assert "0.80" in result[0].score_rationale
        assert "0.70" in result[0].score_rationale

    def test_multiple_articles_scored_independently(self):
        articles = [_article("m1", "a.com"), _article("m2", "b.com")]
        scores = {
            "m1": {"relevance_score": 0.9, "relevance_rationale": "High."},
            "m2": {"relevance_score": 0.1, "relevance_rationale": "Low."},
        }
        result = _combine_scores(
            articles, scores,
            reputation_map={"a.com": 0.8, "b.com": 0.3},
            score_threshold=0.5,
        )
        assert len(result) == 2
        assert result[0].combined_score > result[1].combined_score

    def test_all_articles_pass_when_highly_relevant(self):
        articles = [_article("p1", "a.com"), _article("p2", "a.com")]
        scores = {
            "p1": {"relevance_score": 0.9, "relevance_rationale": "Relevant."},
            "p2": {"relevance_score": 0.85, "relevance_rationale": "Relevant."},
        }
        result = _combine_scores(articles, scores, reputation_map={"a.com": 0.8}, score_threshold=0.5)
        assert all(a.passed_threshold for a in result)

    def test_no_articles_pass_with_high_threshold(self):
        articles = [_article("f1", "a.com")]
        scores = {"f1": {"relevance_score": 0.1, "relevance_rationale": "Irrelevant."}}
        # (0.1 × 0.65) + (0.1 × 0.35) = 0.1 < 0.8
        result = _combine_scores(articles, scores, reputation_map={"a.com": 0.1}, score_threshold=0.8)
        assert all(not a.passed_threshold for a in result)

    def test_returned_article_ids_match_input_order(self):
        articles = [_article("first"), _article("second"), _article("third")]
        scores = {
            "first":  {"relevance_score": 0.5, "relevance_rationale": ""},
            "second": {"relevance_score": 0.5, "relevance_rationale": ""},
            "third":  {"relevance_score": 0.5, "relevance_rationale": ""},
        }
        result = _combine_scores(articles, scores, {}, score_threshold=0.5)
        assert [a.article_id for a in result] == ["first", "second", "third"]


# ---------------------------------------------------------------------------
# _parse_relevance_output — JSON parsing and fallback behavior
# ---------------------------------------------------------------------------

class TestParseRelevanceOutput:

    def test_clean_json_array(self):
        raw = json.dumps([
            {"article_id": "abc", "relevance_score": 0.8, "relevance_rationale": "Direct match."}
        ])
        result = _parse_relevance_output(raw)
        assert "abc" in result
        assert result["abc"]["relevance_score"] == pytest.approx(0.8)
        assert result["abc"]["relevance_rationale"] == "Direct match."

    def test_multiple_articles_all_parsed(self):
        raw = json.dumps([
            {"article_id": "a1", "relevance_score": 0.9, "relevance_rationale": "Strong."},
            {"article_id": "a2", "relevance_score": 0.3, "relevance_rationale": "Weak."},
        ])
        result = _parse_relevance_output(raw)
        assert len(result) == 2
        assert result["a1"]["relevance_score"] == pytest.approx(0.9)
        assert result["a2"]["relevance_score"] == pytest.approx(0.3)

    def test_json_embedded_in_markdown_code_fence(self):
        # LLMs often wrap JSON in markdown blocks
        raw = (
            "Here are the scores:\n"
            "```json\n"
            '[{"article_id": "abc", "relevance_score": 0.7, "relevance_rationale": "Good."}]\n'
            "```"
        )
        result = _parse_relevance_output(raw)
        assert "abc" in result

    def test_json_preceded_by_prose(self):
        raw = (
            "I have evaluated each article as follows:\n\n"
            '[{"article_id": "abc", "relevance_score": 0.6, "relevance_rationale": "OK."}]'
        )
        result = _parse_relevance_output(raw)
        assert "abc" in result

    def test_completely_invalid_json_returns_empty(self):
        result = _parse_relevance_output("This is not JSON at all.")
        assert result == {}

    def test_empty_string_returns_empty(self):
        result = _parse_relevance_output("")
        assert result == {}

    def test_empty_array_returns_empty(self):
        result = _parse_relevance_output("[]")
        assert result == {}

    def test_item_missing_article_id_is_skipped(self):
        raw = json.dumps([{"relevance_score": 0.8, "relevance_rationale": "No id present."}])
        result = _parse_relevance_output(raw)
        assert result == {}

    def test_item_missing_relevance_score_uses_default(self):
        raw = json.dumps([{"article_id": "abc", "relevance_rationale": "No score given."}])
        result = _parse_relevance_output(raw)
        assert result["abc"]["relevance_score"] == pytest.approx(DEFAULT_REPUTATION)

    def test_item_missing_rationale_falls_back_to_empty_string(self):
        raw = json.dumps([{"article_id": "abc", "relevance_score": 0.7}])
        result = _parse_relevance_output(raw)
        assert result["abc"]["relevance_rationale"] == ""

    def test_score_cast_to_float_when_string(self):
        # LLM may return numeric values as JSON strings
        raw = json.dumps([{"article_id": "abc", "relevance_score": "0.8", "relevance_rationale": "Good."}])
        result = _parse_relevance_output(raw)
        assert isinstance(result["abc"]["relevance_score"], float)
        assert result["abc"]["relevance_score"] == pytest.approx(0.8)

    def test_result_keyed_by_article_id(self):
        raw = json.dumps([
            {"article_id": "id-one",   "relevance_score": 0.5, "relevance_rationale": "Mid."},
            {"article_id": "id-two",   "relevance_score": 0.7, "relevance_rationale": "Good."},
        ])
        result = _parse_relevance_output(raw)
        assert set(result.keys()) == {"id-one", "id-two"}


# ---------------------------------------------------------------------------
# _apply_retry_adjustments — threshold modification logic
# ---------------------------------------------------------------------------

class TestApplyRetryAdjustments:

    def _input(self, threshold=0.5, instruction=None) -> ScoringTaskInput:
        return _scoring_input(threshold=threshold, instruction=instruction)

    def _instruction(self, reason_code: RetryReasonCode, params: dict | None = None) -> RetryInstruction:
        return RetryInstruction(
            node=NodeName.SCORING,
            reason_code=reason_code,
            parameter_adjustment=params or {},
        )

    def test_no_instruction_returns_original_threshold(self):
        assert _apply_retry_adjustments(self._input(threshold=0.5)) == pytest.approx(0.5)

    def test_below_score_threshold_lowers_by_0_1(self):
        task_input = self._input(threshold=0.5, instruction=self._instruction(RetryReasonCode.BELOW_SCORE_THRESHOLD))
        assert _apply_retry_adjustments(task_input) == pytest.approx(0.4)

    def test_below_score_threshold_uses_explicit_param(self):
        task_input = self._input(
            threshold=0.5,
            instruction=self._instruction(RetryReasonCode.BELOW_SCORE_THRESHOLD, {"score_threshold": 0.35}),
        )
        assert _apply_retry_adjustments(task_input) == pytest.approx(0.35)

    def test_below_score_threshold_floor_is_0_3(self):
        # 0.35 - 0.1 = 0.25, but floor clamps to 0.3
        task_input = self._input(threshold=0.35, instruction=self._instruction(RetryReasonCode.BELOW_SCORE_THRESHOLD))
        assert _apply_retry_adjustments(task_input) == pytest.approx(0.3)

    def test_below_score_threshold_already_at_floor_stays(self):
        task_input = self._input(threshold=0.3, instruction=self._instruction(RetryReasonCode.BELOW_SCORE_THRESHOLD))
        assert _apply_retry_adjustments(task_input) == pytest.approx(0.3)

    def test_low_quality_articles_raises_by_0_1(self):
        task_input = self._input(threshold=0.5, instruction=self._instruction(RetryReasonCode.LOW_QUALITY_ARTICLES))
        assert _apply_retry_adjustments(task_input) == pytest.approx(0.6)

    def test_low_quality_articles_uses_explicit_param(self):
        task_input = self._input(
            threshold=0.5,
            instruction=self._instruction(RetryReasonCode.LOW_QUALITY_ARTICLES, {"score_threshold": 0.75}),
        )
        assert _apply_retry_adjustments(task_input) == pytest.approx(0.75)

    def test_low_quality_articles_ceiling_is_0_8(self):
        # 0.75 + 0.1 = 0.85, but ceiling clamps to 0.8
        task_input = self._input(threshold=0.75, instruction=self._instruction(RetryReasonCode.LOW_QUALITY_ARTICLES))
        assert _apply_retry_adjustments(task_input) == pytest.approx(0.8)

    def test_low_quality_articles_already_at_ceiling_stays(self):
        task_input = self._input(threshold=0.8, instruction=self._instruction(RetryReasonCode.LOW_QUALITY_ARTICLES))
        assert _apply_retry_adjustments(task_input) == pytest.approx(0.8)

    def test_unhandled_reason_code_leaves_threshold_unchanged(self):
        # INSUFFICIENT_ARTICLES is not a scoring concern — threshold must not change
        task_input = self._input(threshold=0.5, instruction=self._instruction(RetryReasonCode.INSUFFICIENT_ARTICLES))
        assert _apply_retry_adjustments(task_input) == pytest.approx(0.5)

    def test_source_fetch_failure_leaves_threshold_unchanged(self):
        task_input = self._input(threshold=0.6, instruction=self._instruction(RetryReasonCode.SOURCE_FETCH_FAILURE))
        assert _apply_retry_adjustments(task_input) == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# run() — integration with _score_relevance patched out
# ---------------------------------------------------------------------------

class TestScoringRun:

    @pytest.fixture
    def two_articles(self):
        return [_article("r1", "a.com"), _article("r2", "b.com")]

    @pytest.fixture
    def mixed_scores(self):
        return {
            "r1": {"relevance_score": 0.9, "relevance_rationale": "Highly relevant."},
            "r2": {"relevance_score": 0.1, "relevance_rationale": "Off-topic."},
        }

    def test_run_returns_scoring_task_result(self, two_articles, mixed_scores):
        task_input = _scoring_input(two_articles, reputation_map={"a.com": 0.8, "b.com": 0.3})
        with patch("node_definitions.scoring._score_relevance", return_value=mixed_scores):
            result = run(task_input)
        assert isinstance(result, ScoringTaskResult)

    def test_run_propagates_run_id(self, two_articles, mixed_scores):
        task_input = _scoring_input(two_articles, reputation_map={"a.com": 0.8, "b.com": 0.3})
        with patch("node_definitions.scoring._score_relevance", return_value=mixed_scores):
            result = run(task_input)
        assert result.run_id == "test-run-001"

    def test_run_splits_passed_and_filtered(self, two_articles, mixed_scores):
        task_input = _scoring_input(two_articles, reputation_map={"a.com": 0.8, "b.com": 0.3})
        with patch("node_definitions.scoring._score_relevance", return_value=mixed_scores):
            result = run(task_input)
        # r1: (0.9×0.65) + (0.8×0.35) = 0.865 → pass
        # r2: (0.1×0.65) + (0.3×0.35) = 0.170 → fail
        assert result.high_quality_count == 1
        assert result.low_quality_count == 1

    def test_run_high_quality_count_equals_passed_articles_len(self, two_articles):
        scores = {
            "r1": {"relevance_score": 0.9, "relevance_rationale": "High."},
            "r2": {"relevance_score": 0.8, "relevance_rationale": "High."},
        }
        task_input = _scoring_input(two_articles, reputation_map={"a.com": 0.8, "b.com": 0.3})
        with patch("node_definitions.scoring._score_relevance", return_value=scores):
            result = run(task_input)
        assert result.high_quality_count == len(result.passed_articles)

    def test_run_low_quality_count_equals_filtered_articles_len(self, two_articles):
        scores = {
            "r1": {"relevance_score": 0.1, "relevance_rationale": "Low."},
            "r2": {"relevance_score": 0.1, "relevance_rationale": "Low."},
        }
        task_input = _scoring_input(two_articles, reputation_map={"a.com": 0.1, "b.com": 0.1})
        with patch("node_definitions.scoring._score_relevance", return_value=scores):
            result = run(task_input)
        assert result.low_quality_count == len(result.filtered_articles)

    def test_run_total_equals_input_article_count(self, two_articles, mixed_scores):
        task_input = _scoring_input(two_articles, reputation_map={"a.com": 0.8, "b.com": 0.3})
        with patch("node_definitions.scoring._score_relevance", return_value=mixed_scores):
            result = run(task_input)
        total = len(result.passed_articles) + len(result.filtered_articles)
        assert total == len(task_input.articles)

    def test_run_empty_articles_returns_empty_result(self):
        task_input = _scoring_input(articles=[], reputation_map={})
        with patch("node_definitions.scoring._score_relevance", return_value={}):
            result = run(task_input)
        assert result.scored_articles == []
        assert result.passed_articles == []
        assert result.filtered_articles == []
        assert result.high_quality_count == 0
        assert result.low_quality_count == 0

    def test_run_llm_failure_does_not_drop_articles(self, two_articles):
        # _score_relevance returns {} — all articles should still be scored with fallback
        task_input = _scoring_input(two_articles, reputation_map={"a.com": 0.8, "b.com": 0.3})
        with patch("node_definitions.scoring._score_relevance", return_value={}):
            result = run(task_input)
        assert len(result.scored_articles) == 2

    def test_run_llm_failure_uses_default_relevance_score(self, two_articles):
        task_input = _scoring_input(two_articles, reputation_map={})
        with patch("node_definitions.scoring._score_relevance", return_value={}):
            result = run(task_input)
        for article in result.scored_articles:
            assert article.relevance_score == DEFAULT_REPUTATION

    def test_run_applies_retry_threshold_before_scoring(self):
        # borderline article: combined ≈ 0.4675 fails at 0.5 but passes after BELOW_SCORE_THRESHOLD drops to 0.4
        article = _article("edge", "a.com")
        instruction = RetryInstruction(
            node=NodeName.SCORING,
            reason_code=RetryReasonCode.BELOW_SCORE_THRESHOLD,
            parameter_adjustment={},
        )
        task_input = _scoring_input(
            articles=[article],
            reputation_map={"a.com": 0.5},
            threshold=0.5,
            instruction=instruction,
        )
        mock_scores = {"edge": {"relevance_score": 0.45, "relevance_rationale": "Borderline."}}
        with patch("node_definitions.scoring._score_relevance", return_value=mock_scores):
            result = run(task_input)
        # threshold dropped to 0.4; (0.45×0.65)+(0.5×0.35)=0.4675 > 0.4 → passes
        assert result.high_quality_count == 1
