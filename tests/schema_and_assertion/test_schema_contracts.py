# tests/tier3/test_schema_contracts.py
#
# Validates that Pydantic contracts enforce their invariants:
# - required fields reject missing data
# - bounded floats reject out-of-range values
# - aliases and optional defaults work as specified

import pytest
from pydantic import ValidationError

from contracts.primitives import (
    ArticleRaw,
    ArticleScored,
    GateDecision,
    NodeName,
    RetryInstruction,
    RetryReasonCode,
    RunStatus,
    SupervisorDecision,
    SupervisorRoute,
    TrendStrength,
)
from contracts.nodes import (
    DeliveryTaskResult,
    EngineerProfile,
    FetchTaskResult,
    OrchestratorContext,
    PublishTaskInput,
    PublishTaskResult,
    ScoringTaskResult,
    SynthesisTaskResult,
    TopicTaskResult,
    TrendSnapshot,
    TrendTaskResult,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _article_raw(**overrides) -> dict:
    base = {
        "article_id": "abc123",
        "url": "https://simonwillison.net/post",
        "title": "Some Title",
        "source_domain": "simonwillison.net",
        "published_at": "2026-01-01T00:00:00",
        "summary": "Article summary here.",
    }
    base.update(overrides)
    return base


def _article_scored(**overrides) -> dict:
    base = {
        "article_id": "abc123",
        "url": "https://simonwillison.net/post",
        "title": "Some Title",
        "source_domain": "simonwillison.net",
        "published_at": "2026-01-01T00:00:00",
        "summary": "Article summary here.",
        "relevance_score": 0.8,
        "reputation_score": 0.7,
        "combined_score": 0.755,
        "passed_threshold": True,
        "score_rationale": "Relevance 0.80 × 0.65 + Reputation 0.70 × 0.35 = 0.755 (pass).",
    }
    base.update(overrides)
    return base


def _trend_snapshot(**overrides) -> dict:
    base = {
        "trend_id": "llm-routing",
        "name": "LLM Routing",
        "strength": 0.65,
        "strength_band": TrendStrength.STRONG,
        "platform_relevance": 0.8,
        "key_signals": ["routing", "dispatch"],
        "last_reinforced": "2026-05-01",
    }
    base.update(overrides)
    return base


def _engineer_profile(**overrides) -> dict:
    base = {
        "name": "Sam",
        "focus_areas": ["platform engineering", "AI agents"],
        "background_summary": "Senior engineer moving into AI architecture.",
        "experience_level": "senior engineer",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# ArticleRaw
# ---------------------------------------------------------------------------

class TestArticleRaw:

    def test_valid_construction(self):
        ArticleRaw(**_article_raw())

    def test_fetch_error_defaults_to_empty_string(self):
        a = ArticleRaw(**_article_raw())
        assert a.fetch_error == ""

    def test_fetch_error_can_be_set(self):
        a = ArticleRaw(**_article_raw(fetch_error="timeout"))
        assert a.fetch_error == "timeout"

    def test_missing_title_raises(self):
        data = _article_raw()
        del data["title"]
        with pytest.raises(ValidationError):
            ArticleRaw(**data)

    def test_missing_article_id_raises(self):
        data = _article_raw()
        del data["article_id"]
        with pytest.raises(ValidationError):
            ArticleRaw(**data)

    def test_missing_source_domain_raises(self):
        data = _article_raw()
        del data["source_domain"]
        with pytest.raises(ValidationError):
            ArticleRaw(**data)

    def test_published_at_accepts_empty_string(self):
        a = ArticleRaw(**_article_raw(published_at=""))
        assert a.published_at == ""


# ---------------------------------------------------------------------------
# ArticleScored — score bound enforcement
# ---------------------------------------------------------------------------

class TestArticleScored:

    def test_valid_construction(self):
        ArticleScored(**_article_scored())

    def test_relevance_score_above_1_raises(self):
        with pytest.raises(ValidationError):
            ArticleScored(**_article_scored(relevance_score=1.01))

    def test_relevance_score_below_0_raises(self):
        with pytest.raises(ValidationError):
            ArticleScored(**_article_scored(relevance_score=-0.01))

    def test_reputation_score_above_1_raises(self):
        with pytest.raises(ValidationError):
            ArticleScored(**_article_scored(reputation_score=1.1))

    def test_reputation_score_below_0_raises(self):
        with pytest.raises(ValidationError):
            ArticleScored(**_article_scored(reputation_score=-0.1))

    def test_combined_score_above_1_raises(self):
        with pytest.raises(ValidationError):
            ArticleScored(**_article_scored(combined_score=1.001))

    def test_combined_score_below_0_raises(self):
        with pytest.raises(ValidationError):
            ArticleScored(**_article_scored(combined_score=-0.001))

    def test_boundary_values_accepted(self):
        ArticleScored(**_article_scored(relevance_score=0.0, reputation_score=0.0, combined_score=0.0))
        ArticleScored(**_article_scored(relevance_score=1.0, reputation_score=1.0, combined_score=1.0))

    def test_passed_threshold_false_is_valid(self):
        a = ArticleScored(**_article_scored(passed_threshold=False))
        assert a.passed_threshold is False


# ---------------------------------------------------------------------------
# RetryInstruction — alias handling
# ---------------------------------------------------------------------------

class TestRetryInstruction:

    def test_construction_with_alias_node(self):
        ri = RetryInstruction(
            node=NodeName.SCORING,
            reason_code=RetryReasonCode.BELOW_SCORE_THRESHOLD,
        )
        assert ri.target_node == NodeName.SCORING

    def test_construction_with_field_name_target_node(self):
        # populate_by_name=True allows using the Python field name
        ri = RetryInstruction(
            target_node=NodeName.SCORING,
            reason_code=RetryReasonCode.BELOW_SCORE_THRESHOLD,
        )
        assert ri.target_node == NodeName.SCORING

    def test_parameter_adjustment_defaults_to_empty_dict(self):
        ri = RetryInstruction(node=NodeName.TOPIC, reason_code=RetryReasonCode.LOW_CONFIDENCE)
        assert ri.parameter_adjustment == {}

    def test_parameter_adjustment_accepts_arbitrary_keys(self):
        ri = RetryInstruction(
            node=NodeName.SCORING,
            reason_code=RetryReasonCode.BELOW_SCORE_THRESHOLD,
            parameter_adjustment={"score_threshold": 0.35, "min_articles": 5},
        )
        assert ri.parameter_adjustment["score_threshold"] == 0.35

    def test_missing_reason_code_raises(self):
        with pytest.raises(ValidationError):
            RetryInstruction(node=NodeName.SCORING)


# ---------------------------------------------------------------------------
# SupervisorDecision
# ---------------------------------------------------------------------------

class TestSupervisorDecision:

    def test_valid_proceed_without_retry_instruction(self):
        sd = SupervisorDecision(
            supervisor=NodeName.INPUT_SUPERVISOR,
            route=SupervisorRoute.PROCEED,
            rework_count=0,
            rationale="All checks passed.",
        )
        assert sd.retry_instruction is None

    def test_valid_rework_with_retry_instruction(self):
        ri = RetryInstruction(
            node=NodeName.TOPIC,
            reason_code=RetryReasonCode.LOW_CONFIDENCE,
        )
        sd = SupervisorDecision(
            supervisor=NodeName.INPUT_SUPERVISOR,
            route=SupervisorRoute.REWORK,
            rework_count=1,
            rationale="Weak articles.",
            retry_instruction=ri,
        )
        assert sd.retry_instruction is not None
        assert sd.retry_instruction.reason_code == RetryReasonCode.LOW_CONFIDENCE

    def test_missing_rationale_raises(self):
        with pytest.raises(ValidationError):
            SupervisorDecision(
                supervisor=NodeName.INPUT_SUPERVISOR,
                route=SupervisorRoute.PROCEED,
                rework_count=0,
            )

    def test_rework_count_can_be_zero(self):
        sd = SupervisorDecision(
            supervisor=NodeName.OUTPUT_SUPERVISOR,
            route=SupervisorRoute.PROCEED,
            rework_count=0,
            rationale="First pass pass.",
        )
        assert sd.rework_count == 0


# ---------------------------------------------------------------------------
# GateDecision
# ---------------------------------------------------------------------------

class TestGateDecision:

    def test_issues_defaults_to_empty_list(self):
        gd = GateDecision(node_evaluated=NodeName.SCORING, passed=True)
        assert gd.issues == []

    def test_retry_count_defaults_to_zero(self):
        gd = GateDecision(node_evaluated=NodeName.FETCH, passed=False)
        assert gd.retry_count == 0

    def test_failed_gate_with_issues(self):
        gd = GateDecision(
            node_evaluated=NodeName.SCORING,
            passed=False,
            issues=["Too few articles passed threshold."],
            retry_count=1,
        )
        assert len(gd.issues) == 1
        assert gd.retry_count == 1


# ---------------------------------------------------------------------------
# TopicTaskResult — confidence bounds
# ---------------------------------------------------------------------------

class TestTopicTaskResult:

    def test_valid_construction(self):
        TopicTaskResult(
            topic="multi-agent orchestration",
            focus_angle="apply to platform teams",
            rationale="Active trend and no recent coverage.",
            confidence=0.85,
        )

    def test_confidence_above_1_raises(self):
        with pytest.raises(ValidationError):
            TopicTaskResult(
                topic="t", focus_angle="f", rationale="r", confidence=1.01
            )

    def test_confidence_below_0_raises(self):
        with pytest.raises(ValidationError):
            TopicTaskResult(
                topic="t", focus_angle="f", rationale="r", confidence=-0.01
            )

    def test_confidence_boundary_values_accepted(self):
        TopicTaskResult(topic="t", focus_angle="f", rationale="r", confidence=0.0)
        TopicTaskResult(topic="t", focus_angle="f", rationale="r", confidence=1.0)


# ---------------------------------------------------------------------------
# TrendSnapshot — strength bounds
# ---------------------------------------------------------------------------

class TestTrendSnapshot:

    def test_valid_construction(self):
        TrendSnapshot(**_trend_snapshot())

    def test_strength_above_1_raises(self):
        with pytest.raises(ValidationError):
            TrendSnapshot(**_trend_snapshot(strength=1.01))

    def test_strength_below_0_raises(self):
        with pytest.raises(ValidationError):
            TrendSnapshot(**_trend_snapshot(strength=-0.01))

    def test_platform_relevance_above_1_raises(self):
        with pytest.raises(ValidationError):
            TrendSnapshot(**_trend_snapshot(platform_relevance=1.1))

    def test_platform_relevance_below_0_raises(self):
        with pytest.raises(ValidationError):
            TrendSnapshot(**_trend_snapshot(platform_relevance=-0.1))

    def test_key_signals_can_be_empty_list(self):
        snap = TrendSnapshot(**_trend_snapshot(key_signals=[]))
        assert snap.key_signals == []


# ---------------------------------------------------------------------------
# ScoringTaskResult — internal count consistency invariants
# ---------------------------------------------------------------------------

class TestScoringTaskResultConsistency:

    def _make_scored_article(self, article_id: str, passed: bool) -> ArticleScored:
        return ArticleScored(**_article_scored(
            article_id=article_id,
            url=f"https://x.com/{article_id}",
            passed_threshold=passed,
        ))

    def test_high_quality_count_matches_passed_articles_len(self):
        passed = [self._make_scored_article("p1", True), self._make_scored_article("p2", True)]
        filtered = [self._make_scored_article("f1", False)]
        result = ScoringTaskResult(
            run_id="test",
            scored_articles=passed + filtered,
            passed_articles=passed,
            filtered_articles=filtered,
            high_quality_count=len(passed),
            low_quality_count=len(filtered),
        )
        assert result.high_quality_count == len(result.passed_articles)

    def test_low_quality_count_matches_filtered_articles_len(self):
        passed = [self._make_scored_article("p1", True)]
        filtered = [self._make_scored_article("f1", False), self._make_scored_article("f2", False)]
        result = ScoringTaskResult(
            run_id="test",
            scored_articles=passed + filtered,
            passed_articles=passed,
            filtered_articles=filtered,
            high_quality_count=1,
            low_quality_count=2,
        )
        assert result.low_quality_count == len(result.filtered_articles)

    def test_empty_scoring_result_is_valid(self):
        result = ScoringTaskResult(
            run_id="empty-run",
            scored_articles=[],
            passed_articles=[],
            filtered_articles=[],
            high_quality_count=0,
            low_quality_count=0,
        )
        assert result.high_quality_count == 0
        assert result.low_quality_count == 0


# ---------------------------------------------------------------------------
# SynthesisTaskResult
# ---------------------------------------------------------------------------

class TestSynthesisTaskResult:

    def test_valid_minimal_construction(self):
        SynthesisTaskResult(
            run_id="test",
            digest_html="<h1>Digest</h1>",
            digest_summary="Summary.",
            new_signals=[],
            trend_confirmations=[],
        )

    def test_new_signals_and_confirmations_default_to_empty(self):
        result = SynthesisTaskResult(
            run_id="test",
            digest_html="<h1>D</h1>",
            digest_summary="S.",
            new_signals=[],
            trend_confirmations=[],
        )
        assert result.new_signals == []
        assert result.trend_confirmations == []


# ---------------------------------------------------------------------------
# DeliveryTaskResult
# ---------------------------------------------------------------------------

class TestDeliveryTaskResult:

    def test_successful_delivery(self):
        result = DeliveryTaskResult(run_id="r", sent=True, message_id="ses-msg-001")
        assert result.sent is True
        assert result.error == ""

    def test_failed_delivery(self):
        result = DeliveryTaskResult(run_id="r", sent=False, error="SES throttled.")
        assert result.sent is False
        assert result.message_id == ""

    def test_message_id_defaults_to_empty_string(self):
        result = DeliveryTaskResult(run_id="r", sent=True)
        assert result.message_id == ""


# ---------------------------------------------------------------------------
# TrendTaskResult
# ---------------------------------------------------------------------------

class TestTrendTaskResult:

    def test_valid_completed_run(self):
        result = TrendTaskResult(
            run_id="r",
            run_status=RunStatus.COMPLETED,
            trends_updated=["llm-routing"],
            trends_created=[],
            trends_archived=[],
            source_reputations_updated=["simonwillison.net"],
        )
        assert result.run_status == RunStatus.COMPLETED
        assert result.error == ""

    def test_error_defaults_to_empty_string(self):
        result = TrendTaskResult(
            run_id="r",
            run_status=RunStatus.FAILED,
            trends_updated=[],
            trends_created=[],
            trends_archived=[],
            source_reputations_updated=[],
        )
        assert result.error == ""


# ---------------------------------------------------------------------------
# EngineerProfile
# ---------------------------------------------------------------------------

class TestEngineerProfile:

    def test_valid_construction(self):
        EngineerProfile(**_engineer_profile())

    def test_missing_name_raises(self):
        data = _engineer_profile()
        del data["name"]
        with pytest.raises(ValidationError):
            EngineerProfile(**data)

    def test_focus_areas_can_be_empty_list(self):
        profile = EngineerProfile(**_engineer_profile(focus_areas=[]))
        assert profile.focus_areas == []


# ---------------------------------------------------------------------------
# PublishTaskInput / PublishTaskResult
# ---------------------------------------------------------------------------

class TestPublishContracts:

    def _valid_input(self, **overrides) -> dict:
        base = {
            "run_id":      "2026-06-01",
            "digest_html": "<h1>Title</h1><p>Body.</p>",
            "topic":       "multi-agent orchestration",
            "bucket":      "my-site.com",
            "cf_dist_id":  "ABCDEF123",
            "domain":      "my-site.com",
            "author_name": "Test Author",
        }
        base.update(overrides)
        return base

    def test_valid_publish_task_input(self):
        task = PublishTaskInput(**self._valid_input())
        assert task.run_id == "2026-06-01"
        assert task.bucket == "my-site.com"

    def test_missing_required_field_raises(self):
        data = self._valid_input()
        del data["bucket"]
        with pytest.raises(ValidationError):
            PublishTaskInput(**data)

    def test_empty_bucket_is_valid(self):
        task = PublishTaskInput(**self._valid_input(bucket=""))
        assert task.bucket == ""

    def test_publish_result_published_true(self):
        result = PublishTaskResult(
            run_id="2026-06-01",
            published=True,
            article_url="https://my-site.com/articles/2026-06-01.html",
        )
        assert result.published is True
        assert result.skipped is False
        assert result.error == ""

    def test_publish_result_skipped(self):
        result = PublishTaskResult(run_id="2026-06-01", published=False, skipped=True)
        assert result.published is False
        assert result.skipped is True
        assert result.article_url == ""

    def test_publish_result_failed(self):
        result = PublishTaskResult(
            run_id="2026-06-01",
            published=False,
            error="S3 PutObject failed: AccessDenied",
        )
        assert result.published is False
        assert result.skipped is False
        assert "AccessDenied" in result.error

    def test_publish_result_defaults(self):
        result = PublishTaskResult(run_id="r", published=True)
        assert result.article_url == ""
        assert result.skipped is False
        assert result.error == ""
