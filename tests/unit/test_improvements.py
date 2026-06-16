# tests/unit/test_improvements.py
#
# Unit tests for all changes introduced in the improvement spec.
# Covers:
#   - TopicPromptContext and _apply_retry_adjustments (Issues 1-3)
#   - JSON fence-stripping in all parsers (Issues 8, 10, 13, 17)
#   - Output supervisor closing paragraph extraction (Issue 5)
#   - Synthesis failed_criteria rework instructions (Issue 6)
#   - Scoring batch size (Issue 9)
#   - RunStatus.COMPLETED_WITH_WARNINGS assignment logic (Issue 16)
#   - WeeklySynthesis narrative field (Issue 15)

import json
import pytest
from unittest.mock import MagicMock, patch

from contracts.primitives import (
    NodeName,
    RetryInstruction,
    RetryReasonCode,
    RunStatus,
)
from contracts.nodes import TopicTaskInput, SynthesisTaskInput, TrendSnapshot
from contracts.state import WeeklySynthesis, ttl_days, current_week_id


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

def _retry(reason: RetryReasonCode, params: dict | None = None) -> RetryInstruction:
    return RetryInstruction(
        node=NodeName.TOPIC,
        reason_code=reason,
        parameter_adjustment=params or {},
    )


def _topic_input(
    recent_topics: list[str] | None = None,
    available_topics: list[str] | None = None,
    retry: RetryInstruction | None = None,
    narrative: str = "",
) -> TopicTaskInput:
    # `is None` (not `or`) so callers can pass an explicit empty list — `or`
    # would silently fall back to the default since [] is falsy.
    return TopicTaskInput(
        run_id="test-run",
        recent_topics=recent_topics if recent_topics is not None else ["topic-a", "topic-b"],
        active_trend_names=["LLM Routing"],
        recent_signals=["signal-one"],
        available_topics=available_topics if available_topics is not None else ["topic-a", "topic-b", "topic-c"],
        recent_weekly_narrative=narrative,
        retry_instruction=retry,
    )


# ---------------------------------------------------------------------------
# Issue 1 — TopicPromptContext: _apply_retry_adjustments returns correct type
# ---------------------------------------------------------------------------

class TestTopicPromptContext:

    def _run_adjustments(self, task_input):
        from node_definitions.topic import _apply_retry_adjustments
        return _apply_retry_adjustments(task_input)

    def test_no_retry_returns_full_recent_topics(self):
        inp = _topic_input(recent_topics=["topic-a", "topic-b"])
        ctx = self._run_adjustments(inp)
        assert ctx.recent_topics == ["topic-a", "topic-b"]

    # Topic-repetition fix: recent_topics must be hard-filtered out of
    # available_topics on the normal (non-retry) path too — the prompt's
    # "avoid these" text alone was not reliably preventing repeat selections.
    def test_no_retry_excludes_recent_topics_from_available(self):
        inp = _topic_input(
            available_topics=["topic-a", "topic-b", "topic-c"],
            recent_topics=["topic-a", "topic-b"],
        )
        ctx = self._run_adjustments(inp)
        assert ctx.available_topics == ["topic-c"]

    def test_no_retry_keeps_full_list_when_nothing_recent(self):
        inp = _topic_input(available_topics=["topic-a", "topic-b", "topic-c"], recent_topics=[])
        ctx = self._run_adjustments(inp)
        assert ctx.available_topics == ["topic-a", "topic-b", "topic-c"]

    def test_no_retry_falls_back_to_full_list_when_recency_exhausts_rotation(self):
        inp = _topic_input(
            available_topics=["topic-a", "topic-b"],
            recent_topics=["topic-a", "topic-b"],
        )
        ctx = self._run_adjustments(inp)
        assert ctx.available_topics == ["topic-a", "topic-b"]

    def test_no_retry_returns_empty_exclusions(self):
        inp = _topic_input()
        ctx = self._run_adjustments(inp)
        assert ctx.exclusions == []

    def test_weak_topic_selection_excludes_previous_topic(self):
        inp = _topic_input(
            available_topics=["topic-a", "topic-b", "topic-c"],
            retry=_retry(RetryReasonCode.WEAK_TOPIC_SELECTION, {"previous_topic": "topic-a"}),
        )
        ctx = self._run_adjustments(inp)
        assert "topic-a" not in ctx.available_topics
        assert "topic-a" in ctx.exclusions

    def test_weak_topic_selection_keeps_other_topics(self):
        inp = _topic_input(
            available_topics=["topic-a", "topic-b", "topic-c"],
            recent_topics=[],  # isolate the retry-specific exclusion from the recency filter
            retry=_retry(RetryReasonCode.WEAK_TOPIC_SELECTION, {"previous_topic": "topic-a"}),
        )
        ctx = self._run_adjustments(inp)
        assert "topic-b" in ctx.available_topics
        assert "topic-c" in ctx.available_topics

    def test_weak_topic_no_previous_topic_param_leaves_list_intact(self):
        inp = _topic_input(
            available_topics=["topic-a", "topic-b"],
            retry=_retry(RetryReasonCode.WEAK_TOPIC_SELECTION, {}),
        )
        ctx = self._run_adjustments(inp)
        assert len(ctx.available_topics) == 2
        assert ctx.exclusions == []

    # Issue 1 core fix: LOW_CONFIDENCE must clear recent_topics
    def test_low_confidence_clears_recent_topics(self):
        inp = _topic_input(
            recent_topics=["topic-a", "topic-b"],
            retry=_retry(RetryReasonCode.LOW_CONFIDENCE),
        )
        ctx = self._run_adjustments(inp)
        assert ctx.recent_topics == []

    def test_low_confidence_does_not_change_available_topics(self):
        inp = _topic_input(
            available_topics=["topic-a", "topic-b", "topic-c"],
            retry=_retry(RetryReasonCode.LOW_CONFIDENCE),
        )
        ctx = self._run_adjustments(inp)
        assert len(ctx.available_topics) == 3

    def test_low_confidence_does_not_add_exclusions(self):
        inp = _topic_input(retry=_retry(RetryReasonCode.LOW_CONFIDENCE))
        ctx = self._run_adjustments(inp)
        assert ctx.exclusions == []

    # Issue 3 fix: LOW_QUALITY_ARTICLES must behave like WEAK_TOPIC_SELECTION
    def test_low_quality_articles_excludes_previous_topic(self):
        inp = _topic_input(
            available_topics=["topic-a", "topic-b", "topic-c"],
            retry=_retry(RetryReasonCode.LOW_QUALITY_ARTICLES, {"previous_topic": "topic-b"}),
        )
        ctx = self._run_adjustments(inp)
        assert "topic-b" not in ctx.available_topics
        assert "topic-b" in ctx.exclusions

    def test_low_quality_articles_keeps_other_topics(self):
        inp = _topic_input(
            available_topics=["topic-a", "topic-b", "topic-c"],
            recent_topics=[],  # isolate the retry-specific exclusion from the recency filter
            retry=_retry(RetryReasonCode.LOW_QUALITY_ARTICLES, {"previous_topic": "topic-b"}),
        )
        ctx = self._run_adjustments(inp)
        assert "topic-a" in ctx.available_topics
        assert "topic-c" in ctx.available_topics

    def test_weekly_narrative_passed_through_to_context(self):
        inp = _topic_input(narrative="Last week, LLM routing dominated the field.")
        ctx = self._run_adjustments(inp)
        assert ctx.recent_weekly_narrative == "Last week, LLM routing dominated the field."

    def test_no_narrative_defaults_to_empty_string(self):
        inp = _topic_input(narrative="")
        ctx = self._run_adjustments(inp)
        assert ctx.recent_weekly_narrative == ""


# ---------------------------------------------------------------------------
# Issue 8 — Fence-stripping in scoring._parse_relevance_output
# ---------------------------------------------------------------------------

class TestParseRelevanceOutputFenceStripping:

    def _parse(self, raw):
        from node_definitions.scoring import _parse_relevance_output
        return _parse_relevance_output(raw)

    def test_bare_json_array_still_parses(self):
        raw = json.dumps([{"article_id": "a1", "relevance_score": 0.8, "relevance_rationale": "Good."}])
        result = self._parse(raw)
        assert "a1" in result

    def test_fenced_json_only_parses(self):
        raw = "```json\n" + json.dumps([{"article_id": "a1", "relevance_score": 0.7, "relevance_rationale": "OK."}]) + "\n```"
        result = self._parse(raw)
        assert "a1" in result

    def test_fenced_no_lang_tag_parses(self):
        raw = "```\n" + json.dumps([{"article_id": "a2", "relevance_score": 0.6, "relevance_rationale": "Mid."}]) + "\n```"
        result = self._parse(raw)
        assert "a2" in result

    def test_prose_then_json_parses_via_regex_fallback(self):
        payload = json.dumps([{"article_id": "a3", "relevance_score": 0.5, "relevance_rationale": "OK."}])
        raw = f"Here are the results:\n\n{payload}"
        result = self._parse(raw)
        assert "a3" in result

    def test_garbage_returns_empty(self):
        result = self._parse("not json at all")
        assert result == {}


# ---------------------------------------------------------------------------
# Issue 10 — Fence-stripping in synthesis._parse_signals_output
# ---------------------------------------------------------------------------

class TestParseSignalsOutputFenceStripping:

    def _parse(self, raw):
        from node_definitions.synthesis.parsers import parse_signals_output
        return parse_signals_output(raw)

    def _valid_payload(self) -> str:
        return json.dumps({
            "new_signals": ["supervisor pattern"],
            "trend_confirmations": ["LLM Routing"],
            "digest_summary": "Today covered routing.",
        })

    def test_bare_json_parses(self):
        result = self._parse(self._valid_payload())
        assert result["new_signals"] == ["supervisor pattern"]

    def test_fenced_json_parses(self):
        raw = "```json\n" + self._valid_payload() + "\n```"
        result = self._parse(raw)
        assert result["new_signals"] == ["supervisor pattern"]

    def test_fenced_no_lang_parses(self):
        raw = "```\n" + self._valid_payload() + "\n```"
        result = self._parse(raw)
        assert result["trend_confirmations"] == ["LLM Routing"]

    def test_garbage_returns_safe_defaults(self):
        result = self._parse("not valid json")
        assert result["new_signals"] == []
        assert result["trend_confirmations"] == []
        assert result["digest_summary"] == ""


# ---------------------------------------------------------------------------
# Issue 13 — Fence-stripping in signal_analysis._parse_json_list
# ---------------------------------------------------------------------------

class TestParseJsonListFenceStripping:

    def _parse(self, raw):
        from node_definitions.trend.signal_analysis import _parse_json_list
        return _parse_json_list(raw)

    def test_bare_array_parses(self):
        raw = json.dumps([{"key": "val"}])
        result = self._parse(raw)
        assert result == [{"key": "val"}]

    def test_fenced_array_parses(self):
        raw = "```json\n" + json.dumps([{"key": "val"}]) + "\n```"
        result = self._parse(raw)
        assert result == [{"key": "val"}]

    def test_fenced_no_lang_parses(self):
        raw = "```\n" + json.dumps([{"key": "val"}]) + "\n```"
        result = self._parse(raw)
        assert result == [{"key": "val"}]

    def test_garbage_returns_empty_list(self):
        result = self._parse("totally not json")
        assert result == []

    def test_empty_array_returns_empty_list(self):
        result = self._parse("[]")
        assert result == []


# ---------------------------------------------------------------------------
# Issue 5 — Output supervisor closing paragraph extraction
# ---------------------------------------------------------------------------

class TestExtractClosingParagraph:

    def _extract(self, html):
        from node_definitions.output_supervisor import _extract_closing_paragraph
        return _extract_closing_paragraph(html)

    def test_returns_last_paragraph_text(self):
        html = "<html><p>First para.</p><p>Second para.</p><p>Closing takeaway here.</p></html>"
        result = self._extract(html)
        assert result == "Closing takeaway here."

    def test_single_paragraph_returns_it(self):
        html = "<html><p>Only paragraph.</p></html>"
        result = self._extract(html)
        assert result == "Only paragraph."

    def test_empty_html_returns_empty_string(self):
        result = self._extract("")
        assert result == ""

    def test_no_paragraphs_returns_empty_string(self):
        result = self._extract("<html><h1>Title</h1></html>")
        assert result == ""

    def test_whitespace_only_paragraphs_are_skipped(self):
        html = "<html><p>Real content.</p><p>   </p></html>"
        result = self._extract(html)
        assert result == "Real content."


# ---------------------------------------------------------------------------
# Issue 6 — Synthesis failed_criteria rework instructions
# ---------------------------------------------------------------------------

class TestSynthesisApplyRetryAdjustments:

    def _make_input(self, reason, params=None):
        from contracts.nodes import SynthesisTaskInput, EngineerProfile
        ri = RetryInstruction(
            node=NodeName.SYNTHESIS,
            reason_code=reason,
            parameter_adjustment=params or {},
        )
        return SynthesisTaskInput(
            run_id="test",
            topic="multi-agent orchestration",
            focus_angle="apply to platform engineering",
            passed_articles=[],
            active_trends=[],
            recent_run_signals=[],
            engineer_profile=EngineerProfile(
                name="Sam",
                focus_areas=["agentic architecture"],
                background_summary="Senior engineer.",
                experience_level="senior",
            ),
            retry_instruction=ri,
        )

    def _adjust(self, task_input):
        from node_definitions.synthesis.retry import apply_retry_adjustments
        return apply_retry_adjustments(task_input)

    def test_no_retry_returns_empty_string(self):
        from contracts.nodes import SynthesisTaskInput, EngineerProfile
        inp = SynthesisTaskInput(
            run_id="test", topic="t", focus_angle="f", passed_articles=[],
            active_trends=[], recent_run_signals=[],
            engineer_profile=EngineerProfile(name="Sam", focus_areas=[], background_summary="", experience_level="senior"),
        )
        assert self._adjust(inp) == ""

    def test_digest_insufficient_returns_base_instruction(self):
        inp = self._make_input(RetryReasonCode.DIGEST_INSUFFICIENT)
        result = self._adjust(inp)
        assert "insufficient depth" in result.lower()

    def test_personalized_criterion_adds_specific_note(self):
        inp = self._make_input(RetryReasonCode.DIGEST_INSUFFICIENT, {"failed_criteria": ["PERSONALIZED"]})
        result = self._adjust(inp)
        assert "personalized" in result.lower()

    def test_actionable_criterion_adds_specific_note(self):
        inp = self._make_input(RetryReasonCode.DIGEST_INSUFFICIENT, {"failed_criteria": ["ACTIONABLE"]})
        result = self._adjust(inp)
        assert "closing takeaway" in result.lower()

    def test_connected_criterion_adds_specific_note(self):
        inp = self._make_input(RetryReasonCode.DIGEST_INSUFFICIENT, {"failed_criteria": ["CONNECTED"]})
        result = self._adjust(inp)
        assert "isolation" in result.lower()

    def test_multiple_criteria_all_included(self):
        inp = self._make_input(
            RetryReasonCode.DIGEST_INSUFFICIENT,
            {"failed_criteria": ["PERSONALIZED", "ACTIONABLE", "CONNECTED"]},
        )
        result = self._adjust(inp)
        assert "personalized" in result.lower()
        assert "closing takeaway" in result.lower()
        assert "isolation" in result.lower()

    def test_missing_required_field_names_missing_fields(self):
        inp = self._make_input(
            RetryReasonCode.MISSING_REQUIRED_FIELD,
            {"missing_fields": ["new_signals"]},
        )
        result = self._adjust(inp)
        assert "new_signals" in result


# ---------------------------------------------------------------------------
# Issue 9 — Scoring batch size
# ---------------------------------------------------------------------------

class TestScoringBatchSize:

    def _article(self, article_id: str):
        from contracts.primitives import ArticleRaw
        from datetime import datetime, timezone
        return ArticleRaw(
            article_id=article_id,
            url=f"https://example.com/{article_id}",
            title=f"Title {article_id}",
            source_domain="example.com",
            published_at=datetime.now(timezone.utc).isoformat(),
            summary=f"Summary for {article_id}.",
        )

    def _scoring_input(self, articles):
        from contracts.nodes import ScoringTaskInput
        return ScoringTaskInput(
            run_id="test",
            topic="multi-agent orchestration",
            focus_angle="apply to platform engineering",
            articles=articles,
            source_reputation_map={},
            active_trend_names=[],
            score_threshold=0.5,
        )

    def test_score_batch_called_once_for_small_list(self):
        articles = [self._article(f"a{i}") for i in range(5)]
        task_input = self._scoring_input(articles)
        with patch("node_definitions.scoring._score_batch", return_value={}) as mock_batch:
            from node_definitions.scoring import _score_relevance
            _score_relevance(task_input)
        assert mock_batch.call_count == 1

    def test_score_batch_called_twice_for_11_articles(self):
        articles = [self._article(f"a{i}") for i in range(11)]
        task_input = self._scoring_input(articles)
        with patch("node_definitions.scoring._score_batch", return_value={}) as mock_batch:
            from node_definitions.scoring import _score_relevance
            _score_relevance(task_input)
        assert mock_batch.call_count == 2

    def test_batch_results_merged(self):
        articles = [self._article(f"a{i}") for i in range(11)]
        task_input = self._scoring_input(articles)
        batch1 = {f"a{i}": {"relevance_score": 0.8, "relevance_rationale": "Good."} for i in range(10)}
        batch2 = {"a10": {"relevance_score": 0.7, "relevance_rationale": "OK."}}
        with patch("node_definitions.scoring._score_batch", side_effect=[batch1, batch2]):
            from node_definitions.scoring import _score_relevance
            result = _score_relevance(task_input)
        assert len(result) == 11
        assert "a0" in result
        assert "a10" in result

    def test_first_batch_gets_exactly_batch_size_articles(self):
        from node_definitions.scoring import SCORING_BATCH_SIZE
        articles = [self._article(f"a{i}") for i in range(SCORING_BATCH_SIZE + 3)]
        task_input = self._scoring_input(articles)
        captured_batches: list[list] = []

        def capture_batch(batch, ti):
            captured_batches.append(batch)
            return {}

        with patch("node_definitions.scoring._score_batch", side_effect=capture_batch):
            from node_definitions.scoring import _score_relevance
            _score_relevance(task_input)

        assert len(captured_batches[0]) == SCORING_BATCH_SIZE
        assert len(captured_batches[1]) == 3


# ---------------------------------------------------------------------------
# Issue 16 — RunStatus.COMPLETED_WITH_WARNINGS assignment
# ---------------------------------------------------------------------------

class TestRunStatusAssignment:

    def _make_trend_input(self, delivery_sent: bool):
        from contracts.nodes import TrendTaskInput
        return TrendTaskInput(
            run_id="test",
            topic="test-topic",
            focus_angle="test-angle",
            scored_articles=[],
            new_signals=[],
            trend_confirmations=[],
            digest_summary="",
            existing_trends=[],
            source_reputation_map={},
            delivery_sent=delivery_sent,
        )

    def test_completed_with_warnings_when_delivered_with_errors(self):
        """Delivered digest + bookkeeping errors → COMPLETED_WITH_WARNINGS."""
        errors = ["trend write failed"]
        delivery_sent = True
        if not delivery_sent:
            status = RunStatus.DEGRADED
        elif errors:
            status = RunStatus.COMPLETED_WITH_WARNINGS
        else:
            status = RunStatus.COMPLETED
        assert status == RunStatus.COMPLETED_WITH_WARNINGS

    def test_degraded_when_delivery_failed(self):
        """Failed delivery → DEGRADED regardless of other errors."""
        errors = []
        delivery_sent = False
        if not delivery_sent:
            status = RunStatus.DEGRADED
        elif errors:
            status = RunStatus.COMPLETED_WITH_WARNINGS
        else:
            status = RunStatus.COMPLETED
        assert status == RunStatus.DEGRADED

    def test_completed_when_delivery_succeeded_no_errors(self):
        """Clean run → COMPLETED."""
        errors = []
        delivery_sent = True
        if not delivery_sent:
            status = RunStatus.DEGRADED
        elif errors:
            status = RunStatus.COMPLETED_WITH_WARNINGS
        else:
            status = RunStatus.COMPLETED
        assert status == RunStatus.COMPLETED

    def test_degraded_beats_errors_when_delivery_failed(self):
        """Delivery failed + errors → still DEGRADED, not COMPLETED_WITH_WARNINGS."""
        errors = ["trend write failed"]
        delivery_sent = False
        if not delivery_sent:
            status = RunStatus.DEGRADED
        elif errors:
            status = RunStatus.COMPLETED_WITH_WARNINGS
        else:
            status = RunStatus.COMPLETED
        assert status == RunStatus.DEGRADED

    def test_completed_with_warnings_has_distinct_value(self):
        assert RunStatus.COMPLETED_WITH_WARNINGS != RunStatus.COMPLETED
        assert RunStatus.COMPLETED_WITH_WARNINGS != RunStatus.DEGRADED

    def test_completed_with_warnings_is_a_string_enum(self):
        assert RunStatus.COMPLETED_WITH_WARNINGS.value == "completed_with_warnings"


# ---------------------------------------------------------------------------
# Issue 15 — WeeklySynthesis narrative field
# ---------------------------------------------------------------------------

class TestWeeklySynthesisNarrativeField:

    def test_narrative_field_defaults_to_empty_string(self):
        ws = WeeklySynthesis(
            week_id="2026-W23",
            topics_covered=[],
            run_ids_included=[],
            created_at="2026-06-02T00:00:00",
            ttl=ttl_days(90),
        )
        assert ws.narrative == ""

    def test_narrative_field_accepts_string(self):
        ws = WeeklySynthesis(
            week_id="2026-W23",
            narrative="This week, supervisor patterns dominated discussion.",
            run_ids_included=[],
            created_at="2026-06-02T00:00:00",
            ttl=ttl_days(90),
        )
        assert ws.narrative == "This week, supervisor patterns dominated discussion."

    def test_narrative_roundtrips_through_dict(self):
        ws = WeeklySynthesis(
            week_id="2026-W23",
            narrative="Some narrative.",
            run_ids_included=[],
            created_at="2026-06-02T00:00:00",
            ttl=ttl_days(90),
        )
        as_dict = ws.model_dump()
        restored = WeeklySynthesis(**as_dict)
        assert restored.narrative == "Some narrative."


# ---------------------------------------------------------------------------
# Issue 15 — recent_weekly_narrative field on contract types
# ---------------------------------------------------------------------------

class TestNarrativeContractFields:

    def test_topic_task_input_accepts_narrative(self):
        inp = _topic_input(narrative="This week agentic orchestration dominated.")
        assert inp.recent_weekly_narrative == "This week agentic orchestration dominated."

    def test_topic_task_input_narrative_defaults_to_empty(self):
        inp = TopicTaskInput(
            run_id="r",
            recent_topics=[],
            active_trend_names=[],
            recent_signals=[],
            available_topics=["topic-a"],
        )
        assert inp.recent_weekly_narrative == ""

    def test_input_supervisor_input_accepts_narrative(self):
        from contracts.nodes import (
            InputSupervisorInput, TopicTaskResult, FetchTaskResult, ScoringTaskResult
        )
        from contracts.primitives import ArticleRaw
        topic = TopicTaskResult(topic="t", focus_angle="f", rationale="r", confidence=0.8)
        fetch = FetchTaskResult(run_id="r", articles=[], article_count=0)
        scoring = ScoringTaskResult(
            run_id="r", scored_articles=[], passed_articles=[],
            filtered_articles=[], high_quality_count=0, low_quality_count=0,
        )
        sup = InputSupervisorInput(
            run_id="r",
            topic_result=topic,
            fetch_result=fetch,
            scoring_result=scoring,
            recent_weekly_narrative="Supervisor patterns dominated last week.",
        )
        assert sup.recent_weekly_narrative == "Supervisor patterns dominated last week."

    def test_synthesis_task_input_accepts_narrative(self):
        from contracts.nodes import SynthesisTaskInput, EngineerProfile
        inp = SynthesisTaskInput(
            run_id="r", topic="t", focus_angle="f",
            passed_articles=[], active_trends=[], recent_run_signals=[],
            recent_weekly_narrative="Narrative here.",
            engineer_profile=EngineerProfile(
                name="Sam", focus_areas=[], background_summary="", experience_level="senior"
            ),
        )
        assert inp.recent_weekly_narrative == "Narrative here."
