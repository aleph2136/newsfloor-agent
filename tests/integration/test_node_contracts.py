# tests/integration/test_node_contracts.py
"""
Integration tests verifying the input/output contracts at each node boundary.

Two test layers:

Layer 1 — Wiring (nodes.py)
    Each graph node function is called directly with a minimal state dict.
    The downstream run() is patched and call_args are inspected to verify
    the correct typed input contract was assembled from state. The returned
    dict is also checked for the correct output keys.
    Catches: wrong field mapped from state, wrong contract type, missing key.

Layer 2 — Node definition internals (node_definitions/*.py)
    The actual run() functions in supervisor node definitions are called with
    real typed inputs and mocked CrewAI components. This exercises code paths
    that the wiring tests never reach because they mock run() entirely.
    Catches: NameError from wrong variable name inside run(), incorrect
    variable scope in the LLM evaluation path (e.g. task_input vs supervisor_input).
"""

import pytest
from unittest.mock import MagicMock, patch

from contracts.nodes import (
    DeliveryTaskInput,
    DeliveryTaskResult,
    EngineerProfile,
    FetchTaskInput,
    FetchTaskResult,
    InputSupervisorInput,
    OrchestratorContext,
    OutputSupervisorInput,
    PublishTaskInput,
    PublishTaskResult,
    ScoringTaskInput,
    ScoringTaskResult,
    SynthesisTaskInput,
    SynthesisTaskResult,
    TopicTaskInput,
    TopicTaskResult,
    TrendTaskInput,
    TrendTaskResult,
)
from contracts.primitives import (
    ArticleRaw,
    ArticleScored,
    NodeName,
    RunStatus,
    SupervisorDecision,
    SupervisorRoute,
)
from nodes import (
    delivery_node,
    fetch_node,
    input_supervisor,
    output_supervisor,
    publish_node,
    scoring_node,
    synthesis_node,
    topic_node,
    trend_node,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def run_id():
    return "2026-05-27"


@pytest.fixture
def engineer_profile():
    return EngineerProfile(
        name="Sam",
        focus_areas=["agentic architecture", "observability", "platform engineering"],
        background_summary="Senior engineer focused on AI systems.",
        experience_level="senior engineer",
    )


@pytest.fixture
def topic_result():
    return TopicTaskResult(
        topic="Agentic system governance and guardrails",
        focus_angle="Apply governance patterns to production agentic systems.",
        rationale="High signal in current trends.",
        confidence=0.85,
    )


@pytest.fixture
def article_raw():
    return ArticleRaw(
        article_id="abc123",
        url="https://example.com/article-1",
        title="Building Reliable Agentic Systems",
        source_domain="example.com",
        published_at="2026-05-27T00:00:00Z",
        summary="A deep dive into reliability patterns for agentic pipelines.",
    )


@pytest.fixture
def article_scored():
    return ArticleScored(
        article_id="abc123",
        url="https://example.com/article-1",
        title="Building Reliable Agentic Systems",
        source_domain="example.com",
        published_at="2026-05-27T00:00:00Z",
        summary="A deep dive into reliability patterns for agentic pipelines.",
        relevance_score=0.85,
        reputation_score=0.70,
        recency_score=1.0,
        combined_score=0.80,
        passed_threshold=True,
        score_rationale="Directly relevant to topic with trusted source.",
    )


@pytest.fixture
def fetch_result(run_id, article_raw):
    return FetchTaskResult(
        run_id=run_id,
        articles=[article_raw] * 5,
        fetch_errors=[],
        article_count=5,
    )


@pytest.fixture
def scoring_result(run_id, article_scored):
    passed   = [article_scored] * 3
    filtered = [article_scored] * 2
    return ScoringTaskResult(
        run_id=run_id,
        scored_articles=passed + filtered,
        passed_articles=passed,
        filtered_articles=filtered,
        high_quality_count=3,
        low_quality_count=2,
    )


@pytest.fixture
def synthesis_result(run_id):
    # Must be >= 800 chars and contain <h1>, <h2>, <em> to pass output supervisor
    # structural checks and exercise the LLM evaluation path in layer 2 tests.
    body = (
        "<html><body>"
        "<h1>Agentic Governance Digest</h1>"
        "<h2>Building Reliable Agentic Systems</h2>"
        "<p>Governance patterns for agentic pipelines require supervisor nodes that "
        "can evaluate intermediate outputs and route rework when quality falls short. "
        "The key insight is that governance must be built into the graph topology, not "
        "bolted on as post-hoc validation. Systems that treat quality gates as first-class "
        "nodes are more auditable and correctable than those that rely on retry logic alone.</p>"
        "<em>Why this matters: embedding quality gates in the graph makes failure modes "
        "visible and controllable rather than silent.</em>"
        "<h2>Trend Signals</h2>"
        "<ul><li>Supervisor-critic patterns gaining adoption</li>"
        "<li>Structured output contracts becoming standard practice</li></ul>"
        "<p>Take one governance pattern from today's digest and apply it to your "
        "current agentic project this week.</p>"
        "</body></html>"
    )
    # Pad to guarantee length >= MIN_DIGEST_LENGTH (800)
    body = body + (" " * max(0, 800 - len(body)))
    return SynthesisTaskResult(
        run_id=run_id,
        digest_html=body,
        digest_summary="Today's digest covered governance patterns for agentic systems.",
        new_signals=["supervisor-critic patterns", "structured output contracts"],
        trend_confirmations=["Agentic System Governance"],
    )


@pytest.fixture
def mock_context(engineer_profile):
    ctx = MagicMock(spec=OrchestratorContext)
    ctx.recent_topics            = ["Multi-agent orchestration"]
    ctx.active_trends            = []
    ctx.recent_weekly_signals    = ["supervisor patterns", "output contracts"]
    ctx.source_reputation_map    = {"example.com": 0.75}
    ctx.recent_run_signals       = ["governance", "reliability"]
    ctx.recent_weekly_narrative  = ""
    ctx.seen_article_ids         = ["abc123", "def456"]
    ctx.source_last_contributed  = {"example.com": "2026-06-01"}
    ctx.engineer_profile         = engineer_profile
    return ctx


@pytest.fixture
def delivery_result(run_id):
    return DeliveryTaskResult(run_id=run_id, sent=True, message_id="msg-001")


@pytest.fixture
def publish_result(run_id):
    return PublishTaskResult(
        run_id=run_id,
        published=True,
        article_url=f"https://example.com/articles/{run_id}.html",
    )


@pytest.fixture
def trend_result(run_id):
    return TrendTaskResult(
        run_id=run_id,
        run_status=RunStatus.COMPLETED,
        trends_updated=[],
        trends_created=[],
        trends_archived=[],
        source_reputations_updated=[],
    )


def _proceed_decision(supervisor: NodeName) -> SupervisorDecision:
    return SupervisorDecision(
        supervisor=supervisor,
        route=SupervisorRoute.PROCEED,
        rework_count=0,
        rationale="All good.",
    )


# ---------------------------------------------------------------------------
# Layer 1: Wiring — nodes.py assembles the correct input contracts
# ---------------------------------------------------------------------------

class TestTopicNodeWiring:

    @patch("node_definitions.topic.run")
    def test_assembles_topic_task_input(self, mock_run, run_id, topic_result, mock_context):
        mock_run.return_value = topic_result
        state = {
            "run_id": run_id,
            "rework_counts": {},
            "context": mock_context,
            "active_retry_instruction": None,
        }

        topic_node(state)

        arg = mock_run.call_args[0][0]
        assert isinstance(arg, TopicTaskInput)
        assert arg.run_id == run_id
        assert arg.recent_topics == mock_context.recent_topics
        assert arg.recent_signals == mock_context.recent_weekly_signals
        assert arg.retry_instruction is None

    @patch("node_definitions.topic.run")
    def test_returns_topic_result_key(self, mock_run, run_id, topic_result, mock_context):
        mock_run.return_value = topic_result
        state = {"run_id": run_id, "rework_counts": {}, "context": mock_context, "active_retry_instruction": None}

        result = topic_node(state)

        assert result["topic_result"] is topic_result
        assert result["active_retry_instruction"] is None


class TestFetchNodeWiring:

    @patch("node_definitions.fetch.run")
    def test_assembles_fetch_task_input(self, mock_run, run_id, topic_result, fetch_result):
        mock_run.return_value = fetch_result
        state = {
            "run_id": run_id,
            "rework_counts": {},
            "topic_result": topic_result,
            "active_retry_instruction": None,
        }

        fetch_node(state)

        arg = mock_run.call_args[0][0]
        assert isinstance(arg, FetchTaskInput)
        assert arg.run_id == run_id
        assert arg.topic == topic_result.topic
        assert arg.focus_angle == topic_result.focus_angle
        assert arg.retry_instruction is None

    @patch("node_definitions.fetch.run")
    def test_passes_seen_article_ids_from_context(self, mock_run, run_id, topic_result, fetch_result, mock_context):
        mock_run.return_value = fetch_result
        state = {
            "run_id": run_id,
            "rework_counts": {},
            "topic_result": topic_result,
            "context": mock_context,
            "active_retry_instruction": None,
        }

        fetch_node(state)

        arg = mock_run.call_args[0][0]
        assert arg.seen_article_ids == mock_context.seen_article_ids
        assert arg.source_last_contributed == mock_context.source_last_contributed

    @patch("node_definitions.fetch.run")
    def test_defaults_to_empty_when_context_is_none(self, mock_run, run_id, topic_result, fetch_result):
        mock_run.return_value = fetch_result
        state = {
            "run_id": run_id,
            "rework_counts": {},
            "topic_result": topic_result,
            "active_retry_instruction": None,
            # no "context" key
        }

        fetch_node(state)

        arg = mock_run.call_args[0][0]
        assert arg.seen_article_ids == []
        assert arg.source_last_contributed == {}

    @patch("node_definitions.fetch.run")
    def test_returns_fetch_result_key(self, mock_run, run_id, topic_result, fetch_result):
        mock_run.return_value = fetch_result
        state = {"run_id": run_id, "rework_counts": {}, "topic_result": topic_result, "active_retry_instruction": None}

        result = fetch_node(state)

        assert result["fetch_result"] is fetch_result
        assert result["active_retry_instruction"] is None


class TestScoringNodeWiring:

    @patch("node_definitions.scoring.run")
    def test_assembles_scoring_task_input(self, mock_run, run_id, topic_result, fetch_result, scoring_result, mock_context):
        mock_run.return_value = scoring_result
        state = {
            "run_id": run_id,
            "rework_counts": {},
            "topic_result": topic_result,
            "fetch_result": fetch_result,
            "context": mock_context,
            "active_retry_instruction": None,
        }

        scoring_node(state)

        arg = mock_run.call_args[0][0]
        assert isinstance(arg, ScoringTaskInput)
        assert arg.run_id == run_id
        assert arg.topic == topic_result.topic
        assert arg.focus_angle == topic_result.focus_angle
        assert arg.articles == fetch_result.articles
        assert arg.source_reputation_map == mock_context.source_reputation_map
        assert arg.retry_instruction is None

    @patch("node_definitions.scoring.run")
    def test_returns_scoring_result_key(self, mock_run, run_id, topic_result, fetch_result, scoring_result, mock_context):
        mock_run.return_value = scoring_result
        state = {"run_id": run_id, "rework_counts": {}, "topic_result": topic_result, "fetch_result": fetch_result, "context": mock_context, "active_retry_instruction": None}

        result = scoring_node(state)

        assert result["scoring_result"] is scoring_result
        assert result["active_retry_instruction"] is None


class TestInputSupervisorNodeWiring:

    @patch("node_definitions.input_supervisor.run")
    def test_assembles_input_supervisor_input(self, mock_run, run_id, topic_result, fetch_result, scoring_result):
        mock_run.return_value = _proceed_decision(NodeName.INPUT_SUPERVISOR)
        state = {
            "run_id": run_id,
            "rework_counts": {},
            "topic_result": topic_result,
            "fetch_result": fetch_result,
            "scoring_result": scoring_result,
        }

        input_supervisor(state)

        arg = mock_run.call_args[0][0]
        assert isinstance(arg, InputSupervisorInput)
        assert arg.run_id == run_id
        assert arg.topic_result is topic_result
        assert arg.fetch_result is fetch_result
        assert arg.scoring_result is scoring_result
        assert arg.rework_count == 0

    @patch("node_definitions.input_supervisor.run")
    def test_reads_rework_count_from_state(self, mock_run, run_id, topic_result, fetch_result, scoring_result):
        mock_run.return_value = _proceed_decision(NodeName.INPUT_SUPERVISOR)
        state = {
            "run_id": run_id,
            "rework_counts": {NodeName.INPUT_SUPERVISOR.value: 1},
            "topic_result": topic_result,
            "fetch_result": fetch_result,
            "scoring_result": scoring_result,
        }

        input_supervisor(state)

        arg = mock_run.call_args[0][0]
        assert arg.rework_count == 1

    @patch("node_definitions.input_supervisor.run")
    def test_returns_decision_key(self, mock_run, run_id, topic_result, fetch_result, scoring_result):
        decision = _proceed_decision(NodeName.INPUT_SUPERVISOR)
        mock_run.return_value = decision
        state = {"run_id": run_id, "rework_counts": {}, "topic_result": topic_result, "fetch_result": fetch_result, "scoring_result": scoring_result}

        result = input_supervisor(state)

        assert result["input_supervisor_decision"] is decision


class TestSynthesisNodeWiring:

    @patch("node_definitions.synthesis.run")
    def test_assembles_synthesis_task_input(self, mock_run, run_id, topic_result, scoring_result, synthesis_result, mock_context):
        mock_run.return_value = synthesis_result
        state = {
            "run_id": run_id,
            "rework_counts": {},
            "topic_result": topic_result,
            "scoring_result": scoring_result,
            "context": mock_context,
            "active_retry_instruction": None,
        }

        synthesis_node(state)

        arg = mock_run.call_args[0][0]
        assert isinstance(arg, SynthesisTaskInput)
        assert arg.run_id == run_id
        assert arg.topic == topic_result.topic
        assert arg.focus_angle == topic_result.focus_angle
        assert arg.passed_articles == scoring_result.passed_articles
        assert arg.engineer_profile == mock_context.engineer_profile
        assert arg.retry_instruction is None

    @patch("node_definitions.synthesis.run")
    def test_returns_synthesis_result_key(self, mock_run, run_id, topic_result, scoring_result, synthesis_result, mock_context):
        mock_run.return_value = synthesis_result
        state = {"run_id": run_id, "rework_counts": {}, "topic_result": topic_result, "scoring_result": scoring_result, "context": mock_context, "active_retry_instruction": None}

        result = synthesis_node(state)

        assert result["synthesis_result"] is synthesis_result
        assert result["active_retry_instruction"] is None


class TestOutputSupervisorNodeWiring:

    @patch("node_definitions.output_supervisor.run")
    def test_assembles_output_supervisor_input(self, mock_run, run_id, topic_result, synthesis_result, mock_context):
        mock_run.return_value = _proceed_decision(NodeName.OUTPUT_SUPERVISOR)
        state = {
            "run_id": run_id,
            "rework_counts": {},
            "topic_result": topic_result,
            "synthesis_result": synthesis_result,
            "context": mock_context,
        }

        output_supervisor(state)

        arg = mock_run.call_args[0][0]
        assert isinstance(arg, OutputSupervisorInput)
        assert arg.run_id == run_id
        assert arg.synthesis_result is synthesis_result
        assert arg.topic == topic_result.topic
        assert arg.focus_angle == topic_result.focus_angle
        assert arg.engineer_profile == mock_context.engineer_profile
        assert arg.rework_count == 0

    @patch("node_definitions.output_supervisor.run")
    def test_reads_rework_count_from_state(self, mock_run, run_id, topic_result, synthesis_result, mock_context):
        mock_run.return_value = _proceed_decision(NodeName.OUTPUT_SUPERVISOR)
        state = {
            "run_id": run_id,
            "rework_counts": {NodeName.OUTPUT_SUPERVISOR.value: 1},
            "topic_result": topic_result,
            "synthesis_result": synthesis_result,
            "context": mock_context,
        }

        output_supervisor(state)

        arg = mock_run.call_args[0][0]
        assert arg.rework_count == 1

    @patch("node_definitions.output_supervisor.run")
    def test_returns_decision_key(self, mock_run, run_id, topic_result, synthesis_result, mock_context):
        decision = _proceed_decision(NodeName.OUTPUT_SUPERVISOR)
        mock_run.return_value = decision
        state = {"run_id": run_id, "rework_counts": {}, "topic_result": topic_result, "synthesis_result": synthesis_result, "context": mock_context}

        result = output_supervisor(state)

        assert result["output_supervisor_decision"] is decision


class TestDeliveryNodeWiring:

    @patch("node_definitions.delivery.run")
    def test_assembles_delivery_task_input(self, mock_run, run_id, topic_result, synthesis_result, delivery_result):
        mock_run.return_value = delivery_result
        state = {
            "run_id": run_id,
            "rework_counts": {},
            "topic_result": topic_result,
            "synthesis_result": synthesis_result,
        }

        delivery_node(state)

        arg = mock_run.call_args[0][0]
        assert isinstance(arg, DeliveryTaskInput)
        assert arg.run_id == run_id
        assert arg.digest_html == synthesis_result.digest_html
        assert arg.topic == topic_result.topic

    @patch("node_definitions.delivery.run")
    def test_returns_delivery_result_key(self, mock_run, run_id, topic_result, synthesis_result, delivery_result):
        mock_run.return_value = delivery_result
        state = {"run_id": run_id, "rework_counts": {}, "topic_result": topic_result, "synthesis_result": synthesis_result}

        result = delivery_node(state)

        assert result["delivery_result"] is delivery_result


class TestPublishNodeWiring:

    @patch("node_definitions.publish.run")
    def test_assembles_publish_task_input(self, mock_run, run_id, topic_result, synthesis_result, publish_result):
        mock_run.return_value = publish_result
        with patch("nodes.settings") as mock_settings:
            mock_settings.personal_site_bucket   = "test-bucket"
            mock_settings.personal_site_cf_dist_id = "CF123"
            mock_settings.personal_site_domain   = "example.com"
            mock_settings.personal_site_author_name = "Test Author"
            state = {
                "run_id":           run_id,
                "rework_counts":    {},
                "topic_result":     topic_result,
                "synthesis_result": synthesis_result,
            }
            publish_node(state)

        arg = mock_run.call_args[0][0]
        assert isinstance(arg, PublishTaskInput)
        assert arg.run_id      == run_id
        assert arg.digest_html == synthesis_result.digest_html
        assert arg.topic       == topic_result.topic
        assert arg.bucket      == "test-bucket"
        assert arg.cf_dist_id  == "CF123"
        assert arg.domain      == "example.com"
        assert arg.author_name == "Test Author"

    @patch("node_definitions.publish.run")
    def test_returns_publish_result_key(self, mock_run, run_id, topic_result, synthesis_result, publish_result):
        mock_run.return_value = publish_result
        with patch("nodes.settings") as mock_settings:
            mock_settings.personal_site_bucket      = "test-bucket"
            mock_settings.personal_site_cf_dist_id  = "CF123"
            mock_settings.personal_site_domain      = "example.com"
            mock_settings.personal_site_author_name = "Test Author"
            state = {
                "run_id":           run_id,
                "rework_counts":    {},
                "topic_result":     topic_result,
                "synthesis_result": synthesis_result,
            }
            result = publish_node(state)

        assert result["publish_result"] is publish_result

    @patch("node_definitions.publish.run")
    def test_skips_gracefully_when_bucket_empty(self, mock_run, run_id, topic_result, synthesis_result):
        """publish.run returns skipped=True when bucket is empty — no S3/CF calls made."""
        mock_run.return_value = PublishTaskResult(run_id=run_id, published=False, skipped=True)
        with patch("nodes.settings") as mock_settings:
            mock_settings.personal_site_bucket      = ""
            mock_settings.personal_site_cf_dist_id  = ""
            mock_settings.personal_site_domain      = ""
            mock_settings.personal_site_author_name = ""
            state = {
                "run_id":           run_id,
                "rework_counts":    {},
                "topic_result":     topic_result,
                "synthesis_result": synthesis_result,
            }
            result = publish_node(state)

        assert result["publish_result"].published is False
        assert result["publish_result"].skipped is True

    def test_returns_error_result_when_upstream_missing(self, run_id):
        """publish_node degrades gracefully when synthesis_result is absent."""
        state = {
            "run_id":           run_id,
            "rework_counts":    {},
            "topic_result":     None,
            "synthesis_result": None,
        }
        result = publish_node(state)

        assert result["publish_result"].published is False
        assert result["publish_result"].error != ""


class TestTrendNodeWiring:

    @patch("node_definitions.trend.run")
    def test_assembles_trend_task_input(self, mock_run, run_id, topic_result, scoring_result, synthesis_result, mock_context, delivery_result, trend_result):
        mock_run.return_value = trend_result
        state = {
            "run_id": run_id,
            "rework_counts": {},
            "topic_result": topic_result,
            "scoring_result": scoring_result,
            "synthesis_result": synthesis_result,
            "context": mock_context,
            "delivery_result": delivery_result,
        }

        trend_node(state)

        arg = mock_run.call_args[0][0]
        assert isinstance(arg, TrendTaskInput)
        assert arg.run_id == run_id
        assert arg.topic == topic_result.topic
        assert arg.focus_angle == topic_result.focus_angle
        assert arg.scored_articles == scoring_result.scored_articles
        assert arg.new_signals == synthesis_result.new_signals
        assert arg.trend_confirmations == synthesis_result.trend_confirmations
        assert arg.digest_summary == synthesis_result.digest_summary
        assert arg.delivery_sent == delivery_result.sent

    @patch("node_definitions.trend.run")
    def test_maps_rework_counts_to_input(self, mock_run, run_id, topic_result, scoring_result, synthesis_result, mock_context, delivery_result, trend_result):
        mock_run.return_value = trend_result
        state = {
            "run_id": run_id,
            "rework_counts": {
                NodeName.INPUT_SUPERVISOR.value: 1,
                NodeName.OUTPUT_SUPERVISOR.value: 2,
            },
            "topic_result": topic_result,
            "scoring_result": scoring_result,
            "synthesis_result": synthesis_result,
            "context": mock_context,
            "delivery_result": delivery_result,
        }

        trend_node(state)

        arg = mock_run.call_args[0][0]
        assert arg.input_rework_count == 1
        assert arg.output_rework_count == 2

    @patch("node_definitions.trend.run")
    def test_returns_trend_result_and_run_status(self, mock_run, run_id, topic_result, scoring_result, synthesis_result, mock_context, delivery_result, trend_result):
        mock_run.return_value = trend_result
        state = {"run_id": run_id, "rework_counts": {}, "topic_result": topic_result, "scoring_result": scoring_result, "synthesis_result": synthesis_result, "context": mock_context, "delivery_result": delivery_result}

        result = trend_node(state)

        assert result["trend_result"] is trend_result
        assert result["run_status"] == RunStatus.COMPLETED


# ---------------------------------------------------------------------------
# Layer 2: Node definition internals — supervisor LLM evaluation paths
# ---------------------------------------------------------------------------
# These tests call the actual run() functions in the supervisor node definitions
# with mocked CrewAI components. They exercise the internal _evaluate_with_llm
# path that the wiring tests never reach.
#
# Specifically guards against the bug where `task_input.run_id` was used inside
# _evaluate_with_llm instead of `supervisor_input.run_id`. Python evaluates
# kickoff_crew's arguments before calling it, so the wrong variable name would
# raise NameError: name 'task_input' is not defined before kickoff_crew fires.
# ---------------------------------------------------------------------------

class TestInputSupervisorLLMPath:
    """The LLM evaluation path in input_supervisor.run() executes without NameError."""

    def _make_input(self, run_id, topic_result, fetch_result, scoring_result) -> InputSupervisorInput:
        return InputSupervisorInput(
            run_id=run_id,
            topic_result=topic_result,
            fetch_result=fetch_result,
            scoring_result=scoring_result,
            rework_count=0,
        )

    @patch("node_definitions.input_supervisor.kickoff_crew")
    @patch("node_definitions.input_supervisor.Task")
    @patch("node_definitions.input_supervisor.Crew")
    @patch("node_definitions.input_supervisor.Agent")
    @patch("node_definitions.input_supervisor.LLM")
    def test_no_name_error_in_kickoff_args(
        self, mock_LLM, mock_Agent, mock_Crew, mock_Task, mock_kickoff_crew,
        run_id, topic_result, fetch_result, scoring_result,
    ):
        """
        kickoff_crew must be called with supervisor_input.run_id, not task_input.run_id.
        Before the fix, argument evaluation raised NameError before kickoff_crew fired.
        """
        mock_Task.return_value.output = None  # triggers the no-output fallback → PROCEED

        from node_definitions.input_supervisor import run as supervisor_run
        result = supervisor_run(self._make_input(run_id, topic_result, fetch_result, scoring_result))

        assert isinstance(result, SupervisorDecision)
        mock_kickoff_crew.assert_called_once()
        # The third positional argument to kickoff_crew is run_id
        assert mock_kickoff_crew.call_args[0][2] == run_id

    @patch("node_definitions.input_supervisor.kickoff_crew")
    @patch("node_definitions.input_supervisor.Task")
    @patch("node_definitions.input_supervisor.Crew")
    @patch("node_definitions.input_supervisor.Agent")
    @patch("node_definitions.input_supervisor.LLM")
    def test_returns_proceed_when_crew_has_no_output(
        self, mock_LLM, mock_Agent, mock_Crew, mock_Task, mock_kickoff_crew,
        run_id, topic_result, fetch_result, scoring_result,
    ):
        mock_Task.return_value.output = None

        from node_definitions.input_supervisor import run as supervisor_run
        result = supervisor_run(self._make_input(run_id, topic_result, fetch_result, scoring_result))

        assert result.route == SupervisorRoute.PROCEED
        assert result.supervisor == NodeName.INPUT_SUPERVISOR

    @patch("node_definitions.input_supervisor.kickoff_crew")
    @patch("node_definitions.input_supervisor.Task")
    @patch("node_definitions.input_supervisor.Crew")
    @patch("node_definitions.input_supervisor.Agent")
    @patch("node_definitions.input_supervisor.LLM")
    def test_parses_proceed_from_crew_output(
        self, mock_LLM, mock_Agent, mock_Crew, mock_Task, mock_kickoff_crew,
        run_id, topic_result, fetch_result, scoring_result,
    ):
        mock_Task.return_value.output.raw = (
            '{"decision": "PROCEED", "reason_code": null, "rationale": "Strong input."}'
        )

        from node_definitions.input_supervisor import run as supervisor_run
        result = supervisor_run(self._make_input(run_id, topic_result, fetch_result, scoring_result))

        assert result.route == SupervisorRoute.PROCEED

    @patch("node_definitions.input_supervisor.kickoff_crew")
    @patch("node_definitions.input_supervisor.Task")
    @patch("node_definitions.input_supervisor.Crew")
    @patch("node_definitions.input_supervisor.Agent")
    @patch("node_definitions.input_supervisor.LLM")
    def test_structural_gate_fires_before_llm_on_thin_scoring(
        self, mock_LLM, mock_Agent, mock_Crew, mock_Task, mock_kickoff_crew,
        run_id, topic_result, fetch_result,
    ):
        """Structural check catches < MIN_PASSED_ARTICLES before the LLM is invoked."""
        thin_scoring = ScoringTaskResult(
            run_id=run_id,
            scored_articles=[],
            passed_articles=[],
            filtered_articles=[],
            high_quality_count=1,  # below MIN_PASSED_ARTICLES=2
            low_quality_count=9,
        )

        from node_definitions.input_supervisor import run as supervisor_run
        result = supervisor_run(
            InputSupervisorInput(
                run_id=run_id,
                topic_result=topic_result,
                fetch_result=fetch_result,
                scoring_result=thin_scoring,
                rework_count=0,
            )
        )

        assert result.route == SupervisorRoute.REWORK
        mock_kickoff_crew.assert_not_called()


class TestOutputSupervisorLLMPath:
    """The LLM evaluation path in output_supervisor.run() executes without NameError."""

    def _make_input(self, run_id, topic_result, synthesis_result, engineer_profile) -> OutputSupervisorInput:
        return OutputSupervisorInput(
            run_id=run_id,
            synthesis_result=synthesis_result,
            topic=topic_result.topic,
            focus_angle=topic_result.focus_angle,
            engineer_profile=engineer_profile,
            rework_count=0,
        )

    @patch("node_definitions.output_supervisor.kickoff_crew")
    @patch("node_definitions.output_supervisor.Task")
    @patch("node_definitions.output_supervisor.Crew")
    @patch("node_definitions.output_supervisor.Agent")
    @patch("node_definitions.output_supervisor.LLM")
    def test_no_name_error_in_kickoff_args(
        self, mock_LLM, mock_Agent, mock_Crew, mock_Task, mock_kickoff_crew,
        run_id, topic_result, synthesis_result, engineer_profile,
    ):
        """
        kickoff_crew must be called with supervisor_input.run_id, not task_input.run_id.
        Before the fix, argument evaluation raised NameError before kickoff_crew fired.
        """
        mock_Task.return_value.output = None

        from node_definitions.output_supervisor import run as supervisor_run
        result = supervisor_run(self._make_input(run_id, topic_result, synthesis_result, engineer_profile))

        assert isinstance(result, SupervisorDecision)
        mock_kickoff_crew.assert_called_once()
        assert mock_kickoff_crew.call_args[0][2] == run_id

    @patch("node_definitions.output_supervisor.kickoff_crew")
    @patch("node_definitions.output_supervisor.Task")
    @patch("node_definitions.output_supervisor.Crew")
    @patch("node_definitions.output_supervisor.Agent")
    @patch("node_definitions.output_supervisor.LLM")
    def test_returns_proceed_when_crew_has_no_output(
        self, mock_LLM, mock_Agent, mock_Crew, mock_Task, mock_kickoff_crew,
        run_id, topic_result, synthesis_result, engineer_profile,
    ):
        mock_Task.return_value.output = None

        from node_definitions.output_supervisor import run as supervisor_run
        result = supervisor_run(self._make_input(run_id, topic_result, synthesis_result, engineer_profile))

        assert result.route == SupervisorRoute.PROCEED
        assert result.supervisor == NodeName.OUTPUT_SUPERVISOR

    @patch("node_definitions.output_supervisor.kickoff_crew")
    @patch("node_definitions.output_supervisor.Task")
    @patch("node_definitions.output_supervisor.Crew")
    @patch("node_definitions.output_supervisor.Agent")
    @patch("node_definitions.output_supervisor.LLM")
    def test_parses_proceed_from_crew_output(
        self, mock_LLM, mock_Agent, mock_Crew, mock_Task, mock_kickoff_crew,
        run_id, topic_result, synthesis_result, engineer_profile,
    ):
        mock_Task.return_value.output.raw = (
            '{"decision": "PROCEED", "reason_code": null, '
            '"failed_criteria": [], "rationale": "Digest is strong."}'
        )

        from node_definitions.output_supervisor import run as supervisor_run
        result = supervisor_run(self._make_input(run_id, topic_result, synthesis_result, engineer_profile))

        assert result.route == SupervisorRoute.PROCEED

    @patch("node_definitions.output_supervisor.kickoff_crew")
    @patch("node_definitions.output_supervisor.Task")
    @patch("node_definitions.output_supervisor.Crew")
    @patch("node_definitions.output_supervisor.Agent")
    @patch("node_definitions.output_supervisor.LLM")
    def test_structural_gate_fires_before_llm_on_short_digest(
        self, mock_LLM, mock_Agent, mock_Crew, mock_Task, mock_kickoff_crew,
        run_id, topic_result, engineer_profile,
    ):
        """Structural check catches digest < MIN_DIGEST_LENGTH before the LLM is invoked."""
        short_synthesis = SynthesisTaskResult(
            run_id=run_id,
            digest_html="<h1>Short</h1>",  # far below MIN_DIGEST_LENGTH=800
            digest_summary="Too short.",
            new_signals=["some-signal"],
            trend_confirmations=[],
        )

        from node_definitions.output_supervisor import run as supervisor_run
        result = supervisor_run(
            OutputSupervisorInput(
                run_id=run_id,
                synthesis_result=short_synthesis,
                topic=topic_result.topic,
                focus_angle=topic_result.focus_angle,
                engineer_profile=engineer_profile,
                rework_count=0,
            )
        )

        assert result.route == SupervisorRoute.REWORK
        mock_kickoff_crew.assert_not_called()
