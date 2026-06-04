# tests/integration/test_graph_flow.py
import pytest
from unittest.mock import patch, MagicMock

from graph import build_graph
from contracts.primitives import (
    NodeName,
    RetryInstruction,
    RetryReasonCode,
    RunStatus,
    SupervisorDecision,
    SupervisorRoute,
)
from contracts.nodes import (
    EngineerProfile,
    OrchestratorContext,
    TopicTaskResult,
    FetchTaskResult,
    ScoringTaskResult,
    SynthesisTaskResult,
    DeliveryTaskResult,
    PublishTaskResult,
    TrendTaskResult,
)

@pytest.fixture
def compiled_graph():
    return build_graph()

@pytest.fixture
def mock_engineer_profile():
    return EngineerProfile(
        name="Sam",
        focus_areas=["platform engineering", "AI agents", "AWS"],
        background_summary="Senior engineer moving into AI architecture.",
        experience_level="senior engineer",
    )

@pytest.fixture
def mock_context(mock_engineer_profile):
    context = MagicMock(spec=OrchestratorContext)
    context.recent_topics            = []
    context.active_trends            = []
    context.recent_weekly_signals    = []
    context.source_reputation_map    = {}
    context.recent_run_signals       = []
    context.recent_weekly_narrative  = ""
    context.seen_article_ids         = []
    context.source_last_contributed  = {}
    context.engineer_profile         = mock_engineer_profile
    return context

@pytest.fixture
def mock_topic_result():
    return TopicTaskResult(
        topic="multi-agent orchestration patterns",
        focus_angle="Implement supervisor router nodes.",
        rationale="Highly relevant to current design work.",
        confidence=0.9
    )

@pytest.fixture
def mock_fetch_result():
    return FetchTaskResult(
        run_id="test-run",
        articles=[],
        fetch_errors=[],
        article_count=0
    )

@pytest.fixture
def mock_scoring_result():
    return ScoringTaskResult(
        run_id="test-run",
        scored_articles=[],
        passed_articles=[],
        filtered_articles=[],
        high_quality_count=0,
        low_quality_count=0
    )

@pytest.fixture
def mock_synthesis_result():
    return SynthesisTaskResult(
        run_id="test-run",
        digest_html="<h1>Test Digest</h1><h2>Section</h2><em>Takeaway</em>",
        digest_summary="Summary of the digest.",
        new_signals=[],
        trend_confirmations=[]
    )

@pytest.fixture
def mock_publish_result():
    return PublishTaskResult(
        run_id="test-run",
        published=True,
        article_url="https://example.com/articles/test-run.html",
    )


@pytest.fixture
def mock_trend_result():
    return TrendTaskResult(
        run_id="test-run",
        run_status=RunStatus.COMPLETED,
        trends_updated=[],
        trends_created=[],
        trends_archived=[],
        source_reputations_updated=[],
        error=""
    )


@patch("data.load_context.run")
@patch("node_definitions.topic.run")
@patch("node_definitions.fetch.run")
@patch("node_definitions.scoring.run")
@patch("node_definitions.input_supervisor.run")
@patch("node_definitions.synthesis.run")
@patch("node_definitions.output_supervisor.run")
@patch("node_definitions.delivery.run")
@patch("node_definitions.publish.run")
@patch("node_definitions.trend.run")
def test_graph_successful_happy_path(
    mock_trend,
    mock_publish,
    mock_delivery,
    mock_output_sup,
    mock_synthesis,
    mock_input_sup,
    mock_scoring,
    mock_fetch,
    mock_topic,
    mock_load_context,
    compiled_graph,
    mock_context,
    mock_topic_result,
    mock_fetch_result,
    mock_scoring_result,
    mock_synthesis_result,
    mock_publish_result,
    mock_trend_result,
):
    """
    Test the standard happy path where load_context loads the context,
    each reasoning node executes successfully, both supervisors return PROCEED,
    and delivery + trend write completes.
    """
    # 1. Setup mock returns for all run functions
    mock_load_context.return_value = mock_context
    mock_topic.return_value = mock_topic_result
    mock_fetch.return_value = mock_fetch_result
    mock_scoring.return_value = mock_scoring_result
    
    mock_input_sup.return_value = SupervisorDecision(
        supervisor=NodeName.TOPIC, # input supervisor gate
        route=SupervisorRoute.PROCEED,
        rework_count=0,
        rationale="Input is high quality.",
    )
    
    mock_synthesis.return_value = mock_synthesis_result
    
    mock_output_sup.return_value = SupervisorDecision(
        supervisor=NodeName.SYNTHESIS, # output supervisor gate
        route=SupervisorRoute.PROCEED,
        rework_count=0,
        rationale="Output newsletter matches quality criteria.",
    )
    
    mock_delivery.return_value = MagicMock()
    mock_publish.return_value = mock_publish_result
    mock_trend.return_value = mock_trend_result

    # 2. Invoke the graph
    initial_state = {
        "run_id": "happy-run-2026",
        "rework_counts": {}
    }

    final_state = compiled_graph.invoke(initial_state)

    # 3. Assertions
    # Verify every mock was called exactly once in order
    mock_load_context.assert_called_once()
    mock_topic.assert_called_once()
    mock_fetch.assert_called_once()
    mock_scoring.assert_called_once()
    mock_input_sup.assert_called_once()
    mock_synthesis.assert_called_once()
    mock_output_sup.assert_called_once()
    mock_delivery.assert_called_once()
    mock_publish.assert_called_once()
    mock_trend.assert_called_once()

    # Check final state updates
    assert final_state["run_status"] == RunStatus.COMPLETED
    assert final_state["topic_result"].topic == "multi-agent orchestration patterns"
    assert final_state["synthesis_result"].digest_html.startswith("<h1>")
    assert final_state["publish_result"].published is True


@patch("data.load_context.run")
@patch("node_definitions.topic.run")
@patch("node_definitions.fetch.run")
@patch("node_definitions.scoring.run")
@patch("node_definitions.input_supervisor.run")
@patch("node_definitions.synthesis.run")
@patch("node_definitions.output_supervisor.run")
@patch("node_definitions.delivery.run")
@patch("node_definitions.publish.run")
@patch("node_definitions.trend.run")
def test_graph_input_rework_loop_limit(
    mock_trend,
    mock_publish,
    mock_delivery,
    mock_output_sup,
    mock_synthesis,
    mock_input_sup,
    mock_scoring,
    mock_fetch,
    mock_topic,
    mock_load_context,
    compiled_graph,
    mock_context,
    mock_topic_result,
    mock_fetch_result,
    mock_scoring_result,
    mock_synthesis_result,
    mock_publish_result,
    mock_trend_result,
):
    """
    Test that the graph handles REWORK requests from the input supervisor.
    If the supervisor repeatedly requests REWORK, the routing logic in 
    route_input_supervisor must catch this when the rework counter reaches 2,
    log a warning, and force proceed to synthesis rather than looping forever.
    """
    # 1. Setup mock returns
    mock_load_context.return_value = mock_context
    mock_topic.return_value = mock_topic_result
    mock_fetch.return_value = mock_fetch_result
    mock_scoring.return_value = mock_scoring_result
    
    # Input supervisor ALWAYS returns REWORK
    mock_input_sup.return_value = SupervisorDecision(
        supervisor=NodeName.TOPIC,
        route=SupervisorRoute.REWORK,
        rework_count=0, # supervisor returns 0, the node increment will add 1
        rationale="Topic and articles are weak. Try another selection.",
        retry_instruction=RetryInstruction(
            target_node=NodeName.TOPIC,
            reason_code=RetryReasonCode.LOW_CONFIDENCE,
            parameter_adjustment={"previous_topic": "multi-agent orchestration patterns"}
        )
    )
    
    mock_synthesis.return_value = mock_synthesis_result
    
    mock_output_sup.return_value = SupervisorDecision(
        supervisor=NodeName.SYNTHESIS,
        route=SupervisorRoute.PROCEED,
        rework_count=0,
        rationale="Proceed."
    )
    
    mock_delivery.return_value = MagicMock()
    mock_publish.return_value = mock_publish_result
    mock_trend.return_value = mock_trend_result

    # 2. Invoke the graph
    initial_state = {
        "run_id": "rework-loop-run",
        "rework_counts": {}
    }

    final_state = compiled_graph.invoke(initial_state)

    # 3. Assertions
    # Since max reworks is 2, the loop should run:
    # Pass 1: load_context -> topic -> fetch -> scoring -> input_supervisor (returns REWORK, rework_count set to 1)
    # Pass 2: route_input_supervisor redirects to topic -> fetch -> scoring -> input_supervisor (returns REWORK, rework_count set to 2)
    # Pass 3: route_input_supervisor sees rework_counts >= 2 -> redirects to synthesis -> output_supervisor (PROCEED) -> delivery -> publish -> trend

    # Thus, topic_node should be called exactly 2 times
    assert mock_topic.call_count == 2
    assert mock_fetch.call_count == 2
    assert mock_scoring.call_count == 2

    # input_supervisor should be called 2 times
    assert mock_input_sup.call_count == 2

    # synthesis, output_supervisor, delivery, publish, and trend should be called exactly once
    mock_synthesis.assert_called_once()
    mock_output_sup.assert_called_once()
    mock_delivery.assert_called_once()
    mock_publish.assert_called_once()
    mock_trend.assert_called_once()

    # Rework counts should be accumulated in the state
    assert final_state["rework_counts"][NodeName.INPUT_SUPERVISOR.value] == 2
    assert final_state["run_status"] == RunStatus.COMPLETED


@patch("data.load_context.run")
@patch("node_definitions.topic.run")
@patch("node_definitions.fetch.run")
@patch("node_definitions.scoring.run")
@patch("node_definitions.input_supervisor.run")
@patch("node_definitions.synthesis.run")
@patch("node_definitions.output_supervisor.run")
@patch("node_definitions.delivery.run")
@patch("node_definitions.publish.run")
@patch("node_definitions.trend.run")
def test_graph_output_rework_loop_limit(
    mock_trend,
    mock_publish,
    mock_delivery,
    mock_output_sup,
    mock_synthesis,
    mock_input_sup,
    mock_scoring,
    mock_fetch,
    mock_topic,
    mock_load_context,
    compiled_graph,
    mock_context,
    mock_topic_result,
    mock_fetch_result,
    mock_scoring_result,
    mock_synthesis_result,
    mock_publish_result,
    mock_trend_result,
):
    """
    Test that the graph handles REWORK from the output supervisor.
    After 2 reworks, route_output_supervisor forces proceed to delivery
    rather than looping forever.

    Expected execution order:
      Pass 1: load_context → topic → fetch → scoring → input_supervisor (PROCEED)
               → synthesis → output_supervisor (REWORK, count=1)
      Pass 2: → synthesis → output_supervisor (REWORK, count=2)
      Pass 3: route_output_supervisor sees count >= 2 → forced to delivery → trend
    """
    mock_load_context.return_value = mock_context
    mock_topic.return_value = mock_topic_result
    mock_fetch.return_value = mock_fetch_result
    mock_scoring.return_value = mock_scoring_result

    mock_input_sup.return_value = SupervisorDecision(
        supervisor=NodeName.INPUT_SUPERVISOR,
        route=SupervisorRoute.PROCEED,
        rework_count=0,
        rationale="Input is fine.",
    )

    mock_synthesis.return_value = mock_synthesis_result

    # Output supervisor ALWAYS returns REWORK
    mock_output_sup.return_value = SupervisorDecision(
        supervisor=NodeName.OUTPUT_SUPERVISOR,
        route=SupervisorRoute.REWORK,
        rework_count=0,
        rationale="Digest needs improvement.",
    )

    mock_delivery.return_value = MagicMock()
    mock_publish.return_value = mock_publish_result
    mock_trend.return_value = mock_trend_result

    initial_state = {"run_id": "output-rework-run", "rework_counts": {}}
    final_state = compiled_graph.invoke(initial_state)

    # Input stage runs once
    mock_load_context.assert_called_once()
    mock_topic.assert_called_once()
    mock_fetch.assert_called_once()
    mock_scoring.assert_called_once()
    mock_input_sup.assert_called_once()

    # Synthesis and output_supervisor each run twice before forced proceed
    assert mock_synthesis.call_count == 2
    assert mock_output_sup.call_count == 2

    # Delivery, publish, and trend run once after the forced proceed
    mock_delivery.assert_called_once()
    mock_publish.assert_called_once()
    mock_trend.assert_called_once()

    assert final_state["rework_counts"][NodeName.OUTPUT_SUPERVISOR.value] == 2
    assert final_state["run_status"] == RunStatus.COMPLETED


@patch("data.load_context.run")
@patch("node_definitions.topic.run")
@patch("node_definitions.fetch.run")
@patch("node_definitions.scoring.run")
@patch("node_definitions.input_supervisor.run")
@patch("node_definitions.synthesis.run")
@patch("node_definitions.output_supervisor.run")
@patch("node_definitions.delivery.run")
@patch("node_definitions.publish.run")
@patch("node_definitions.trend.run")
def test_graph_both_supervisors_rework_once(
    mock_trend,
    mock_publish,
    mock_delivery,
    mock_output_sup,
    mock_synthesis,
    mock_input_sup,
    mock_scoring,
    mock_fetch,
    mock_topic,
    mock_load_context,
    compiled_graph,
    mock_context,
    mock_topic_result,
    mock_fetch_result,
    mock_scoring_result,
    mock_synthesis_result,
    mock_publish_result,
    mock_trend_result,
):
    """
    Test that rework counts for each supervisor accumulate independently
    when each supervisor reworks exactly once before proceeding.

    Expected execution order:
      Pass 1: load_context → topic → fetch → scoring → input_supervisor (REWORK, count=1)
      Pass 2: → topic → fetch → scoring → input_supervisor (PROCEED)
               → synthesis → output_supervisor (REWORK, count=1)
      Pass 3: → synthesis → output_supervisor (PROCEED)
               → delivery → trend

    Final rework_counts: {input_supervisor: 1, output_supervisor: 1}
    """
    mock_load_context.return_value = mock_context
    mock_topic.return_value = mock_topic_result
    mock_fetch.return_value = mock_fetch_result
    mock_scoring.return_value = mock_scoring_result

    mock_input_sup.side_effect = [
        SupervisorDecision(
            supervisor=NodeName.INPUT_SUPERVISOR,
            route=SupervisorRoute.REWORK,
            rework_count=0,
            rationale="Weak articles — try another topic.",
            retry_instruction=RetryInstruction(
                target_node=NodeName.TOPIC,
                reason_code=RetryReasonCode.LOW_CONFIDENCE,
                parameter_adjustment={"previous_topic": "multi-agent orchestration patterns"},
            ),
        ),
        SupervisorDecision(
            supervisor=NodeName.INPUT_SUPERVISOR,
            route=SupervisorRoute.PROCEED,
            rework_count=0,
            rationale="Input is now acceptable.",
        ),
    ]

    mock_synthesis.return_value = mock_synthesis_result

    mock_output_sup.side_effect = [
        SupervisorDecision(
            supervisor=NodeName.OUTPUT_SUPERVISOR,
            route=SupervisorRoute.REWORK,
            rework_count=0,
            rationale="Digest needs more depth.",
        ),
        SupervisorDecision(
            supervisor=NodeName.OUTPUT_SUPERVISOR,
            route=SupervisorRoute.PROCEED,
            rework_count=0,
            rationale="Digest is now acceptable.",
        ),
    ]

    mock_delivery.return_value = MagicMock()
    mock_publish.return_value = mock_publish_result
    mock_trend.return_value = mock_trend_result

    initial_state = {"run_id": "both-rework-run", "rework_counts": {}}
    final_state = compiled_graph.invoke(initial_state)

    # Input stage loops once: each node called twice
    assert mock_topic.call_count == 2
    assert mock_fetch.call_count == 2
    assert mock_scoring.call_count == 2
    assert mock_input_sup.call_count == 2

    # Output stage loops once: each node called twice
    assert mock_synthesis.call_count == 2
    assert mock_output_sup.call_count == 2

    # Terminal nodes run exactly once
    mock_delivery.assert_called_once()
    mock_publish.assert_called_once()
    mock_trend.assert_called_once()

    # Verify both supervisors accumulated their rework counts independently
    assert final_state["rework_counts"][NodeName.INPUT_SUPERVISOR.value] == 1
    assert final_state["rework_counts"][NodeName.OUTPUT_SUPERVISOR.value] == 1
    assert final_state["run_status"] == RunStatus.COMPLETED
