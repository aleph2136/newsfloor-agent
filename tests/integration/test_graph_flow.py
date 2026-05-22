# tests/integration/test_graph_flow.py
import pytest
from unittest.mock import patch, MagicMock

from graph import build_graph
from contracts.primitives import NodeName, SupervisorRoute, RetryReasonCode, RunStatus
from contracts.nodes import (
    OrchestratorContext,
    TopicTaskResult,
    FetchTaskResult,
    ScoringTaskResult,
    SynthesisTaskResult,
    DeliveryTaskResult,
    TrendTaskResult,
)
from contracts.primitives import SupervisorDecision, RetryInstruction

@pytest.fixture
def compiled_graph():
    return build_graph()

@pytest.fixture
def mock_context():
    context = MagicMock(spec=OrchestratorContext)
    context.recent_topics = []
    context.active_trends = []
    context.recent_weekly_signals = []
    context.source_reputation_map = {}
    context.recent_run_signals = []
    context.engineer_profile = MagicMock()
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
@patch("node_definitions.trend.run")
def test_graph_successful_happy_path(
    mock_trend,
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
    mock_trend.assert_called_once()

    # Check final state updates
    assert final_state["run_status"] == RunStatus.COMPLETED
    assert final_state["topic_result"].topic == "multi-agent orchestration patterns"
    assert final_state["synthesis_result"].digest_html.startswith("<h1>")


@patch("data.load_context.run")
@patch("node_definitions.topic.run")
@patch("node_definitions.fetch.run")
@patch("node_definitions.scoring.run")
@patch("node_definitions.input_supervisor.run")
@patch("node_definitions.synthesis.run")
@patch("node_definitions.output_supervisor.run")
@patch("node_definitions.delivery.run")
@patch("node_definitions.trend.run")
def test_graph_input_rework_loop_limit(
    mock_trend,
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
            reason_code=RetryReasonCode.WEAK_TOPIC_SELECTION,
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
    # Pass 3: route_input_supervisor sees rework_counts >= 2 -> redirects to synthesis -> output_supervisor (PROCEED) -> delivery -> trend
    
    # Thus, topic_node should be called exactly 2 times
    assert mock_topic.call_count == 2
    assert mock_fetch.call_count == 2
    assert mock_scoring.call_count == 2
    
    # input_supervisor should be called 2 times
    assert mock_input_sup.call_count == 2
    
    # synthesis, output_supervisor, delivery, and trend should be called exactly once
    mock_synthesis.assert_called_once()
    mock_output_sup.assert_called_once()
    mock_delivery.assert_called_once()
    mock_trend.assert_called_once()
    
    # Rework counts should be accumulated in the state
    assert final_state["rework_counts"][NodeName.INPUT_SUPERVISOR.value] == 2
    assert final_state["run_status"] == RunStatus.COMPLETED
