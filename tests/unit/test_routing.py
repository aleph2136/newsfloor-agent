# tests/unit/test_routing.py
#
# Tests for the two conditional edge routing functions in graph/nodes.py.
# These are pure functions — they read supervisor decisions and rework counts
# from state and return the name of the next node to execute. No I/O, no LLM.

import pytest

from contracts.primitives import NodeName, SupervisorDecision, SupervisorRoute
from nodes import route_input_supervisor, route_output_supervisor


# ---------------------------------------------------------------------------
# State builders
# ---------------------------------------------------------------------------

def _input_state(rework_counts=None, decision=None) -> dict:
    return {
        "run_id": "test-run",
        "rework_counts": rework_counts or {},
        "input_supervisor_decision": decision,
    }


def _output_state(rework_counts=None, decision=None) -> dict:
    return {
        "run_id": "test-run",
        "rework_counts": rework_counts or {},
        "output_supervisor_decision": decision,
    }


def _input_proceed() -> SupervisorDecision:
    return SupervisorDecision(
        supervisor=NodeName.INPUT_SUPERVISOR,
        route=SupervisorRoute.PROCEED,
        rework_count=0,
        rationale="Quality gate passed.",
    )


def _input_rework() -> SupervisorDecision:
    return SupervisorDecision(
        supervisor=NodeName.INPUT_SUPERVISOR,
        route=SupervisorRoute.REWORK,
        rework_count=0,
        rationale="Not enough articles.",
    )


def _output_proceed() -> SupervisorDecision:
    return SupervisorDecision(
        supervisor=NodeName.OUTPUT_SUPERVISOR,
        route=SupervisorRoute.PROCEED,
        rework_count=0,
        rationale="Digest is high quality.",
    )


def _output_rework() -> SupervisorDecision:
    return SupervisorDecision(
        supervisor=NodeName.OUTPUT_SUPERVISOR,
        route=SupervisorRoute.REWORK,
        rework_count=0,
        rationale="Digest needs improvement.",
    )


# ---------------------------------------------------------------------------
# route_input_supervisor
# ---------------------------------------------------------------------------

def test_route_input_supervisor_proceed_goes_to_synthesis():
    state = _input_state(decision=_input_proceed())
    assert route_input_supervisor(state) == NodeName.SYNTHESIS.value


def test_route_input_supervisor_rework_goes_to_topic():
    state = _input_state(
        rework_counts={NodeName.INPUT_SUPERVISOR.value: 1},
        decision=_input_rework(),
    )
    assert route_input_supervisor(state) == NodeName.TOPIC.value


def test_route_input_supervisor_first_rework_with_empty_counts_goes_to_topic():
    state = _input_state(rework_counts={}, decision=_input_rework())
    assert route_input_supervisor(state) == NodeName.TOPIC.value


def test_route_input_supervisor_at_max_reworks_forces_synthesis():
    state = _input_state(
        rework_counts={NodeName.INPUT_SUPERVISOR.value: 2},
        decision=_input_rework(),
    )
    assert route_input_supervisor(state) == NodeName.SYNTHESIS.value


def test_route_input_supervisor_above_max_reworks_forces_synthesis():
    state = _input_state(
        rework_counts={NodeName.INPUT_SUPERVISOR.value: 3},
        decision=_input_rework(),
    )
    assert route_input_supervisor(state) == NodeName.SYNTHESIS.value


def test_route_input_supervisor_no_decision_defaults_to_synthesis():
    state = _input_state(decision=None)
    assert route_input_supervisor(state) == NodeName.SYNTHESIS.value


def test_route_input_supervisor_no_decision_with_high_rework_count_still_synthesis():
    state = _input_state(
        rework_counts={NodeName.INPUT_SUPERVISOR.value: 5},
        decision=None,
    )
    assert route_input_supervisor(state) == NodeName.SYNTHESIS.value


# ---------------------------------------------------------------------------
# route_output_supervisor
# ---------------------------------------------------------------------------

def test_route_output_supervisor_proceed_goes_to_delivery():
    state = _output_state(decision=_output_proceed())
    assert route_output_supervisor(state) == NodeName.DELIVERY.value


def test_route_output_supervisor_rework_goes_to_synthesis():
    state = _output_state(
        rework_counts={NodeName.OUTPUT_SUPERVISOR.value: 1},
        decision=_output_rework(),
    )
    assert route_output_supervisor(state) == NodeName.SYNTHESIS.value


def test_route_output_supervisor_first_rework_with_empty_counts_goes_to_synthesis():
    state = _output_state(rework_counts={}, decision=_output_rework())
    assert route_output_supervisor(state) == NodeName.SYNTHESIS.value


def test_route_output_supervisor_at_max_reworks_forces_delivery():
    state = _output_state(
        rework_counts={NodeName.OUTPUT_SUPERVISOR.value: 2},
        decision=_output_rework(),
    )
    assert route_output_supervisor(state) == NodeName.DELIVERY.value


def test_route_output_supervisor_above_max_reworks_forces_delivery():
    state = _output_state(
        rework_counts={NodeName.OUTPUT_SUPERVISOR.value: 5},
        decision=_output_rework(),
    )
    assert route_output_supervisor(state) == NodeName.DELIVERY.value


def test_route_output_supervisor_no_decision_defaults_to_delivery():
    state = _output_state(decision=None)
    assert route_output_supervisor(state) == NodeName.DELIVERY.value


def test_route_output_supervisor_unrelated_rework_counts_not_counted():
    # input_supervisor rework count should not affect output routing
    state = _output_state(
        rework_counts={NodeName.INPUT_SUPERVISOR.value: 5},
        decision=_output_rework(),
    )
    assert route_output_supervisor(state) == NodeName.SYNTHESIS.value
