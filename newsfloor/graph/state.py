"""
graph/state.py
 
The canonical LangGraph state definition for the digest pipeline.
 
Why this file exists separately from contracts/nodes.py
────────────────────────────────────────────────────────
LangGraph requires state to be a TypedDict — it reads, merges, and checkpoints
it using that interface. Pydantic BaseModel does not satisfy that requirement.
 
contracts/nodes.GraphState (BaseModel) is the documented shape — readable,
type-checked by mypy, and useful in tests. This TypedDict mirrors it exactly
and is what the actual graph uses at runtime. If you add a field to one,
add it to the other.
 
Reducer note
────────────
Most fields use the default "last write wins" reducer — the node just
returns the new value and LangGraph replaces the old one. The only
exception is rework_counts, which uses a custom reducer that merges
dicts so a supervisor incrementing one key doesn't wipe out counts
from the other supervisor.
"""
from __future__ import annotations
from typing import TypedDict, Annotated
from contracts.primitives import RunStatus, SupervisorDecision, RetryInstruction
from contracts.nodes import (
    OrchestratorContext,
    TopicTaskResult,
    FetchTaskResult,
    ScoringTaskResult,
    SynthesisTaskResult,
    DeliveryTaskResult,
    TrendTaskResult,
)

def merge_rework_counts(existing: dict[str, int], update: dict[str, int]) -> dict[str, int]:
    """
    Custom reducer for rework_counts.
 
    LangGraph calls this whenever a node returns a rework_counts update.
    We merge rather than replace so each supervisor's count is independent.
 
    Example:
        existing = {"input_supervisor": 1}
        update   = {"output_supervisor": 1}
        result   = {"input_supervisor": 1, "output_supervisor": 1}
    """
    merged = dict(existing)
    for key, value in update.items():
        merged[key] = merged.get(key, 0) + value
    return merged

class DigestGraphState(TypedDict):
    """
    LangGraph state for the digest pipeline.
    Every node receives this full state and returns a partial update
    containing only the fields it changed.
 
    Field naming convention
    ───────────────────────
    <node>_result         → output of that reasoning node
    <supervisor>_decision → output of that supervisor node
    active_retry_*        → current retry instruction, cleared after use
    """
    run_id: str

    # Loaded once at start by load_context -- read-only for all other nodes
    context: OrchestratorContext | None

    # Reasoning node results
    topic_result:     TopicTaskResult     | None
    fetch_result:     FetchTaskResult     | None
    scoring_result:   ScoringTaskResult   | None
    synthesis_result: SynthesisTaskResult | None
    delivery_result:  DeliveryTaskResult  | None
    trend_result:     TrendTaskResult     | None

    # Supervisor decisions - one slot per supervisor
    input_supervisor_decision:  SupervisorDecision | None
    output_supervisor_decision: SupervisorDecision | None

    # Active retry instruction - written by supervisor, read by target node,
    # then cleared. Prevents a stale instruction from a prior rework cycle
    # from being applied on a later pass.object
    active_retry_instruction: RetryInstruction | None

    # Rework counts - keyed by NodeName.value, mered across supervisors
    rework_counts: Annotated[dict[str, int], merge_rework_counts]

    # Writen by trend_node at end of run
    run_status: RunStatus