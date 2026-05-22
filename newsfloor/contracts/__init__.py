"""
contracts/__init__.py
 
Public surface of the contracts package.
Import everything from here rather than from submodules directly.
 
Usage:
    from contracts import TopicTaskInput, GraphState, SupervisorDecision
"""
 
from .primitives import (
    RunStatus,
    NodeName,
    RetryReasonCode,
    TrendStrength,
    ArticleRaw,
    ArticleScored,
    SupervisorRoute,
    RetryInstruction,
    SupervisorDecision,
)
 
from .nodes import (
    OrchestratorContext,
    GraphState,
    TopicTaskInput,
    TopicTaskResult,
    FetchTaskInput,
    FetchTaskResult,
    ScoringTaskInput,
    ScoringTaskResult,
    InputSupervisorInput,
    SynthesisTaskInput,
    SynthesisTaskResult,
    OutputSupervisorInput,
    DeliveryTaskInput,
    DeliveryTaskResult,
    TrendTaskInput,
    TrendTaskResult,
    TrendSnapshot,
    EngineerProfile,
)
 
from .state import (
    RunRecord,
    WeeklySynthesis,
    TrendRecord,
    SourceRecord,
    ttl_days,
    current_week_id,
)
 
__all__ = [
    # Primitives
    "RunStatus", "NodeName", "RetryReasonCode", "TrendStrength",
    "ArticleRaw", "ArticleScored",
    "SupervisorRoute", "RetryInstruction", "SupervisorDecision",
    # Node contracts
    "OrchestratorContext", "GraphState",
    "TopicTaskInput", "TopicTaskResult",
    "FetchTaskInput", "FetchTaskResult",
    "ScoringTaskInput", "ScoringTaskResult",
    "InputSupervisorInput",
    "SynthesisTaskInput", "SynthesisTaskResult",
    "OutputSupervisorInput",
    "DeliveryTaskInput", "DeliveryTaskResult",
    "TrendTaskInput", "TrendTaskResult",
    "TrendSnapshot", "EngineerProfile",
    # State / DynamoDB records
    "RunRecord", "WeeklySynthesis", "TrendRecord", "SourceRecord",
    "ttl_days", "current_week_id",
]