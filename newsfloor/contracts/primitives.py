"""
primitives.py

Shared building blocks used across node contracts.
These are teh smallest units of structured data in the system.
No node-specific logic lives here.
"""

from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, Field, HttpUrl

class RunStatus(str, Enum):
    """Tracks the overall health of a daily run."""
    IN_PROGRESS = "in_progress"
    COMPLETED   = "complete"
    DEGRADED    = "degraded" # one or more gates failed
    FAILED      = "failed" # could not recover

class NodeName(str, Enum):
    """ 
    Canonical names for every node in the pipeline.
    Used in gate decisions and retry instructions so nothing is
    stringly typed.
    """
    LOAD_CONTEXT      = "load_context"
    TOPIC             = "topic"
    FETCH             = "fetch"
    SCORING           = "scoring"
    INPUT_SUPERVISOR  = "input_supervisor"
    SYNTHESIS         = "synthesis"
    OUTPUT_SUPERVISOR = "output_supervisor"
    DELIVERY          = "delivery"
    TREND             = "trend"

class RetryReasonCode(str, Enum):
    """
    Structured reason codes the orchestrator uses when sending a retry instruction.
    Nodes receive a reason code + parameter adjustment — never a free-text prompt.
    """
    INSUFFICIENT_ARTICLES  = "insufficient_articles"
    LOW_CONFIDENCE         = "low_confidence"
    LOW_QUALITY_ARTICLES   = "low_quality_articles"
    MISSING_REQUIRED_FIELD = "missing_required_field"
    SOURCE_FETCH_FAILURE   = "source_fetch_failure"
    BELOW_SCORE_THRESHOLD  = "below_score_threshold"
    TREND_WRITE_FAILURE    = "trend_write_failure"
    DIGEST_INSUFFICIENT    = "digest_insufficient"
    WEAK_TOPIC_SELECTION   = "weak_topic_selection"

class TrendStrength(str, Enum):
    """
    Human-readable band for a trend's current strength score.
    Derived from the float strength value — used in digest copy and logging.
    """
    EMERGING  = "emerging"   # 0.1 – 0.39
    GROWING   = "growing"    # 0.4 – 0.64
    STRONG    = "strong"     # 0.65 – 0.84
    DOMINANT  = "dominant"   # 0.85 – 1.0

class ArticleRaw(BaseModel):
    """
    An article as it arrives from the Fetch node — no scoring applied yet.
    Every field the Scoring node will need is captured here so it never
    has to re-fetch.
    """
    article_id:    str      = Field(description="SHA-256 hash of the URL. Stable across runs.")
    url:           str      = Field(description="Canonical article URL.")
    title:         str
    source_domain: str      = Field(description="e.g. 'simonwillison.net'")
    published_at:  str      = Field(description="ISO 8601 date string. Empty string if unavailable.")
    summary:       str      = Field(description="First 500 chars of article body or RSS description.")
    fetch_error:   str      = Field(default="", description="Non-empty if this article had a partial fetch failure.")

class ArticleScored(BaseModel):
    """
    An article after the Scoring node has evaluated it.
    Extends ArticleRaw with scores and a pass/fail decision.
    """
    article_id:         str
    url:                str
    title:              str
    source_domain:      str
    published_at:       str
    summary:            str
 
    relevance_score:    float = Field(ge=0.0, le=1.0, description="How relevant to today's topic and focus angle.")
    reputation_score:   float = Field(ge=0.0, le=1.0, description="Source domain reputation at time of scoring.")
    combined_score:     float = Field(ge=0.0, le=1.0, description="Weighted combination used for filtering.")
    passed_threshold:   bool  = Field(description="True if combined_score meets the orchestrator's minimum.")
    score_rationale:    str   = Field(description="One sentence explaining the combined score.")
 
# ---------------------------------------------------------------------------
# Orchestrator gate types
# ---------------------------------------------------------------------------
 
class GateDecision(BaseModel):
    """
    The orchestrator's verdict after evaluating a node's output.
    Every transition between nodes produces exactly one of these.
    """
    node_evaluated: NodeName
    passed:         bool
    issues:         list[str] = Field(default_factory=list, description="Specific failures. Empty if passed.")
    retry_count:    int       = Field(default=0, description="How many times this node has been retried this run.")
 
 
class RetryInstruction(BaseModel):
    """
    Sent back to a node when the gate does not pass.
    Contains a reason code and a parameter adjustment dict — no free text.
    The receiving node pattern-matches on reason_code to adjust its behavior.
    """
    target_node:         NodeName = Field(alias="node")
    reason_code:         RetryReasonCode
    parameter_adjustment: dict = Field(
        default_factory=dict,
        description=(
            "Key/value pairs the node should apply on retry. "
            "e.g. {'min_articles': 5} or {'score_threshold': 0.4}. "
            "Never contains natural language instructions."
        )
    )

    model_config = {
        "populate_by_name": True
    }


class SupervisorRoute(str, Enum):
    """Routing outcomes for supervisors."""
    PROCEED = "proceed"
    REWORK  = "rework"


class SupervisorDecision(BaseModel):
    """
    The outcome of a supervisor's holistic quality gate evaluation.
    Determines whether the graph proceeds or loops back for rework.
    """
    supervisor:        NodeName
    route:             SupervisorRoute
    rework_count:      int
    rationale:         str
    retry_instruction: RetryInstruction | None = None