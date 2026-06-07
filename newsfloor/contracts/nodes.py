"""
nodes.py
 
Typed input/output contracts for every node in the pipeline.
 
Reading guide
─────────────
Each reasoning node has a TaskInput and a TaskResult.
Each supervisor node has a SupervisorInput and produces a SupervisorDecision
(defined in primitives.py — it is shared across both supervisors).
 
GraphState is the single typed dict that LangGraph threads through every node.
Every node reads what it needs from GraphState and writes its result back into it.
Nodes never pass data to each other directly.
 
Changes from the CrewAI orchestrator design
───────────────────────────────────────────
- All `gate` fields removed from TaskResult models. The graph topology
  replaces the orchestrator's gate loop — supervisors write SupervisorDecision
  into GraphState and LangGraph routes on it.
- OrchestratorContext added — assembled by load_context, read by nodes that
  need historical state. Replaces the implicit context the orchestrator held.
- GraphState added — the single source of truth for a run's state.
"""
 
from __future__ import annotations
from typing import Annotated
from pydantic import BaseModel, Field
from langgraph.graph import add_messages
from .primitives import (
    ArticleRaw,
    ArticleScored,
    NodeName,
    RetryInstruction,
    RunStatus,
    SupervisorDecision,
    TrendStrength,
)


# ---------------------------------------------------------------------------
# Structured digest data model
# The new multi-tiered JSON format produced by the synthesis node.
# Consumed by publish (HTML rendering) and delivery (plain text email).
# ---------------------------------------------------------------------------

class VisualAssets(BaseModel):
    mermaid_diagram: str = Field(default="", description="Valid Mermaid.js flowchart or sequence diagram syntax.")
    code_block:      str = Field(default="", description="Illustrative Python or TypeScript code snippet.")


class DigestContentBlock(BaseModel):
    section_id:        str            = Field(description="Unique block ID, e.g. 'block_1'.")
    section_title:     str            = Field(description="The specific technical concept or pattern.")
    tier_1_hook:       str            = Field(description="1-sentence main takeaway for a senior engineer.")
    tier_2_bullets:    list[str]      = Field(description="2-3 bullets; each starts with **bold anchor** text.")
    tier_3_deep_dive:  str            = Field(description="Dense technical elaboration, 1-2 paragraphs max.")
    visual_assets:     VisualAssets   = Field(default_factory=VisualAssets)


class DigestMetadata(BaseModel):
    title:                 str = Field(description="Specific, compelling headline.")
    date:                  str = Field(description="ISO date, e.g. '2026-06-06'.")
    summary_hook:          str = Field(description="1-sentence hook: the key question or tension.")
    overall_trend_context: str = Field(description="1-sentence industry movement this content reflects.")


class DigestStructured(BaseModel):
    """
    The structured JSON document produced by the synthesis writer agent.
    Replaces the flat HTML string for downstream rendering and delivery.
    """
    article_id:     str                   = Field(description="Unique article ID, e.g. 'YYYY-MM-DD-slug'.")
    metadata:       DigestMetadata
    content_blocks: list[DigestContentBlock] = Field(default_factory=list)
 
 
# ---------------------------------------------------------------------------
# Orchestrator context
# Assembled once by load_context at run start. Read-only after that.
# Contains everything loaded from DynamoDB that nodes need as background.
# ---------------------------------------------------------------------------
 
class OrchestratorContext(BaseModel):
    """
    Snapshot of historical state loaded from DynamoDB at the start of each run.
    Assembled by load_context, then passed through GraphState as read-only context.
    No node writes to this — it represents the world as it was before this run.
    """
    active_trends:           list[TrendSnapshot]  = Field(
        description="Trend records with strength > 0.3, ordered by strength desc."
    )
    source_reputation_map:   dict[str, float]     = Field(
        description="Current reputation score for every known domain."
    )
    recent_topics:           list[str]            = Field(
        description="Topics covered in the last 30 days, most recent first."
    )
    recent_run_signals:      list[str]            = Field(
        description="Raw trend signals from the last 7 run records."
    )
    recent_weekly_signals:   list[str]            = Field(
        description="Recurring signals from the last 2 weekly synthesis records."
    )
    recent_weekly_narrative: str                  = Field(
        default="",
        description=(
            "LLM-generated narrative from the most recent WeeklySynthesis record. "
            "Injected into topic, input_supervisor, and synthesis prompts to give "
            "each node longitudinal context about last week's momentum."
        )
    )
    seen_article_ids:        list[str]            = Field(
        default_factory=list,
        description="article_id hashes seen in the last 14 days. Passed to fetch to prevent article recurrence."
    )
    source_last_contributed: dict[str, str]       = Field(
        default_factory=dict,
        description="ISO date when each domain last had an article pass scoring. Used for fetch-time rotation weighting."
    )
    engineer_profile:        EngineerProfile
 
 
# ---------------------------------------------------------------------------
# GraphState
# The single typed dict LangGraph threads through every node.
# Every field is Optional — nodes only populate their own result field.
# ---------------------------------------------------------------------------
 
class GraphState(BaseModel):
    """
    Shared state for the entire LangGraph run.
 
    LangGraph note: in the actual graph definition this will be declared as
    a TypedDict rather than a BaseModel so LangGraph can manage it natively.
    It is defined here as a BaseModel for documentation and type-checking
    purposes. The graph module will mirror these fields as a TypedDict.
 
    Naming convention: <node_name>_result holds that node's TaskResult.
    Supervisor decisions are stored separately so they are always easy to find.
    """
    run_id:               str
 
    # Loaded at start — read-only for the rest of the run
    context:              OrchestratorContext | None = None
 
    # Reasoning node results — populated as each node completes
    topic_result:         TopicTaskResult    | None = None
    fetch_result:         FetchTaskResult    | None = None
    scoring_result:       ScoringTaskResult  | None = None
    synthesis_result:     SynthesisTaskResult| None = None
    delivery_result:      DeliveryTaskResult | None = None
    publish_result:       PublishTaskResult  | None = None
    trend_result:         TrendTaskResult    | None = None
 
    # Supervisor decisions — one slot per supervisor
    input_supervisor_decision:  SupervisorDecision | None = None
    output_supervisor_decision: SupervisorDecision | None = None
 
    # Active retry instruction — written by a supervisor, cleared after the
    # target node reads it. Prevents stale instructions from a prior rework
    # from being applied on a subsequent pass.
    active_retry_instruction: RetryInstruction | None = None
 
    # Rework tracking — keyed by NodeName value, incremented by supervisors
    rework_counts: dict[str, int] = Field(default_factory=dict)
 
    # Final run status — written by trend_node at end of run
    run_status: RunStatus = RunStatus.IN_PROGRESS
 
 
# ---------------------------------------------------------------------------
# Topic Node
# ---------------------------------------------------------------------------
 
class TopicTaskInput(BaseModel):
    """Everything the Topic node needs to make an informed selection."""
    run_id:                  str
    recent_topics:           list[str]        = Field(description="Topics covered in the last 30 days.")
    active_trend_names:      list[str]        = Field(description="Names of trends with strength > 0.3.")
    recent_signals:          list[str]        = Field(description="Raw signals from the last two weekly syntheses.")
    available_topics:        list[str]        = Field(description="Full rotation list the node may select from.")
    recent_weekly_narrative: str              = Field(
        default="",
        description="LLM narrative from last Monday's WeeklySynthesis. Injected into the strategist prompt."
    )
    retry_instruction:       RetryInstruction | None = Field(
        default=None,
        description="Populated on rework. Node pattern-matches on reason_code to adjust behavior."
    )
 
 
class TopicTaskResult(BaseModel):
    """The Topic node's selection and its reasoning."""
    topic:       str   = Field(description="The topic selected for today's digest.")
    focus_angle: str   = Field(description="The specific lens to apply. e.g. 'apply to platform engineering'.")
    rationale:   str   = Field(description="Why this topic now, given recent coverage and active trends.")
    confidence:  float = Field(ge=0.0, le=1.0, description="Node's self-assessed confidence in this selection.")
 
 
# ---------------------------------------------------------------------------
# Fetch Node
# ---------------------------------------------------------------------------
 
class FetchTaskInput(BaseModel):
    """What the Fetch node needs to retrieve articles."""
    run_id:                  str
    topic:                   str
    focus_angle:             str
    sources:                 list[str]        = Field(description="Curated list of RSS feed URLs or known article endpoints.")
    min_articles:            int              = Field(default=3)
    max_articles:            int              = Field(default=10)
    seen_article_ids:        list[str]        = Field(default_factory=list, description="Article IDs seen in recent runs. Skipped during fetch.")
    source_last_contributed: dict[str, str]   = Field(default_factory=dict, description="Domain → ISO date of last passing article. Used for rotation weighting.")
    retry_instruction:       RetryInstruction | None = Field(default=None)
 
 
class FetchTaskResult(BaseModel):
    """Raw articles returned from sources, before any scoring."""
    run_id:        str
    articles:      list[ArticleRaw]
    fetch_errors:  list[str] = Field(default_factory=list, description="URLs that failed with reason.")
    article_count: int
 
 
# ---------------------------------------------------------------------------
# Scoring Node
# ---------------------------------------------------------------------------
 
class ScoringTaskInput(BaseModel):
    """Articles plus context the Scoring node needs to evaluate them."""
    run_id:                str
    topic:                 str
    focus_angle:           str
    articles:              list[ArticleRaw]
    source_reputation_map: dict[str, float] = Field(
        description="Reputation scores keyed by domain. e.g. {'simonwillison.net': 0.82}"
    )
    active_trend_names:    list[str] = Field(
        description="High-strength trend names used to boost relevance scoring."
    )
    score_threshold:       float     = Field(default=0.5)
    retry_instruction:     RetryInstruction | None = Field(default=None)
 
 
class ScoringTaskResult(BaseModel):
    """Scored articles split into passed and filtered sets."""
    run_id:             str
    scored_articles:    list[ArticleScored]
    passed_articles:    list[ArticleScored] = Field(description="Articles that met the score threshold.")
    filtered_articles:  list[ArticleScored] = Field(description="Articles below threshold. Kept for trend logging.")
    high_quality_count: int
    low_quality_count:  int
 
 
# ---------------------------------------------------------------------------
# Input Supervisor
# Evaluates topic + fetch + scoring as a unit before synthesis begins.
# Rework routes back to topic_node — a thin result may mean wrong topic,
# not just a bad fetch.
# ---------------------------------------------------------------------------
 
class InputSupervisorInput(BaseModel):
    """
    The full picture of the input stage, passed to the input supervisor.
    Supervisor evaluates these together — not each in isolation.
    """
    run_id:                  str
    topic_result:            TopicTaskResult
    fetch_result:            FetchTaskResult
    scoring_result:          ScoringTaskResult
    recent_weekly_narrative: str = Field(
        default="",
        description="LLM narrative from last Monday's WeeklySynthesis. Injected into the LLM evaluator prompt."
    )
    rework_count:            int = Field(default=0, description="How many times this supervisor has reworked this run.")
    max_reworks:             int = Field(default=2)
 
 
# SupervisorDecision is the output — defined in primitives.py
# InputSupervisor returns: SupervisorDecision
 
 
# ---------------------------------------------------------------------------
# Synthesis Node
# ---------------------------------------------------------------------------
 
class SynthesisTaskInput(BaseModel):
    """Scored articles plus historical context for writing the digest."""
    run_id:                  str
    topic:                   str
    focus_angle:             str
    passed_articles:         list[ArticleScored]
    active_trends:           list[TrendSnapshot]  = Field(
        description="Full trend snapshots for contextual reasoning."
    )
    recent_run_signals:      list[str]            = Field(
        description="Raw signals from the last 7 run records."
    )
    recent_weekly_narrative: str                  = Field(
        default="",
        description="LLM narrative from last Monday's WeeklySynthesis. Injected into contextualizer and writer prompts."
    )
    engineer_profile:        EngineerProfile
    retry_instruction:       RetryInstruction | None = Field(default=None)
 
 
class SynthesisTaskResult(BaseModel):
    """The finished digest and signals extracted for the Trend node."""
    run_id:              str
    digest_html:         str                    = Field(description="HTML representation generated from digest_json. Used by the output supervisor for quality evaluation.")
    digest_json:         DigestStructured | None = Field(default=None, description="Structured JSON digest. Used by publish (HTML rendering) and delivery (plain text email).")
    digest_summary:      str                    = Field(description="Plain text 3-5 sentence summary for DynamoDB storage.")
    new_signals:         list[str]              = Field(description="Trend signals extracted from today's articles.")
    trend_confirmations: list[str]              = Field(description="Names of existing trends this run reinforces.")
 
 
# ---------------------------------------------------------------------------
# Output Supervisor
# Evaluates the synthesis result before delivery.
# Rework routes back to synthesis_node only — articles are not re-fetched.
# ---------------------------------------------------------------------------
 
class OutputSupervisorInput(BaseModel):
    """
    The synthesis result plus the context it was generated from.
    Supervisor checks whether the output reflects the input faithfully.
    """
    run_id:           str
    synthesis_result: SynthesisTaskResult
    topic:            str
    focus_angle:      str
    engineer_profile: EngineerProfile
    rework_count:     int = Field(default=0)
    max_reworks:      int = Field(default=2)
 
 
# SupervisorDecision is the output — defined in primitives.py
# OutputSupervisor returns: SupervisorDecision
 
 
# ---------------------------------------------------------------------------
# Delivery Node
# ---------------------------------------------------------------------------
 
class DeliveryTaskInput(BaseModel):
    """Everything needed to send the digest email via SES."""
    run_id:          str
    digest_html:     str
    digest_json:     DigestStructured | None = Field(default=None, description="Structured JSON digest for plain-text email formatting.")
    topic:           str
    recipient_email: str
    sender_email:    str
    article_url:     str = Field(default="", description="Published article URL to include in the email footer.")
 
 
class DeliveryTaskResult(BaseModel):
    """Confirmation of send attempt — success or failure, never reworked."""
    run_id:     str
    sent:       bool
    message_id: str = Field(default="", description="SES message ID if sent successfully.")
    error:      str = Field(default="", description="Non-empty if sending failed.")
 
 
# ---------------------------------------------------------------------------
# Publish Node
# ---------------------------------------------------------------------------

class PublishTaskInput(BaseModel):
    """Everything the Publish node needs to upload the article to the personal site."""
    run_id:      str
    digest_html: str
    digest_json: DigestStructured | None = Field(default=None, description="Structured JSON digest for progressive-disclosure HTML rendering.")
    topic:       str
    bucket:      str = Field(description="S3 bucket name. Empty string skips publish.")
    cf_dist_id:  str = Field(description="CloudFront distribution ID for cache invalidation.")
    domain:      str = Field(description="Site domain, e.g. 'sam-griffith.dev'. Used to build article URLs.")
    author_name: str = Field(description="Author name rendered in the article template sidebar.")


class PublishTaskResult(BaseModel):
    """Outcome of the publish attempt."""
    run_id:      str
    published:   bool
    skipped:     bool  = Field(default=False, description="True if publish was skipped due to missing config.")
    article_url: str   = Field(default="")
    error:       str   = Field(default="")


# ---------------------------------------------------------------------------
# Trend Node
# ---------------------------------------------------------------------------
 
class TrendTaskInput(BaseModel):
    """
    The full picture of today's run passed to the Trend node for state updates.
    This node is the only one that writes to DynamoDB.
    """
    run_id:                str
    topic:                 str
    focus_angle:           str
    scored_articles:       list[ArticleScored]
    new_signals:           list[str]
    trend_confirmations:   list[str]
    digest_summary:        str
    existing_trends:       list[TrendSnapshot]
    source_reputation_map: dict[str, float]
    delivery_sent:         bool = Field(description="Whether delivery succeeded — written into RunRecord.")
    input_rework_count:    int  = Field(default=0, description="Total reworks by input supervisor this run.")
    output_rework_count:   int  = Field(default=0, description="Total reworks by output supervisor this run.")
 
 
class TrendTaskResult(BaseModel):
    """What the Trend node wrote to DynamoDB."""
    run_id:                     str
    run_status:                 RunStatus
    trends_updated:             list[str] = Field(description="Trend IDs that had strength adjusted.")
    trends_created:             list[str] = Field(description="Trend IDs created for the first time.")
    trends_archived:            list[str] = Field(description="Trend IDs moved to archived state.")
    source_reputations_updated: list[str] = Field(description="Domains whose reputation score changed.")
    error:                      str       = Field(default="")
 
 
# ---------------------------------------------------------------------------
# Supporting models
# ---------------------------------------------------------------------------
 
class TrendSnapshot(BaseModel):
    """
    A read-only view of a TrendRecord passed into nodes as context.
    Nodes never write trends — only the Trend node does that via DynamoDB.
    """
    trend_id:           str
    name:               str
    strength:           float = Field(ge=0.0, le=1.0)
    strength_band:      TrendStrength
    platform_relevance: float = Field(ge=0.0, le=1.0)
    key_signals:        list[str]
    last_reinforced:    str   = Field(description="ISO 8601 date.")
 
 
class EngineerProfile(BaseModel):
    """
    Static context about the digest recipient.
    Passed into Synthesis and the output supervisor so output is evaluated
    against the same personalization criteria it was generated with.
    """
    name:               str
    focus_areas:        list[str] = Field(description="e.g. ['platform engineering', 'AI agents', 'AWS']")
    background_summary: str       = Field(description="Brief paragraph used to personalize synthesis output.")
    experience_level:   str       = Field(description="e.g. 'senior engineer moving into AI architecture'")