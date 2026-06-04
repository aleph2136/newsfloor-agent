"""
state.py
 
DynamoDB record schemas — the persistent state layer.
 
Three tiers
───────────
Tier 1 — RunRecord         TTL 30 days   One record per daily run
Tier 2 — WeeklySynthesis   TTL 90 days   One record per week, written by Trend node on Mondays
Tier 3 — TrendRecord       Permanent     Named durable trends with decaying strength scores
         SourceRecord       Permanent     Per-domain reputation scores
 
These are NOT node contracts — no node receives or returns these directly.
The Trend node reads/writes them via a DynamoDB service layer (built in Phase 2).
The Orchestrator reads TrendRecord and SourceRecord at startup to build context.
 
DynamoDB key design
───────────────────
RunRecord          PK: run_id          (e.g. "2026-05-17")
WeeklySynthesis    PK: week_id         (e.g. "2026-W20")
TrendRecord        PK: trend_id        (e.g. "multi-agent-coordination")
SourceRecord       PK: domain          (e.g. "simonwillison.net")
"""

from __future__ import annotations
import time
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from .primitives import RunStatus, TrendStrength

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ttl_days(days: int) -> int:
    """Returns a unix timestamp N days from now. Used to set DynamoDB TTL."""
    return int((datetime.utcnow() + timedelta(days=days)).timestamp())

def current_week_id() -> str:
    """Returns ISO week string e.g. '2026-W20'."""
    now = datetime.utcnow()
    return f"{now.year}-W{now.isocalendar()[1]:02d}"

# ---------------------------------------------------------------------------
# Tier 1 — Run Records (TTL: 30 days)
# ---------------------------------------------------------------------------

class RunRecord(BaseModel):
    """
    One record per daily run. Written progressively as nodes complete.
    The Trend node writes the final state of this record at end of run.
 
    DynamoDB table: digest-run-records
    PK: run_id
    TTL attribute: ttl
    """
    run_id:              str         = Field(description="ISO date string. e.g. '2026-05-17'")
    status:              RunStatus   = Field(default=RunStatus.IN_PROGRESS)
    topic:               str         = Field(default="")
    focus_angle:         str         = Field(default="")
    articles_fetched:    int         = Field(default=0)
    articles_passed:     int         = Field(default=0)
    new_signals:         list[str]   = Field(default_factory=list)
    trend_confirmations: list[str]   = Field(default_factory=list)
    digest_summary:      str         = Field(default="")
    article_ids_used:    list[str]   = Field(
        default_factory=list,
        description="article_id hashes of articles that passed scoring this run. Used for cross-run dedup."
    )
    orchestrator_notes:  list[str]   = Field(
        default_factory=list,
        description="Gate failures and degraded-mode decisions logged here for observability."
    )
    created_at:          str         = Field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at:        str         = Field(default="")
    ttl:                 int         = Field(default_factory=lambda: ttl_days(30))

# ---------------------------------------------------------------------------
# Tier 2 — Weekly Synthesis (TTL: 90 days)
# ---------------------------------------------------------------------------

class WeeklySynthesis(BaseModel):
    """
    Written by the Trend node on the first run of each Monday.
    Distills the past 7 run records into pattern signals.

    DynamoDB table: digest-weekly-synthesis
    PK: week_id
    TTL attribute: ttl
    """
    week_id:              str       = Field(description="e.g. '2026-W20'")
    topics_covered:       list[str] = Field(default_factory=list)
    recurring_signals:    list[str] = Field(
        default_factory=list,
        description="Signals that appeared in 3 or more runs this week."
    )
    emerging_concepts:    list[str] = Field(
        default_factory=list,
        description="Terms not present in the prior week's synthesis."
    )
    fading_concepts:      list[str] = Field(
        default_factory=list,
        description="Terms present last week but absent this week."
    )
    source_reputation_deltas: dict[str, float] = Field(
        default_factory=dict,
        description="Change in reputation score this week keyed by domain. e.g. {'simonwillison.net': +0.05}"
    )
    narrative:            str       = Field(
        default="",
        description=(
            "LLM-generated 3-5 sentence pattern narrative for the week. "
            "Read by topic_node, input_supervisor, and synthesis_node in subsequent runs "
            "to provide longitudinal context — which topics are saturated, which are "
            "emerging, and what the field was preoccupied with last week."
        )
    )
    run_ids_included:     list[str] = Field(default_factory=list, description="The run_ids that fed this synthesis.")
    created_at:           str       = Field(default_factory=lambda: datetime.utcnow().isoformat())
    ttl:                  int       = Field(default_factory=lambda: ttl_days(90))

# ---------------------------------------------------------------------------
# Tier 3 — Trend Records (Permanent)
# ---------------------------------------------------------------------------

class TrendRecord(BaseModel):
    """
    A named durable trend identified by the system over time.
    Strength decays if not reinforced. Archived when strength < 0.1.
 
    DynamoDB table: digest-trends
    PK: trend_id
    No TTL — managed manually via archived flag and periodic cleanup.
    """
    trend_id:            str           = Field(description="Slugified name. e.g. 'multi-agent-coordination'")
    name:                str           = Field(description="Human readable. e.g. 'Multi-Agent Coordination'")
    first_observed:      str           = Field(description="ISO date when the system first identified this trend.")
    last_reinforced:     str           = Field(description="ISO date of most recent run that confirmed this trend.")
    strength:            float         = Field(ge=0.0, le=1.0, description="Current strength score.")
    strength_band:       TrendStrength
    platform_relevance:  float         = Field(ge=0.0, le=1.0, description="Relevance to platform engineering specifically.")
    related_topics:      list[str]     = Field(default_factory=list)
    key_signals:         list[str]     = Field(description="The actual phrases and concepts driving this trend.")
    evidence_weeks:      list[str]     = Field(default_factory=list, description="week_ids that have reinforced this trend.")
    times_reinforced:    int           = Field(default=1)
    archived:            bool          = Field(default=False, description="True when strength drops below 0.1.")
    archived_at:         str           = Field(default="")
    created_at:          str           = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at:          str           = Field(default_factory=lambda: datetime.utcnow().isoformat())
 
    def updated_strength(self, was_reinforced: bool) -> float:
        """
        Calculates the new strength score after a run.
        Called by the Trend node — never by any spoke node.
 
        Decay:  -0.15 per run without reinforcement
        Boost:  +0.25 per run with reinforcement
        Bounds: clamped to [0.0, 1.0]
        """
        DECAY = 0.15
        BOOST = 0.25
        if was_reinforced:
            return min(1.0, self.strength + BOOST)
        return max(0.0, self.strength - DECAY)
 
    def to_band(self, strength: float) -> TrendStrength:
        """Maps a strength float to its human-readable band."""
        if strength >= 0.85:
            return TrendStrength.DOMINANT
        if strength >= 0.65:
            return TrendStrength.STRONG
        if strength >= 0.40:
            return TrendStrength.GROWING
        return TrendStrength.EMERGING

# ---------------------------------------------------------------------------
# Tier 3 — Source Reputation Records (Permanent)
# ---------------------------------------------------------------------------

class SourceRecord(BaseModel):
    """
    Per-domain quality score that improves or decays over time.
    Read by the Scoring node at the start of each run.
    Updated by the Trend node at the end of each run.
 
    DynamoDB table: digest-sources
    PK: domain
    No TTL — sources are few and small.
    """
    domain:                    str     = Field(description="e.g. 'simonwillison.net'")
    reputation_score:          float   = Field(ge=0.0, le=1.0, default=0.5, description="Starts neutral at 0.5.")
    total_articles_seen:       int     = Field(default=0)
    avg_relevance_of_articles: float   = Field(ge=0.0, le=1.0, default=0.5)
    last_seen:                 str     = Field(default="")
    last_contributed_date:     str     = Field(
        default="",
        description="ISO date when this source last had an article pass scoring. Used for fetch-time rotation weighting."
    )
    first_seen:                str     = Field(default_factory=lambda: datetime.utcnow().isoformat())
    created_at:                str     = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at:                str     = Field(default_factory=lambda: datetime.utcnow().isoformat())
 
    def updated_reputation(self, new_article_relevance: float) -> float:
        """
        Rolling average update. Each new article nudges the score
        toward the article's relevance rather than replacing it.
        Weight of 0.2 means recent articles matter but don't dominate.
        """
        RECENCY_WEIGHT = 0.2
        return round(
            (self.reputation_score * (1 - RECENCY_WEIGHT)) + (new_article_relevance * RECENCY_WEIGHT),
            4
        )