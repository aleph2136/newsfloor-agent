"""
nodes/load_context.py
 
Loads all historical state from DynamoDB and assembles OrchestratorContext.
 
This is the only node that reads from DynamoDB at run start.
No LLM call — pure I/O. Everything it produces is read-only for
the rest of the run.
 
What it builds
──────────────
OrchestratorContext contains:
  - active_trends        TrendSnapshots for nodes that reason about trends
  - source_reputation_map  Domain → score dict for the Scoring node
  - recent_topics        Last 30 days of topics — Topic node avoids repeats
  - recent_run_signals   Raw signals from last 7 runs — Synthesis context
  - recent_weekly_signals  Distilled signals from last 2 weekly records
  - engineer_profile     Static config — personalizes Synthesis output
"""
from __future__ import annotations
import logging
from datetime import datetime, timedelta

from config import settings
from config_loader import load_profile
from contracts.nodes import (
    OrchestratorContext,
    TrendSnapshot,
)
from contracts.primitives import TrendStrength
from .db import DynamoDBService
 
logger = logging.getLogger(__name__)
 

def run() -> OrchestratorContext:
    """
    Reads DynamoDB and assembles the OrchestratorContext for this run.
 
    Returns a fully populated OrchestratorContext.
    On any read failure the relevant field defaults to empty — the run
    continues with reduced context rather than failing entirely.
    """
    logger.info("load_context: reading DynamoDB state")

    db = DynamoDBService()

    # --- Active trends ---
    trend_records = db.get_active_trends(min_strength=0.3)
    active_trends = [_to_snapshot(t) for t in trend_records]
    logger.info(f"load_context: {len(active_trends)} active trends loaded")

    # --- Source reputation map and rotation tracking ---
    source_records = db.get_all_sources()
    source_reputation_map    = {s.domain: s.reputation_score for s in source_records}
    source_last_contributed  = {s.domain: s.last_contributed_date for s in source_records if s.last_contributed_date}
    logger.info(f"load_context: {len(source_reputation_map)} source reputation scores loaded")

    # --- Recent topics (last 30 days) ---
    recent_runs = db.get_recent_runs(days=30)
    recent_topics = [r.topic for r in recent_runs if r.topic]

    # --- Seen article IDs (last 14 days) for cross-run deduplication ---
    cutoff = datetime.utcnow() - timedelta(days=14)
    seen_article_ids: list[str] = []
    for run in recent_runs:
        if run.created_at and run.created_at >= cutoff.isoformat():
            seen_article_ids.extend(run.article_ids_used)
    logger.info(f"load_context: {len(seen_article_ids)} article IDs loaded for dedup window")

    # --- Recent run signals (last 7 runs) ---
    recent_run_signals = []
    for run in recent_runs[:7]:
        recent_run_signals.extend(run.new_signals)

    # --- Recent weekly signals and narrative (last 2 weekly synthesis records) ---
    weekly_records = db.get_recent_weekly_syntheses(count=2)
    recent_weekly_signals = []
    for week in weekly_records:
        recent_weekly_signals.extend(week.recurring_signals)
        recent_weekly_signals.extend(week.emerging_concepts)

    # Narrative from the most recent weekly record only — it's a single coherent
    # paragraph written for a specific week and is not meaningfully combinable.
    # Falls back to empty string if no weekly record exists yet (early runs).
    recent_weekly_narrative = weekly_records[0].narrative if weekly_records else ""

    logger.info("load_context: context assembled successfully")

    return OrchestratorContext(
        active_trends            = active_trends,
        source_reputation_map    = source_reputation_map,
        source_last_contributed  = source_last_contributed,
        recent_topics            = recent_topics,
        seen_article_ids         = seen_article_ids,
        recent_run_signals       = recent_run_signals,
        recent_weekly_signals    = recent_weekly_signals,
        recent_weekly_narrative  = recent_weekly_narrative,
        engineer_profile         = load_profile(),
    )

def _to_snapshot(record) -> TrendSnapshot:
    """Converts a TrendRecord to a TrendSnapshot for passing into node contracts."""
    return TrendSnapshot(
        trend_id           = record.trend_id,
        name               = record.name,
        strength           = record.strength,
        strength_band      = record.strength_band,
        platform_relevance = record.platform_relevance,
        key_signals        = record.key_signals,
        last_reinforced    = record.last_reinforced,
    )