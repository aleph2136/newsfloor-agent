"""
nodes/trend/__init__.py
 
Public entry point for the trend node.
graph/nodes.py imports only: from nodes.trend import run
 
Internal structure
──────────────────
strength.py         Deterministic strength/decay math
db_writer.py        All DynamoDB write operations
signal_analysis.py  LLM reasoning — signal clustering and trend classification
weekly_synthesis.py LLM reasoning — weekly narrative and synthesis record
 
Execution order
───────────────
1. Deterministic: apply strength updates to all existing trends
2. Deterministic: update source reputation scores
   → Logs prior and post scores for any domain with delta >= 0.1
3. Conditional LLM: cluster and classify new signals (if any)
4. Conditional LLM: weekly synthesis (Mondays only)
5. Deterministic: write RunRecord — always last
"""
 
from __future__ import annotations
import logging
from datetime import datetime, timezone
 
from crewai.llm import LLM
 
from config import settings
from contracts.nodes import TrendTaskInput, TrendTaskResult
from contracts.primitives import RunStatus
from data.db import DynamoDBService
 
from .db_writer import (
    write_new_trends,
    write_run_record,
    write_source_updates,
    write_trend_updates,
)
from .signal_analysis import classify_new_trends, cluster_signals
from .strength import apply_strength_update
from .weekly_synthesis import write_weekly_synthesis
 
logger = logging.getLogger(__name__)
 
 
def run(task_input: TrendTaskInput) -> TrendTaskResult:
    """
    Runs all state updates for the completed run.
    Never raises — failures are captured in TrendTaskResult and logged.
    """
    logger.info({
        "node":            "trend",
        "run_id":          task_input.run_id,
        "new_signals":     len(task_input.new_signals),
        "confirmations":   len(task_input.trend_confirmations),
        "articles_scored": len(task_input.scored_articles),
        "delivery_sent":   task_input.delivery_sent,
    })
 
    db         = DynamoDBService()
    llm        = LLM(model=settings.bedrock_model_trend,        max_retries=1)
    llm_weekly = LLM(model=settings.bedrock_model_trend_weekly, max_retries=1)
    now        = datetime.now(timezone.utc).isoformat()
 
    all_errors:                 list[str] = []
    trends_updated:             list[str] = []
    trends_archived:            list[str] = []
    trends_created:             list[str] = []
    source_reputations_updated: list[str] = []
 
    # -------------------------------------------------------------------------
    # 1. Strength updates on all existing trends
    # -------------------------------------------------------------------------
    all_trends      = db.get_all_trends()
    confirmed_names = set(task_input.trend_confirmations)
 
    updated_trend_records = [
        apply_strength_update(
            trend              = t,
            was_reinforced     = t.name in confirmed_names,
            now                = now,
            archive_threshold  = settings.trend_archive_threshold,
        )
        for t in all_trends
        if not t.archived
    ]
 
    updated, archived, errors = write_trend_updates(db, updated_trend_records)
    trends_updated.extend(updated)
    trends_archived.extend(archived)
    all_errors.extend(errors)
 
    # -------------------------------------------------------------------------
    # 2. Source reputation updates
    # Logs prior → post scores for any domain with delta >= 0.1
    # so you can investigate in CloudWatch without paying for LLM analysis.
    # -------------------------------------------------------------------------
    source_updated, reputation_deltas, errors = write_source_updates(
        db              = db,
        scored_articles = task_input.scored_articles,
        now             = now,
    )
    source_reputations_updated.extend(source_updated)
    all_errors.extend(errors)
 
    # Log reputation changes that are worth knowing about
    for domain, delta in reputation_deltas.items():
        logger.info({
            "node":             "trend",
            "action":           "reputation_delta",
            "domain":           domain,
            "delta":            delta,
            "message":          (
                f"{domain} reputation {'increased' if delta > 0 else 'decreased'} "
                f"by {abs(delta):.3f} this run"
            ),
        })
 
    # -------------------------------------------------------------------------
    # 3. Signal clustering and new trend classification (LLM — if signals exist)
    # -------------------------------------------------------------------------
    if task_input.new_signals:
        try:
            clusters = cluster_signals(
                llm             = llm,
                new_signals     = task_input.new_signals,
                existing_trends = all_trends,
                topic           = task_input.topic,
            )
 
            new_clusters = [c for c in clusters if c.get("is_new", True)]
 
            if new_clusters:
                existing_ids = {t.trend_id for t in all_trends}
                new_records  = classify_new_trends(
                    llm          = llm,
                    clusters     = new_clusters,
                    topic        = task_input.topic,
                    existing_ids = existing_ids,
                    run_id       = task_input.run_id,
                    now          = now,
                )
                created, errors = write_new_trends(db, new_records)
                trends_created.extend(created)
                all_errors.extend(errors)
 
        except Exception as e:
            error_msg = f"signal analysis failed: {e}"
            all_errors.append(error_msg)
            logger.error({"node": "trend", "error": error_msg})
 
    # -------------------------------------------------------------------------
    # 4. Weekly synthesis (LLM — Mondays only)
    # -------------------------------------------------------------------------
    if _is_monday():
        try:
            write_weekly_synthesis(db=db, llm=llm_weekly, now=now)
        except Exception as e:
            error_msg = f"weekly synthesis failed: {e}"
            all_errors.append(error_msg)
            logger.error({"node": "trend", "error": error_msg})
 
    # -------------------------------------------------------------------------
    # 5. RunRecord — always last
    # -------------------------------------------------------------------------
    run_status = (
        RunStatus.DEGRADED
        if (all_errors or not task_input.delivery_sent)
        else RunStatus.COMPLETED
    )
 
    run_status, errors = write_run_record(
        db         = db,
        task_input = task_input,
        run_status = run_status,
        notes      = all_errors,
        now        = now,
    )
    all_errors.extend(errors)
 
    logger.info({
        "node":                       "trend",
        "run_id":                     task_input.run_id,
        "trends_updated":             len(trends_updated),
        "trends_created":             len(trends_created),
        "trends_archived":            len(trends_archived),
        "source_reputations_updated": len(source_reputations_updated),
        "reputation_deltas_logged":   len(reputation_deltas),
        "errors":                     len(all_errors),
        "run_status":                 run_status.value,
    })
 
    return TrendTaskResult(
        run_id                     = task_input.run_id,
        run_status                 = run_status,
        trends_updated             = trends_updated,
        trends_created             = trends_created,
        trends_archived            = trends_archived,
        source_reputations_updated = source_reputations_updated,
        error                      = "; ".join(all_errors) if all_errors else "",
    )
 
 
def _is_monday() -> bool:
    return datetime.now(timezone.utc).weekday() == 0
 