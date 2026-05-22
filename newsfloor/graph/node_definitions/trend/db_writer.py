"""
nodes/trend/db_writer.py

All DynamoDB write operations for the trend node.

Each function handles one category of writes and wraps failures
independently — a failed trend write does not block source updates
or the run record write.

Returns lists of succeeded IDs and error strings rather than raising,
so run() can assemble a complete TrendTaskResult even when some writes fail.
"""

from __future__ import annotations
import logging
from datetime import datetime, timezone

from contracts.primitives import RunStatus
from contracts.state import RunRecord, SourceRecord, TrendRecord, ttl_days
from data.db import DynamoDBService

logger = logging.getLogger(__name__)


def write_trend_updates(
    db:         DynamoDBService,
    trends:     list[TrendRecord],
) -> tuple[list[str], list[str], list[str]]:
    """
    Writes updated TrendRecords to DynamoDB.

    Returns:
        updated:  trend_ids successfully updated (not archived)
        archived: trend_ids successfully archived
        errors:   error strings for failed writes
    """
    updated:  list[str] = []
    archived: list[str] = []
    errors:   list[str] = []

    for trend in trends:
        try:
            db.put_trend(trend)
            if trend.archived:
                archived.append(trend.trend_id)
                logger.info({
                    "action":   "archived",
                    "trend_id": trend.trend_id,
                    "strength": trend.strength,
                })
            else:
                updated.append(trend.trend_id)
        except Exception as e:
            error_msg = f"trend write failed for {trend.trend_id}: {e}"
            errors.append(error_msg)
            logger.error({"action": "trend_write_failed", "trend_id": trend.trend_id, "error": str(e)})

    return updated, archived, errors


def write_new_trends(
    db:      DynamoDBService,
    trends:  list[TrendRecord],
) -> tuple[list[str], list[str]]:
    """
    Writes newly created TrendRecords to DynamoDB.

    Returns:
        created: trend_ids successfully written
        errors:  error strings for failed writes
    """
    created: list[str] = []
    errors:  list[str] = []

    for trend in trends:
        try:
            db.put_trend(trend)
            created.append(trend.trend_id)
            logger.info({
                "action":   "trend_created",
                "trend_id": trend.trend_id,
                "name":     trend.name,
            })
        except Exception as e:
            error_msg = f"new trend write failed for {trend.trend_id}: {e}"
            errors.append(error_msg)
            logger.error({"action": "new_trend_write_failed",
                          "trend_id": trend.trend_id, "error": str(e)})

    return created, errors


def write_source_updates(
    db:              DynamoDBService,
    scored_articles: list,
    now:             str,
) -> tuple[list[str], dict[str, float], list[str]]:
    """
    Updates SourceRecord reputation scores for every domain in scored_articles.
    Creates new SourceRecords for domains seen for the first time.

    Returns:
        updated:           domain names successfully written
        reputation_deltas: {domain: delta} for domains with change >= 0.1
        errors:            error strings for failed writes
    """
    updated:             list[str]        = []
    reputation_deltas:   dict[str, float] = {}
    errors:              list[str]        = []

    # Deduplicate articles per domain — update reputation once per domain per run
    # using the highest relevance score seen from that domain today.
    # A domain that published 3 articles today shouldn't get 3 reputation updates.
    best_per_domain: dict[str, float] = {}
    for article in scored_articles:
        domain = article.source_domain
        if article.relevance_score > best_per_domain.get(domain, 0.0):
            best_per_domain[domain] = article.relevance_score

    for domain, relevance_score in best_per_domain.items():
        try:
            source      = db.get_or_create_source(domain)
            prior_score = source.reputation_score
            new_rep     = source.updated_reputation(relevance_score)
            delta       = round(new_rep - prior_score, 4)

            updated_source = source.model_copy(update={
                "reputation_score":    new_rep,
                "total_articles_seen": source.total_articles_seen + 1,
                "last_seen":           now,
                "updated_at":          now,
            })

            db.put_source(updated_source)
            updated.append(domain)

            if abs(delta) >= 0.1:
                reputation_deltas[domain] = delta

        except Exception as e:
            error_msg = f"source update failed for {domain}: {e}"
            errors.append(error_msg)
            logger.error({"action": "source_write_failed", "domain": domain, "error": str(e)})

    return updated, reputation_deltas, errors


def write_run_record(
    db:         DynamoDBService,
    task_input,
    run_status: RunStatus,
    notes:      list[str],
    now:        str,
) -> tuple[RunStatus, list[str]]:
    """
    Writes the completed RunRecord for today's run.

    Returns the final run_status (may be downgraded to DEGRADED on failure)
    and any additional errors encountered.
    """
    errors: list[str] = []

    try:
        run_record = RunRecord(
            run_id              = task_input.run_id,
            status              = run_status,
            topic               = task_input.topic,
            focus_angle         = task_input.focus_angle,
            articles_fetched    = len(task_input.scored_articles),
            articles_passed     = len([
                a for a in task_input.scored_articles if a.passed_threshold
            ]),
            new_signals         = task_input.new_signals,
            trend_confirmations = task_input.trend_confirmations,
            digest_summary      = task_input.digest_summary,
            orchestrator_notes  = notes,
            completed_at        = now,
            ttl                 = ttl_days(30),
        )
        db.put_run_record(run_record)
        logger.info({
            "action":     "run_record_written",
            "run_id":     task_input.run_id,
            "run_status": run_status.value,
        })
    except Exception as e:
        error_msg = f"run record write failed: {e}"
        errors.append(error_msg)
        logger.error({"action": "run_record_write_failed", "error": str(e)})
        run_status = RunStatus.DEGRADED

    return run_status, errors