"""
nodes/db.py
 
DynamoDB service layer — the only file in the project that talks to DynamoDB.
 
All reads happen via load_context at run start.
All writes happen via trend_node at run end.
No other node touches this file.
 
Each method maps to exactly one table and one operation type.
Nothing here knows about CrewAI, LangGraph, or business logic.
"""

from __future__ import annotations
import logging
from datetime import datetime, timedelta, timezone
from typing import Any
 
import boto3
from boto3.dynamodb.conditions import Attr
from botocore.exceptions import ClientError
 
from config import settings
from contracts.state import (
    RunRecord,
    WeeklySynthesis,
    TrendRecord,
    SourceRecord
)
 
logger = logging.getLogger(__name__)

class DynamoDBService:
    """
    Thin wrapper around boto3 DynamoDB resource.
 
    Instantiated once by load_context and trend_node.
    All methods return typed Pydantic models — callers never
    touch raw DynamoDB dicts.
    """

    def __init__(self) -> None:
        resource = boto3.resource("dynamodb", region_name = settings.aws_region)
        self._runs      = resource.Table(settings.dynamodb_runs_table)
        self._weekly    = resource.Table(settings.dynamodb_weekly_table)
        self._trends    = resource.Table(settings.dynamodb_trends_table)
        self._sources   = resource.Table(settings.dynamodb_sources_table)

    def get_recent_runs(self, days: int = 30) -> list[RunRecord]:
        """
        Returns all run records from the last N days, most recent first.
        Used by load_context to build recent_topics and recent_run_signals.

        Scan is acceptable here — the table never exceeds ~30 records
        due to the 30-day TTL.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
        try:
            response = self._runs.scan(
                FilterExpression=(
                    Attr("status").is_in(["complete", "degraded"]) &
                    Attr("run_id").gte(cutoff)
                )
            )
            records = [RunRecord(**item) for item in response.get("Items", [])]
            return sorted(records, key=lambda r: r.run_id, reverse=True)
        except ClientError as e:
            logger.error({"method": "get_recent_runs", "error": str(e)})
            return []

    def try_claim_run(self, run_id: str) -> bool:
        """
        Attempts to claim this run_id with a conditional DynamoDB write.
        Returns True if the claim succeeded — this invocation should proceed.
        Returns False if a record already exists — another invocation claimed it
        first (e.g. an EventBridge Lambda retry after a timeout).

        On unexpected DynamoDB error, returns True so the run proceeds rather
        than being silently blocked by a transient infra issue.
        """
        from contracts.primitives import RunStatus
        record = RunRecord(run_id=run_id, status=RunStatus.IN_PROGRESS)
        try:
            self._runs.put_item(
                Item=record.model_dump(),
                ConditionExpression="attribute_not_exists(run_id)",
            )
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                logger.warning({
                    "method":  "try_claim_run",
                    "run_id":  run_id,
                    "message": "Run record already exists — skipping duplicate invocation",
                })
                return False
            logger.error({"method": "try_claim_run", "run_id": run_id, "error": str(e)})
            return True  # allow run on unexpected error rather than silently blocking

    def put_run_record(self, record: RunRecord) -> None:
        """writes or overwrites a run record."""
        try:
            self._runs.put_item(Item=record.model_dump())
        except ClientError as e:
            logger.error({"method": "put_run_record", "run_id": record.run_id, "error": str(e)})
            raise

    # -------------------------------------------------------------------------
    # Weekly Synthesis
    # -------------------------------------------------------------------------
    def get_recent_weekly_syntheses(self, count: int = 2) -> list[WeeklySynthesis]:
        """
        Returns the most recent N weekly synthesis records.
        Used by load_context to build recent_weekly_signals.
        """
        try:
            response = self._weekly.scan()
            records = [WeeklySynthesis(**item) for item in response.get("Items", [])]
            return sorted(records, key=lambda r: r.week_id, reverse=True)[:count]
        except ClientError as e:
            logger.error({"method": "get_recent_weekly_syntheses", "error": str(e)})
            return []
    
    def put_weekly_synthesis(self, record: WeeklySynthesis) -> None:
        try:
            self._weekly.put_item(Item=record.model_dump())
        except ClientError as e:
            logger.error({"method": "put_weekly_synthesis", "week_id": record.week_id, "error": str(e)})
            raise

    # -------------------------------------------------------------------------
    # Trend Records
    # -------------------------------------------------------------------------
    def get_active_trends(self, min_strength: float = 0.3) -> list[TrendRecord]:
        """
        Returns all non-archived trends above the strength threshold.
        Used by load_context — passed as TrendSnapshots into nodes that need them.
        """
        try:
            response = self._trends.scan(
                FilterExpression=Attr("archived").eq(False) & Attr("strength").gte(min_strength)
            )
            return [TrendRecord(**item) for item in response.get("Items", [])]
        except ClientError as e:
            logger.error({"method": "get_active_trends", "error": str(e)})
            return []
        
    def get_all_trends(self) -> list[TrendRecord]:
        """
        Returns all trend records including archived ones.
        Used by trend_node to apply decay across the full set.
        """
        try:
            response = self._trends.scan()
            return [TrendRecord(**item) for item in response.get("Items", [])]
        except ClientError as e:
            logger.error({"method": "get_all_trends", "error": str(e)})
            return []

    def put_trend(self, record: TrendRecord) -> None:
        try:
            self._trends.put_item(Item=record.model_dump())
        except ClientError as e:
            logger.error({"method": "put_trend", "trend_id": record.trend_id, "error": str(e)})
            raise
    
    # -------------------------------------------------------------------------
    # Source Records
    # -------------------------------------------------------------------------
    def get_all_sources(self) -> list[SourceRecord]:
        try:
            response = self._sources.scan()
            return [SourceRecord(**item) for item in response.get("Items", [])]
        except ClientError as e:
            logger.error({"method": "get_all_sources", "error": str(e)})
            return []
    
    def put_source(self, record: SourceRecord) -> None:
        try:
            self._sources.put_item(Item=record.model_dump())
        except ClientError as e:
            logger.error({"method": "put_source", "domain": record.domain, "error": str(e)})
            raise
    
    def get_or_create_source(self, domain: str) -> SourceRecord:
        """
        Returns an existing source record or creates a new one with
        a neutral starting reputation of 0.5.
        Used by trend_node when a new domain appears for the first time.
        """
        try:
            response = self._sources.get_item(Key={"domain": domain})
            item = response.get("Item")
            if item:
                return SourceRecord(**item)
        except ClientError as e:
            logger.error({"method": "get_or_create_source", "domain": domain, "error": str(e)})
        
        return SourceRecord(
            domain      = domain,
            first_seen  = datetime.now(timezone.utc).isoformat(),
        )