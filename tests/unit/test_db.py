# tests/unit/test_db.py
#
# Unit tests for DynamoDBService. All boto3 calls are patched — no real AWS
# connections are made. Covers mark_run_failed (new) and try_claim_run
# (the method that writes IN_PROGRESS and whose interaction with mark_run_failed
# is the subject of the status-leak bug fix).

from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from data.db import DynamoDBService
from contracts.primitives import RunStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _client_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": f"{code} test error"}}, "Operation")


def _patched_db() -> tuple[MagicMock, MagicMock]:
    """Returns (mock_boto3, mock_runs_table). Apply as @patch('data.db.boto3')."""
    mock_table = MagicMock()
    mock_resource = MagicMock()
    mock_resource.Table.return_value = mock_table
    mock_boto3 = MagicMock()
    mock_boto3.resource.return_value = mock_resource
    return mock_boto3, mock_table


# ---------------------------------------------------------------------------
# mark_run_failed
# ---------------------------------------------------------------------------

class TestMarkRunFailed:
    @patch("data.db.boto3")
    def test_calls_update_item_with_failed_status(self, mock_boto3):
        mock_boto3, mock_table = _patched_db()
        with patch("data.db.boto3", mock_boto3):
            DynamoDBService().mark_run_failed("2026-06-14")

        mock_table.update_item.assert_called_once()
        kwargs = mock_table.update_item.call_args[1]
        assert kwargs["Key"] == {"run_id": "2026-06-14"}
        assert kwargs["ExpressionAttributeValues"][":s"] == RunStatus.FAILED.value

    @patch("data.db.boto3")
    def test_sets_completed_at_timestamp(self, mock_boto3):
        mock_boto3, mock_table = _patched_db()
        with patch("data.db.boto3", mock_boto3):
            DynamoDBService().mark_run_failed("2026-06-14")

        kwargs = mock_table.update_item.call_args[1]
        completed_at = kwargs["ExpressionAttributeValues"].get(":t", "")
        assert completed_at != "", "completed_at must be set on failure"

    @patch("data.db.boto3")
    def test_does_not_overwrite_other_fields(self, mock_boto3):
        """update_item (not put_item) must be used to preserve partial run data."""
        mock_boto3, mock_table = _patched_db()
        with patch("data.db.boto3", mock_boto3):
            DynamoDBService().mark_run_failed("2026-06-14")

        mock_table.update_item.assert_called_once()
        mock_table.put_item.assert_not_called()

    @patch("data.db.boto3")
    def test_swallows_client_error_without_raising(self, mock_boto3):
        mock_boto3, mock_table = _patched_db()
        mock_table.update_item.side_effect = _client_error("ProvisionedThroughputExceededException")
        with patch("data.db.boto3", mock_boto3):
            # Should not raise — caller (handler) must still return its 500
            DynamoDBService().mark_run_failed("2026-06-14")


# ---------------------------------------------------------------------------
# try_claim_run
# ---------------------------------------------------------------------------

class TestTryClaimRun:
    @patch("data.db.boto3")
    def test_returns_true_and_writes_in_progress_on_first_claim(self, mock_boto3):
        mock_boto3, mock_table = _patched_db()
        with patch("data.db.boto3", mock_boto3):
            result = DynamoDBService().try_claim_run("2026-06-14")

        assert result is True
        mock_table.put_item.assert_called_once()
        item = mock_table.put_item.call_args[1]["Item"]
        assert item["status"] == RunStatus.IN_PROGRESS.value

    @patch("data.db.boto3")
    def test_uses_conditional_write_to_prevent_duplicates(self, mock_boto3):
        mock_boto3, mock_table = _patched_db()
        with patch("data.db.boto3", mock_boto3):
            DynamoDBService().try_claim_run("2026-06-14")

        kwargs = mock_table.put_item.call_args[1]
        assert "ConditionExpression" in kwargs

    @patch("data.db.boto3")
    def test_returns_false_when_record_already_exists(self, mock_boto3):
        mock_boto3, mock_table = _patched_db()
        mock_table.put_item.side_effect = _client_error("ConditionalCheckFailedException")
        with patch("data.db.boto3", mock_boto3):
            result = DynamoDBService().try_claim_run("2026-06-14")

        assert result is False

    @patch("data.db.boto3")
    def test_returns_true_on_unexpected_dynamo_error(self, mock_boto3):
        """Transient infra errors should not silently block the run."""
        mock_boto3, mock_table = _patched_db()
        mock_table.put_item.side_effect = _client_error("ProvisionedThroughputExceededException")
        with patch("data.db.boto3", mock_boto3):
            result = DynamoDBService().try_claim_run("2026-06-14")

        assert result is True


# ---------------------------------------------------------------------------
# get_recent_runs — status filter regression tests
#
# Root cause of topic repetition: the filter used "complete" (no 'd') but
# RunStatus.COMPLETED.value == "completed". Every successful run was silently
# excluded from recent_topics, making the exclusion list effectively empty.
# These tests pin the exact status values included/excluded so a future typo
# or enum rename is caught immediately.
# ---------------------------------------------------------------------------

def _scan_filter_values(mock_table: MagicMock) -> list[str]:
    """
    Extracts the list of status values from the DynamoDB scan FilterExpression.

    The expression is structured as an And(In(..., [values]), Gte(...)):
      filter_expr._values == (In_condition, Gte_condition)
      In_condition._values == (Attr("status"), [list_of_statuses])
    """
    kwargs = mock_table.scan.call_args[1]
    filter_expr = kwargs["FilterExpression"]
    # _values[0] is the In(...) condition; _values[1] is the list of status strings
    status_condition = filter_expr._values[0]
    return list(status_condition._values[1])


class TestGetRecentRuns:

    @patch("data.db.boto3")
    def test_filter_includes_completed_status(self, mock_boto3):
        mock_boto3, mock_table = _patched_db()
        mock_table.scan.return_value = {"Items": []}
        with patch("data.db.boto3", mock_boto3):
            DynamoDBService().get_recent_runs(days=30)
        values = _scan_filter_values(mock_table)
        assert RunStatus.COMPLETED.value in values, (
            f"'completed' must be in the filter — got {values}"
        )

    @patch("data.db.boto3")
    def test_filter_includes_completed_with_warnings_status(self, mock_boto3):
        mock_boto3, mock_table = _patched_db()
        mock_table.scan.return_value = {"Items": []}
        with patch("data.db.boto3", mock_boto3):
            DynamoDBService().get_recent_runs(days=30)
        values = _scan_filter_values(mock_table)
        assert RunStatus.COMPLETED_WITH_WARNINGS.value in values, (
            f"'completed_with_warnings' must be in the filter — got {values}"
        )

    @patch("data.db.boto3")
    def test_filter_includes_degraded_status(self, mock_boto3):
        mock_boto3, mock_table = _patched_db()
        mock_table.scan.return_value = {"Items": []}
        with patch("data.db.boto3", mock_boto3):
            DynamoDBService().get_recent_runs(days=30)
        values = _scan_filter_values(mock_table)
        assert RunStatus.DEGRADED.value in values, (
            f"'degraded' must be in the filter — got {values}"
        )

    @patch("data.db.boto3")
    def test_filter_excludes_in_progress(self, mock_boto3):
        mock_boto3, mock_table = _patched_db()
        mock_table.scan.return_value = {"Items": []}
        with patch("data.db.boto3", mock_boto3):
            DynamoDBService().get_recent_runs(days=30)
        values = _scan_filter_values(mock_table)
        assert RunStatus.IN_PROGRESS.value not in values, (
            f"'in_progress' must NOT be in the filter — got {values}"
        )

    @patch("data.db.boto3")
    def test_filter_excludes_failed(self, mock_boto3):
        mock_boto3, mock_table = _patched_db()
        mock_table.scan.return_value = {"Items": []}
        with patch("data.db.boto3", mock_boto3):
            DynamoDBService().get_recent_runs(days=30)
        values = _scan_filter_values(mock_table)
        assert RunStatus.FAILED.value not in values, (
            f"'failed' must NOT be in the filter — got {values}"
        )

    @patch("data.db.boto3")
    def test_returns_records_with_completed_status(self, mock_boto3):
        """
        A run stored with status='completed' must be returned by get_recent_runs.
        This is the direct regression test for the 'complete' vs 'completed' typo.
        """
        mock_boto3, mock_table = _patched_db()
        from datetime import datetime, timezone, timedelta
        completed_item = {
            "run_id": "2026-06-23",
            "status": RunStatus.COMPLETED.value,
            "topic": "structured outputs and contract-driven agents",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "ttl": 9999999999,
        }
        mock_table.scan.return_value = {"Items": [completed_item]}
        with patch("data.db.boto3", mock_boto3):
            results = DynamoDBService().get_recent_runs(days=30)
        assert len(results) == 1
        assert results[0].topic == "structured outputs and contract-driven agents"
