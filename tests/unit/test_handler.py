# tests/unit/test_handler.py
#
# Unit tests for the Lambda entry point. All DynamoDB and graph calls are
# patched — no real AWS connections or LangGraph execution occurs.
#
# Key behaviors under test:
#   - Pipeline crash writes FAILED status to DynamoDB before returning 500
#   - A DB error inside mark_run_failed does not mask the original pipeline error
#   - force=True overrides an IN_PROGRESS record (stale from a prior crash)
#   - Idempotency guard skips already-claimed runs without invoking the graph
#   - Insufficient Lambda time remaining exits early with 202

import json
from unittest.mock import MagicMock, patch

import pytest

from handler import lambda_handler
from contracts.primitives import RunStatus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ctx():
    mock = MagicMock()
    mock.aws_request_id = "req-test-abc"
    mock.get_remaining_time_in_millis.return_value = 300_000  # 5 min — plenty of time
    return mock


@pytest.fixture
def ctx_low_time():
    mock = MagicMock()
    mock.aws_request_id = "req-test-xyz"
    mock.get_remaining_time_in_millis.return_value = 30_000  # under 60s threshold
    return mock


def _db(*, claim=True, existing=None):
    """Build a mock DynamoDBService instance with sensible defaults."""
    db = MagicMock()
    db.try_claim_run.return_value = claim
    db.get_run_record.return_value = existing
    return db


# ---------------------------------------------------------------------------
# Pipeline crash → FAILED status
# ---------------------------------------------------------------------------

class TestPipelineCrash:
    @patch("data.db.DynamoDBService")
    @patch("handler.digest_graph")
    def test_writes_failed_status_on_unhandled_exception(self, mock_graph, mock_db_cls, ctx):
        db = _db()
        mock_db_cls.return_value = db
        mock_graph.invoke.side_effect = RuntimeError("crew blew up")

        result = lambda_handler({}, ctx)

        assert result["statusCode"] == 500
        db.mark_run_failed.assert_called_once()

    @patch("data.db.DynamoDBService")
    @patch("handler.digest_graph")
    def test_500_body_contains_original_error_message(self, mock_graph, mock_db_cls, ctx):
        db = _db()
        mock_db_cls.return_value = db
        mock_graph.invoke.side_effect = RuntimeError("toolConfig not supported")

        result = lambda_handler({}, ctx)

        body = json.loads(result["body"])
        assert "toolConfig not supported" in body["error"]

    @patch("data.db.DynamoDBService")
    @patch("handler.digest_graph")
    def test_returns_original_error_even_when_mark_failed_raises(self, mock_graph, mock_db_cls, ctx):
        """A secondary DB failure must not mask the pipeline error in the response."""
        db = _db()
        mock_db_cls.return_value = db
        mock_graph.invoke.side_effect = RuntimeError("original pipeline error")
        db.mark_run_failed.side_effect = Exception("DynamoDB unavailable")

        result = lambda_handler({}, ctx)

        assert result["statusCode"] == 500
        body = json.loads(result["body"])
        assert "original pipeline error" in body["error"]

    @patch("data.db.DynamoDBService")
    @patch("handler.digest_graph")
    def test_does_not_call_mark_run_failed_on_success(self, mock_graph, mock_db_cls, ctx):
        db = _db()
        mock_db_cls.return_value = db
        mock_graph.invoke.return_value = {"run_status": RunStatus.COMPLETED}

        lambda_handler({}, ctx)

        db.mark_run_failed.assert_not_called()


# ---------------------------------------------------------------------------
# force flag
# ---------------------------------------------------------------------------

class TestForceFlag:
    @patch("data.db.DynamoDBService")
    @patch("handler.digest_graph")
    def test_force_overrides_stale_in_progress_and_proceeds(self, mock_graph, mock_db_cls, ctx):
        """The core regression: force=True must not return 409 on a stale IN_PROGRESS record."""
        db = _db(existing=MagicMock(status=RunStatus.IN_PROGRESS))
        mock_db_cls.return_value = db
        mock_graph.invoke.return_value = {"run_status": RunStatus.COMPLETED}

        result = lambda_handler({"force": True}, ctx)

        assert result["statusCode"] == 200
        mock_graph.invoke.assert_called_once()

    @patch("data.db.DynamoDBService")
    @patch("handler.digest_graph")
    def test_force_deletes_prior_record_before_rerun(self, mock_graph, mock_db_cls, ctx):
        db = _db(existing=MagicMock(status=RunStatus.IN_PROGRESS))
        mock_db_cls.return_value = db
        mock_graph.invoke.return_value = {"run_status": RunStatus.COMPLETED}

        lambda_handler({"force": True}, ctx)

        db.delete_run_record.assert_called_once()

    @patch("data.db.DynamoDBService")
    @patch("handler.digest_graph")
    def test_force_on_completed_run_deletes_and_reruns(self, mock_graph, mock_db_cls, ctx):
        db = _db(existing=MagicMock(status=RunStatus.COMPLETED))
        mock_db_cls.return_value = db
        mock_graph.invoke.return_value = {"run_status": RunStatus.COMPLETED}

        result = lambda_handler({"force": True}, ctx)

        assert result["statusCode"] == 200
        db.delete_run_record.assert_called_once()

    @patch("data.db.DynamoDBService")
    @patch("handler.digest_graph")
    def test_no_force_does_not_delete_existing_record(self, mock_graph, mock_db_cls, ctx):
        db = _db()
        mock_db_cls.return_value = db
        mock_graph.invoke.return_value = {"run_status": RunStatus.COMPLETED}

        lambda_handler({}, ctx)

        db.delete_run_record.assert_not_called()


# ---------------------------------------------------------------------------
# Idempotency guard
# ---------------------------------------------------------------------------

class TestIdempotency:
    @patch("data.db.DynamoDBService")
    @patch("handler.digest_graph")
    def test_skips_run_when_already_claimed(self, mock_graph, mock_db_cls, ctx):
        db = _db(claim=False)
        mock_db_cls.return_value = db

        result = lambda_handler({}, ctx)

        assert result["statusCode"] == 200
        body = json.loads(result["body"])
        assert "already claimed" in body["status"]
        mock_graph.invoke.assert_not_called()

    @patch("data.db.DynamoDBService")
    @patch("handler.digest_graph")
    def test_proceeds_when_claim_succeeds(self, mock_graph, mock_db_cls, ctx):
        db = _db(claim=True)
        mock_db_cls.return_value = db
        mock_graph.invoke.return_value = {"run_status": RunStatus.COMPLETED}

        result = lambda_handler({}, ctx)

        assert result["statusCode"] == 200
        mock_graph.invoke.assert_called_once()


# ---------------------------------------------------------------------------
# Time guard
# ---------------------------------------------------------------------------

class TestTimeGuard:
    def test_returns_202_when_insufficient_time_remaining(self, ctx_low_time):
        result = lambda_handler({}, ctx_low_time)

        assert result["statusCode"] == 202
        body = json.loads(result["body"])
        assert "insufficient" in body["status"]
