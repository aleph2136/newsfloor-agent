# tests/unit/test_delivery.py
#
# Unit tests for the delivery node. All network I/O is patched — no SMTP
# connections are made. Tests cover subject extraction, HTML-to-text
# conversion, the happy-path send, and the failure-return contract.

import smtplib
from unittest.mock import MagicMock, patch

import pytest

from node_definitions.delivery import (
    _extract_subject,
    _html_to_text,
    run,
)
from contracts.nodes import DeliveryTaskInput, DeliveryTaskResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _task_input(**overrides) -> DeliveryTaskInput:
    base = {
        "run_id":          "test-run-001",
        "digest_html":     "<h1>Agent Routing Patterns</h1><p>Summary here.</p>",
        "topic":           "agent routing patterns",
        "recipient_email": "recipient@gmail.com",
        "sender_email":    "sender@gmail.com",
    }
    base.update(overrides)
    return DeliveryTaskInput(**base)


# ---------------------------------------------------------------------------
# _extract_subject
# ---------------------------------------------------------------------------

class TestExtractSubject:

    def test_extracts_h1_as_subject(self):
        html = "<h1>Agent Routing Patterns</h1><p>Body.</p>"
        assert _extract_subject(html, "agent routing") == "Digest: Agent Routing Patterns"

    def test_strips_nested_tags_from_h1(self):
        html = "<h1><em>Bold</em> Title</h1>"
        assert _extract_subject(html, "topic") == "Digest: Bold Title"

    def test_fallback_when_no_h1(self):
        html = "<p>No heading here.</p>"
        result = _extract_subject(html, "multi-agent orchestration")
        assert result == "AI Agentic Engineering Digest — Multi-Agent Orchestration"

    def test_fallback_when_h1_is_empty(self):
        html = "<h1>   </h1><p>Content.</p>"
        result = _extract_subject(html, "llm routing")
        assert result == "AI Agentic Engineering Digest — Llm Routing"

    def test_case_insensitive_h1_match(self):
        html = "<H1>Case Insensitive</H1>"
        assert _extract_subject(html, "topic") == "Digest: Case Insensitive"


# ---------------------------------------------------------------------------
# _html_to_text
# ---------------------------------------------------------------------------

class TestHtmlToText:

    def test_strips_paragraph_tags(self):
        result = _html_to_text("<p>Hello world</p>")
        assert "Hello world" in result
        assert "<p>" not in result

    def test_converts_headings_to_newlines(self):
        result = _html_to_text("<h2>Section</h2><p>Body.</p>")
        assert "Section" in result
        assert "<h2>" not in result

    def test_converts_list_items(self):
        result = _html_to_text("<ul><li>Item one</li><li>Item two</li></ul>")
        assert "- Item one" in result
        assert "- Item two" in result

    def test_collapses_excessive_newlines(self):
        result = _html_to_text("<h1>A</h1>\n\n\n\n<p>B</p>")
        assert "\n\n\n" not in result

    def test_strips_all_remaining_tags(self):
        result = _html_to_text("<div><span>text</span></div>")
        assert "<" not in result
        assert "text" in result


# ---------------------------------------------------------------------------
# run() — happy path
# ---------------------------------------------------------------------------

class TestRunHappyPath:

    @patch("node_definitions.delivery.smtplib.SMTP_SSL")
    def test_returns_sent_true_on_success(self, mock_smtp_class):
        mock_server = MagicMock()
        mock_smtp_class.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp_class.return_value.__exit__  = MagicMock(return_value=False)

        result = run(_task_input())

        assert isinstance(result, DeliveryTaskResult)
        assert result.sent is True
        assert result.error == ""
        assert result.message_id != ""

    @patch("node_definitions.delivery.smtplib.SMTP_SSL")
    def test_message_id_is_a_uuid_string(self, mock_smtp_class):
        mock_server = MagicMock()
        mock_smtp_class.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp_class.return_value.__exit__  = MagicMock(return_value=False)

        result = run(_task_input())

        import uuid
        uuid.UUID(result.message_id)  # raises if not a valid UUID

    @patch("node_definitions.delivery.smtplib.SMTP_SSL")
    def test_run_id_echoed_in_result(self, mock_smtp_class):
        mock_server = MagicMock()
        mock_smtp_class.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp_class.return_value.__exit__  = MagicMock(return_value=False)

        result = run(_task_input(run_id="my-specific-run"))

        assert result.run_id == "my-specific-run"

    @patch("node_definitions.delivery.smtplib.SMTP_SSL")
    def test_login_called_with_sender_credentials(self, mock_smtp_class):
        mock_server = MagicMock()
        mock_smtp_class.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp_class.return_value.__exit__  = MagicMock(return_value=False)

        with patch("node_definitions.delivery.settings") as mock_settings:
            mock_settings.smtp_app_token = "app-password-123"
            run(_task_input(sender_email="sender@gmail.com"))

        mock_server.login.assert_called_once_with("sender@gmail.com", "app-password-123")

    @patch("node_definitions.delivery.smtplib.SMTP_SSL")
    def test_sendmail_called_with_correct_addresses(self, mock_smtp_class):
        mock_server = MagicMock()
        mock_smtp_class.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp_class.return_value.__exit__  = MagicMock(return_value=False)

        run(_task_input(sender_email="from@gmail.com", recipient_email="to@example.com"))

        call_args = mock_server.sendmail.call_args
        assert call_args[0][0] == "from@gmail.com"
        assert call_args[0][1] == "to@example.com"


# ---------------------------------------------------------------------------
# run() — failure path
# ---------------------------------------------------------------------------

class TestRunFailurePath:

    @patch("node_definitions.delivery.smtplib.SMTP_SSL")
    def test_returns_sent_false_on_smtp_error(self, mock_smtp_class):
        mock_smtp_class.side_effect = smtplib.SMTPConnectError(421, "Service unavailable")

        result = run(_task_input())

        assert result.sent is False
        assert result.message_id == ""
        assert "421" in result.error or "Service" in result.error

    @patch("node_definitions.delivery.smtplib.SMTP_SSL")
    def test_error_message_captured_in_result(self, mock_smtp_class):
        mock_smtp_class.side_effect = smtplib.SMTPAuthenticationError(535, b"Bad credentials")

        result = run(_task_input())

        assert result.sent is False
        assert result.error != ""

    @patch("node_definitions.delivery.smtplib.SMTP_SSL")
    def test_run_never_raises(self, mock_smtp_class):
        mock_smtp_class.side_effect = RuntimeError("unexpected error")

        # Should not raise — run() catches all exceptions
        result = run(_task_input())
        assert result.sent is False
