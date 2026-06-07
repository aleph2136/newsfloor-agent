# tests/unit/test_delivery.py
#
# Unit tests for the delivery node. All network I/O is patched — no SMTP
# connections are made. Tests cover subject extraction, HTML-to-text
# conversion, the JSON plain-text format, the happy-path send, and the
# failure-return contract.

import smtplib
from unittest.mock import MagicMock, patch

import pytest

from node_definitions.delivery import (
    _extract_subject,
    _html_to_text,
    _json_to_plain_text,
    _split_bold_bullet,
    run,
)
from contracts.nodes import (
    DeliveryTaskInput,
    DeliveryTaskResult,
    DigestContentBlock,
    DigestMetadata,
    DigestStructured,
    VisualAssets,
)


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


def _minimal_digest_json() -> DigestStructured:
    return DigestStructured(
        article_id="2026-06-06-test",
        metadata=DigestMetadata(
            title="Supervisor Patterns at Scale",
            date="2026-06-06",
            summary_hook="Why long-running agents lose coherence.",
            overall_trend_context="The industry is moving toward strict state boundaries.",
        ),
        content_blocks=[
            DigestContentBlock(
                section_id="block_1",
                section_title="Soft State Nudges Fail",
                tier_1_hook="Soft system prompts cannot hold constraints over 10+ steps.",
                tier_2_bullets=[
                    "**State drift accumulates** silently before exceptions surface.",
                    "**Validation loops** must wrap LLM steps for schema fidelity.",
                ],
                tier_3_deep_dive="Technical deep dive here.",
                visual_assets=VisualAssets(),
            )
        ],
    )


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

    def test_uses_digest_json_title_when_available(self):
        dj = _minimal_digest_json()
        result = _extract_subject("<h1>Old HTML Title</h1>", "topic", digest_json=dj)
        assert result == "Digest: Supervisor Patterns at Scale"

    def test_json_title_takes_precedence_over_h1(self):
        dj = _minimal_digest_json()
        result = _extract_subject("<h1>HTML Title</h1>", "topic", digest_json=dj)
        assert "Supervisor Patterns at Scale" in result
        assert "HTML Title" not in result


# ---------------------------------------------------------------------------
# _split_bold_bullet
# ---------------------------------------------------------------------------

class TestSplitBoldBullet:

    def test_splits_standard_bold_anchor(self):
        anchor, body = _split_bold_bullet("**State drift accumulates** silently inside workflows.")
        assert anchor == "State drift accumulates"
        assert body == "silently inside workflows."

    def test_returns_empty_when_no_bold(self):
        anchor, body = _split_bold_bullet("No bold here just text.")
        assert anchor == ""
        assert body == ""

    def test_strips_whitespace_from_anchor(self):
        anchor, body = _split_bold_bullet("**  Padded anchor  ** rest of text.")
        assert anchor == "Padded anchor"

    def test_handles_multi_word_anchor(self):
        anchor, body = _split_bold_bullet("**Deterministic validation loops** must wrap LLM steps.")
        assert anchor == "Deterministic validation loops"
        assert body == "must wrap LLM steps."


# ---------------------------------------------------------------------------
# _json_to_plain_text
# ---------------------------------------------------------------------------

class TestJsonToPlainText:

    def test_includes_title_in_header(self):
        dj = _minimal_digest_json()
        result = _json_to_plain_text(dj)
        assert "Supervisor Patterns at Scale" in result

    def test_includes_overall_trend_context(self):
        dj = _minimal_digest_json()
        result = _json_to_plain_text(dj)
        assert "strict state boundaries" in result

    def test_formats_block_with_hook(self):
        dj = _minimal_digest_json()
        result = _json_to_plain_text(dj)
        assert "Soft State Nudges Fail" in result
        assert "Soft system prompts cannot hold constraints" in result

    def test_formats_bullets_as_anchor_arrow_body(self):
        dj = _minimal_digest_json()
        result = _json_to_plain_text(dj)
        assert "[State drift accumulates] ->" in result
        assert "[Validation loops] ->" in result

    def test_includes_article_url_when_provided(self):
        dj = _minimal_digest_json()
        result = _json_to_plain_text(dj, "https://sam-griffith.dev/articles/2026-06-06.html")
        assert "https://sam-griffith.dev/articles/2026-06-06.html" in result

    def test_no_url_line_when_url_empty(self):
        dj = _minimal_digest_json()
        result = _json_to_plain_text(dj, "")
        assert "https://" not in result

    def test_no_html_tags_in_output(self):
        dj = _minimal_digest_json()
        result = _json_to_plain_text(dj)
        # The "->" arrow is intentional formatting, but no HTML tags should appear
        import re
        assert not re.search(r"<[a-zA-Z/][^>]*>", result)

    def test_no_markdown_bold_markers_in_output(self):
        dj = _minimal_digest_json()
        result = _json_to_plain_text(dj)
        assert "**" not in result


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

        mock_smtp_class.return_value.login.assert_called_once_with("sender@gmail.com", "app-password-123")

    @patch("node_definitions.delivery.smtplib.SMTP_SSL")
    def test_sendmail_called_with_correct_addresses(self, mock_smtp_class):
        mock_server = MagicMock()
        mock_smtp_class.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp_class.return_value.__exit__  = MagicMock(return_value=False)

        run(_task_input(sender_email="from@gmail.com", recipient_email="to@example.com"))

        call_args = mock_smtp_class.return_value.sendmail.call_args
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
