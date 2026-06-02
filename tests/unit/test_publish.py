# tests/unit/test_publish.py
#
# Unit tests for the publish node. All S3, CloudFront, and template I/O are
# patched. Pure helper functions are tested in isolation. run() is tested via
# mocked boto3 + render functions so no real AWS calls or file reads occur.
#
# Note: run() calls pub_date.strftime("%-d %B %Y") which is Linux-only. The
# happy-path run() tests patch the date module to avoid this platform issue.

import json
from datetime import date as real_date
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from node_definitions.publish import (
    _extract_body,
    _extract_excerpt,
    _extract_title,
    _generate_article_list_html,
    _load_manifest,
    _render_sitemap,
    _update_manifest,
    run,
)
from contracts.nodes import PublishTaskInput, PublishTaskResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _task_input(**overrides) -> PublishTaskInput:
    base = {
        "run_id":      "2026-06-01",
        "digest_html": "<h1>Agent Routing Patterns</h1><p>First paragraph here.</p><p>More.</p>",
        "topic":       "agent-routing-patterns",
        "bucket":      "my-site.com",
        "cf_dist_id":  "ABCDEF123",
        "domain":      "my-site.com",
        "author_name": "Test Author",
    }
    base.update(overrides)
    return PublishTaskInput(**base)


def _client_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": "test error"}}, "GetObject")


def _manifest_entry(date_str: str) -> dict:
    return {
        "date":    date_str,
        "title":   f"Article {date_str}",
        "excerpt": "Excerpt.",
        "url":     f"/articles/{date_str}.html",
    }


# ---------------------------------------------------------------------------
# _extract_title
# ---------------------------------------------------------------------------

class TestExtractTitle:

    def test_extracts_h1_content(self):
        assert _extract_title("<h1>Agent Routing Patterns</h1><p>Body.</p>", "fallback") == "Agent Routing Patterns"

    def test_strips_nested_tags_from_h1(self):
        assert _extract_title("<h1><em>Bold</em> Title</h1>", "fallback") == "Bold Title"

    def test_case_insensitive_h1_match(self):
        assert _extract_title("<H1>Upper Case Heading</H1>", "fallback") == "Upper Case Heading"

    def test_fallback_title_cases_topic(self):
        # replace("-", " ") then .title(): "multi-agent orchestration" → "Multi Agent Orchestration"
        result = _extract_title("<p>No heading.</p>", "multi-agent orchestration")
        assert result == "Multi Agent Orchestration"

    def test_fallback_replaces_dashes_with_spaces(self):
        result = _extract_title("<p>no h1</p>", "llm-routing")
        assert result == "Llm Routing"

    def test_fallback_when_h1_is_whitespace_only(self):
        result = _extract_title("<h1>   </h1><p>Content.</p>", "llm-routing")
        assert result == "Llm Routing"


# ---------------------------------------------------------------------------
# _extract_excerpt
# ---------------------------------------------------------------------------

class TestExtractExcerpt:

    def test_extracts_first_paragraph(self):
        html = "<p>First paragraph text.</p><p>Second paragraph.</p>"
        assert _extract_excerpt(html) == "First paragraph text."

    def test_strips_inner_tags(self):
        html = "<p><strong>Bold</strong> text here.</p>"
        result = _extract_excerpt(html)
        assert "Bold text here." in result
        assert "<strong>" not in result

    def test_truncates_to_160_characters(self):
        html = f"<p>{'x' * 200}</p>"
        assert len(_extract_excerpt(html)) == 160

    def test_does_not_truncate_short_text(self):
        html = "<p>Short text.</p>"
        assert _extract_excerpt(html) == "Short text."

    def test_returns_empty_string_when_no_paragraph(self):
        assert _extract_excerpt("<h1>Only a heading.</h1>") == ""


# ---------------------------------------------------------------------------
# _extract_body
# ---------------------------------------------------------------------------

class TestExtractBody:

    def test_removes_leading_h1(self):
        html = "<h1>Title</h1><p>Body paragraph.</p>"
        result = _extract_body(html)
        assert "<h1>" not in result
        assert "Body paragraph." in result

    def test_removes_only_first_h1(self):
        html = "<h1>First</h1><p>Middle.</p><h1>Second</h1>"
        result = _extract_body(html)
        assert "First" not in result
        assert "<h1>Second</h1>" in result
        assert "Middle." in result

    def test_no_h1_returns_html_unchanged(self):
        html = "<p>Just a paragraph.</p>"
        assert _extract_body(html) == "<p>Just a paragraph.</p>"

    def test_multiline_h1_stripped(self):
        html = "<h1>\n  Multi\n  Line\n</h1><p>Body.</p>"
        result = _extract_body(html)
        assert "<h1>" not in result
        assert "Body." in result


# ---------------------------------------------------------------------------
# _load_manifest
# ---------------------------------------------------------------------------

class TestLoadManifest:

    def test_returns_empty_list_on_no_such_key(self):
        mock_s3 = MagicMock()
        mock_s3.get_object.side_effect = _client_error("NoSuchKey")
        assert _load_manifest(mock_s3, "bucket") == []

    def test_returns_empty_list_on_404(self):
        mock_s3 = MagicMock()
        mock_s3.get_object.side_effect = _client_error("404")
        assert _load_manifest(mock_s3, "bucket") == []

    def test_returns_empty_list_on_access_denied(self):
        mock_s3 = MagicMock()
        mock_s3.get_object.side_effect = _client_error("AccessDenied")
        assert _load_manifest(mock_s3, "bucket") == []

    def test_returns_empty_list_on_generic_exception(self):
        mock_s3 = MagicMock()
        mock_s3.get_object.side_effect = RuntimeError("connection reset")
        assert _load_manifest(mock_s3, "bucket") == []

    def test_returns_parsed_manifest_on_success(self):
        mock_s3 = MagicMock()
        manifest_data = [_manifest_entry("2026-05-01")]
        mock_s3.get_object.return_value = {
            "Body": MagicMock(read=MagicMock(return_value=json.dumps(manifest_data).encode()))
        }
        result = _load_manifest(mock_s3, "bucket")
        assert len(result) == 1
        assert result[0]["date"] == "2026-05-01"


# ---------------------------------------------------------------------------
# _update_manifest
# ---------------------------------------------------------------------------

class TestUpdateManifest:

    def test_prepends_new_entry_to_empty_manifest(self):
        result = _update_manifest([], "2026-06-01", "New Entry", "excerpt")
        assert len(result) == 1
        assert result[0]["date"] == "2026-06-01"

    def test_new_entry_sorts_to_top(self):
        manifest = [_manifest_entry("2026-05-01")]
        result = _update_manifest(manifest, "2026-06-01", "Newer", "excerpt")
        assert result[0]["date"] == "2026-06-01"

    def test_deduplicates_by_date(self):
        manifest = [_manifest_entry("2026-06-01")]
        manifest[0]["title"] = "Stale"
        result = _update_manifest(manifest, "2026-06-01", "Fresh", "new excerpt")
        assert [e["date"] for e in result].count("2026-06-01") == 1
        assert result[0]["title"] == "Fresh"

    def test_sorts_descending_by_date(self):
        manifest = [_manifest_entry("2026-04-01"), _manifest_entry("2026-05-15")]
        result = _update_manifest(manifest, "2026-06-01", "Latest", "e")
        assert result[0]["date"] == "2026-06-01"
        assert result[1]["date"] == "2026-05-15"
        assert result[2]["date"] == "2026-04-01"

    def test_entry_url_uses_correct_format(self):
        result = _update_manifest([], "2026-06-01", "Title", "excerpt")
        assert result[0]["url"] == "/articles/2026-06-01.html"


# ---------------------------------------------------------------------------
# _render_sitemap
# ---------------------------------------------------------------------------

class TestRenderSitemap:

    def test_includes_articles_within_90_days(self):
        today = real_date(2026, 6, 1)
        manifest = [_manifest_entry("2026-05-01")]  # 31 days ago
        xml = _render_sitemap(manifest, "my-site.com", today)
        assert "2026-05-01.html" in xml

    def test_excludes_articles_older_than_89_days(self):
        today = real_date(2026, 6, 1)
        manifest = [_manifest_entry("2026-01-01")]  # 151 days ago
        xml = _render_sitemap(manifest, "my-site.com", today)
        assert "2026-01-01.html" not in xml

    def test_always_includes_root_and_index_urls(self):
        xml = _render_sitemap([], "my-site.com", real_date(2026, 6, 1))
        assert "https://my-site.com/" in xml
        assert "https://my-site.com/articles/index.html" in xml

    def test_output_is_valid_xml_envelope(self):
        xml = _render_sitemap([], "my-site.com", real_date(2026, 6, 1))
        assert xml.startswith('<?xml version="1.0"')
        assert "<urlset" in xml


# ---------------------------------------------------------------------------
# _generate_article_list_html
# ---------------------------------------------------------------------------

class TestGenerateArticleListHtml:
    # _featured_article_html and _older_articles_html call
    # date.fromisoformat(s).strftime("%-d ...") which is Linux-only.
    # Patch the date class in the module to make these tests cross-platform.

    def _date_mock(self):
        m = MagicMock()
        m.fromisoformat.side_effect = lambda s: MagicMock(strftime=MagicMock(return_value="1 Jun"))
        return m

    def test_empty_list_returns_empty_state_message(self):
        result = _generate_article_list_html([], real_date(2026, 6, 1))
        assert "No articles yet" in result

    def test_today_entry_rendered_with_featured_markup(self):
        entries = [_manifest_entry("2026-06-01")]
        with patch("node_definitions.publish.date", self._date_mock()):
            result = _generate_article_list_html(entries, real_date(2026, 6, 1))
        assert "Today" in result
        assert "Latest" in result

    def test_older_entries_wrapped_in_details_element(self):
        entries = [_manifest_entry("2026-05-01"), _manifest_entry("2026-05-15")]
        with patch("node_definitions.publish.date", self._date_mock()):
            result = _generate_article_list_html(entries, real_date(2026, 6, 1))
        assert "<details" in result
        assert "previous article" in result

    def test_single_older_article_uses_singular_label(self):
        entries = [_manifest_entry("2026-05-01")]
        with patch("node_definitions.publish.date", self._date_mock()):
            result = _generate_article_list_html(entries, real_date(2026, 6, 1))
        assert "1 previous article" in result
        assert "1 previous articles" not in result

    def test_multiple_older_articles_uses_plural_label(self):
        entries = [_manifest_entry("2026-05-01"), _manifest_entry("2026-05-10")]
        with patch("node_definitions.publish.date", self._date_mock()):
            result = _generate_article_list_html(entries, real_date(2026, 6, 1))
        assert "2 previous articles" in result


# ---------------------------------------------------------------------------
# run() — skip behavior (no AWS calls, no platform-specific code)
# ---------------------------------------------------------------------------

class TestRunSkipBehavior:

    def test_returns_skipped_when_bucket_is_empty(self):
        result = run(_task_input(bucket=""))
        assert result.published is False
        assert result.skipped is True
        assert result.error == ""

    def test_run_id_echoed_in_skip_result(self):
        result = run(_task_input(bucket="", run_id="2026-06-01"))
        assert result.run_id == "2026-06-01"


# ---------------------------------------------------------------------------
# run() — happy path (boto3 + template helpers + date patched)
# ---------------------------------------------------------------------------

class TestRunSuccessPath:

    def _patched_run(self, task_input=None):
        """Run with all external dependencies mocked. Returns (result, mock_s3, mock_cf)."""
        if task_input is None:
            task_input = _task_input()
        mock_s3 = MagicMock()
        mock_cf = MagicMock()
        mock_s3.get_object.side_effect = _client_error("NoSuchKey")

        mock_date = MagicMock()
        mock_date.fromisoformat.return_value.strftime.return_value = "1 June 2026"

        with patch("node_definitions.publish._render_article", return_value="<html>art</html>"), \
             patch("node_definitions.publish._render_index", return_value="<html>idx</html>"), \
             patch("node_definitions.publish._render_sitemap", return_value="<?xml?><urlset/>"), \
             patch("node_definitions.publish.date", mock_date), \
             patch("node_definitions.publish.boto3.client",
                   side_effect=lambda svc, **kw: mock_cf if svc == "cloudfront" else mock_s3):
            result = run(task_input)

        return result, mock_s3, mock_cf

    def test_returns_published_true(self):
        result, _, _ = self._patched_run()
        assert isinstance(result, PublishTaskResult)
        assert result.published is True
        assert result.skipped is False
        assert result.error == ""

    def test_article_url_uses_domain_and_run_id(self):
        result, _, _ = self._patched_run(_task_input(domain="sam-griffith.dev", run_id="2026-06-01"))
        assert result.article_url == "https://sam-griffith.dev/articles/2026-06-01.html"

    def test_s3_put_called_for_all_four_artifacts(self):
        _, mock_s3, _ = self._patched_run()
        keys = [call[1]["Key"] for call in mock_s3.put_object.call_args_list]
        assert any("articles/2026-06-01.html" in k for k in keys)
        assert any("manifest.json" in k for k in keys)
        assert any("articles/index.html" in k for k in keys)
        assert any("sitemap.xml" in k for k in keys)

    def test_cloudfront_invalidation_covers_three_paths(self):
        _, _, mock_cf = self._patched_run()
        batch = mock_cf.create_invalidation.call_args[1]["InvalidationBatch"]
        paths = batch["Paths"]["Items"]
        assert "/articles/2026-06-01.html" in paths
        assert "/articles/index.html" in paths
        assert "/sitemap.xml" in paths


# ---------------------------------------------------------------------------
# run() — error path
# ---------------------------------------------------------------------------

class TestRunErrorPath:

    @patch("node_definitions.publish.date")
    @patch("node_definitions.publish.boto3.client")
    @patch("node_definitions.publish._render_article", return_value="<html/>")
    def test_returns_published_false_on_s3_error(self, _mock_render, mock_boto, mock_date):
        mock_date.fromisoformat.return_value.strftime.return_value = "1 June 2026"
        mock_s3 = MagicMock()
        mock_boto.return_value = mock_s3
        mock_s3.put_object.side_effect = RuntimeError("S3 unavailable")

        result = run(_task_input())

        assert result.published is False
        assert result.skipped is False
        assert result.error != ""

    @patch("node_definitions.publish.date")
    @patch("node_definitions.publish.boto3.client")
    @patch("node_definitions.publish._render_article", return_value="<html/>")
    def test_error_message_captured_in_result(self, _mock_render, mock_boto, mock_date):
        mock_date.fromisoformat.return_value.strftime.return_value = "1 June 2026"
        mock_s3 = MagicMock()
        mock_boto.return_value = mock_s3
        mock_s3.put_object.side_effect = RuntimeError("AccessDenied: no s3:PutObject permission")

        result = run(_task_input())

        assert "AccessDenied" in result.error

    @patch("node_definitions.publish.date")
    @patch("node_definitions.publish.boto3.client")
    @patch("node_definitions.publish._render_article", return_value="<html/>")
    def test_run_never_raises(self, _mock_render, mock_boto, mock_date):
        mock_date.fromisoformat.return_value.strftime.return_value = "1 June 2026"
        mock_boto.side_effect = Exception("catastrophic boto3 failure")

        result = run(_task_input())

        assert isinstance(result, PublishTaskResult)
        assert result.published is False
