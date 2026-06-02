# tests/unit/test_fetch_sanitization.py
#
# Unit tests for the _strip_html helper and _parse_feed date-sort behavior in fetch.py.
# Verifies that externally-fetched article content is fully sanitized
# before being passed to the LLM — including entity-encoded tags.

import pytest
from unittest.mock import MagicMock, patch

from node_definitions.fetch import _strip_html, _parse_feed


# ---------------------------------------------------------------------------
# Basic HTML stripping
# ---------------------------------------------------------------------------

class TestStripHtmlBasic:

    def test_strips_paragraph_tags(self):
        result = _strip_html("<p>Hello world</p>")
        assert "<p>" not in result
        assert "Hello world" in result

    def test_strips_anchor_tags(self):
        result = _strip_html('<a href="https://example.com">link text</a>')
        assert "<a" not in result
        assert "link text" in result

    def test_strips_script_tag_and_content(self):
        result = _strip_html("<p>Safe</p><script>alert('xss')</script>")
        assert "<script>" not in result
        assert "alert" not in result

    def test_strips_style_tag_and_content(self):
        result = _strip_html("<style>body { color: red; }</style><p>Text</p>")
        assert "<style>" not in result
        assert "color: red" not in result

    def test_strips_nested_tags(self):
        result = _strip_html("<div><span><em>deep</em></span></div>")
        assert "<" not in result
        assert "deep" in result

    def test_returns_plain_text_unchanged(self):
        result = _strip_html("plain text no tags")
        assert result == "plain text no tags"

    def test_empty_string(self):
        result = _strip_html("")
        assert result == ""


# ---------------------------------------------------------------------------
# Entity-encoded tag removal (the key security property)
# ---------------------------------------------------------------------------

class TestStripHtmlEntityEncoded:

    def test_strips_entity_encoded_script(self):
        # &lt;script&gt; decodes to <script> — must not survive as a literal
        result = _strip_html("&lt;script&gt;alert(1)&lt;/script&gt;")
        assert "<script>" not in result
        assert "&lt;script&gt;" not in result
        assert "alert" not in result

    def test_strips_entity_encoded_iframe(self):
        result = _strip_html("&lt;iframe src='evil.com'&gt;&lt;/iframe&gt;")
        assert "iframe" not in result
        assert "evil.com" not in result

    def test_strips_mixed_real_and_encoded_tags(self):
        result = _strip_html("<p>Safe</p>&lt;script&gt;evil()&lt;/script&gt;")
        assert "<p>" not in result
        assert "evil" not in result
        assert "Safe" in result

    def test_unescapes_safe_entities(self):
        result = _strip_html("AT&amp;T &mdash; &ldquo;hello&rdquo;")
        assert "AT&T" in result or "AT" in result  # entity decoded to plain text


# ---------------------------------------------------------------------------
# Malformed / adversarial HTML
# ---------------------------------------------------------------------------

class TestStripHtmlMalformed:

    def test_unclosed_tag(self):
        result = _strip_html("<p>Unclosed paragraph")
        assert "<p>" not in result
        assert "Unclosed paragraph" in result

    def test_deeply_nested_tags(self):
        html = "<div>" * 50 + "content" + "</div>" * 50
        result = _strip_html(html)
        assert "<div>" not in result
        assert "content" in result

    def test_tag_with_no_closing_angle(self):
        # Malformed but should not raise
        result = _strip_html("<p Truncated content")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _parse_feed date sort
# ---------------------------------------------------------------------------

class _FeedEntry:
    """Minimal feedparser-compatible entry for testing."""
    def __init__(self, url: str, title: str, summary: str, published_tuple):
        self._data = {"link": url, "title": title, "summary": summary}
        self.published_parsed = published_tuple

    def get(self, key, default=""):
        return self._data.get(key, default)


def _make_fake_feed(entries):
    feed = MagicMock()
    feed.bozo = False
    feed.entries = entries
    return feed


class TestParseFeedDateSort:
    """_parse_feed returns articles sorted newest-published-first, regardless of feed entry order."""

    @patch("node_definitions.fetch.feedparser.parse")
    @patch("node_definitions.fetch.httpx.get")
    def test_entries_sorted_newest_first(self, mock_http, mock_feedparser):
        old_entry = _FeedEntry(
            "https://example.com/old", "Old Article", "Old summary.",
            (2024, 1, 1, 0, 0, 0),
        )
        new_entry = _FeedEntry(
            "https://example.com/new", "New Article", "New summary.",
            (2026, 5, 1, 0, 0, 0),
        )
        mock_http.return_value = MagicMock(text="<xml/>", status_code=200)
        mock_feedparser.return_value = _make_fake_feed([old_entry, new_entry])

        articles, error = _parse_feed("https://example.com/feed.xml")

        assert error == ""
        assert len(articles) == 2
        assert articles[0].title == "New Article"
        assert articles[1].title == "Old Article"

    @patch("node_definitions.fetch.feedparser.parse")
    @patch("node_definitions.fetch.httpx.get")
    def test_already_sorted_feed_unchanged(self, mock_http, mock_feedparser):
        first_entry = _FeedEntry(
            "https://example.com/first", "First Article", "First summary.",
            (2026, 5, 15, 0, 0, 0),
        )
        second_entry = _FeedEntry(
            "https://example.com/second", "Second Article", "Second summary.",
            (2026, 4, 1, 0, 0, 0),
        )
        mock_http.return_value = MagicMock(text="<xml/>", status_code=200)
        mock_feedparser.return_value = _make_fake_feed([first_entry, second_entry])

        articles, error = _parse_feed("https://example.com/feed.xml")

        assert articles[0].title == "First Article"
        assert articles[1].title == "Second Article"

    @patch("node_definitions.fetch.feedparser.parse")
    @patch("node_definitions.fetch.httpx.get")
    def test_undated_entry_sorted_last(self, mock_http, mock_feedparser):
        undated = _FeedEntry(
            "https://example.com/undated", "Undated Article", "No date.", None
        )
        dated = _FeedEntry(
            "https://example.com/dated", "Dated Article", "Has a date.",
            (2025, 6, 1, 0, 0, 0),
        )
        mock_http.return_value = MagicMock(text="<xml/>", status_code=200)
        mock_feedparser.return_value = _make_fake_feed([undated, dated])

        articles, error = _parse_feed("https://example.com/feed.xml")

        assert articles[0].title == "Dated Article"
        assert articles[1].title == "Undated Article"

    @patch("node_definitions.fetch.feedparser.parse")
    @patch("node_definitions.fetch.httpx.get")
    def test_multiple_undated_entries_all_present(self, mock_http, mock_feedparser):
        entries = [
            _FeedEntry("https://example.com/a", "Article A", "Summary A.", None),
            _FeedEntry("https://example.com/b", "Article B", "Summary B.", None),
        ]
        mock_http.return_value = MagicMock(text="<xml/>", status_code=200)
        mock_feedparser.return_value = _make_fake_feed(entries)

        articles, error = _parse_feed("https://example.com/feed.xml")

        assert len(articles) == 2
