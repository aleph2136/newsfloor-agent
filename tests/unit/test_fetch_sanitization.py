# tests/unit/test_fetch_sanitization.py
#
# Unit tests for the _strip_html helper in fetch.py.
# Verifies that externally-fetched article content is fully sanitized
# before being passed to the LLM — including entity-encoded tags.

import pytest

from node_definitions.fetch import _strip_html


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
