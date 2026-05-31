# tests/unit/test_synthesis_sanitization.py
#
# Unit tests for the HTML sanitization and markdown-fence stripping helpers
# in synthesis.py. No LLM or CrewAI calls — pure function tests.

import pytest

from node_definitions.synthesis import (
    _sanitize_digest_html,
    _strip_markdown_fences,
)


# ---------------------------------------------------------------------------
# _strip_markdown_fences
# ---------------------------------------------------------------------------

class TestStripMarkdownFences:

    def test_strips_html_fenced_block(self):
        raw = "```html\n<h1>Title</h1>\n```"
        assert _strip_markdown_fences(raw) == "<h1>Title</h1>"

    def test_strips_plain_fenced_block(self):
        raw = "```\n<h1>Title</h1>\n```"
        assert _strip_markdown_fences(raw) == "<h1>Title</h1>"

    def test_passthrough_when_no_fences(self):
        raw = "<h1>Title</h1><p>Body.</p>"
        assert _strip_markdown_fences(raw) == raw

    def test_case_insensitive_html_label(self):
        raw = "```HTML\n<h1>Title</h1>\n```"
        assert _strip_markdown_fences(raw) == "<h1>Title</h1>"

    def test_preserves_interior_backtick_content(self):
        raw = "```html\n<p>Use `code` here</p>\n```"
        assert _strip_markdown_fences(raw) == "<p>Use `code` here</p>"

    def test_strips_trailing_whitespace_around_fences(self):
        raw = "```html\n<h1>Title</h1>\n```   "
        assert _strip_markdown_fences(raw) == "<h1>Title</h1>"

    def test_multiline_html_preserved(self):
        raw = "```html\n<h1>T</h1>\n<p>Body.</p>\n```"
        result = _strip_markdown_fences(raw)
        assert "<h1>T</h1>" in result
        assert "<p>Body.</p>" in result


# ---------------------------------------------------------------------------
# _sanitize_digest_html — blocked tags removed
# ---------------------------------------------------------------------------

class TestSanitizeDigestHtmlBlockedTags:

    def test_removes_script_tag_and_content(self):
        html = "<p>Safe</p><script>alert('xss')</script>"
        result = _sanitize_digest_html(html)
        assert "<script>" not in result
        assert "alert" not in result

    def test_removes_style_tag(self):
        html = "<p>Text</p><style>body { color: red; }</style>"
        result = _sanitize_digest_html(html)
        assert "<style>" not in result
        assert "color: red" not in result

    def test_removes_iframe(self):
        html = '<p>Text</p><iframe src="https://evil.com"></iframe>'
        result = _sanitize_digest_html(html)
        assert "iframe" not in result

    def test_removes_object_tag(self):
        html = '<p>Text</p><object data="file.swf"></object>'
        result = _sanitize_digest_html(html)
        assert "object" not in result

    def test_removes_embed_tag(self):
        html = '<p>Text</p><embed src="plugin.swf" />'
        result = _sanitize_digest_html(html)
        assert "embed" not in result

    def test_removes_form_and_input(self):
        html = '<form action="/post"><input type="text" name="q"></form><p>Safe</p>'
        result = _sanitize_digest_html(html)
        assert "form" not in result
        assert "input" not in result

    def test_removes_svg(self):
        html = "<p>Text</p><svg><script>alert(1)</script></svg>"
        result = _sanitize_digest_html(html)
        assert "svg" not in result

    def test_removes_template_tag(self):
        html = "<template><p>Hidden</p></template><p>Visible</p>"
        result = _sanitize_digest_html(html)
        assert "template" not in result

    def test_preserves_safe_content(self):
        html = "<h1>Title</h1><p>Body <a href='https://example.com'>link</a>.</p>"
        result = _sanitize_digest_html(html)
        assert "Title" in result
        assert "Body" in result
        assert "https://example.com" in result


# ---------------------------------------------------------------------------
# _sanitize_digest_html — event-handler attributes stripped
# ---------------------------------------------------------------------------

class TestSanitizeDigestHtmlEventAttributes:

    def test_removes_onclick(self):
        html = '<p onclick="evil()">Click me</p>'
        result = _sanitize_digest_html(html)
        assert "onclick" not in result
        assert "Click me" in result

    def test_removes_onerror(self):
        html = '<img src="x" onerror="alert(1)" />'
        result = _sanitize_digest_html(html)
        assert "onerror" not in result

    def test_removes_onload(self):
        html = '<body onload="steal()">'
        result = _sanitize_digest_html(html)
        assert "onload" not in result

    def test_removes_onmouseover(self):
        html = '<a href="https://ok.com" onmouseover="track()">link</a>'
        result = _sanitize_digest_html(html)
        assert "onmouseover" not in result
        assert "https://ok.com" in result

    def test_removes_arbitrary_on_prefix_attribute(self):
        html = '<div onfoo="bar()">content</div>'
        result = _sanitize_digest_html(html)
        assert "onfoo" not in result
        assert "content" in result


# ---------------------------------------------------------------------------
# _sanitize_digest_html — javascript: URL schemes stripped
# ---------------------------------------------------------------------------

class TestSanitizeDigestHtmlJavascriptUrls:

    def test_removes_javascript_href(self):
        html = '<a href="javascript:alert(1)">click</a>'
        result = _sanitize_digest_html(html)
        assert "javascript:" not in result
        assert "click" in result

    def test_removes_javascript_src(self):
        html = '<img src="javascript:evil()" />'
        result = _sanitize_digest_html(html)
        assert "javascript:" not in result

    def test_preserves_https_href(self):
        html = '<a href="https://example.com">link</a>'
        result = _sanitize_digest_html(html)
        assert 'href="https://example.com"' in result or "https://example.com" in result

    def test_javascript_with_leading_whitespace_stripped(self):
        html = '<a href="  javascript:alert(1)">xss</a>'
        result = _sanitize_digest_html(html)
        assert "javascript:" not in result
