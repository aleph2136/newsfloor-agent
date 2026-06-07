# tests/unit/test_synthesis_json.py
#
# Unit tests for the new JSON parsing and HTML generation helpers in synthesis.py.
# No LLM or CrewAI calls — pure function tests.

import pytest

from node_definitions.synthesis import (
    _digest_json_to_html,
    _parse_digest_json,
    _strip_markdown_fences,
)
from contracts.nodes import DigestContentBlock, DigestMetadata, DigestStructured, VisualAssets


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _minimal_structured() -> DigestStructured:
    return DigestStructured(
        article_id="2026-06-06-test",
        metadata=DigestMetadata(
            title="Test Title",
            date="2026-06-06",
            summary_hook="A concise hook.",
            overall_trend_context="The field is moving toward structured outputs.",
        ),
        content_blocks=[
            DigestContentBlock(
                section_id="block_1",
                section_title="Supervisor Patterns",
                tier_1_hook="Soft prompts cannot hold state constraints.",
                tier_2_bullets=[
                    "**State drift accumulates** silently in long-running workflows.",
                    "**Deterministic validation** must wrap LLM steps to maintain schema fidelity.",
                ],
                tier_3_deep_dive="When agents run over long horizons their context degrades. By treating each step as a database-backed transaction delta, developers can pinpoint failures.",
                visual_assets=VisualAssets(
                    mermaid_diagram="graph TD;\n  A[Agent] -->|No Memory| B(Drift);",
                    code_block="class CheckpointedAgent:\n    pass",
                ),
            )
        ],
    )


_VALID_JSON = """
{
  "article_id": "2026-06-06-test",
  "metadata": {
    "title": "Test Title",
    "date": "2026-06-06",
    "summary_hook": "A concise hook.",
    "overall_trend_context": "Moving toward structured outputs."
  },
  "content_blocks": [
    {
      "section_id": "block_1",
      "section_title": "Supervisor Patterns",
      "tier_1_hook": "Soft prompts cannot hold state.",
      "tier_2_bullets": [
        "**State drift accumulates** silently.",
        "**Validation loops** must wrap LLM steps."
      ],
      "tier_3_deep_dive": "Deep technical details here.",
      "visual_assets": {
        "mermaid_diagram": "graph TD; A-->B;",
        "code_block": "def foo(): pass"
      }
    }
  ]
}
"""


# ---------------------------------------------------------------------------
# _parse_digest_json
# ---------------------------------------------------------------------------

class TestParseDigestJson:

    def test_parses_valid_json(self):
        result = _parse_digest_json(_VALID_JSON, "2026-06-06")
        assert result.metadata.title == "Test Title"
        assert result.metadata.summary_hook == "A concise hook."

    def test_parses_content_blocks(self):
        result = _parse_digest_json(_VALID_JSON, "2026-06-06")
        assert len(result.content_blocks) == 1
        block = result.content_blocks[0]
        assert block.section_title == "Supervisor Patterns"
        assert len(block.tier_2_bullets) == 2

    def test_parses_visual_assets(self):
        result = _parse_digest_json(_VALID_JSON, "2026-06-06")
        block = result.content_blocks[0]
        assert "graph TD" in block.visual_assets.mermaid_diagram
        assert "def foo" in block.visual_assets.code_block

    def test_returns_fallback_on_invalid_json(self):
        result = _parse_digest_json("NOT VALID JSON AT ALL", "2026-06-06")
        assert result.article_id == "2026-06-06"
        assert result.metadata.title == "Today's AI Engineering Digest"

    def test_returns_fallback_on_empty_string(self):
        result = _parse_digest_json("", "2026-06-06")
        assert result.article_id == "2026-06-06"

    def test_tolerates_missing_optional_fields(self):
        minimal = '{"article_id": "x", "metadata": {"title": "T", "date": "2026-06-06", "summary_hook": "", "overall_trend_context": ""}, "content_blocks": []}'
        result = _parse_digest_json(minimal, "2026-06-06")
        assert result.metadata.title == "T"
        assert result.content_blocks == []

    def test_strips_json_fences_before_parsing(self):
        fenced = "```json\n" + _VALID_JSON.strip() + "\n```"
        result = _parse_digest_json(fenced, "2026-06-06")
        assert result.metadata.title == "Test Title"

    def test_extracts_json_from_surrounding_prose(self):
        with_prose = "Here is the JSON:\n" + _VALID_JSON.strip() + "\nEnd of output."
        result = _parse_digest_json(with_prose, "2026-06-06")
        assert result.metadata.title == "Test Title"


# ---------------------------------------------------------------------------
# _digest_json_to_html
# ---------------------------------------------------------------------------

class TestDigestJsonToHtml:

    def test_includes_h1_with_title(self):
        ds = _minimal_structured()
        html = _digest_json_to_html(ds)
        assert "<h1>Test Title</h1>" in html

    def test_includes_h2_for_each_block(self):
        ds = _minimal_structured()
        html = _digest_json_to_html(ds)
        assert "<h2>" in html
        assert "Supervisor Patterns" in html

    def test_includes_em_for_tier1_hook(self):
        ds = _minimal_structured()
        html = _digest_json_to_html(ds)
        assert "<em>" in html
        assert "Soft prompts cannot hold state constraints." in html

    def test_includes_tier2_bullets(self):
        ds = _minimal_structured()
        html = _digest_json_to_html(ds)
        assert "State drift accumulates" in html

    def test_includes_summary_hook(self):
        ds = _minimal_structured()
        html = _digest_json_to_html(ds)
        assert "A concise hook." in html

    def test_includes_trend_context(self):
        ds = _minimal_structured()
        html = _digest_json_to_html(ds)
        assert "The field is moving toward structured outputs." in html

    def test_escapes_html_in_content(self):
        ds = _minimal_structured()
        ds.metadata.title = "<script>alert(1)</script>"
        html = _digest_json_to_html(ds)
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_output_meets_min_supervisor_length(self):
        # The output supervisor requires >= 800 chars. A real digest will be
        # much larger, but even a minimal single-block digest should be close.
        ds = _minimal_structured()
        # Pad tier3 to ensure the threshold is met
        ds.content_blocks[0].tier_3_deep_dive = "A" * 600
        html = _digest_json_to_html(ds)
        assert len(html) > 800

    def test_required_html_markers_present(self):
        # The output supervisor checks for <h1>, <h2>, <em>.
        ds = _minimal_structured()
        html = _digest_json_to_html(ds).lower()
        assert "<h1>" in html
        assert "<h2>" in html
        assert "<em>" in html
