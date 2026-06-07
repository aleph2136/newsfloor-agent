"""
nodes/synthesis/parsers.py

JSON parsing for digest output and signal extraction.

Public interface
────────────────
  parse_digest_json(raw, run_id)  DigestStructured from writer JSON output
  parse_signals_output(raw)       dict from signal extractor JSON output
  strip_markdown_fences(text)     Removes ```json/``` wrappers from LLM output
"""

from __future__ import annotations
import json
import logging
import re

from contracts.nodes import (
    DigestContentBlock,
    DigestMetadata,
    DigestStructured,
    VisualAssets,
)

logger = logging.getLogger(__name__)


def parse_digest_json(raw: str, run_id: str) -> DigestStructured:
    """
    Parses the writer's JSON output into a DigestStructured object.
    Returns a minimal fallback on failure so the run never crashes here.
    """
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in writer output")
        data = json.loads(match.group())

        metadata = DigestMetadata(
            title                 = data.get("metadata", {}).get("title", "Today's AI Digest"),
            date                  = data.get("metadata", {}).get("date", run_id),
            summary_hook          = data.get("metadata", {}).get("summary_hook", ""),
            overall_trend_context = data.get("metadata", {}).get("overall_trend_context", ""),
        )

        blocks = []
        for i, b in enumerate(data.get("content_blocks", [])):
            visual_raw = b.get("visual_assets", {})
            blocks.append(DigestContentBlock(
                section_id       = b.get("section_id", f"block_{i+1}"),
                section_title    = b.get("section_title", ""),
                tier_1_hook      = b.get("tier_1_hook", ""),
                tier_2_bullets   = b.get("tier_2_bullets", []),
                tier_3_deep_dive = b.get("tier_3_deep_dive", ""),
                visual_assets    = VisualAssets(
                    mermaid_diagram = visual_raw.get("mermaid_diagram", ""),
                    code_block      = visual_raw.get("code_block", ""),
                ),
            ))

        return DigestStructured(
            article_id     = data.get("article_id", run_id),
            metadata       = metadata,
            content_blocks = blocks,
        )

    except Exception as exc:
        logger.warning({"node": "synthesis", "warning": f"Could not parse digest JSON: {exc} — using minimal fallback"})
        return DigestStructured(
            article_id = run_id,
            metadata   = DigestMetadata(
                title="Today's AI Engineering Digest",
                date=run_id,
                summary_hook="",
                overall_trend_context="",
            ),
            content_blocks=[DigestContentBlock(
                section_id="block_1",
                section_title="Digest",
                tier_1_hook="",
                tier_2_bullets=[f"**Raw output** {raw[:300]}"],
                tier_3_deep_dive="",
            )],
        )


def parse_signals_output(raw_output: str) -> dict:
    """
    Parses the signal extractor's JSON output.
    Returns safe defaults on parse failure so the run never crashes here.

    Fences are stripped first — LLMs frequently wrap JSON in markdown blocks.
    """
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_output.strip(), flags=re.IGNORECASE)
    try:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        data  = json.loads(match.group()) if match else {}
        return {
            "new_signals":         data.get("new_signals", []),
            "trend_confirmations": data.get("trend_confirmations", []),
            "digest_summary":      data.get("digest_summary", ""),
        }
    except (json.JSONDecodeError, AttributeError):
        logger.warning({"node": "synthesis", "warning": "Could not parse signal extractor output"})
        return {
            "new_signals":         [],
            "trend_confirmations": [],
            "digest_summary":      "",
        }


def strip_markdown_fences(text: str) -> str:
    """Removes ```json/```html/``` fences that LLMs sometimes wrap around output."""
    return re.sub(
        r"^```(?:json|html)?\s*\n?(.*?)\n?```\s*$",
        r"\1",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()
