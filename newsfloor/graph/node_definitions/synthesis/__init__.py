"""
nodes/synthesis/__init__.py

Public entry point for the synthesis node.
nodes.py imports only: from node_definitions.synthesis import run

Internal structure
──────────────────
crew.py     Agent definitions, task prompts, and crew assembly
parsers.py  JSON parsing for digest output and signal extraction
html.py     HTML generation and sanitization
retry.py    Retry instruction building for rework runs

Crew design
───────────
Three agents with distinct responsibilities:

  TrendContextualizer   Reviews active trends and recent signals alongside
                        the passed articles. Identifies which trends today's
                        content confirms, challenges, or extends. Produces
                        a trend analysis brief the writer uses as context.

  DigestWriter          Writes the structured JSON digest using the passed
                        articles and the trend context brief. Personalizes to
                        the engineer profile. Uses Gemini for near-zero cost.
                        Produces a DigestStructured JSON document with tiered
                        content blocks (hook → bullets → deep dive + visuals).

  SignalExtractor       Reads the finished digest and extracts discrete trend
                        signals — specific phrases, concepts, or patterns that
                        should be tracked over time. Produces new_signals and
                        trend_confirmations for the Trend node.

Rework behavior
───────────────
  DIGEST_INSUFFICIENT    → rewrite with stricter depth requirements
  MISSING_REQUIRED_FIELD → regenerate with explicit field checklist
"""

from __future__ import annotations
import logging

from crewai.llm import LLM

from config import settings
from contracts.nodes import SynthesisTaskInput, SynthesisTaskResult
from node_definitions.crew_utils import kickoff_crew

from .crew import build_synthesis_crew
from .html import digest_json_to_html, sanitize_digest_html
from .parsers import parse_digest_json, parse_signals_output, strip_markdown_fences
from .retry import apply_retry_adjustments

logger = logging.getLogger(__name__)


def run(task_input: SynthesisTaskInput) -> SynthesisTaskResult:
    """
    Runs the synthesis crew and returns a SynthesisTaskResult.
    """
    logger.info({
        "node":           "synthesis",
        "topic":          task_input.topic,
        "articles_count": len(task_input.passed_articles),
        "active_trends":  len(task_input.active_trends),
        "has_retry":      task_input.retry_instruction is not None,
    })

    depth_instruction = apply_retry_adjustments(task_input)

    # Gemini for the writer — near-zero cost at equal quality for JSON synthesis.
    # Bedrock Maverick for support agents (contextualizer + signal extractor).
    llm_writer  = LLM(model=settings.gemini_model_synthesis, max_retries=1)
    llm_support = LLM(model=settings.bedrock_model_synthesis_support, max_retries=1)

    crew, write_task, extract_task = build_synthesis_crew(
        task_input        = task_input,
        depth_instruction = depth_instruction,
        llm_writer        = llm_writer,
        llm_support       = llm_support,
    )

    kickoff_crew(
        crew, "synthesis", task_input.run_id,
        [settings.gemini_model_synthesis, settings.bedrock_model_synthesis_support],
    )

    # -------------------------------------------------------------------------
    # Parse results — guard against partial crew failure
    # -------------------------------------------------------------------------
    if not write_task.output or not write_task.output.raw:
        raise RuntimeError("Synthesis crew write task produced no output.")

    raw_json    = strip_markdown_fences(write_task.output.raw.strip())
    digest_json = parse_digest_json(raw_json, task_input.run_id)
    digest_html = digest_json_to_html(digest_json)
    digest_html = sanitize_digest_html(digest_html)

    if not extract_task.output or not extract_task.output.raw:
        logger.warning({"node": "synthesis", "warning": "Signal extractor produced no output — using empty signals"})
        signals_output: dict = {"new_signals": [], "trend_confirmations": [], "digest_summary": ""}
    else:
        signals_output = parse_signals_output(extract_task.output.raw)

    logger.info({
        "node":                "synthesis",
        "digest_html_length":  len(digest_html),
        "content_blocks":      len(digest_json.content_blocks) if digest_json else 0,
        "new_signals":         len(signals_output["new_signals"]),
        "trend_confirmations": len(signals_output["trend_confirmations"]),
    })

    return SynthesisTaskResult(
        run_id              = task_input.run_id,
        digest_html         = digest_html,
        digest_json         = digest_json,
        digest_summary      = signals_output["digest_summary"],
        new_signals         = signals_output["new_signals"],
        trend_confirmations = signals_output["trend_confirmations"],
    )
