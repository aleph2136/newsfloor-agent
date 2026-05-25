"""
crew_utils.py

Shared utilities for CrewAI crew execution in node definitions.
"""

from __future__ import annotations
import logging
import time

from crewai import Crew

logger = logging.getLogger(__name__)


def kickoff_crew(
    crew:   Crew,
    node:   str,
    run_id: str | None,
    models: list[str],
) -> None:
    """
    Calls crew.kickoff() and logs token usage, latency, and model for the given graph node.

    Logs a structured 'llm_usage' event after a successful kickoff. Any exception
    from kickoff() propagates to the caller unchanged — no metrics are logged on failure.

    Args:
        crew:   The CrewAI Crew to execute.
        node:   Name of the graph node (or sub-function) making this call.
        run_id: The pipeline run identifier. Pass None when not available (e.g. trend helpers).
        models: List of model IDs used by this crew's agents.
    """
    t0 = time.perf_counter()
    crew.kickoff()
    latency_ms = round((time.perf_counter() - t0) * 1000)

    m = crew.usage_metrics
    entry: dict = {
        "event":                 "llm_usage",
        "node":                  node,
        "model":                 models,
        "prompt_tokens":         m.prompt_tokens         if m else 0,
        "completion_tokens":     m.completion_tokens     if m else 0,
        "cached_tokens":         m.cached_prompt_tokens  if m else 0,
        "cache_creation_tokens": m.cache_creation_tokens if m else 0,
        "total_tokens":          m.total_tokens          if m else 0,
        "successful_requests":   m.successful_requests   if m else 0,
        "latency_ms":            latency_ms,
    }
    if run_id is not None:
        entry["run_id"] = run_id

    logger.info(entry)
