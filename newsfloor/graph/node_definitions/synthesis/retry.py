"""
nodes/synthesis/retry.py

Retry instruction building for synthesis rework runs.

Public interface
────────────────
  apply_retry_adjustments(task_input)  Returns depth instruction string for the writer task
"""

from __future__ import annotations

from contracts.nodes import SynthesisTaskInput
from contracts.primitives import RetryReasonCode


def apply_retry_adjustments(task_input: SynthesisTaskInput) -> str:
    """
    Returns an additional instruction string for the writer task on rework.
    Empty string on first pass — only adds instructions when retrying.
    """
    instruction = task_input.retry_instruction
    if instruction is None:
        return ""

    reason = instruction.reason_code
    params = instruction.parameter_adjustment

    if reason == RetryReasonCode.DIGEST_INSUFFICIENT:
        base = (
            "The previous digest was rejected for insufficient depth. "
            "Each content block must have a tier_3_deep_dive of at least 3 sentences. "
            "The tier_1_hook must connect explicitly to agentic architecture or "
            "engineering governance — not a general observation."
        )

        failed = params.get("failed_criteria", [])
        criterion_notes: list[str] = []

        if "PERSONALIZED" in failed:
            criterion_notes.append(
                "The previous digest was not sufficiently personalized — "
                "every section must connect directly to agentic architecture, "
                "governance, or observability as they apply to the reader's work."
            )
        if "ACTIONABLE" in failed:
            criterion_notes.append(
                "The previous closing takeaway was too generic — "
                "name a specific pattern, tradeoff, or decision the reader can act on."
            )
        if "CONNECTED" in failed:
            criterion_notes.append(
                "The previous digest treated articles in isolation — "
                "explicitly connect at least two content blocks to each other or to an active trend."
            )

        if criterion_notes:
            return base + " Additionally: " + " ".join(criterion_notes)
        return base

    if reason == RetryReasonCode.MISSING_REQUIRED_FIELD:
        missing = params.get("missing_fields", [])
        return (
            f"The previous digest was missing required fields: {', '.join(missing)}. "
            "Ensure every required section is present and complete."
        )

    return ""
