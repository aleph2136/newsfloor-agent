"""
nodes/trend/strength.py

Deterministic strength and decay math for TrendRecords.

No LLM, no I/O — pure functions that take a TrendRecord and return
updated values. Isolated here so the math is easy to find, read,
and unit test independently of the rest of the trend node.
"""

from __future__ import annotations
from datetime import datetime, timezone

from contracts.primitives import TrendStrength
from contracts.state import TrendRecord, current_week_id


def apply_strength_update(
    trend:          TrendRecord,
    was_reinforced: bool,
    now:            str,
    archive_threshold: float,
) -> TrendRecord:
    """
    Returns a new TrendRecord with updated strength, band, and metadata.
    Does not write to DynamoDB — that is the caller's responsibility.

    Args:
        trend:              The existing TrendRecord to update.
        was_reinforced:     True if today's content confirmed this trend.
        now:                ISO 8601 timestamp string for updated_at fields.
        archive_threshold:  Strength below which the trend is archived.
    """
    new_strength   = _calculate_strength(trend.strength, was_reinforced)
    new_band       = _to_band(new_strength)
    should_archive = new_strength < archive_threshold

    return trend.model_copy(update={
        "strength":         new_strength,
        "strength_band":    new_band,
        "archived":         should_archive,
        "archived_at":      now if should_archive else "",
        "last_reinforced":  now if was_reinforced else trend.last_reinforced,
        "times_reinforced": trend.times_reinforced + (1 if was_reinforced else 0),
        "evidence_weeks": (
            list({*trend.evidence_weeks, current_week_id()})
            if was_reinforced else trend.evidence_weeks
        ),
        "updated_at": now,
    })


def _calculate_strength(current: float, was_reinforced: bool) -> float:
    """
    Applies boost or decay to a strength score.

    Boost:  +0.25 when reinforced — meaningful signal, not a small nudge
    Decay:  -0.15 when not reinforced — gradual fade, not sudden drop
    Bounds: clamped to [0.0, 1.0]

    At these rates a new trend (0.25) needs ~3 reinforcements to reach
    STRONG (0.65), and a STRONG trend survives ~5 non-reinforced runs
    before archiving below 0.1.
    """
    BOOST = 0.25
    DECAY = 0.15

    if was_reinforced:
        return min(1.0, round(current + BOOST, 4))
    return max(0.0, round(current - DECAY, 4))


def _to_band(strength: float) -> TrendStrength:
    """Maps a strength float to its human-readable band."""
    if strength >= 0.85:
        return TrendStrength.DOMINANT
    if strength >= 0.65:
        return TrendStrength.STRONG
    if strength >= 0.40:
        return TrendStrength.GROWING
    return TrendStrength.EMERGING