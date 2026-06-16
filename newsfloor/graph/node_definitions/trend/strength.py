"""
nodes/trend/strength.py

Deterministic strength and decay math for TrendRecords.

No LLM, no I/O — pure functions that take a TrendRecord and return
updated values. Isolated here so the math is easy to find, read,
and unit test independently of the rest of the trend node.
"""

from __future__ import annotations
from datetime import datetime

from contracts.primitives import TrendStrength
from contracts.state import TrendRecord, current_week_id


def apply_strength_update(
    trend:              TrendRecord,
    was_reinforced:     bool,
    now:                str,
    archive_threshold:  float,
    decay_rate_per_day: float,
    boost_rate:         float,
) -> TrendRecord:
    """
    Returns a new TrendRecord with updated strength, band, and metadata.
    Does not write to DynamoDB — that is the caller's responsibility.

    Args:
        trend:              The existing TrendRecord to update.
        was_reinforced:     True if today's content confirmed this trend.
        now:                ISO 8601 timestamp string for updated_at fields.
        archive_threshold:  Strength below which the trend is archived.
        decay_rate_per_day: Strength lost per elapsed day without reinforcement.
        boost_rate:         Strength gained when reinforced (instantaneous).
    """
    elapsed_days   = _elapsed_days(trend.updated_at, now)
    new_strength   = _calculate_strength(trend.strength, was_reinforced, elapsed_days, decay_rate_per_day, boost_rate)
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


def _calculate_strength(
    current:            float,
    was_reinforced:      bool,
    elapsed_days:        float,
    decay_rate_per_day:  float,
    boost_rate:          float,
) -> float:
    """
    Applies boost or decay to a strength score.

    Boost:  +boost_rate when reinforced — a flat, instantaneous bump.
            Reinforcement is concrete same-day evidence, so it isn't
            time-scaled the way decay is.
    Decay:  -(elapsed_days * decay_rate_per_day) when not reinforced.
            Calendar-time-based rather than a flat amount per call —
            a trend's only chance to be reinforced is tied to which topic
            gets selected, and topic rotation guarantees long gaps between
            chances for any given trend. Keying decay to elapsed time
            instead of call count means those gaps don't read as "fading"
            on their own; only genuinely long stretches of silence do.
    Bounds: clamped to [0.0, 1.0]
    """
    if was_reinforced:
        return min(1.0, round(current + boost_rate, 4))
    decay_amount = max(0.0, elapsed_days) * decay_rate_per_day
    return max(0.0, round(current - decay_amount, 4))


def _to_band(strength: float) -> TrendStrength:
    """Maps a strength float to its human-readable band."""
    if strength >= 0.85:
        return TrendStrength.DOMINANT
    if strength >= 0.65:
        return TrendStrength.STRONG
    if strength >= 0.40:
        return TrendStrength.GROWING
    return TrendStrength.EMERGING


def _elapsed_days(earlier_iso: str, later_iso: str) -> float:
    """
    Returns elapsed days between two ISO timestamps.

    Strips tzinfo before diffing because the codebase mixes naive
    (`datetime.utcnow()`) and aware (`datetime.now(timezone.utc)`) ISO
    strings — comparing those directly raises. Clamped to >= 0 so clock
    skew or an out-of-order call never produces a strength *increase*
    via decay.
    """
    earlier = datetime.fromisoformat(earlier_iso).replace(tzinfo=None)
    later   = datetime.fromisoformat(later_iso).replace(tzinfo=None)
    return max(0.0, (later - earlier).total_seconds() / 86400)
