# tests/unit/test_strength.py
import pytest

from config import settings
from contracts.primitives import TrendStrength
from contracts.state import TrendRecord, current_week_id
from node_definitions.trend.strength import (
    _calculate_strength,
    _elapsed_days,
    _to_band,
    apply_strength_update,
)

NOW = "2026-05-22T12:00:00"
ARCHIVE_THRESHOLD = 0.1
DECAY_RATE = 0.01   # per day — matches config.py's trend_decay_rate_per_day default
BOOST_RATE = 0.25   # matches config.py's trend_boost_rate default


@pytest.fixture
def base_trend():
    return TrendRecord(
        trend_id="llm-routing",
        name="LLM Routing",
        first_observed="2026-01-01T00:00:00",
        last_reinforced="2026-04-01T00:00:00",
        # 30 days before NOW — one full topic_recency_window gap, the
        # scenario this whole fix exists for.
        updated_at="2026-04-22T12:00:00",
        strength=0.5,
        strength_band=TrendStrength.GROWING,
        platform_relevance=0.7,
        key_signals=["router", "dispatch"],
        times_reinforced=2,
        evidence_weeks=["2026-W14"],
    )


# ---------------------------------------------------------------------------
# _elapsed_days
# ---------------------------------------------------------------------------

def test_elapsed_days_basic_gap():
    assert _elapsed_days("2026-05-01T00:00:00", "2026-05-31T00:00:00") == pytest.approx(30.0)


def test_elapsed_days_zero_gap():
    assert _elapsed_days(NOW, NOW) == pytest.approx(0.0)


def test_elapsed_days_negative_gap_clamps_to_zero():
    # `later` before `earlier` — clock skew or out-of-order call.
    assert _elapsed_days("2026-05-31T00:00:00", "2026-05-01T00:00:00") == pytest.approx(0.0)


def test_elapsed_days_handles_mixed_naive_and_aware_strings():
    # datetime.utcnow().isoformat() (naive) vs datetime.now(timezone.utc).isoformat() (aware)
    naive  = "2026-05-01T00:00:00"
    aware  = "2026-05-02T00:00:00+00:00"
    assert _elapsed_days(naive, aware) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _calculate_strength
# ---------------------------------------------------------------------------

def test_calculate_strength_boost_ignores_elapsed_days():
    # Boost is a flat, instantaneous bump — elapsed time shouldn't matter.
    result = _calculate_strength(0.5, was_reinforced=True, elapsed_days=45, decay_rate_per_day=DECAY_RATE, boost_rate=BOOST_RATE)
    assert result == pytest.approx(0.75)


def test_calculate_strength_decay_scales_with_elapsed_days():
    # 0.5 - (30 * 0.01) = 0.2
    result = _calculate_strength(0.5, was_reinforced=False, elapsed_days=30, decay_rate_per_day=DECAY_RATE, boost_rate=BOOST_RATE)
    assert result == pytest.approx(0.2)


def test_calculate_strength_decay_zero_elapsed_days_is_a_no_op():
    result = _calculate_strength(0.5, was_reinforced=False, elapsed_days=0, decay_rate_per_day=DECAY_RATE, boost_rate=BOOST_RATE)
    assert result == pytest.approx(0.5)


def test_calculate_strength_decay_fractional_day():
    # 0.5 - (0.5 * 0.01) = 0.495
    result = _calculate_strength(0.5, was_reinforced=False, elapsed_days=0.5, decay_rate_per_day=DECAY_RATE, boost_rate=BOOST_RATE)
    assert result == pytest.approx(0.495)


def test_calculate_strength_clamp_upper():
    assert _calculate_strength(0.9, was_reinforced=True, elapsed_days=0, decay_rate_per_day=DECAY_RATE, boost_rate=BOOST_RATE) == pytest.approx(1.0)
    assert _calculate_strength(1.0, was_reinforced=True, elapsed_days=0, decay_rate_per_day=DECAY_RATE, boost_rate=BOOST_RATE) == pytest.approx(1.0)


def test_calculate_strength_clamp_lower():
    assert _calculate_strength(0.1, was_reinforced=False, elapsed_days=200, decay_rate_per_day=DECAY_RATE, boost_rate=BOOST_RATE) == pytest.approx(0.0)
    assert _calculate_strength(0.0, was_reinforced=False, elapsed_days=30, decay_rate_per_day=DECAY_RATE, boost_rate=BOOST_RATE) == pytest.approx(0.0)


def test_calculate_strength_rounds_to_4_decimals():
    result = _calculate_strength(0.333, was_reinforced=True, elapsed_days=0, decay_rate_per_day=DECAY_RATE, boost_rate=BOOST_RATE)
    assert result == round(result, 4)


# ---------------------------------------------------------------------------
# Decay calibration — automates the manual sanity check used to pick
# settings.trend_decay_rate_per_day. These read the *real* settings (not
# the local DECAY_RATE/BOOST_RATE constants used above) so a future change
# to config.py's defaults is caught here instead of silently drifting away
# from the fade timeline documented in config.py's "Trend node" comment.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "starting_strength, idle_days, expected_strength, scenario",
    [
        (1.0,  30, 0.7, "DOMINANT trend survives one full topic_recency_window (30 days) idle comfortably"),
        (1.0,  90, 0.1, "a never-reinforced trend fully decays to exactly the archive threshold after 90 days"),
        (0.25, 30, 0.0, "a brand-new, single-confirmation EMERGING trend left idle for one rotation window fades out entirely"),
    ],
)
def test_decay_calibration_matches_designed_timeline(starting_strength, idle_days, expected_strength, scenario):
    result = _calculate_strength(
        starting_strength,
        was_reinforced=False,
        elapsed_days=idle_days,
        decay_rate_per_day=settings.trend_decay_rate_per_day,
        boost_rate=settings.trend_boost_rate,
    )
    assert result == pytest.approx(expected_strength), scenario


def test_decay_calibration_strong_trend_stays_active_through_one_rotation_window():
    """
    A STRONG trend (0.65) idle for one full topic_recency_window (30 days)
    must stay above trend_active_min_strength — otherwise topic rotation
    alone (not genuine fading) would drop it out of active_trends before
    its own excluded topic is even eligible again.
    """
    result = _calculate_strength(
        0.65,
        was_reinforced=False,
        elapsed_days=30,
        decay_rate_per_day=settings.trend_decay_rate_per_day,
        boost_rate=settings.trend_boost_rate,
    )
    assert result > settings.trend_active_min_strength


def test_decay_calibration_archive_threshold_reached_at_designed_fade_days():
    """
    Rate-agnostic version of the 90-day check above: derives the fade
    window from the settings themselves rather than hardcoding 90, so this
    keeps passing (and keeps meaning the same thing) even if the decay
    rate or archive threshold are deliberately retuned later.
    """
    fade_days = (1.0 - settings.trend_archive_threshold) / settings.trend_decay_rate_per_day
    result = _calculate_strength(
        1.0,
        was_reinforced=False,
        elapsed_days=fade_days,
        decay_rate_per_day=settings.trend_decay_rate_per_day,
        boost_rate=settings.trend_boost_rate,
    )
    assert result == pytest.approx(settings.trend_archive_threshold)


# ---------------------------------------------------------------------------
# _to_band
# ---------------------------------------------------------------------------

def test_to_band_dominant():
    assert _to_band(0.85) == TrendStrength.DOMINANT
    assert _to_band(1.0) == TrendStrength.DOMINANT


def test_to_band_strong():
    assert _to_band(0.65) == TrendStrength.STRONG
    assert _to_band(0.849) == TrendStrength.STRONG


def test_to_band_growing():
    assert _to_band(0.40) == TrendStrength.GROWING
    assert _to_band(0.649) == TrendStrength.GROWING


def test_to_band_emerging():
    assert _to_band(0.0) == TrendStrength.EMERGING
    assert _to_band(0.399) == TrendStrength.EMERGING


def test_to_band_boundary_dominant_strong():
    assert _to_band(0.849) == TrendStrength.STRONG
    assert _to_band(0.85) == TrendStrength.DOMINANT


def test_to_band_boundary_strong_growing():
    assert _to_band(0.649) == TrendStrength.GROWING
    assert _to_band(0.65) == TrendStrength.STRONG


def test_to_band_boundary_growing_emerging():
    assert _to_band(0.399) == TrendStrength.EMERGING
    assert _to_band(0.40) == TrendStrength.GROWING


# ---------------------------------------------------------------------------
# apply_strength_update — reinforced
# ---------------------------------------------------------------------------

def _apply(trend, was_reinforced, now=NOW):
    return apply_strength_update(
        trend,
        was_reinforced=was_reinforced,
        now=now,
        archive_threshold=ARCHIVE_THRESHOLD,
        decay_rate_per_day=DECAY_RATE,
        boost_rate=BOOST_RATE,
    )


def test_apply_strength_update_reinforced_updates_strength(base_trend):
    updated = _apply(base_trend, was_reinforced=True)
    assert updated.strength == pytest.approx(0.75)


def test_apply_strength_update_reinforced_updates_band(base_trend):
    updated = _apply(base_trend, was_reinforced=True)
    assert updated.strength_band == TrendStrength.STRONG


def test_apply_strength_update_reinforced_increments_times_reinforced(base_trend):
    updated = _apply(base_trend, was_reinforced=True)
    assert updated.times_reinforced == base_trend.times_reinforced + 1


def test_apply_strength_update_reinforced_sets_last_reinforced(base_trend):
    updated = _apply(base_trend, was_reinforced=True)
    assert updated.last_reinforced == NOW


def test_apply_strength_update_reinforced_not_archived(base_trend):
    updated = _apply(base_trend, was_reinforced=True)
    assert updated.archived is False
    assert updated.archived_at == ""


def test_apply_strength_update_reinforced_adds_current_week_to_evidence(base_trend):
    updated = _apply(base_trend, was_reinforced=True)
    assert current_week_id() in updated.evidence_weeks


def test_apply_strength_update_reinforced_preserves_existing_evidence_weeks(base_trend):
    updated = _apply(base_trend, was_reinforced=True)
    assert "2026-W14" in updated.evidence_weeks


def test_apply_strength_update_reinforced_deduplicates_evidence_weeks():
    this_week = current_week_id()
    trend = TrendRecord(
        trend_id="t", name="T", first_observed="2026-01-01", last_reinforced="2026-01-01",
        strength=0.5, strength_band=TrendStrength.GROWING, platform_relevance=0.5,
        key_signals=["s"], evidence_weeks=[this_week],
    )
    updated = _apply(trend, was_reinforced=True)
    assert updated.evidence_weeks.count(this_week) == 1


# ---------------------------------------------------------------------------
# apply_strength_update — not reinforced (calendar-time-based decay)
# ---------------------------------------------------------------------------

def test_apply_strength_update_not_reinforced_updates_strength(base_trend):
    # base_trend.updated_at is 30 days before NOW: 0.5 - (30 * 0.01) = 0.2
    updated = _apply(base_trend, was_reinforced=False)
    assert updated.strength == pytest.approx(0.2)


def test_apply_strength_update_not_reinforced_updates_band(base_trend):
    updated = _apply(base_trend, was_reinforced=False)
    assert updated.strength_band == TrendStrength.EMERGING


def test_apply_strength_update_not_reinforced_does_not_change_times_reinforced(base_trend):
    updated = _apply(base_trend, was_reinforced=False)
    assert updated.times_reinforced == base_trend.times_reinforced


def test_apply_strength_update_not_reinforced_preserves_last_reinforced(base_trend):
    updated = _apply(base_trend, was_reinforced=False)
    assert updated.last_reinforced == base_trend.last_reinforced


def test_apply_strength_update_not_reinforced_does_not_change_evidence_weeks(base_trend):
    updated = _apply(base_trend, was_reinforced=False)
    assert updated.evidence_weeks == base_trend.evidence_weeks


def test_apply_strength_update_not_reinforced_zero_elapsed_time_is_a_no_op(base_trend):
    # Same instant as the trend's last update — no time has passed, so no decay.
    updated = _apply(base_trend, was_reinforced=False, now=base_trend.updated_at)
    assert updated.strength == pytest.approx(base_trend.strength)


# ---------------------------------------------------------------------------
# apply_strength_update — archiving
# ---------------------------------------------------------------------------

def test_apply_strength_update_archives_when_strength_drops_below_threshold():
    trend = TrendRecord(
        trend_id="t", name="T", first_observed="2026-01-01", last_reinforced="2026-01-01",
        updated_at="2026-04-22T12:00:00",  # 30 days before NOW
        strength=0.35, strength_band=TrendStrength.GROWING, platform_relevance=0.5,
        key_signals=["s"],
    )
    # 0.35 - (30 * 0.01) = 0.05, below threshold 0.1
    updated = _apply(trend, was_reinforced=False)
    assert updated.archived is True
    assert updated.archived_at == NOW


def test_apply_strength_update_does_not_archive_when_above_threshold():
    trend = TrendRecord(
        trend_id="t", name="T", first_observed="2026-01-01", last_reinforced="2026-01-01",
        updated_at="2026-04-22T12:00:00",  # 30 days before NOW
        strength=0.5, strength_band=TrendStrength.GROWING, platform_relevance=0.5,
        key_signals=["s"],
    )
    # 0.5 - (30 * 0.01) = 0.2, above threshold 0.1
    updated = _apply(trend, was_reinforced=False)
    assert updated.archived is False
    assert updated.archived_at == ""


def test_apply_strength_update_archives_a_never_reinforced_dominant_trend_after_90_days():
    trend = TrendRecord(
        trend_id="t", name="T", first_observed="2026-01-01", last_reinforced="2026-01-01",
        updated_at="2026-02-21T12:00:00",  # 90 days before NOW
        strength=1.0, strength_band=TrendStrength.DOMINANT, platform_relevance=0.5,
        key_signals=["s"],
    )
    # 1.0 - (90 * 0.01) = 0.1 — at the threshold, not below it, so not yet archived.
    updated = _apply(trend, was_reinforced=False)
    assert updated.strength == pytest.approx(0.1)
    assert updated.archived is False


def test_apply_strength_update_sets_updated_at(base_trend):
    updated = _apply(base_trend, was_reinforced=True)
    assert updated.updated_at == NOW


# ---------------------------------------------------------------------------
# apply_strength_update — immutability
# ---------------------------------------------------------------------------

def test_apply_strength_update_does_not_mutate_original(base_trend):
    original_strength = base_trend.strength
    original_times = base_trend.times_reinforced
    _apply(base_trend, was_reinforced=True)
    assert base_trend.strength == original_strength
    assert base_trend.times_reinforced == original_times
