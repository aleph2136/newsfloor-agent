# tests/unit/test_strength.py
import pytest

from contracts.primitives import TrendStrength
from contracts.state import TrendRecord, current_week_id
from node_definitions.trend.strength import (
    _calculate_strength,
    _to_band,
    apply_strength_update,
)

NOW = "2026-05-22T12:00:00"
ARCHIVE_THRESHOLD = 0.1


@pytest.fixture
def base_trend():
    return TrendRecord(
        trend_id="llm-routing",
        name="LLM Routing",
        first_observed="2026-01-01T00:00:00",
        last_reinforced="2026-04-01T00:00:00",
        strength=0.5,
        strength_band=TrendStrength.GROWING,
        platform_relevance=0.7,
        key_signals=["router", "dispatch"],
        times_reinforced=2,
        evidence_weeks=["2026-W14"],
    )


# ---------------------------------------------------------------------------
# _calculate_strength
# ---------------------------------------------------------------------------

def test_calculate_strength_boost():
    assert _calculate_strength(0.5, was_reinforced=True) == pytest.approx(0.75)


def test_calculate_strength_decay():
    assert _calculate_strength(0.5, was_reinforced=False) == pytest.approx(0.35)


def test_calculate_strength_clamp_upper():
    assert _calculate_strength(0.9, was_reinforced=True) == pytest.approx(1.0)
    assert _calculate_strength(1.0, was_reinforced=True) == pytest.approx(1.0)


def test_calculate_strength_clamp_lower():
    assert _calculate_strength(0.1, was_reinforced=False) == pytest.approx(0.0)
    assert _calculate_strength(0.0, was_reinforced=False) == pytest.approx(0.0)


def test_calculate_strength_rounds_to_4_decimals():
    result = _calculate_strength(0.333, was_reinforced=True)
    assert result == round(result, 4)


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

def test_apply_strength_update_reinforced_updates_strength(base_trend):
    updated = apply_strength_update(base_trend, was_reinforced=True, now=NOW, archive_threshold=ARCHIVE_THRESHOLD)
    assert updated.strength == pytest.approx(0.75)


def test_apply_strength_update_reinforced_updates_band(base_trend):
    updated = apply_strength_update(base_trend, was_reinforced=True, now=NOW, archive_threshold=ARCHIVE_THRESHOLD)
    assert updated.strength_band == TrendStrength.STRONG


def test_apply_strength_update_reinforced_increments_times_reinforced(base_trend):
    updated = apply_strength_update(base_trend, was_reinforced=True, now=NOW, archive_threshold=ARCHIVE_THRESHOLD)
    assert updated.times_reinforced == base_trend.times_reinforced + 1


def test_apply_strength_update_reinforced_sets_last_reinforced(base_trend):
    updated = apply_strength_update(base_trend, was_reinforced=True, now=NOW, archive_threshold=ARCHIVE_THRESHOLD)
    assert updated.last_reinforced == NOW


def test_apply_strength_update_reinforced_not_archived(base_trend):
    updated = apply_strength_update(base_trend, was_reinforced=True, now=NOW, archive_threshold=ARCHIVE_THRESHOLD)
    assert updated.archived is False
    assert updated.archived_at == ""


def test_apply_strength_update_reinforced_adds_current_week_to_evidence(base_trend):
    updated = apply_strength_update(base_trend, was_reinforced=True, now=NOW, archive_threshold=ARCHIVE_THRESHOLD)
    assert current_week_id() in updated.evidence_weeks


def test_apply_strength_update_reinforced_preserves_existing_evidence_weeks(base_trend):
    updated = apply_strength_update(base_trend, was_reinforced=True, now=NOW, archive_threshold=ARCHIVE_THRESHOLD)
    assert "2026-W14" in updated.evidence_weeks


def test_apply_strength_update_reinforced_deduplicates_evidence_weeks():
    this_week = current_week_id()
    trend = TrendRecord(
        trend_id="t", name="T", first_observed="2026-01-01", last_reinforced="2026-01-01",
        strength=0.5, strength_band=TrendStrength.GROWING, platform_relevance=0.5,
        key_signals=["s"], evidence_weeks=[this_week],
    )
    updated = apply_strength_update(trend, was_reinforced=True, now=NOW, archive_threshold=ARCHIVE_THRESHOLD)
    assert updated.evidence_weeks.count(this_week) == 1


# ---------------------------------------------------------------------------
# apply_strength_update — not reinforced
# ---------------------------------------------------------------------------

def test_apply_strength_update_not_reinforced_updates_strength(base_trend):
    updated = apply_strength_update(base_trend, was_reinforced=False, now=NOW, archive_threshold=ARCHIVE_THRESHOLD)
    assert updated.strength == pytest.approx(0.35)


def test_apply_strength_update_not_reinforced_updates_band(base_trend):
    updated = apply_strength_update(base_trend, was_reinforced=False, now=NOW, archive_threshold=ARCHIVE_THRESHOLD)
    assert updated.strength_band == TrendStrength.EMERGING


def test_apply_strength_update_not_reinforced_does_not_change_times_reinforced(base_trend):
    updated = apply_strength_update(base_trend, was_reinforced=False, now=NOW, archive_threshold=ARCHIVE_THRESHOLD)
    assert updated.times_reinforced == base_trend.times_reinforced


def test_apply_strength_update_not_reinforced_preserves_last_reinforced(base_trend):
    updated = apply_strength_update(base_trend, was_reinforced=False, now=NOW, archive_threshold=ARCHIVE_THRESHOLD)
    assert updated.last_reinforced == base_trend.last_reinforced


def test_apply_strength_update_not_reinforced_does_not_change_evidence_weeks(base_trend):
    updated = apply_strength_update(base_trend, was_reinforced=False, now=NOW, archive_threshold=ARCHIVE_THRESHOLD)
    assert updated.evidence_weeks == base_trend.evidence_weeks


# ---------------------------------------------------------------------------
# apply_strength_update — archiving
# ---------------------------------------------------------------------------

def test_apply_strength_update_archives_when_strength_drops_below_threshold():
    trend = TrendRecord(
        trend_id="t", name="T", first_observed="2026-01-01", last_reinforced="2026-01-01",
        strength=0.15, strength_band=TrendStrength.EMERGING, platform_relevance=0.5,
        key_signals=["s"],
    )
    # 0.15 - 0.15 = 0.0, below threshold 0.1
    updated = apply_strength_update(trend, was_reinforced=False, now=NOW, archive_threshold=ARCHIVE_THRESHOLD)
    assert updated.archived is True
    assert updated.archived_at == NOW


def test_apply_strength_update_does_not_archive_when_above_threshold():
    trend = TrendRecord(
        trend_id="t", name="T", first_observed="2026-01-01", last_reinforced="2026-01-01",
        strength=0.3, strength_band=TrendStrength.EMERGING, platform_relevance=0.5,
        key_signals=["s"],
    )
    # 0.3 - 0.15 = 0.15, above threshold 0.1
    updated = apply_strength_update(trend, was_reinforced=False, now=NOW, archive_threshold=ARCHIVE_THRESHOLD)
    assert updated.archived is False
    assert updated.archived_at == ""


def test_apply_strength_update_sets_updated_at(base_trend):
    updated = apply_strength_update(base_trend, was_reinforced=True, now=NOW, archive_threshold=ARCHIVE_THRESHOLD)
    assert updated.updated_at == NOW


# ---------------------------------------------------------------------------
# apply_strength_update — immutability
# ---------------------------------------------------------------------------

def test_apply_strength_update_does_not_mutate_original(base_trend):
    original_strength = base_trend.strength
    original_times = base_trend.times_reinforced
    apply_strength_update(base_trend, was_reinforced=True, now=NOW, archive_threshold=ARCHIVE_THRESHOLD)
    assert base_trend.strength == original_strength
    assert base_trend.times_reinforced == original_times
