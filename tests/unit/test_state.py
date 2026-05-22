# tests/unit/test_state.py
import re
import time
from datetime import datetime, timedelta

import pytest

from contracts.primitives import TrendStrength
from contracts.state import SourceRecord, TrendRecord, current_week_id, ttl_days
from state import merge_rework_counts


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_trend():
    return TrendRecord(
        trend_id="multi-agent-coordination",
        name="Multi-Agent Coordination",
        first_observed="2026-01-01",
        last_reinforced="2026-05-01",
        strength=0.5,
        strength_band=TrendStrength.GROWING,
        platform_relevance=0.8,
        key_signals=["orchestration", "crew routing"],
        times_reinforced=3,
    )


@pytest.fixture
def base_source():
    return SourceRecord(domain="simonwillison.net")

def test_merge_rework_counts_empty_existing():
    """Test that when existing is empty, the update dictionary is returned exactly."""
    existing = {}
    update = {"input_supervisor": 1}
    
    result = merge_rework_counts(existing, update)
    assert result == {"input_supervisor": 1}

def test_merge_rework_counts_combines_and_sums():
    """Test that existing keys are accumulated correctly."""
    existing = {"input_supervisor": 1}
    update = {"input_supervisor": 1, "output_supervisor": 1}
    
    result = merge_rework_counts(existing, update)
    assert result == {"input_supervisor": 2, "output_supervisor": 1}

def test_merge_rework_counts_does_not_mutate_inputs():
    """Verify that the reducer is a pure function and does not mutate the inputs."""
    existing = {"input_supervisor": 1}
    update = {"output_supervisor": 2}

    result = merge_rework_counts(existing, update)

    assert existing == {"input_supervisor": 1}
    assert update == {"output_supervisor": 2}
    assert result == {"input_supervisor": 1, "output_supervisor": 2}


# ---------------------------------------------------------------------------
# ttl_days
# ---------------------------------------------------------------------------

def test_ttl_days_returns_future_timestamp():
    # ttl_days uses datetime.utcnow().timestamp() which on some platforms
    # treats naive datetimes as local time — compare using the same call path.
    expected = int((datetime.utcnow() + timedelta(days=30)).timestamp())
    result = ttl_days(30)
    assert abs(result - expected) <= 2


def test_ttl_days_zero_is_approximately_now():
    expected = int(datetime.utcnow().timestamp())
    result = ttl_days(0)
    assert abs(result - expected) <= 2


def test_ttl_days_returns_int():
    assert isinstance(ttl_days(7), int)


# ---------------------------------------------------------------------------
# current_week_id
# ---------------------------------------------------------------------------

def test_current_week_id_format():
    assert re.match(r"^\d{4}-W\d{2}$", current_week_id())


def test_current_week_id_matches_current_utc_week():
    now = datetime.utcnow()
    expected = f"{now.year}-W{now.isocalendar()[1]:02d}"
    assert current_week_id() == expected


# ---------------------------------------------------------------------------
# TrendRecord.updated_strength
# ---------------------------------------------------------------------------

def test_trend_updated_strength_reinforced(base_trend):
    # 0.5 + 0.25 = 0.75
    assert base_trend.updated_strength(was_reinforced=True) == pytest.approx(0.75)


def test_trend_updated_strength_decay(base_trend):
    # 0.5 - 0.15 = 0.35
    assert base_trend.updated_strength(was_reinforced=False) == pytest.approx(0.35)


def test_trend_updated_strength_boost_clamped_at_one():
    trend = TrendRecord(
        trend_id="t", name="T", first_observed="2026-01-01", last_reinforced="2026-01-01",
        strength=0.9, strength_band=TrendStrength.DOMINANT, platform_relevance=0.5,
        key_signals=["signal"],
    )
    assert trend.updated_strength(was_reinforced=True) == pytest.approx(1.0)


def test_trend_updated_strength_decay_clamped_at_zero():
    trend = TrendRecord(
        trend_id="t", name="T", first_observed="2026-01-01", last_reinforced="2026-01-01",
        strength=0.1, strength_band=TrendStrength.EMERGING, platform_relevance=0.5,
        key_signals=["signal"],
    )
    assert trend.updated_strength(was_reinforced=False) == pytest.approx(0.0)


def test_trend_updated_strength_does_not_mutate(base_trend):
    original = base_trend.strength
    base_trend.updated_strength(was_reinforced=True)
    assert base_trend.strength == original


# ---------------------------------------------------------------------------
# TrendRecord.to_band
# ---------------------------------------------------------------------------

def test_to_band_dominant(base_trend):
    assert base_trend.to_band(0.85) == TrendStrength.DOMINANT
    assert base_trend.to_band(1.0) == TrendStrength.DOMINANT


def test_to_band_strong(base_trend):
    assert base_trend.to_band(0.65) == TrendStrength.STRONG
    assert base_trend.to_band(0.84) == TrendStrength.STRONG


def test_to_band_growing(base_trend):
    assert base_trend.to_band(0.40) == TrendStrength.GROWING
    assert base_trend.to_band(0.64) == TrendStrength.GROWING


def test_to_band_emerging(base_trend):
    assert base_trend.to_band(0.0) == TrendStrength.EMERGING
    assert base_trend.to_band(0.39) == TrendStrength.EMERGING


def test_to_band_boundary_dominant_strong(base_trend):
    assert base_trend.to_band(0.849) == TrendStrength.STRONG
    assert base_trend.to_band(0.85) == TrendStrength.DOMINANT


def test_to_band_boundary_growing_emerging(base_trend):
    assert base_trend.to_band(0.399) == TrendStrength.EMERGING
    assert base_trend.to_band(0.40) == TrendStrength.GROWING


# ---------------------------------------------------------------------------
# SourceRecord.updated_reputation
# ---------------------------------------------------------------------------

def test_source_updated_reputation_high_article(base_source):
    # 0.5 * 0.8 + 1.0 * 0.2 = 0.6
    assert base_source.updated_reputation(1.0) == pytest.approx(0.6)


def test_source_updated_reputation_low_article(base_source):
    # 0.5 * 0.8 + 0.0 * 0.2 = 0.4
    assert base_source.updated_reputation(0.0) == pytest.approx(0.4)


def test_source_updated_reputation_neutral_article_unchanged(base_source):
    # 0.5 * 0.8 + 0.5 * 0.2 = 0.5 (no change)
    assert base_source.updated_reputation(0.5) == pytest.approx(0.5)


def test_source_updated_reputation_rounds_to_4_decimals():
    source = SourceRecord(domain="example.com", reputation_score=0.3333)
    result = source.updated_reputation(0.7)
    assert result == round(result, 4)


def test_source_updated_reputation_does_not_mutate(base_source):
    original = base_source.reputation_score
    base_source.updated_reputation(1.0)
    assert base_source.reputation_score == original
