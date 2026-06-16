# tests/unit/test_state.py
import re
import time
from datetime import datetime, timedelta

import pytest

from contracts.state import RunRecord, SourceRecord, current_week_id, ttl_days
from state import merge_rework_counts


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# RunRecord — article_ids_used field
# ---------------------------------------------------------------------------

def test_run_record_article_ids_used_defaults_to_empty():
    record = RunRecord(run_id="2026-06-04")
    assert record.article_ids_used == []


def test_run_record_article_ids_used_accepts_list():
    record = RunRecord(run_id="2026-06-04", article_ids_used=["abc123", "def456"])
    assert record.article_ids_used == ["abc123", "def456"]


def test_run_record_article_ids_used_roundtrips_through_dict():
    record = RunRecord(run_id="2026-06-04", article_ids_used=["aaa", "bbb"])
    restored = RunRecord(**record.model_dump())
    assert restored.article_ids_used == ["aaa", "bbb"]


# ---------------------------------------------------------------------------
# SourceRecord — last_contributed_date field
# ---------------------------------------------------------------------------

def test_source_last_contributed_date_defaults_to_empty(base_source):
    assert base_source.last_contributed_date == ""


def test_source_last_contributed_date_accepts_iso_date():
    source = SourceRecord(domain="example.com", last_contributed_date="2026-06-04")
    assert source.last_contributed_date == "2026-06-04"


def test_source_last_contributed_date_roundtrips_through_model_copy():
    source = SourceRecord(domain="example.com")
    updated = source.model_copy(update={"last_contributed_date": "2026-06-04"})
    assert updated.last_contributed_date == "2026-06-04"
    assert source.last_contributed_date == ""  # original unchanged
