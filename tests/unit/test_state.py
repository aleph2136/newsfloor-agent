# tests/unit/test_state.py
import pytest
from state import merge_rework_counts

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
