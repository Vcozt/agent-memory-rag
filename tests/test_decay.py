"""Tests for memory decay system."""

from __future__ import annotations

import time
from agent_memory_rag.decay import (
    DecayConfig,
    calculate_decay,
    apply_decay_to_results,
    find_stale_chunks,
)


def test_calculate_decay_just_accessed():
    """Chunks accessed just now should have decay ~1.0."""
    now = time.time()
    assert calculate_decay(now, now) == 1.0


def test_calculate_decay_half_life():
    """At half-life, decay should be ~0.5."""
    now = time.time()
    half_life_days = 30
    days_30_ago = now - (half_life_days * 86400)
    decay = calculate_decay(days_30_ago, now, half_life_days=half_life_days)
    assert 0.49 <= decay <= 0.51


def test_calculate_decay_never_accessed():
    """Never accessed (0) should default to 0.5."""
    assert calculate_decay(0, time.time()) == 0.5


def test_calculate_decay_old_memory():
    """Old memories should have low decay but above minimum."""
    now = time.time()
    year_ago = now - (365 * 86400)
    decay = calculate_decay(year_ago, now)
    assert decay == 0.05  # Minimum floor


def test_calculate_decay_respects_minimum():
    """Decay should never go below min_decay_factor."""
    now = time.time()
    very_old = now - (1000 * 86400)
    decay = calculate_decay(very_old, now, min_decay_factor=0.1)
    assert decay == 0.1


def test_apply_decay_to_results_sorts_by_decayed_score():
    """Results should be re-sorted by decayed score (highest first)."""
    now = time.time()
    results = [
        {"chunk_id": "a", "score": 10.0, "last_accessed": now - 86400 * 60},  # 60 days old, lower similarity
        {"chunk_id": "b", "score": 20.0, "last_accessed": now},  # just accessed, higher similarity
    ]
    apply_decay_to_results(results, now=now)
    # b should be first (just accessed, high similarity)
    # a should be second (old, but lower raw similarity too)
    assert results[0]["chunk_id"] == "b"
    assert results[1]["chunk_id"] == "a"
    # Verify decay fields were added
    assert "decay_factor" in results[0]
    assert "similarity" in results[0]
    assert "decayed_score" in results[0]


def test_apply_decay_to_results_empty():
    """Empty results list should not crash."""
    results = []
    apply_decay_to_results(results)
    assert results == []


def test_find_stale_chunks():
    """Chunks below threshold should be identified as stale."""
    now = time.time()
    chunks = [
        {"chunk_id": "a", "last_accessed": now},  # Just accessed
        {"chunk_id": "b", "last_accessed": now - 86400 * 60},  # 60 days old
    ]
    stale = find_stale_chunks(chunks, threshold=0.3)
    # a has decay ~1.0 (not stale), b has decay ~0.25 (stale)
    assert len(stale) == 1
    assert stale[0]["chunk_id"] == "b"
