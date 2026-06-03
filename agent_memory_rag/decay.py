"""Memory decay system — exponential decay on unused chunks with safety protections.

Core idea: chunks that haven't been accessed in a while get lower priority.
Protected tags (security, decision) get slower decay to prevent data loss.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field


@dataclass
class DecayConfig:
    """Configuration for memory decay."""

    # Half-life in days: after this many days without access, score is halved
    half_life_days: float = 30.0

    # Minimum decay factor: prevents very old memories from going to zero
    min_decay_factor: float = 0.05

    # Cleanup threshold: chunks below this decay factor are candidates for removal
    cleanup_threshold: float = 0.1

    # Whether decay is enabled (can be toggled off)
    enabled: bool = True

    # Tags that get slower decay (longer half-life)
    protected_tags: list[str] = field(default_factory=lambda: [
        "topic:security",
        "topic:agent",
        "context:decision",
        "source_type:config",
    ])

    # Half-life multiplier for protected chunks (2x = twice as slow)
    protected_half_life_multiplier: float = 2.0

    # Minimum decay factor for protected chunks (never goes below this)
    protected_min_decay: float = 0.2


def calculate_decay(
    last_accessed: float,
    now: float | None = None,
    half_life_days: float = 30.0,
    min_decay_factor: float = 0.05,
    is_protected: bool = False,
    protected_multiplier: float = 2.0,
    protected_min: float = 0.2,
) -> float:
    """Calculate decay multiplier for a chunk based on last access time.

    Protected chunks decay slower and have a higher floor.

    Args:
        last_accessed: Unix timestamp of last access
        now: Current time (defaults to time.time())
        half_life_days: Days until score halves
        min_decay_factor: Minimum allowed factor
        is_protected: Whether this chunk has protected tags
        protected_multiplier: Half-life multiplier for protected chunks
        protected_min: Minimum decay for protected chunks

    Returns:
        Multiplier between min_decay_factor and 1.0
    """
    if now is None:
        now = time.time()

    if last_accessed <= 0:
        base = 0.5
    else:
        days_since = (now - last_accessed) / 86400.0
        if days_since <= 0:
            return 1.0
        effective_half_life = half_life_days * (protected_multiplier if is_protected else 1.0)
        base = math.pow(2, -days_since / effective_half_life)

    floor = protected_min if is_protected else min_decay_factor
    return max(floor, base)


def is_protected_chunk(tags: list[str], protected_tags: list[str]) -> bool:
    """Check if a chunk has any protected tags."""
    if not protected_tags or not tags:
        return False
    tag_set = set(tags)
    for pt in protected_tags:
        if pt in tag_set:
            return True
    return False


def apply_decay_to_results(
    results: list[dict],
    now: float | None = None,
    half_life_days: float = 30.0,
    min_decay_factor: float = 0.05,
    protected_tags: list[str] | None = None,
    protected_multiplier: float = 2.0,
    protected_min: float = 0.2,
) -> list[dict]:
    """Apply decay factor to search results in-place.

    Respects protected tags — chunks with protected tags decay slower.
    """
    if now is None:
        now = time.time()

    for r in results:
        last_accessed = r.get("last_accessed", 0)
        original_score = r.get("score", 0.0)
        chunk_tags = r.get("tags", [])

        similarity = 1.0 / (1.0 + original_score) if original_score >= 0 else 0.0
        protected = is_protected_chunk(chunk_tags, protected_tags or [])

        decay = calculate_decay(
            last_accessed, now, half_life_days, min_decay_factor,
            is_protected=protected,
            protected_multiplier=protected_multiplier,
            protected_min=protected_min,
        )

        r["decay_factor"] = round(decay, 4)
        r["original_score"] = round(original_score, 4)
        r["similarity"] = round(similarity, 4)
        r["decayed_score"] = round(similarity * decay, 4)
        r["is_protected"] = protected

    results.sort(key=lambda x: x.get("decayed_score", 0), reverse=True)
    return results


def find_stale_chunks(
    chunks: list[dict],
    now: float | None = None,
    half_life_days: float = 30.0,
    threshold: float = 0.1,
    protected_tags: list[str] | None = None,
    protected_multiplier: float = 2.0,
    protected_min: float = 0.2,
) -> list[dict]:
    """Find chunks below decay threshold (candidates for cleanup).

    Protected chunks use higher thresholds and won't be marked stale
    if they're above protected_min.
    """
    if now is None:
        now = time.time()

    stale = []
    for chunk in chunks:
        last_accessed = chunk.get("last_accessed", 0)
        chunk_tags = chunk.get("tags", [])
        protected = is_protected_chunk(chunk_tags, protected_tags or [])

        decay = calculate_decay(
            last_accessed, now, half_life_days,
            is_protected=protected,
            protected_multiplier=protected_multiplier,
            protected_min=protected_min,
        )

        # Protected chunks only marked stale if below their specific minimum
        effective_threshold = protected_min if protected else threshold
        if decay < effective_threshold:
            stale.append({**chunk, "decay_factor": round(decay, 4), "is_protected": protected})

    return stale
