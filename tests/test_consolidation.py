"""Tests for memory consolidation system (v2)."""

from __future__ import annotations

import tempfile
from pathlib import Path

from agent_memory_rag.consolidation import (
    Insight,
    consolidate,
    discover_daily_notes,
    extract_events_from_note,
    find_matching_memory_entry,
    find_matching_memory_entry_with_dedup,
    _detect_category,
    _estimate_confidence,
    _has_negation,
    _compute_text_similarity,
    _find_stale_insights,
    _load_state,
)


def _create_temp_workspace(files: dict[str, str]) -> Path:
    """Create a temporary workspace with given files."""
    tmp = tempfile.mkdtemp()
    workspace = Path(tmp)
    for name, content in files.items():
        path = workspace / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return workspace


# === discover_daily_notes ===

def test_discover_daily_notes_finds_recent():
    workspace = _create_temp_workspace({
        "memory/2026-06-01.md": "# 2026-06-01\n## Test\nSome content here.",
        "memory/2026-06-02.md": "# 2026-06-02\n## Test\nMore content here.",
        "memory/old-note.md": "# Old\nNot a daily note.",
    })
    notes = discover_daily_notes(workspace / "memory", days_back=7)
    assert len(notes) == 2
    assert all("2026-06" in n.name for n in notes)


def test_discover_daily_notes_respects_after_date():
    workspace = _create_temp_workspace({
        "memory/2026-05-20.md": "# 2026-05-20\n## Test\nSome content.",
        "memory/2026-06-01.md": "# 2026-06-01\n## Test\nMore content.",
    })
    notes = discover_daily_notes(workspace / "memory", days_back=30, after_date="2026-05-25")
    assert len(notes) == 1
    assert notes[0].name == "2026-06-01.md"


# === extract_events_from_note ===

def test_extract_events_from_note():
    workspace = _create_temp_workspace({
        "memory/2026-06-01.md": """# 2026-06-01

## Project shipped

Shipped agent-memory-rag v0.1.0 to GitHub. MIT license, public repo.

## Bug fix

Fixed a critical bug in the search module. The issue was incorrect vector dimension handling.
""",
    })
    events = extract_events_from_note(workspace / "memory" / "2026-06-01.md")
    assert len(events) >= 1
    categories = [e["category"] for e in events]
    assert "project" in categories


# === _detect_category ===

def test_detect_category():
    assert _detect_category("Shipped the project to GitHub") == "project"
    assert _detect_category("Fixed a bug in the search") == "correction"
    assert _detect_category("I prefer Python over JavaScript") == "preference"
    assert _detect_category("Decided to use LanceDB") == "decision"


# === _estimate_confidence ===

def test_estimate_confidence_explicit():
    high = _estimate_confidence("Confirmed and verified that it works", "decision", False)
    low = _estimate_confidence("Maybe we could try this approach", "lesson", False)
    assert high > low


def test_estimate_confidence_negation_lowers():
    normal = _estimate_confidence("Decided to use SQLite", "decision", False)
    negated = _estimate_confidence("Decided not to use SQLite", "decision", True)
    assert negated < normal


# === _has_negation ===

def test_has_negation():
    assert _has_negation("We should not use this approach") == True
    assert _has_negation("Don't deploy to production") == True
    assert _has_negation("jangan pernah push ke main") == True
    assert _has_negation("Deploy the project") == False
    assert _has_negation("Shipped the feature") == False


# === _compute_text_similarity ===

def test_compute_text_similarity():
    sim_high = _compute_text_similarity(
        "Shipped agent-memory-rag to GitHub",
        "Shipped agent-memory-rag to GitHub repo"
    )
    sim_low = _compute_text_similarity(
        "Shipped agent-memory-rag to GitHub",
        "Fixed a bug in the search module"
    )
    assert sim_high > sim_low
    assert sim_high > 0.5


def test_compute_text_similarity_identical():
    assert _compute_text_similarity("hello world", "hello world") == 1.0


# === find_matching_memory_entry ===

def test_find_matching_memory_entry():
    memory = """# MEMORY.md

## Projects
- Shipped agent-memory-rag v0.1.0

## Technical Lessons
- LanceDB requires explicit schema
"""
    insight = Insight(
        category="project",
        text="Shipped agent-memory-rag v0.1.0 to GitHub",
        source_file="2026-06-01.md",
        source_date="2026-06-01",
    )
    match = find_matching_memory_entry(insight, memory)
    assert match is not None
    assert "agent-memory-rag" in match


def test_find_matching_memory_entry_no_match():
    memory = "# MEMORY.md\n\n## Projects\n- Something else entirely"
    insight = Insight(
        category="project",
        text="Completely new and unique insight about quantum computing",
        source_file="2026-06-01.md",
        source_date="2026-06-01",
    )
    match = find_matching_memory_entry(insight, memory)
    assert match is None


# === find_matching_memory_entry_with_dedup ===

def test_dedup_within_batch():
    insight1 = Insight(
        category="project",
        text="Shipped agent-memory-rag v0.1.0 to GitHub",
        source_file="2026-06-01.md",
        source_date="2026-06-01",
    )
    insight2 = Insight(
        category="project",
        text="Published agent-memory-rag v0.1.0 on GitHub",
        source_file="2026-06-02.md",
        source_date="2026-06-02",
    )
    # insight2 should be detected as duplicate of insight1
    match = find_matching_memory_entry_with_dedup(
        insight2, [insight1], "", dedup_threshold=0.4
    )
    assert match is not None


# === _find_stale_insights ===

def test_find_stale_insights():
    memory = """# MEMORY.md

## Auto-Consolidated (2026-05-01)
### Decisions Made
- ✅ Old decision that should be pruned
  _Source: 2026-05-01.md (2026-05-01) | confidence: 0.8_

## Auto-Consolidated (2026-06-01)
### Decisions Made
- ✅ Recent decision that should stay
  _Source: 2026-06-01.md (2026-06-01) | confidence: 0.9_
"""
    stale = _find_stale_insights(memory, max_age_days=15)
    # The May 1st insight should be stale (30+ days old)
    assert any("Old decision" in s for s in stale)
    # The June 1st insight should NOT be stale
    assert not any("Recent decision" in s for s in stale)


def test_find_stale_insights_low_confidence():
    memory = """# MEMORY.md

## Auto-Consolidated (2026-04-01)
### Lessons
- 📝 Maybe this could be useful but I'm not sure
  _Source: 2026-04-01.md (2026-04-01) | confidence: 0.3_
"""
    stale = _find_stale_insights(memory, max_age_days=90, min_confidence=0.5)
    assert any("confidence: 0.3" in s for s in stale)


# === _load_state ===

def test_load_state_default():
    workspace = _create_temp_workspace({})
    state = _load_state(workspace)
    assert state["last_consolidated"] is None
    assert state["archived"] == []


def test_load_state_existing():
    workspace = _create_temp_workspace({
        "memory/consolidation-state.json": '{"last_consolidated": "2026-06-01", "archived": ["2026-05-20.md"]}'
    })
    state = _load_state(workspace)
    assert state["last_consolidated"] == "2026-06-01"
    assert "2026-05-20.md" in state["archived"]


# === consolidate (integration) ===

def test_consolidate_dry_run():
    workspace = _create_temp_workspace({
        "memory/2026-06-01.md": """# 2026-06-01

## Shipped project

Shipped agent-memory-rag v0.1.0. This is a new project.
""",
        "MEMORY.md": "# MEMORY.md\n\n## Existing\n- Old entry",
    })
    result = consolidate(workspace, days_back=7, dry_run=True)
    assert result.notes_reviewed >= 1
    assert result.new_insights >= 1
    content = (workspace / "MEMORY.md").read_text()
    assert "Auto-Consolidated" not in content


def test_consolidate_apply():
    workspace = _create_temp_workspace({
        "memory/2026-06-01.md": """# 2026-06-01

## Important decision

Decided to use SQLite for access tracking in agent-memory-rag.
This is a key architectural decision.
""",
        "MEMORY.md": "# MEMORY.md\n\n## Existing\n- Old entry",
    })
    result = consolidate(workspace, days_back=7, dry_run=False)
    assert result.new_insights >= 1
    content = (workspace / "MEMORY.md").read_text()
    assert "Auto-Consolidated" in content


def test_consolidate_archives_notes():
    workspace = _create_temp_workspace({
        "memory/2026-06-01.md": "# 2026-06-01\n\n## Event\nShipped something new and exciting.",
        "MEMORY.md": "# MEMORY.md",
    })
    result = consolidate(workspace, days_back=7, dry_run=False, archive=True)
    assert result.notes_archived == 1
    # Note should be moved to .archive/
    assert not (workspace / "memory" / "2026-06-01.md").exists()
    assert (workspace / "memory" / ".archive" / "2026-06-01.md").exists()


def test_consolidate_incremental():
    workspace = _create_temp_workspace({
        "memory/2026-06-01.md": "# 2026-06-01\n\n## Shipped feature\nDeployed the new authentication system to production today.",
        "memory/2026-06-02.md": "# 2026-06-02\n\n## Fixed database\nCorrected a critical PostgreSQL connection pooling issue.",
        "MEMORY.md": "# MEMORY.md",
    })
    # First run — consolidate everything
    result1 = consolidate(workspace, days_back=7, dry_run=False, archive=False)
    assert result1.new_insights >= 2

    # Second run — should find nothing new (incremental)
    result2 = consolidate(workspace, days_back=7, dry_run=False, archive=False)
    assert result2.new_insights == 0


def test_consolidate_negation():
    workspace = _create_temp_workspace({
        "memory/2026-06-01.md": """# 2026-06-01

## Decision

Decided not to use MongoDB for this project. Too complex.
""",
        "MEMORY.md": "# MEMORY.md",
    })
    result = consolidate(workspace, days_back=7, dry_run=False)
    # Should detect negation
    content = (workspace / "MEMORY.md").read_text()
    assert "🚫" in content  # Negated insight prefix


def test_consolidate_pruning():
    workspace = _create_temp_workspace({
        "memory/2026-06-01.md": "# 2026-06-01\n\n## New\nBrand new insight here for today.",
        "MEMORY.md": """# MEMORY.md

## Auto-Consolidated (2026-05-01)
### Lessons
- 📝 Very uncertain maybe perhaps something
  _Source: old.md (2026-05-01) | confidence: 0.3_
""",
    })
    result = consolidate(workspace, days_back=7, dry_run=False, max_prune_age=15, min_prune_confidence=0.5, min_per_category=0)
    assert result.pruned >= 1
