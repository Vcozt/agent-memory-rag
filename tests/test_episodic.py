"""Tests for episodic tagging system."""

from __future__ import annotations

from agent_memory_rag.episodic import (
    EpisodicTag,
    detect_tags_from_content,
    detect_tags_from_source,
    generate_tags,
    tags_to_json,
    tags_from_json,
    filter_by_tags,
    format_tags_compact,
    format_tags_grouped,
)


def test_detect_tags_from_content_topic():
    """Should detect topic tags from content."""
    tags = detect_tags_from_content("Deployed the agent to production server")
    tag_values = [t.value for t in tags]
    assert "agent" in tag_values
    assert "infra" in tag_values


def test_detect_tags_from_content_context():
    """Should detect context tags from content."""
    tags = detect_tags_from_content("Fixed a critical bug in the search module")
    tag_values = [t.value for t in tags]
    assert "debugging" in tag_values


def test_detect_tags_from_content_shipping():
    """Should detect shipping context."""
    tags = detect_tags_from_content("Shipped the new feature to production")
    tag_values = [t.value for t in tags]
    assert "shipping" in tag_values


def test_detect_tags_from_source_daily_note():
    """Should detect daily note source type."""
    tags = detect_tags_from_source("memory/2026-06-01.md")
    categories = [t.category for t in tags]
    assert "source_type" in categories
    assert "recency" in categories


def test_detect_tags_from_source_config():
    """Should detect config source type."""
    tags = detect_tags_from_source("SOUL.md")
    tag_values = [t.value for t in tags]
    assert "config" in tag_values


def test_generate_tags():
    """Should generate combined tags from source and content."""
    tags = generate_tags("memory/2026-06-01.md", "Shipped agent-memory-rag to GitHub")
    tag_map = {t.category: t.value for t in tags}
    assert tag_map.get("source_type") == "daily_note"
    assert "project" in [t.value for t in tags]


def test_tags_roundtrip():
    """Tags should survive JSON serialization roundtrip."""
    original = [
        EpisodicTag("topic", "agent"),
        EpisodicTag("context", "debugging"),
        EpisodicTag("recency", "today"),
    ]
    json_str = tags_to_json(original)
    restored = tags_from_json(json_str)
    assert len(restored) == 3
    assert restored[0].category == "topic"
    assert restored[0].value == "agent"


def test_tags_from_json_invalid():
    """Invalid JSON should return empty list."""
    assert tags_from_json("not json") == []
    assert tags_from_json("") == []


def test_filter_by_tags_include():
    """Include filter should match chunks with at least one tag per category."""
    tags = [
        EpisodicTag("topic", "agent"),
        EpisodicTag("context", "debugging"),
    ]
    assert filter_by_tags(tags, include={"topic": ["agent"]})
    assert filter_by_tags(tags, include={"topic": ["agent", "project"]})
    assert not filter_by_tags(tags, include={"topic": ["nonexistent"]})


def test_filter_by_tags_exclude():
    """Exclude filter should reject chunks with any matching tag."""
    tags = [
        EpisodicTag("topic", "agent"),
        EpisodicTag("source_type", "daily_note"),
    ]
    # Should reject because it has source_type:daily_note
    assert not filter_by_tags(tags, exclude={"source_type": ["daily_note"]})
    # Should pass because it doesn't have source_type:config
    assert filter_by_tags(tags, exclude={"source_type": ["config"]})


def test_filter_by_tags_combined():
    """Include + exclude should work together."""
    tags = [
        EpisodicTag("topic", "agent"),
        EpisodicTag("source_type", "daily_note"),
    ]
    # Must have topic:agent AND must NOT be daily_note
    assert not filter_by_tags(tags, include={"topic": ["agent"]}, exclude={"source_type": ["daily_note"]})
    # Must have topic:agent AND must NOT be config
    assert filter_by_tags(tags, include={"topic": ["agent"]}, exclude={"source_type": ["config"]})


def test_format_tags_compact():
    """Should format tags as compact string."""
    tags = [EpisodicTag("topic", "agent"), EpisodicTag("context", "debugging")]
    result = format_tags_compact(tags)
    assert "topic:agent" in result
    assert "context:debugging" in result


def test_format_tags_grouped():
    """Should group tags by category."""
    tags = [
        EpisodicTag("topic", "agent"),
        EpisodicTag("topic", "memory"),
        EpisodicTag("context", "debugging"),
    ]
    grouped = format_tags_grouped(tags)
    assert "topic" in grouped
    assert "agent" in grouped["topic"]
    assert "memory" in grouped["topic"]
    assert "context" in grouped
