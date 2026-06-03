"""Episodic tagging — context-aware memory tagging and retrieval.

Adds context tags to memory chunks so retrieval can match on "why"
not just "what". Tags are auto-generated during ingest and can be
filtered during search.

Tag types:
- source_type: daily_note, project_doc, config, decision_log
- topic: tech, infra, agent, memory, project, security
- context: debugging, shipping, planning, learning, debugging
- recency: today, this_week, this_month, older
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


@dataclass
class EpisodicTag:
    """A context tag for a memory chunk."""

    category: str  # source_type, topic, context, recency
    value: str
    confidence: float = 1.0

    def to_string(self) -> str:
        return f"{self.category}:{self.value}"


# Topic detection patterns
TOPIC_PATTERNS = {
    "tech": r"(python|node|npm|docker|nginx|lancedb|sqlite|github|api|httpx|fastembed|pyarrow)",
    "infra": r"(server|vps|deploy|ssh|systemd|cron|tunnel|cloudflare|nginx|certbot)",
    "agent": r"(agent|openclaw|kiro|claude|llm|model|embedding|rag|memory|consolidat)",
    "memory": r"(memory|recall|search|embed|vector|lance|chunk|ingest|decay|episod)",
    "project": r"(project|repo|release|ship|v\d|version|published|deployed|built|created)",
    "security": r"(credential|token|key|password|auth|security|encrypt|secret)",
    "devops": r"(ci|cd|github.action|workflow|pipeline|test|lint|build)",
}

# Context detection patterns
CONTEXT_PATTERNS = {
    "debugging": r"(bug|fix|error|issue|broken|failed|debug|crash|traceback|exception)",
    "shipping": r"(shipped|deployed|released|published|launch|done|completed|merged)",
    "planning": r"(plan|design|architecture|decided|chose|approach|strategy|rfc)",
    "learning": r"(learned|realized|discovered|insight|turns out|found out|noticed)",
    "configuring": r"(config|setup|install|configure|setting|option|env|environment)",
    "reviewing": r"(review|checked|verified|tested|confirmed|validated|audited)",
}

# Source type detection
SOURCE_TYPE_PATTERNS = {
    "daily_note": r"^\d{4}-\d{2}-\d{2}\.md$",
    "project_doc": r"(README|CHANGELOG|CONTRIBUTING|LICENSE|PRD)\.md$",
    "config": r"(TOOLS|SOUL|USER|AGENTS|IDENTITY|HEARTBEAT)\.md$",
    "memory": r"^memory/.*\.md$",
}


def detect_tags_from_content(text: str) -> list[EpisodicTag]:
    """Auto-detect tags from chunk content."""
    tags = []
    text_lower = text.lower()

    # Topic tags
    for topic, pattern in TOPIC_PATTERNS.items():
        if re.search(pattern, text_lower):
            tags.append(EpisodicTag("topic", topic))

    # Context tags
    for context, pattern in CONTEXT_PATTERNS.items():
        if re.search(pattern, text_lower):
            tags.append(EpisodicTag("context", context))

    return tags


def detect_tags_from_source(source_path: str) -> list[EpisodicTag]:
    """Auto-detect tags from source file path/name."""
    tags = []
    filename = Path(source_path).name

    # Source type
    for source_type, pattern in SOURCE_TYPE_PATTERNS.items():
        if re.search(pattern, filename):
            tags.append(EpisodicTag("source_type", source_type))
            break

    # Recency from date in filename
    date_match = re.match(r"(\d{4}-\d{2}-\d{2})", filename)
    if date_match:
        try:
            file_date = datetime.strptime(date_match.group(1), "%Y-%m-%d")
            now = datetime.now()
            delta = now - file_date

            if delta.days == 0:
                tags.append(EpisodicTag("recency", "today"))
            elif delta.days <= 7:
                tags.append(EpisodicTag("recency", "this_week"))
            elif delta.days <= 30:
                tags.append(EpisodicTag("recency", "this_month"))
            else:
                tags.append(EpisodicTag("recency", "older"))
        except ValueError:
            pass

    return tags


def generate_tags(source_path: str, content: str) -> list[EpisodicTag]:
    """Generate all tags for a chunk based on source path and content."""
    tags = []
    tags.extend(detect_tags_from_source(source_path))
    tags.extend(detect_tags_from_content(content))
    return tags


def tags_to_json(tags: list[EpisodicTag]) -> str:
    """Serialize tags to JSON string for storage."""
    return json.dumps([{"c": t.category, "v": t.value, "conf": t.confidence} for t in tags])


def tags_from_json(raw: str) -> list[EpisodicTag]:
    """Deserialize tags from JSON string."""
    try:
        data = json.loads(raw)
        if not isinstance(data, list):
            return []
        return [EpisodicTag(category=d["c"], value=d["v"], confidence=d.get("conf", 1.0)) for d in data]
    except (json.JSONDecodeError, KeyError, TypeError):
        return []


def filter_by_tags(
    tags: list[EpisodicTag],
    include: Optional[dict[str, list[str]]] = None,
    exclude: Optional[dict[str, list[str]]] = None,
) -> bool:
    """Check if tags match include/exclude filters.

    Args:
        tags: Tags on the chunk
        include: {category: [values]} — chunk must have at least one match per category
        exclude: {category: [values]} — chunk must NOT have any match

    Returns:
        True if tags pass the filter
    """
    if not include and not exclude:
        return True

    tag_map: dict[str, set[str]] = {}
    for t in tags:
        tag_map.setdefault(t.category, set()).add(t.value)

    # Check include filters
    if include:
        for category, values in include.items():
            chunk_values = tag_map.get(category, set())
            if not any(v in chunk_values for v in values):
                return False

    # Check exclude filters
    if exclude:
        for category, values in exclude.items():
            chunk_values = tag_map.get(category, set())
            if any(v in chunk_values for v in values):
                return False

    return True


def format_tags_compact(tags: list[EpisodicTag]) -> str:
    """Format tags as compact string for display."""
    if not tags:
        return ""
    return " | ".join(f"{t.category}:{t.value}" for t in tags)


def format_tags_grouped(tags: list[EpisodicTag]) -> dict[str, list[str]]:
    """Group tags by category for display."""
    grouped: dict[str, list[str]] = {}
    for t in tags:
        grouped.setdefault(t.category, []).append(t.value)
    return grouped
