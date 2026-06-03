"""Memory consolidation — auto-extract insights from daily notes into MEMORY.md.

v2: Fixed incremental processing, better dedup, pruning, archive, negation handling.
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
class Insight:
    """A single consolidated insight from daily notes."""

    category: str  # decision, correction, preference, project, lesson, fact
    text: str
    source_file: str
    source_date: str
    confidence: float = 0.7
    tags: list[str] = field(default_factory=list)
    already_in_memory: bool = False
    negated: bool = False  # NEW: tracks negation ("decided NOT to")

    def to_memory_line(self) -> str:
        """Format as a MEMORY.md bullet point."""
        prefix = ""
        if self.negated:
            prefix = "🚫 "  # Negated insights get a warning prefix
        elif self.confidence >= 0.9:
            prefix = "✅ "
        elif self.confidence >= 0.7:
            prefix = "📌 "
        else:
            prefix = "📝 "
        return f"{prefix}{self.text}"


@dataclass
class ConsolidationResult:
    """Result of a consolidation run."""

    notes_reviewed: int
    notes_archived: int  # NEW: how many notes were archived
    insights_found: int
    new_insights: int
    updated_insights: int
    unchanged: int
    pruned: int  # NEW: how many stale insights were pruned
    categories: dict[str, int]
    files_modified: list[str]


# Category detection patterns
CATEGORY_PATTERNS = {
    "decision": [
        r"(?:decided|keputusan|diputuskan|locked|confirmed|final)",
        r"(?:we went with|pilih|switched to|moved to)",
    ],
    "correction": [
        r"(?:koreksi|corrected|fixed|bug|fix|was wrong|nggak bener|salah)",
        r"(?:lesson|pelajaran|learned|turns out|ternyata)",
        r"(?:should have|seharusnya|mistake|error)",
    ],
    "preference": [
        r"(?:prefer|suka|likes?|dislikes?|favorit)",
        r"(?:style|gaya|format|preferensi|biasa)",
        r"(?:always|selalu|never|jangan|don't like)",
    ],
    "project": [
        r"(?:shipped|deployed|released|published|launch)",
        r"(?:project|repo|v0\.|v1\.|version)",
        r"(?:built|created|added|implemented|done)",
    ],
    "lesson": [
        r"(?: learned|注意到|realized|insight|key takeaway)",
        r"(?:important|penting|remember|ingat|note)",
        r"(?:pattern|pola|always happens|terjadi lagi)",
    ],
    "fact": [
        r"(?:server|IP|port|hostname|credential|path|config)",
        r"(?:installed|terinstall|running|active|configured)",
        r"(?:account|akun|token|key|ID)",
    ],
}

# Negation patterns — these flip the meaning
NEGATION_PATTERNS = [
    r"\bnot\b",
    r"\bnever\b",
    r"\bno\b",
    r"\bdon'?t\b",
    r"\bdoesn'?t\b",
    r"\bdidn'?t\b",
    r"\bwon'?t\b",
    r"\bcan'?t\b",
    r"\bcannot\b",
    r"\bshouldn'?t\b",
    r"\bwouldn'?t\b",
    r"\bjangan\b",
    r"\bnggak\b",
    r"\bgak\b",
    r"\btdk\b",
    r"\btidak\b",
]

# Consolidation state file
STATE_FILE = "memory/consolidation-state.json"


def _load_state(workspace: Path) -> dict:
    """Load consolidation state (last consolidated date, archived files)."""
    state_path = workspace / STATE_FILE
    if state_path.exists():
        try:
            return json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"last_consolidated": None, "archived": [], "pruned_ids": []}


def _save_state(workspace: Path, state: dict):
    """Save consolidation state."""
    state_path = workspace / STATE_FILE
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def _has_negation(text: str) -> bool:
    """Check if text contains negation near key action words."""
    text_lower = text.lower()
    # Simple heuristic: if negation pattern appears within 5 words of a decision/correction keyword
    for neg in NEGATION_PATTERNS:
        if re.search(neg, text_lower):
            return True
    return False


def discover_daily_notes(
    memory_dir: Path,
    days_back: int = 7,
    after_date: Optional[str] = None,
) -> list[Path]:
    """Find daily note files from the last N days."""
    if not memory_dir.exists():
        return []

    cutoff = datetime.now() - timedelta(days=days_back)
    after_dt = datetime.strptime(after_date, "%Y-%m-%d") if after_date else None

    notes = []
    for f in memory_dir.glob("????-??-??.md"):
        try:
            date_str = f.stem
            file_date = datetime.strptime(date_str, "%Y-%m-%d")
            if file_date >= cutoff and (after_dt is None or file_date > after_dt):
                notes.append(f)
        except ValueError:
            continue

    return sorted(notes)


def extract_events_from_note(note_path: Path) -> list[dict]:
    """Extract structured events from a daily note file."""
    content = note_path.read_text(encoding="utf-8")
    events = []
    date_str = note_path.stem

    sections = re.split(r"^##\s+", content, flags=re.MULTILINE)

    for section in sections:
        section = section.strip()
        if not section or len(section) < 20:
            continue

        lines = section.split("\n", 1)
        heading = lines[0].strip() if lines else ""
        body = lines[1].strip() if len(lines) > 1 else section

        if len(body) < 15:
            continue

        category = _detect_category(section)
        negated = _has_negation(section)
        confidence = _estimate_confidence(section, category, negated)
        tags = _extract_tags(section)
        insight_text = _extract_key_insight(body, heading)

        if insight_text and len(insight_text) > 10:
            events.append({
                "text": insight_text,
                "category": category,
                "confidence": confidence,
                "tags": tags,
                "source_date": date_str,
                "section_heading": heading,
                "negated": negated,
            })

    return events


def _detect_category(text: str) -> str:
    """Detect the category of an event based on keyword patterns."""
    text_lower = text.lower()
    scores = {}

    for category, patterns in CATEGORY_PATTERNS.items():
        score = 0
        for pattern in patterns:
            if re.search(pattern, text_lower):
                score += 1
        scores[category] = score

    if max(scores.values()) == 0:
        return "fact"

    return max(scores, key=scores.get)


def _estimate_confidence(text: str, category: str, negated: bool) -> float:
    """Estimate confidence based on explicitness and negation."""
    confidence = 0.7

    explicit_markers = [
        r"(?:confirmed|verified|tested|proven|works)",
        r"(?:decided|locked|final|shipped|deployed)",
        r"(?:always|never|must|wajib|jangan pernah)",
    ]
    for marker in explicit_markers:
        if re.search(marker, text.lower()):
            confidence = min(confidence + 0.1, 1.0)

    tentative_markers = [
        r"(?:maybe|might|could|perhaps|mungkin|kayaknya)",
        r"(?:thinking|考虑|plan to|rencana)",
        r"(?:not sure|belum tau|uncertain)",
    ]
    for marker in tentative_markers:
        if re.search(marker, text.lower()):
            confidence = max(confidence - 0.15, 0.3)

    # Negated statements get lower confidence (they're "anti-patterns")
    if negated:
        confidence = max(confidence - 0.2, 0.3)

    if category == "decision":
        confidence = min(confidence + 0.1, 1.0)
    elif category == "correction":
        confidence = min(confidence + 0.15, 1.0)
    elif category == "preference":
        confidence = max(confidence - 0.05, 0.4)

    return round(confidence, 2)


def _extract_tags(text: str) -> list[str]:
    """Extract relevant tags from text."""
    tags = []
    tag_patterns = {
        "tech": r"(python|node|npm|docker|nginx|lancedb|sqlite|github|api)",
        "infra": r"(server|vps|deploy|ssh|systemd|cron|tunnel)",
        "agent": r"(agent|openclaw|kiro|claude|llm|model)",
        "memory": r"(memory|recall|search|embed|rag|consolidat)",
        "project": r"(project|repo|release|ship|v\d|version)",
        "security": r"(credential|token|key|password|auth|security)",
    }
    for tag, pattern in tag_patterns.items():
        if re.search(pattern, text.lower()):
            tags.append(tag)
    return tags


def _extract_key_insight(body: str, heading: str) -> str:
    """Extract the most meaningful insight from a section body."""
    lines = body.split("\n")
    insight_parts = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith("```") or (line.startswith("|") and "---" in line):
            continue
        if re.match(r"^[\-\*\+\s]+$", line):
            continue
        if line.startswith(("- ", "* ", "+ ")) or len(line) > 30:
            cleaned = re.sub(r"^[\-\*\+]\s*", "", line)
            if cleaned and len(cleaned) > 10:
                insight_parts.append(cleaned)
        if len(insight_parts) >= 3:
            break

    if not insight_parts:
        clean_body = re.sub(r"\s+", " ", body)[:200]
        return f"{heading}: {clean_body}" if heading else clean_body

    combined = "; ".join(insight_parts[:2])
    if heading and heading.lower() not in combined.lower():
        combined = f"{heading}: {combined}"
    return combined[:500]


def _compute_text_similarity(a: str, b: str) -> float:
    """Compute word-level Jaccard similarity between two strings."""
    stop_words = {"the", "a", "an", "is", "was", "are", "in", "on", "at", "to",
                  "for", "of", "with", "by", "and", "or", "not", "this", "that",
                  "yang", "dan", "di", "untuk", "dengan", "adalah", "ini", "itu"}
    words_a = set(a.lower().split()) - stop_words
    words_b = set(b.lower().split()) - stop_words
    if not words_a or not words_b:
        return 0.0
    intersection = len(words_a & words_b)
    union = len(words_a | words_b)
    return intersection / union if union > 0 else 0.0


def find_matching_memory_entry(
    insight: Insight,
    memory_content: str,
    threshold: float = 0.35,
) -> Optional[str]:
    """Check if an insight already exists in MEMORY.md using semantic similarity.

    Uses Jaccard word similarity with a reasonable threshold.
    Returns the matching line if found, None otherwise.
    """
    best_match = None
    best_similarity = 0.0

    for line in memory_content.split("\n"):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("<!--"):
            continue

        similarity = _compute_text_similarity(insight.text, line)
        if similarity > best_similarity and similarity > threshold:
            best_similarity = similarity
            best_match = line

    return best_match


def _find_stale_insights(
    memory_content: str,
    max_age_days: int = 30,
    min_confidence: float = 0.5,
) -> list[str]:
    """Find insight lines in MEMORY.md that should be pruned.

    Returns only insight lines (starting with '- ') and their source lines,
    not section headers or empty lines.
    """
    lines_to_prune = []
    in_consolidated_section = False
    current_date = None
    lines_list = memory_content.split("\n")
    i = 0

    while i < len(lines_list):
        line = lines_list[i]

        # Track consolidated sections
        if "Auto-Consolidated" in line:
            in_consolidated_section = True
            date_match = re.search(r"\((\d{4}-\d{2}-\d{2})\)", line)
            if date_match:
                current_date = date_match.group(1)
            i += 1
            continue

        if line.startswith("## ") and "Auto-Consolidated" not in line:
            in_consolidated_section = False
            current_date = None
            i += 1
            continue

        if in_consolidated_section and current_date:
            stripped = line.strip()
            if stripped.startswith("- "):
                should_prune = False

                # Check age
                try:
                    section_date = datetime.strptime(current_date, "%Y-%m-%d")
                    age_days = (datetime.now() - section_date).days
                    if age_days > max_age_days:
                        should_prune = True
                except ValueError:
                    pass

                # Check confidence (look at next line for _Source:)
                if not should_prune and i + 1 < len(lines_list):
                    next_line = lines_list[i + 1].strip()
                    if next_line.startswith("_Source:"):
                        conf_match = re.search(r"confidence:\s*([\d.]+)", next_line)
                        if conf_match:
                            conf = float(conf_match.group(1))
                            if conf < min_confidence:
                                should_prune = True

                if should_prune:
                    lines_to_prune.append(line)
                    # Also add source line if present
                    if i + 1 < len(lines_list) and lines_list[i + 1].strip().startswith("_Source:"):
                        lines_to_prune.append(lines_list[i + 1])

        i += 1

    return lines_to_prune


def _prune_memory(
    memory_path: Path,
    max_age_days: int = 30,
    min_confidence: float = 0.5,
    min_per_category: int = 2,
) -> int:
    """Remove stale insights from MEMORY.md with category safety.

    Args:
        memory_path: Path to MEMORY.md
        max_age_days: Prune insights older than this
        min_confidence: Prune insights below this confidence
        min_per_category: Keep at least this many insights per category

    Returns:
        Count of lines removed
    """
    if not memory_path.exists():
        return 0

    content = memory_path.read_text(encoding="utf-8")
    lines_to_prune = _find_stale_insights(content, max_age_days, min_confidence)

    if not lines_to_prune:
        return 0

    # Count remaining insights per category after pruning
    pruned_set = set(l.strip() for l in lines_to_prune)
    total_by_category: dict[str, int] = {}
    remaining_by_category: dict[str, int] = {}
    current_category = "unknown"

    for line in content.split("\n"):
        stripped = line.strip()
        if stripped.startswith("### "):
            current_category = stripped[4:].lower()
        if stripped.startswith("- ") and current_category:
            total_by_category[current_category] = total_by_category.get(current_category, 0) + 1
            if stripped not in pruned_set:
                remaining_by_category[current_category] = remaining_by_category.get(current_category, 0) + 1

    # Only prune if we won't go below minimum per category
    safe_to_prune = []
    prune_counts: dict[str, int] = {}
    current_category = "unknown"

    for line in content.split("\n"):
        stripped = line.strip()
        if stripped.startswith("### "):
            current_category = stripped[4:].lower()
        if stripped in pruned_set:
            total = total_by_category.get(current_category, 0)
            already_pruned = prune_counts.get(current_category, 0)
            # After this prune, will we still have >= min_per_category?
            if total - already_pruned - 1 >= min_per_category:
                safe_to_prune.append(line)
                prune_counts[current_category] = already_pruned + 1

    if not safe_to_prune:
        return 0

    # Remove safe-to-prune lines
    prune_set = set(l.strip() for l in safe_to_prune)
    new_lines = []
    for line in content.split("\n"):
        if line.strip() in prune_set:
            continue
        new_lines.append(line)

    # Clean up empty sections
    cleaned = "\n".join(new_lines)
    cleaned = re.sub(r"### \w[^\n]*\n(?=\n*(?:###|\Z))", "", cleaned)

    memory_path.write_text(cleaned, encoding="utf-8")
    return len(safe_to_prune)


def _archive_note(note_path: Path, archive_dir: Path):
    """Move a daily note to the archive directory."""
    archive_dir.mkdir(parents=True, exist_ok=True)
    target = archive_dir / note_path.name
    note_path.rename(target)


def find_matching_memory_entry_with_dedup(
    insight: Insight,
    existing_insights: list[Insight],
    memory_content: str,
    dedup_threshold: float = 0.5,
    memory_threshold: float = 0.35,
) -> Optional[str]:
    """Check if insight matches either existing memory or other new insights."""
    # Check against existing memory
    match = find_matching_memory_entry(insight, memory_content, memory_threshold)
    if match:
        return match

    # Check against other new insights in this batch
    for other in existing_insights:
        if other is insight:
            continue
        similarity = _compute_text_similarity(insight.text, other.text)
        if similarity > dedup_threshold:
            return f"[dedup:{similarity:.2f}] {other.text[:80]}"

    return None


def consolidate(
    workspace: Path,
    days_back: int = 7,
    after_date: Optional[str] = None,
    dry_run: bool = True,
    memory_file: str = "MEMORY.md",
    archive: bool = True,
    max_prune_age: int = 30,
    min_prune_confidence: float = 0.5,
    min_per_category: int = 2,
) -> ConsolidationResult:
    """Run consolidation with incremental processing, better dedup, and pruning."""
    memory_dir = workspace / "memory"
    memory_path = workspace / memory_file
    state = _load_state(workspace)

    # Read existing memory
    memory_content = ""
    if memory_path.exists():
        memory_content = memory_path.read_text(encoding="utf-8")

    # INCREMENTAL: only process notes after last consolidation
    effective_after = after_date
    if state.get("last_consolidated") and not after_date:
        effective_after = state["last_consolidated"]

    # Discover daily notes
    notes = discover_daily_notes(memory_dir, days_back, effective_after)

    # Filter out already-archived notes
    archived_set = set(state.get("archived", []))
    notes = [n for n in notes if n.name not in archived_set]

    result = ConsolidationResult(
        notes_reviewed=len(notes),
        notes_archived=0,
        insights_found=0,
        new_insights=0,
        updated_insights=0,
        unchanged=0,
        pruned=0,
        categories={},
        files_modified=[],
    )

    all_insights = []
    batch_insights = []  # Track new insights in this batch for dedup

    for note in notes:
        events = extract_events_from_note(note)
        for event in events:
            insight = Insight(
                category=event["category"],
                text=event["text"],
                source_file=str(note.name),
                source_date=event["source_date"],
                confidence=event["confidence"],
                tags=event["tags"],
                negated=event["negated"],
            )

            # Better dedup: check memory + batch
            match = find_matching_memory_entry_with_dedup(
                insight, batch_insights, memory_content
            )
            if match:
                insight.already_in_memory = True
                result.unchanged += 1
            else:
                result.new_insights += 1
                batch_insights.append(insight)

            all_insights.append(insight)
            result.insights_found += 1
            result.categories[insight.category] = result.categories.get(insight.category, 0) + 1

    # PRUNING: remove stale insights from MEMORY.md
    if not dry_run and max_prune_age > 0:
        pruned = _prune_memory(memory_path, max_prune_age, min_prune_confidence, min_per_category)
        result.pruned = pruned
        if pruned > 0:
            # Refresh memory content after pruning
            memory_content = memory_path.read_text(encoding="utf-8") if memory_path.exists() else ""

    # Write new insights
    if not dry_run and result.new_insights > 0:
        _write_insights_to_memory(memory_path, all_insights, memory_content)
        result.files_modified.append(str(memory_path))

    # ARCHIVE: move processed notes to .archive/
    if not dry_run and archive:
        archive_dir = memory_dir / ".archive"
        for note in notes:
            _archive_note(note, archive_dir)
            result.notes_archived += 1
            state.setdefault("archived", []).append(note.name)

    # Update state
    if not dry_run:
        state["last_consolidated"] = datetime.now().strftime("%Y-%m-%d")
        _save_state(workspace, state)

    return result


def _write_insights_to_memory(
    memory_path: Path,
    insights: list[Insight],
    existing_content: str,
):
    """Write new insights to MEMORY.md, organized by category."""
    new_insights = [i for i in insights if not i.already_in_memory]
    if not new_insights:
        return

    by_category: dict[str, list[Insight]] = {}
    for insight in new_insights:
        by_category.setdefault(insight.category, []).append(insight)

    new_lines = []
    new_lines.append("")
    new_lines.append(f"## Auto-Consolidated ({datetime.now().strftime('%Y-%m-%d')})")
    new_lines.append("")

    category_headers = {
        "decision": "Decisions Made",
        "correction": "Corrections & Lessons",
        "preference": "Preferences",
        "project": "Project Updates",
        "lesson": "Key Lessons",
        "fact": "Facts & Environment",
    }

    for category in ["decision", "correction", "preference", "project", "lesson", "fact"]:
        items = by_category.get(category, [])
        if not items:
            continue
        header = category_headers.get(category, category.title())
        new_lines.append(f"### {header}")
        for item in items:
            new_lines.append(f"- {item.to_memory_line()}")
            new_lines.append(f"  _Source: {item.source_file} ({item.source_date}) | confidence: {item.confidence}_")
        new_lines.append("")

    if existing_content.strip():
        updated = existing_content.rstrip() + "\n" + "\n".join(new_lines)
    else:
        updated = "\n".join(new_lines)

    memory_path.write_text(updated, encoding="utf-8")


def format_consolidation_report(result: ConsolidationResult) -> str:
    """Format consolidation result for display."""
    lines = []
    lines.append("═══ Consolidation Report ═══")
    lines.append(f"Notes reviewed:   {result.notes_reviewed}")
    lines.append(f"Notes archived:   {result.notes_archived}")
    lines.append(f"Insights found:   {result.insights_found}")
    lines.append(f"  New:            {result.new_insights}")
    lines.append(f"  Already in memory: {result.unchanged}")
    lines.append(f"  Updated:        {result.updated_insights}")
    lines.append(f"  Pruned:         {result.pruned}")
    lines.append("")
    lines.append("By category:")
    for cat, count in sorted(result.categories.items(), key=lambda x: -x[1]):
        lines.append(f"  {cat}: {count}")
    if result.files_modified:
        lines.append("")
        lines.append("Files modified:")
        for f in result.files_modified:
            lines.append(f"  {f}")
    return "\n".join(lines)
