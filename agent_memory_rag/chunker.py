"""Markdown-aware chunker. Splits files into semantic chunks while preserving
heading context. Tracks line offsets for source citation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class Chunk:
    """A single chunk of text with source metadata."""

    text: str
    source: str  # path relative to workspace (or absolute)
    heading_path: str  # e.g. "MEMORY.md > Key Decisions"
    line_start: int  # 1-indexed
    line_end: int  # 1-indexed, inclusive
    chunk_id: str  # stable id: source#line_start-line_end

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "source": self.source,
            "heading_path": self.heading_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "chunk_id": self.chunk_id,
        }


def _approx_token_count(text: str) -> int:
    """Approximate token count (chars / 4 is the common rule of thumb)."""
    return max(1, len(text) // 4)


def _heading_level(line: str) -> int:
    """Return markdown heading level (1-6) or 0 if not a heading."""
    stripped = line.lstrip()
    if not stripped.startswith("#"):
        return 0
    i = 0
    while i < len(stripped) and stripped[i] == "#":
        i += 1
    if i > 6:
        return 0
    if i < len(stripped) and stripped[i] != " ":
        return 0
    return i


def _heading_text(line: str) -> str:
    return line.lstrip("#").strip()


def chunk_markdown(
    path: Path,
    source_label: str | None = None,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> Iterator[Chunk]:
    """Yield chunks from a markdown file.

    Strategy:
      1. Walk lines, track heading stack (h1 > h2 > h3 ...).
      2. Accumulate lines until token budget hit, then flush a chunk.
      3. New heading at same/higher level forces a flush boundary so chunks
         don't span across unrelated sections.
      4. Overlap last N tokens between adjacent chunks for context preservation.

    Args:
        path: file to read
        source_label: how to label this file in citations (defaults to path)
        chunk_size: target tokens per chunk
        chunk_overlap: tokens to overlap between adjacent chunks
    """
    label = source_label or str(path)
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return

    lines = text.splitlines()
    heading_stack: list[tuple[int, str]] = []  # (level, text)

    # Buffer state
    buf_lines: list[str] = []
    buf_start_line: int = 0
    buf_tokens: int = 0
    buf_heading_path: str = label

    def heading_path_str() -> str:
        if not heading_stack:
            return label
        return label + " > " + " > ".join(h[1] for h in heading_stack)

    def make_chunk(end_line_inclusive: int) -> Chunk | None:
        if not buf_lines:
            return None
        body = "\n".join(buf_lines).strip()
        if not body:
            return None
        return Chunk(
            text=body,
            source=label,
            heading_path=buf_heading_path,
            line_start=buf_start_line,
            line_end=end_line_inclusive,
            chunk_id=f"{label}#L{buf_start_line}-L{end_line_inclusive}",
        )

    def flush(end_line_inclusive: int) -> Iterator[Chunk]:
        nonlocal buf_lines, buf_start_line, buf_tokens, buf_heading_path
        chunk = make_chunk(end_line_inclusive)
        if chunk:
            yield chunk
        # Compute overlap tail (in lines) for next chunk start
        tail_tokens = 0
        tail_lines: list[str] = []
        for ln in reversed(buf_lines):
            t = _approx_token_count(ln)
            if tail_tokens + t > chunk_overlap:
                break
            tail_lines.insert(0, ln)
            tail_tokens += t
        # Reset buffer with tail as carry-over
        buf_lines = list(tail_lines)
        buf_tokens = tail_tokens
        # Start line of new buffer: just after end_line_inclusive minus tail length
        buf_start_line = max(1, end_line_inclusive - len(tail_lines) + 1)
        buf_heading_path = heading_path_str()

    for idx, line in enumerate(lines, start=1):
        level = _heading_level(line)
        if level > 0:
            # Heading boundary: flush current buffer first
            if buf_lines:
                yield from flush(idx - 1)
            # Update heading stack: pop until we find lower level
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, _heading_text(line)))
            buf_heading_path = heading_path_str()

        if not buf_lines:
            buf_start_line = idx
            buf_heading_path = heading_path_str()

        buf_lines.append(line)
        buf_tokens += _approx_token_count(line)

        if buf_tokens >= chunk_size:
            yield from flush(idx)

    # Final flush
    if buf_lines:
        chunk = make_chunk(len(lines))
        if chunk:
            yield chunk
