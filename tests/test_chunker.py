"""Tests for the markdown chunker."""

from pathlib import Path

import pytest

from agent_memory_rag.chunker import chunk_markdown, _heading_level


def test_heading_level():
    assert _heading_level("# Title") == 1
    assert _heading_level("## Subtitle") == 2
    assert _heading_level("### Section") == 3
    assert _heading_level("normal text") == 0
    assert _heading_level("#hashtag") == 0  # no space after hashes
    assert _heading_level("####### too deep") == 0


def test_chunk_basic(tmp_path: Path):
    f = tmp_path / "test.md"
    f.write_text(
        "# Title\n\n"
        "Some intro text under the title.\n\n"
        "## Section A\n\n"
        "Body of section A goes here. " * 30 + "\n\n"
        "## Section B\n\n"
        "Body of section B is shorter.\n",
        encoding="utf-8",
    )

    chunks = list(chunk_markdown(f, source_label="test.md", chunk_size=200, chunk_overlap=20))
    assert len(chunks) >= 2

    # Every chunk should carry source + heading_path
    for c in chunks:
        assert c.source == "test.md"
        assert c.heading_path.startswith("test.md")
        assert c.line_start >= 1
        assert c.line_end >= c.line_start
        assert c.text.strip()
        assert c.chunk_id == f"test.md#L{c.line_start}-L{c.line_end}"


def test_chunk_heading_boundary(tmp_path: Path):
    f = tmp_path / "test.md"
    f.write_text(
        "# Top\n\nIntro.\n\n## A\n\nA body.\n\n## B\n\nB body.\n",
        encoding="utf-8",
    )

    chunks = list(chunk_markdown(f, chunk_size=10000, chunk_overlap=0))
    # Headings cause flushes, so we should not get a single chunk that mixes A and B body
    paths = {c.heading_path for c in chunks}
    has_a = any(p.endswith("> A") for p in paths)
    has_b = any(p.endswith("> B") for p in paths)
    assert has_a and has_b


def test_chunk_empty_file(tmp_path: Path):
    f = tmp_path / "empty.md"
    f.write_text("", encoding="utf-8")
    chunks = list(chunk_markdown(f))
    assert chunks == []


def test_chunk_id_stable(tmp_path: Path):
    """Same content should always produce the same chunk_ids."""
    f = tmp_path / "stable.md"
    content = "# A\nbody\n## B\nmore body\n"
    f.write_text(content, encoding="utf-8")

    ids_run1 = [c.chunk_id for c in chunk_markdown(f, source_label="stable.md")]
    ids_run2 = [c.chunk_id for c in chunk_markdown(f, source_label="stable.md")]
    assert ids_run1 == ids_run2
