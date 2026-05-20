"""Tests for ingestion file discovery (no embedding)."""
from __future__ import annotations

from pathlib import Path

from agent_memory_rag.config import Config
from agent_memory_rag.ingest import discover_files, _is_excluded


def _make_workspace(tmp: Path) -> Path:
    (tmp / "MEMORY.md").write_text("# memory")
    (tmp / "memory").mkdir()
    (tmp / "memory" / "2026-05-20.md").write_text("# log")
    (tmp / "backups").mkdir()
    (tmp / "backups" / "memory").mkdir()
    (tmp / "backups" / "memory" / "old.md").write_text("# old")
    (tmp / "drafts").mkdir()
    (tmp / "drafts" / "MEMORY.md").write_text("# draft")
    return tmp


def test_discover_files_excludes_default_patterns(tmp_path: Path) -> None:
    root = _make_workspace(tmp_path)
    cfg = Config()
    cfg.ingest.workspace = root
    cfg.ingest.patterns = ["MEMORY.md", "memory/*.md", "backups/**/*.md", "drafts/*.md"]
    # Defaults already exclude backups/** and drafts/**
    files = discover_files(cfg)
    rel = sorted(str(p.relative_to(root)) for p in files)
    assert "MEMORY.md" in rel
    assert "memory/2026-05-20.md" in rel
    assert all("backups/" not in r for r in rel)
    assert all("drafts/" not in r for r in rel)


def test_discover_files_custom_excludes(tmp_path: Path) -> None:
    root = _make_workspace(tmp_path)
    cfg = Config()
    cfg.ingest.workspace = root
    cfg.ingest.patterns = ["MEMORY.md", "memory/*.md"]
    cfg.ingest.exclude_patterns = ["memory/2026-05-20.md"]
    files = discover_files(cfg)
    rel = [str(p.relative_to(root)) for p in files]
    assert "MEMORY.md" in rel
    assert "memory/2026-05-20.md" not in rel


def test_discover_files_empty_excludes_keeps_everything(tmp_path: Path) -> None:
    root = _make_workspace(tmp_path)
    cfg = Config()
    cfg.ingest.workspace = root
    cfg.ingest.patterns = ["MEMORY.md", "memory/*.md", "drafts/*.md"]
    cfg.ingest.exclude_patterns = []
    files = discover_files(cfg)
    rel = sorted(str(p.relative_to(root)) for p in files)
    assert "drafts/MEMORY.md" in rel


def test_is_excluded_outside_root_returns_false(tmp_path: Path) -> None:
    other = tmp_path.parent
    assert _is_excluded(other / "elsewhere.md", tmp_path, ["**"]) is False
