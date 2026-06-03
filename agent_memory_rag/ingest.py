"""Ingestion pipeline: discover files → chunk → embed → store."""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Iterable

from .chunker import chunk_markdown, Chunk
from .config import Config
from .embedder import Embedder, batched
from .store import MemoryStore


def _is_excluded(path: Path, root: Path, exclude_patterns: list[str]) -> bool:
    """Return True if path matches any exclude pattern (relative to root).

    Patterns are matched against the POSIX-style relative path with
    `fnmatch`, so `**` works as a recursive wildcard (matches any sequence
    of path segments, including separators).
    """
    if not exclude_patterns:
        return False
    try:
        rel = path.relative_to(root).as_posix()
    except ValueError:
        return False
    for pat in exclude_patterns:
        # Translate `**` (recursive) into something fnmatch handles.
        # fnmatch treats `*` as matching any chars including `/`, which is
        # fine for our recursive case; we just need to drop the doubled star
        # so a pattern like "backups/**" matches "backups/foo/bar.md".
        normalized = pat.replace("**/", "*/").replace("/**", "/*").replace("**", "*")
        if fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(rel, normalized):
            return True
    return False


def discover_files(config: Config) -> list[Path]:
    """Walk workspace and return matching files based on config.ingest.patterns.

    Files matching `config.ingest.exclude_patterns` are filtered out.
    """
    root = config.ingest.workspace.expanduser().resolve()
    excludes = list(config.ingest.exclude_patterns)
    seen: set[Path] = set()
    files: list[Path] = []
    for pattern in config.ingest.patterns:
        # Path.glob handles both single-segment ("MEMORY.md") and globs ("memory/*.md")
        for p in root.glob(pattern):
            p = p.resolve()
            if not p.is_file() or p in seen:
                continue
            if _is_excluded(p, root, excludes):
                continue
            seen.add(p)
            files.append(p)
    return sorted(files)


def relativize(path: Path, root: Path) -> str:
    """Return path relative to root, or absolute if outside."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def ingest_files(
    config: Config,
    embedder: Embedder,
    store: MemoryStore,
    files: Iterable[Path] | None = None,
    batch_size: int = 32,
    on_progress=None,
) -> dict:
    """Ingest files into the store. Replaces all chunks for each source.

    Returns stats dict: {files, chunks_added, chunks_deleted, sources}
    """
    root = config.ingest.workspace.expanduser().resolve()
    if files is None:
        files = discover_files(config)

    stats = {"files": 0, "chunks_added": 0, "chunks_deleted": 0, "sources": []}

    for path in files:
        source = relativize(path, root)
        # Wipe stale chunks for this source first
        deleted = store.delete_source(source)
        stats["chunks_deleted"] += deleted

        chunks = list(
            chunk_markdown(
                path,
                source_label=source,
                chunk_size=config.ingest.chunk_size,
                chunk_overlap=config.ingest.chunk_overlap,
            )
        )
        if not chunks:
            stats["files"] += 1
            stats["sources"].append({"source": source, "chunks": 0})
            if on_progress:
                on_progress(source, 0)
            continue

        # Embed in batches
        added_for_file = 0
        for batch in batched(chunks, batch_size):
            vectors = embedder.embed([c.text for c in batch])
            added_for_file += store.upsert_chunks(
                (c.to_dict() for c in batch), vectors
            )

        stats["files"] += 1
        stats["chunks_added"] += added_for_file
        stats["sources"].append({"source": source, "chunks": added_for_file})
        if on_progress:
            on_progress(source, added_for_file)

    return stats
