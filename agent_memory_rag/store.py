"""LanceDB-backed vector store for memory chunks with decay support.

Access metadata (last_accessed, access_count) tracked in a separate SQLite
database for fast in-place updates without LanceDB delete+re-insert.
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from pathlib import Path
from typing import Iterable, Optional

import lancedb
import pyarrow as pa

from .decay import DecayConfig, apply_decay_to_results, calculate_decay

TABLE_NAME = "memory_chunks"


def _make_schema(dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("chunk_id", pa.string()),
            pa.field("source", pa.string()),
            pa.field("heading_path", pa.string()),
            pa.field("text", pa.string()),
            pa.field("line_start", pa.int32()),
            pa.field("line_end", pa.int32()),
            pa.field("content_hash", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
            pa.field("tags", pa.string()),  # JSON array of episodic tags
        ]
    )


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class AccessTracker:
    """SQLite-backed access metadata for decay tracking.

    Stores last_accessed (unix timestamp) and access_count per chunk_id.
    Fast in-place updates — no LanceDB schema modification needed.
    """

    def __init__(self, db_path: Path):
        self._conn = sqlite3.connect(str(db_path / "access_metadata.db"))
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS access_log (
                chunk_id TEXT PRIMARY KEY,
                last_accessed REAL NOT NULL DEFAULT 0,
                access_count INTEGER NOT NULL DEFAULT 0
            )
        """)
        self._conn.commit()

    def touch(self, chunk_ids: list[str]):
        """Update last_accessed and access_count for accessed chunks."""
        if not chunk_ids:
            return
        now = time.time()
        self._conn.executemany(
            """INSERT INTO access_log (chunk_id, last_accessed, access_count)
               VALUES (?, ?, 1)
               ON CONFLICT(chunk_id) DO UPDATE SET
                 last_accessed = excluded.last_accessed,
                 access_count = access_log.access_count + 1""",
            [(cid, now) for cid in chunk_ids],
        )
        self._conn.commit()

    def touch_many(self, chunk_ids: list[str], timestamp: float):
        """Set last_accessed for multiple chunks (used during ingest)."""
        if not chunk_ids:
            return
        self._conn.executemany(
            """INSERT INTO access_log (chunk_id, last_accessed, access_count)
               VALUES (?, ?, 0)
               ON CONFLICT(chunk_id) DO UPDATE SET
                 last_accessed = excluded.last_accessed""",
            [(cid, timestamp) for cid in chunk_ids],
        )
        self._conn.commit()

    def get(self, chunk_id: str) -> tuple[float, int]:
        """Get (last_accessed, access_count) for a chunk."""
        row = self._conn.execute(
            "SELECT last_accessed, access_count FROM access_log WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()
        if row:
            return row[0], row[1]
        return 0.0, 0

    def get_batch(self, chunk_ids: list[str]) -> dict[str, tuple[float, int]]:
        """Get access metadata for multiple chunks."""
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" * len(chunk_ids))
        rows = self._conn.execute(
            f"SELECT chunk_id, last_accessed, access_count FROM access_log WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        ).fetchall()
        return {row[0]: (row[1], row[2]) for row in rows}

    def delete(self, chunk_ids: list[str]):
        """Remove access records for deleted chunks."""
        if not chunk_ids:
            return
        placeholders = ",".join("?" * len(chunk_ids))
        self._conn.execute(
            f"DELETE FROM access_log WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        )
        self._conn.commit()

    def get_all(self) -> dict[str, tuple[float, int]]:
        """Get all access records."""
        rows = self._conn.execute(
            "SELECT chunk_id, last_accessed, access_count FROM access_log"
        ).fetchall()
        return {row[0]: (row[1], row[2]) for row in rows}

    def close(self):
        self._conn.close()


class MemoryStore:
    """Wraps a LanceDB table for chunks with vector search and decay support."""

    def __init__(self, db_path: Path, dimension: int, decay_config: DecayConfig | None = None):
        db_path.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(db_path))
        self._dim = dimension
        self._decay = decay_config or DecayConfig()
        self._access = AccessTracker(db_path)
        if TABLE_NAME in self._db.table_names():
            self._table = self._db.open_table(TABLE_NAME)
            self._ensure_clean_schema(dimension)
        else:
            self._table = self._db.create_table(
                TABLE_NAME, schema=_make_schema(dimension)
            )

    def _ensure_clean_schema(self, dim: int):
        """Drop and recreate if table has old schema (missing tags or has decay fields)."""
        try:
            df = self._table.search().limit(1).to_pandas()
            needs_migrate = False
            if "last_accessed" in df.columns:
                needs_migrate = True  # Old schema with decay fields in LanceDB
            if "tags" not in df.columns:
                needs_migrate = True  # Missing tags field
            if needs_migrate:
                self._db.drop_table(TABLE_NAME)
                self._table = self._db.create_table(
                    TABLE_NAME, schema=_make_schema(dim)
                )
        except Exception:
            try:
                self._db.drop_table(TABLE_NAME)
            except Exception:
                pass
            self._table = self._db.create_table(
                TABLE_NAME, schema=_make_schema(dim)
            )

    @property
    def table(self):
        return self._table

    @property
    def decay_config(self) -> DecayConfig:
        return self._decay

    @property
    def access_tracker(self) -> AccessTracker:
        return self._access

    def count(self) -> int:
        return self._table.count_rows()

    def sources(self) -> list[str]:
        """Distinct source values currently in the store."""
        df = self._table.search().select(["source"]).limit(100000).to_pandas()
        return sorted(df["source"].unique().tolist()) if len(df) else []

    def delete_source(self, source: str) -> int:
        """Delete all chunks belonging to a given source. Returns prior count."""
        # Get chunk IDs before deleting (for access tracker cleanup)
        try:
            df = self._table.search().where(
                f"source = '{source.replace(chr(39), chr(39)*2)}'"
            ).select(["chunk_id"]).limit(100000).to_pandas()
            chunk_ids = df["chunk_id"].tolist()
        except Exception:
            chunk_ids = []

        before = self._table.count_rows()
        escaped = source.replace("'", "''")
        self._table.delete(f"source = '{escaped}'")
        after = self._table.count_rows()

        # Clean up access tracker
        self._access.delete(chunk_ids)

        return before - after

    def upsert_chunks(self, chunks: Iterable[dict], vectors: Iterable[list[float]]):
        """Insert chunks with episodic tags. Access metadata tracked in SQLite."""
        from .episodic import generate_tags, tags_to_json

        now = time.time()
        rows = []
        chunk_ids = []
        for chunk, vec in zip(chunks, vectors):
            text = chunk["text"]
            source = chunk["source"]
            # Generate episodic tags from source + content
            tags = generate_tags(source, text)
            tags_json = tags_to_json(tags)
            rows.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "source": source,
                    "heading_path": chunk["heading_path"],
                    "text": text,
                    "line_start": int(chunk["line_start"]),
                    "line_end": int(chunk["line_end"]),
                    "content_hash": _hash_text(text),
                    "vector": vec,
                    "tags": tags_json,
                }
            )
            chunk_ids.append(chunk["chunk_id"])
        if not rows:
            return 0
        self._table.add(rows)

        # Initialize access metadata (last_accessed = ingest time)
        self._access.touch_many(chunk_ids, now)

        return len(rows)

    def search(
        self,
        query_vector: list[float],
        limit: int = 5,
        source_filter: Optional[str] = None,
        decay_config: Optional[DecayConfig] = None,
        tag_filter: Optional[dict[str, list[str]]] = None,
        tag_exclude: Optional[dict[str, list[str]]] = None,
    ) -> list[dict]:
        """Vector similarity search with decay scoring and episodic tag filtering.

        If decay is enabled, fetches extra results for re-ranking.
        If tag_filter/tag_exclude provided, filters results by episodic tags.
        """
        from .episodic import tags_from_json, filter_by_tags

        cfg = decay_config or self._decay
        # Fetch extra for both decay re-ranking and tag filtering
        multiplier = 4 if (tag_filter or tag_exclude) else 3
        fetch_limit = limit * multiplier if cfg.enabled else limit
        q = self._table.search(query_vector).limit(fetch_limit)
        if source_filter:
            escaped = source_filter.replace("'", "''")
            q = q.where(f"source = '{escaped}'")
        df = q.to_pandas()

        # Collect chunk IDs and fetch access metadata from SQLite
        chunk_ids = df["chunk_id"].tolist()
        access_data = self._access.get_batch(chunk_ids)

        results = []
        for _, row in df.iterrows():
            cid = row["chunk_id"]
            last_accessed, access_count = access_data.get(cid, (0.0, 0))
            # Parse episodic tags
            tags_raw = row.get("tags", "[]")
            tags = tags_from_json(tags_raw) if tags_raw else []

            # Apply tag filtering
            if (tag_filter or tag_exclude) and not filter_by_tags(tags, tag_filter, tag_exclude):
                continue

            result = {
                "chunk_id": cid,
                "source": row["source"],
                "heading_path": row["heading_path"],
                "text": row["text"],
                "line_start": int(row["line_start"]),
                "line_end": int(row["line_end"]),
                "score": float(row.get("_distance", 0.0)),
                "last_accessed": last_accessed,
                "access_count": access_count,
                "tags": [t.to_string() for t in tags],
            }
            results.append(result)

        # Apply decay scoring
        if cfg.enabled and results:
            results = apply_decay_to_results(
                results,
                half_life_days=cfg.half_life_days,
                min_decay_factor=cfg.min_decay_factor,
            )
            results = results[:limit]
            # Touch retrieved chunks (update access timestamps)
            touched_ids = [r["chunk_id"] for r in results]
            self._access.touch(touched_ids)

        return results

    def get_all_chunks(self) -> list[dict]:
        """Return all chunks with access metadata for decay analysis."""
        try:
            df = self._table.search().limit(100000).to_pandas()
            chunk_ids = df["chunk_id"].tolist()
            access_data = self._access.get_batch(chunk_ids)

            chunks = []
            for _, row in df.iterrows():
                cid = row["chunk_id"]
                last_accessed, access_count = access_data.get(cid, (0.0, 0))
                chunks.append({
                    "chunk_id": cid,
                    "source": row["source"],
                    "text": row["text"][:100],
                    "last_accessed": last_accessed,
                    "access_count": access_count,
                })
            return chunks
        except Exception:
            return []

    def delete_stale(self, threshold: float = 0.1, half_life_days: float = 30.0) -> int:
        """Remove chunks below decay threshold. Returns count removed."""
        chunks = self.get_all_chunks()
        now = time.time()
        removed = 0
        for chunk in chunks:
            last_accessed = chunk.get("last_accessed", 0)
            decay = calculate_decay(last_accessed, now, half_life_days)
            if decay < threshold:
                try:
                    escaped_id = chunk["chunk_id"].replace("'", "''")
                    self._table.delete(f"chunk_id = '{escaped_id}'")
                    self._access.delete([chunk["chunk_id"]])
                    removed += 1
                except Exception:
                    pass
        return removed
