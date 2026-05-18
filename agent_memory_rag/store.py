"""LanceDB-backed vector store for memory chunks."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Optional

import lancedb
import pyarrow as pa


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
        ]
    )


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class MemoryStore:
    """Wraps a LanceDB table for chunks with vector + keyword search."""

    def __init__(self, db_path: Path, dimension: int):
        db_path.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(db_path))
        self._dim = dimension
        if TABLE_NAME in self._db.table_names():
            self._table = self._db.open_table(TABLE_NAME)
        else:
            self._table = self._db.create_table(
                TABLE_NAME, schema=_make_schema(dimension)
            )

    @property
    def table(self):
        return self._table

    def count(self) -> int:
        return self._table.count_rows()

    def sources(self) -> list[str]:
        """Distinct source values currently in the store."""
        df = self._table.search().select(["source"]).limit(100000).to_pandas()
        return sorted(df["source"].unique().tolist()) if len(df) else []

    def delete_source(self, source: str) -> int:
        """Delete all chunks belonging to a given source. Returns prior count."""
        before = self._table.count_rows()
        # LanceDB delete uses SQL-like predicate
        escaped = source.replace("'", "''")
        self._table.delete(f"source = '{escaped}'")
        after = self._table.count_rows()
        return before - after

    def upsert_chunks(self, chunks: Iterable[dict], vectors: Iterable[list[float]]):
        """Insert chunks. Caller is responsible for deletion of stale rows first."""
        rows = []
        for chunk, vec in zip(chunks, vectors):
            text = chunk["text"]
            rows.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "source": chunk["source"],
                    "heading_path": chunk["heading_path"],
                    "text": text,
                    "line_start": int(chunk["line_start"]),
                    "line_end": int(chunk["line_end"]),
                    "content_hash": _hash_text(text),
                    "vector": vec,
                }
            )
        if not rows:
            return 0
        self._table.add(rows)
        return len(rows)

    def search(
        self,
        query_vector: list[float],
        limit: int = 5,
        source_filter: Optional[str] = None,
    ) -> list[dict]:
        """Vector similarity search. Returns list of dicts with text + metadata + score."""
        q = self._table.search(query_vector).limit(limit)
        if source_filter:
            escaped = source_filter.replace("'", "''")
            q = q.where(f"source = '{escaped}'")
        df = q.to_pandas()
        results = []
        for _, row in df.iterrows():
            results.append(
                {
                    "chunk_id": row["chunk_id"],
                    "source": row["source"],
                    "heading_path": row["heading_path"],
                    "text": row["text"],
                    "line_start": int(row["line_start"]),
                    "line_end": int(row["line_end"]),
                    "score": float(row.get("_distance", 0.0)),
                }
            )
        return results
