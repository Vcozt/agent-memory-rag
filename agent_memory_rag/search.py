"""Search interface: query → embed → vector search → format results."""

from __future__ import annotations

from typing import Optional

from .config import Config
from .embedder import Embedder
from .store import MemoryStore

def search(
    query: str,
    config: Config,
    embedder: Embedder,
    store: MemoryStore,
    limit: int = 5,
    source_filter: Optional[str] = None,
    tag_filter: Optional[dict[str, list[str]]] = None,
    tag_exclude: Optional[dict[str, list[str]]] = None,
) -> list[dict]:
    """Semantic search over ingested memory chunks.

    Args:
        query: natural language query
        config: app config
        embedder: embedding provider
        store: vector store
        limit: max results
        source_filter: restrict to a specific source file
        tag_filter: {category: [values]} — include only matching tags
        tag_exclude: {category: [values]} — exclude matching tags

    Returns:
        List of result dicts with text, source, heading_path, line_start, line_end, score, tags
    """
    query_vec = embedder.embed_one(query)
    results = store.search(
        query_vec,
        limit=limit,
        source_filter=source_filter,
        tag_filter=tag_filter,
        tag_exclude=tag_exclude,
    )
    return results

def format_results(results: list[dict], show_score: bool = False) -> str:
    """Format search results for terminal display."""
    if not results:
        return "No results found."

    lines = []
    for i, r in enumerate(results, 1):
        source_ref = f"{r['source']}#L{r['line_start']}-L{r['line_end']}"
        heading = r.get("heading_path", "")
        score_str = f" (score: {r['score']:.4f})" if show_score else ""
        tags = r.get("tags", [])
        tag_str = f" [{', '.join(tags)}]" if tags else ""

        lines.append(f"─── Result {i}{score_str}{tag_str} ───")
        lines.append(f"Source: {source_ref}")
        if heading and heading != r["source"]:
            lines.append(f"Section: {heading}")
        lines.append("")
        # Truncate long text for display
        text = r["text"]
        if len(text) > 600:
            text = text[:600] + "…"
        lines.append(text)
        lines.append("")

    return "\n".join(lines)
