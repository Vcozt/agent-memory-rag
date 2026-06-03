"""CLI entry point for agent-memory-rag."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from .config import Config, EmbeddingConfig, IngestConfig, DEFAULT_DB_PATH, DEFAULT_WORKSPACE
from .decay import DecayConfig
from .embedder import build_embedder
from .ingest import ingest_files, discover_files
from .search import search as do_search, format_results
from .store import MemoryStore

console = Console()

def _build_config(
    workspace: Path | None,
    db_path: Path | None,
    provider: str,
    model: str | None,
    api_base: str | None,
    api_key: str | None,
    dim: int | None,
    decay_enabled: bool = True,
    half_life: float = 30.0,
) -> Config:
    cfg = Config()
    if workspace:
        cfg.ingest.workspace = workspace
    if db_path:
        cfg.db_path = db_path
    cfg.embedding.provider = provider
    if model:
        cfg.embedding.model = model
    if api_base:
        cfg.embedding.api_base = api_base
    if api_key:
        cfg.embedding.api_key = api_key
    if dim:
        cfg.embedding.dimension = dim
    cfg.decay = DecayConfig(
        enabled=decay_enabled,
        half_life_days=half_life,
    )
    return cfg

def _build_decay_config(
    enabled: bool = True,
    half_life: float = 30.0,
    min_factor: float = 0.05,
    threshold: float = 0.1,
) -> DecayConfig:
    return DecayConfig(
        enabled=enabled,
        half_life_days=half_life,
        min_decay_factor=min_factor,
        cleanup_threshold=threshold,
    )

# Global options shared across commands
def common_options(fn):
    fn = click.option(
        "--db",
        "db_path",
        type=click.Path(path_type=Path),
        default=DEFAULT_DB_PATH,
        help=f"LanceDB path (default: {DEFAULT_DB_PATH})",
    )(fn)
    fn = click.option(
        "--provider",
        type=click.Choice(["local", "api"]),
        default="local",
        help="Embedding provider",
    )(fn)
    fn = click.option("--model", default=None, help="Embedding model name")(fn)
    fn = click.option("--api-base", default=None, help="API base URL (for provider=api)")(fn)
    fn = click.option(
        "--api-key",
        default=None,
        envvar="AMR_API_KEY",
        help="API key (for provider=api). Can also use AMR_API_KEY env var",
    )(fn)
    fn = click.option("--dim", type=int, default=None, help="Embedding dimension override")(fn)
    return fn

def decay_options(fn):
    fn = click.option(
        "--no-decay", "decay_enabled",
        is_flag=True, default=True,
        help="Disable decay scoring",
    )(fn)
    fn = click.option(
        "--half-life",
        type=float, default=30.0,
        help="Decay half-life in days",
    )(fn)
    return fn

@click.group()
@click.version_option(package_name="agent-memory-rag")
def main():
    """agent-memory-rag: semantic search for AI agent workspace memory."""

@main.command()
@click.option(
    "--workspace",
    type=click.Path(path_type=Path),
    default=DEFAULT_WORKSPACE,
    help=f"Workspace root (default: {DEFAULT_WORKSPACE})",
)
@click.option(
    "--pattern",
    "patterns",
    multiple=True,
    help="Glob pattern(s) to ingest (repeatable). Defaults to MEMORY.md, memory/*.md, AGENTS.md, SOUL.md, USER.md, TOOLS.md",
)
@click.option(
    "--exclude",
    "excludes",
    multiple=True,
    help="Glob pattern(s) to exclude from indexing (repeatable). Defaults to backups/**, drafts/**, **/.archive/**",
)
@common_options
def ingest(workspace, patterns, excludes, db_path, provider, model, api_base, api_key, dim):
    """Index memory files into the vector store."""
    cfg = _build_config(workspace, db_path, provider, model, api_base, api_key, dim)
    if patterns:
        cfg.ingest.patterns = list(patterns)
    if excludes:
        cfg.ingest.exclude_patterns = list(excludes)

    files = discover_files(cfg)
    if not files:
        console.print(
            f"[yellow]No files matched patterns under {cfg.ingest.workspace}[/yellow]"
        )
        console.print(f"  patterns: {cfg.ingest.patterns}")
        sys.exit(1)

    console.print(f"[cyan]Discovered {len(files)} files under {cfg.ingest.workspace}[/cyan]")
    for f in files:
        console.print(f"  - {f}")

    console.print(
        f"[cyan]Loading embedder ({cfg.embedding.provider}: {cfg.embedding.model})…[/cyan]"
    )
    embedder = build_embedder(cfg)
    console.print(f"[cyan]Opening store at {cfg.db_path}[/cyan]")
    store = MemoryStore(cfg.db_path, embedder.dimension)

    def on_progress(source, n):
        console.print(f"  [green]✓[/green] {source} → {n} chunks")

    stats = ingest_files(cfg, embedder, store, files=files, on_progress=on_progress)
    console.print()
    console.print(
        f"[bold green]Done.[/bold green] {stats['files']} files, "
        f"+{stats['chunks_added']} chunks added, "
        f"-{stats['chunks_deleted']} stale removed. "
        f"Total in store: {store.count()}"
    )

@main.command()
@click.argument("query")
@click.option("-n", "--limit", default=5, help="Max results")
@click.option("--source", default=None, help="Filter by source file")
@click.option("--show-score", is_flag=True, help="Show similarity and decay scores")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON output")
@click.option("--no-decay", is_flag=True, default=False, help="Disable decay scoring")
@click.option("--half-life", type=float, default=30.0, help="Decay half-life in days")
@click.option("--tag", "tags", multiple=True, help="Filter by tag (format: category:value, e.g. topic:agent)")
@click.option("--exclude-tag", "exclude_tags", multiple=True, help="Exclude by tag (format: category:value)")
@common_options
def search(query, limit, source, show_score, as_json, no_decay, half_life, tags, exclude_tags, db_path, provider, model, api_base, api_key, dim):
    """Search the indexed memory."""
    cfg = _build_config(None, db_path, provider, model, api_base, api_key, dim)
    decay_cfg = _build_decay_config(enabled=not no_decay, half_life=half_life)

    # Parse tag filters
    tag_filter = {}
    tag_exclude = {}
    for t in tags:
        if ":" in t:
            cat, val = t.split(":", 1)
            tag_filter.setdefault(cat, []).append(val)
    for t in exclude_tags:
        if ":" in t:
            cat, val = t.split(":", 1)
            tag_exclude.setdefault(cat, []).append(val)

    embedder = build_embedder(cfg)
    store = MemoryStore(cfg.db_path, embedder.dimension, decay_config=decay_cfg)
    if store.count() == 0:
        console.print(
            "[yellow]Store is empty. Run 'amr ingest' first.[/yellow]"
        )
        sys.exit(1)

    results = do_search(
        query, cfg, embedder, store,
        limit=limit, source_filter=source,
        tag_filter=tag_filter or None,
        tag_exclude=tag_exclude or None,
    )

    if as_json:
        click.echo(json.dumps(results, indent=2, ensure_ascii=False))
        return

    console.print(f"[bold]Query:[/bold] {query}")
    console.print(f"[dim]Searching {store.count()} chunks…[/dim]\n")
    click.echo(format_results(results, show_score=show_score))

@main.command()
@click.option("--half-life", type=float, default=30.0, help="Decay half-life in days")
@click.option("--threshold", type=float, default=0.1, help="Decay factor threshold for stale chunks")
@click.option("--delete", "do_delete", is_flag=True, help="Actually delete stale chunks (dry run by default)")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON output")
@common_options
def decay(half_life, threshold, do_delete, as_json, db_path, provider, model, api_base, api_key, dim):
    """Analyze and optionally clean up stale memory chunks.

    By default, shows which chunks would be removed (dry run).
    Use --delete to actually remove them.
    """
    cfg = _build_config(None, db_path, provider, model, api_base, api_key, dim)
    decay_cfg = _build_decay_config(half_life=half_life, threshold=threshold)

    embedder = build_embedder(cfg)
    store = MemoryStore(cfg.db_path, embedder.dimension, decay_config=decay_cfg)

    if store.count() == 0:
        console.print("[yellow]Store is empty.[/yellow]")
        sys.exit(0)

    chunks = store.get_all_chunks()
    from .decay import find_stale_chunks
    import time

    stale = find_stale_chunks(chunks, half_life_days=half_life, threshold=threshold)

    if as_json:
        click.echo(json.dumps({
            "total_chunks": len(chunks),
            "stale_count": len(stale),
            "threshold": threshold,
            "half_life_days": half_life,
            "stale_chunks": stale,
        }, indent=2, ensure_ascii=False))
        return

    # Summary table
    table = Table(title="Memory Decay Analysis", show_header=True, header_style="bold cyan")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Total chunks", str(len(chunks)))
    table.add_row("Stale (below threshold)", str(len(stale)))
    table.add_row("Threshold", f"{threshold}")
    table.add_row("Half-life", f"{half_life} days")
    table.add_row("Action", "DELETE" if do_delete else "DRY RUN (use --delete to apply)")
    console.print(table)

    if stale:
        console.print(f"\n[dim]Stale chunks (decay < {threshold}):[/dim]")
        for chunk in stale[:20]:
            console.print(
                f"  [red]▼[/red] {chunk['source']} "
                f"(decay={chunk['decay_factor']}, "
                f"preview: {chunk['text'][:60]}…)"
            )
        if len(stale) > 20:
            console.print(f"  [dim]... and {len(stale) - 20} more[/dim]")

        if do_delete:
            removed = store.delete_stale(threshold=threshold, half_life_days=half_life)
            console.print(f"\n[green]✓ Removed {removed} stale chunks[/green]")
        else:
            console.print(f"\n[yellow]Dry run — no chunks removed. Use --delete to apply.[/yellow]")
    else:
        console.print("\n[green]No stale chunks found.[/green]")

@main.command()
@common_options
def status(db_path, provider, model, api_base, api_key, dim):
    """Show what's indexed in the store."""
    cfg = _build_config(None, db_path, provider, model, api_base, api_key, dim)
    if not cfg.db_path.exists():
        console.print(f"[yellow]No store at {cfg.db_path}[/yellow]")
        sys.exit(1)

    store = MemoryStore(cfg.db_path, cfg.embedding.dimension)
    total = store.count()
    sources = store.sources()

    console.print(f"[bold]Store:[/bold] {cfg.db_path}")
    console.print(f"[bold]Total chunks:[/bold] {total}")
    console.print(f"[bold]Sources ({len(sources)}):[/bold]")

    if sources:
        # Get per-source counts
        df = store.table.search().select(["source"]).limit(100000).to_pandas()
        counts = df["source"].value_counts().to_dict()
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Source")
        table.add_column("Chunks", justify="right")
        for src in sources:
            table.add_row(src, str(counts.get(src, 0)))
        console.print(table)

@main.command()
@click.option(
    "--workspace",
    type=click.Path(path_type=Path),
    default=DEFAULT_WORKSPACE,
    help=f"Workspace root (default: {DEFAULT_WORKSPACE})",
)
@click.option(
    "--days",
    type=int,
    default=7,
    help="How many days of notes to review",
)
@click.option(
    "--after",
    default=None,
    help="Only process notes after this date (YYYY-MM-DD)",
)
@click.option(
    "--apply",
    "do_apply",
    is_flag=True,
    help="Actually write to MEMORY.md (dry run by default)",
)
@click.option(
    "--no-archive",
    is_flag=True,
    help="Don't archive processed notes",
)
@click.option(
    "--prune-age",
    type=int,
    default=30,
    help="Prune auto-consolidated insights older than N days (0=disable)",
)
@click.option(
    "--min-confidence",
    type=float,
    default=0.5,
    help="Prune auto-consolidated insights below this confidence",
)
@click.option(
    "--min-per-category",
    type=int,
    default=2,
    help="Keep at least N insights per category when pruning",
)
@click.option("--json", "as_json", is_flag=True, help="Emit JSON output")
@common_options
def consolidate(workspace, days, after, do_apply, no_archive, prune_age, min_confidence, min_per_category, as_json, db_path, provider, model, api_base, api_key, dim):
    """Auto-extract insights from daily notes into MEMORY.md.

    Reviews recent daily notes, extracts decisions, corrections,
    preferences, and lessons, then adds new insights to MEMORY.md.

    By default, runs as dry run (no changes). Use --apply to write.
    """
    from .consolidation import consolidate as do_consolidate, format_consolidation_report

    click.echo(f"Reviewing daily notes from last {days} days…", err=True)
    if after:
        click.echo(f"  After date: {after}", err=True)

    result = do_consolidate(
        workspace=workspace,
        days_back=days,
        after_date=after,
        dry_run=not do_apply,
        archive=not no_archive,
        max_prune_age=prune_age,
        min_prune_confidence=min_confidence,
        min_per_category=min_per_category,
    )

    if as_json:
        click.echo(json.dumps({
            "notes_reviewed": result.notes_reviewed,
            "notes_archived": result.notes_archived,
            "insights_found": result.insights_found,
            "new_insights": result.new_insights,
            "unchanged": result.unchanged,
            "pruned": result.pruned,
            "categories": result.categories,
            "files_modified": result.files_modified,
            "dry_run": not do_apply,
        }, indent=2, ensure_ascii=False))
        return

    console.print(format_consolidation_report(result))

    if not do_apply and result.new_insights > 0:
        console.print(
            "\n[yellow]Dry run — no changes made. Use --apply to write to MEMORY.md.[/yellow]"
        )
    elif do_apply and result.new_insights > 0:
        console.print(
            f"\n[green]✓ Consolidated {result.new_insights} new insights into MEMORY.md[/green]"
        )
    elif result.new_insights == 0:
        console.print("\n[green]No new insights to consolidate.[/green]")

@main.command()
@click.argument("source", required=False)
@click.option("--all", "delete_all", is_flag=True, help="Delete the entire store")
@click.confirmation_option(prompt="Are you sure?")
@common_options
def delete(source, delete_all, db_path, provider, model, api_base, api_key, dim):
    """Delete chunks for a specific source, or the entire store."""
    cfg = _build_config(None, db_path, provider, model, api_base, api_key, dim)

    if delete_all:
        import shutil

        if cfg.db_path.exists():
            shutil.rmtree(cfg.db_path)
            console.print(f"[red]Deleted store at {cfg.db_path}[/red]")
        else:
            console.print("[yellow]Store does not exist[/yellow]")
        return

    if not source:
        console.print("[red]Provide a source path or --all[/red]")
        sys.exit(1)

    store = MemoryStore(cfg.db_path, cfg.embedding.dimension)
    removed = store.delete_source(source)
    console.print(f"[green]Removed {removed} chunks for source '{source}'[/green]")

if __name__ == "__main__":
    main()
