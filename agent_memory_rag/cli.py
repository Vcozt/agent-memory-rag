"""CLI entry point for agent-memory-rag."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from .config import Config, EmbeddingConfig, IngestConfig, DEFAULT_DB_PATH, DEFAULT_WORKSPACE
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
    return cfg


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
@click.option("--show-score", is_flag=True, help="Show similarity score")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON output")
@common_options
def search(query, limit, source, show_score, as_json, db_path, provider, model, api_base, api_key, dim):
    """Search the indexed memory."""
    cfg = _build_config(None, db_path, provider, model, api_base, api_key, dim)

    embedder = build_embedder(cfg)
    store = MemoryStore(cfg.db_path, embedder.dimension)
    if store.count() == 0:
        console.print(
            "[yellow]Store is empty. Run 'amr ingest' first.[/yellow]"
        )
        sys.exit(1)

    results = do_search(query, cfg, embedder, store, limit=limit, source_filter=source)

    if as_json:
        click.echo(json.dumps(results, indent=2, ensure_ascii=False))
        return

    console.print(f"[bold]Query:[/bold] {query}")
    console.print(f"[dim]Searching {store.count()} chunks…[/dim]\n")
    click.echo(format_results(results, show_score=show_score))


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
