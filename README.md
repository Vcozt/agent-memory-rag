# agent-memory-rag

RAG-powered semantic search for AI agent workspace memory files.

Built for the [OpenClaw](https://openclaw.ai) agent workspace pattern: index `MEMORY.md`, daily notes, and other markdown context files into a local vector store, then query them in natural language with proper source citations.

## Why

AI agents like OpenClaw, Claude Code, and Kiro keep their long-term memory in flat markdown files (`MEMORY.md`, `memory/YYYY-MM-DD.md`, `AGENTS.md`, etc). As that memory grows, three problems show up:

1. **Recall gets lossy** — the agent re-reads the same files every session and burns tokens on irrelevant context.
2. **Cross-file connections are missed** — a decision logged in a daily note rarely surfaces when the agent reads `MEMORY.md`.
3. **No grounded citations** — answers about past work are paraphrased, not quoted with line refs.

`agent-memory-rag` fixes all three: chunk → embed → vector store → cite.

## Features

- **Markdown-aware chunking** with heading-path tracking and line-range citations.
- **Local embeddings by default** (multilingual MiniLM via [fastembed](https://github.com/qdrant/fastembed), ONNX, ~80 MB, no torch).
- **OpenAI-compatible API mode** for hosted embedding providers.
- **LanceDB** vector store (file-based, zero-ops, supports filters).
- **CLI-first** with JSON output mode for agent integration.
- **Multi-language** — works on Indonesian + English + ~50 other languages.

## Install

```bash
git clone https://github.com/Vcozt/agent-memory-rag.git
cd agent-memory-rag
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

> Requires Python 3.10+. First run downloads the embedding model (~80 MB) to `~/.cache/`.

## Quick start

```bash
# Index the OpenClaw workspace memory (default)
amr ingest

# Search
amr search "kapan setup 9router?"
amr search "all decisions about Kiro" -n 10

# Filter by source
amr search "tunnel" --source MEMORY.md

# JSON output (for agent tooling)
amr search "embedding setup" --json | jq

# Inspect what's indexed
amr status
```

## Default workspace patterns

By default `amr ingest` looks under `~/.openclaw/workspace/` for:

- `MEMORY.md` — long-term curated memory
- `memory/*.md` — daily raw logs
- `AGENTS.md`, `SOUL.md`, `USER.md`, `TOOLS.md` — agent identity & runtime notes

Override with `--workspace /path/to/workspace --pattern '*.md' --pattern 'docs/*.md'`.

## Embedding providers

### Local (default)

Multilingual MiniLM, 384-dim, runs on CPU. ~80 MB model, ~50 ms per query on a small VPS. Best for personal workspaces and air-gapped use.

### API (OpenAI-compatible)

```bash
amr ingest \
  --provider api \
  --api-base https://api.openai.com/v1 \
  --api-key $OPENAI_API_KEY \
  --model text-embedding-3-small \
  --dim 1536
```

Works with anything that speaks `/v1/embeddings`: OpenAI, Together, Voyage, self-hosted Infinity, etc. The store is keyed by dimension, so don't mix providers in the same `--db` path.

## Re-indexing

`amr ingest` is idempotent per-source: it deletes all existing chunks for each file before re-inserting. Run it after editing memory files; safe to run on a cron / heartbeat.

## Architecture

```
┌──────────────┐   ┌──────────┐   ┌──────────┐   ┌─────────┐
│ markdown     │ → │ chunker  │ → │ embedder │ → │ LanceDB │
│ MEMORY.md    │   │ heading- │   │ local /  │   │ vector  │
│ memory/*.md  │   │ aware    │   │ API      │   │ store   │
└──────────────┘   └──────────┘   └──────────┘   └─────────┘
                                                      ↓
                                                  ┌─────────┐
                                                  │ search  │
                                                  │ + cite  │
                                                  └─────────┘
```

Each chunk carries: `source`, `heading_path`, `line_start`, `line_end`, `chunk_id`. Search results include all of these so the agent can quote with `Source: MEMORY.md#L42-L67`.

## Integrating with OpenClaw

Add a skill or cron job that runs `amr ingest` after memory updates, and query from the agent loop with `amr search "<question>" --json`. The default DB lives at `~/.local/share/agent-memory-rag/lancedb`.

## License

MIT — see [LICENSE](LICENSE).
