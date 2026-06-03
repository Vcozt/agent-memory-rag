"""Configuration and defaults for agent-memory-rag."""

from pathlib import Path
from pydantic import BaseModel, Field
from typing import Optional

from .decay import DecayConfig


# Default paths
DEFAULT_WORKSPACE = Path.home() / ".openclaw" / "workspace"
DEFAULT_DB_PATH = Path.home() / ".local" / "share" / "agent-memory-rag" / "lancedb"

# Embedding
DEFAULT_LOCAL_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_EMBEDDING_DIM = 384

# Chunking
DEFAULT_CHUNK_SIZE = 512  # tokens approx (chars / 4)
DEFAULT_CHUNK_OVERLAP = 64


class EmbeddingConfig(BaseModel):
    """Embedding provider configuration."""

    provider: str = Field(default="local", description="local | api")
    model: str = Field(default=DEFAULT_LOCAL_MODEL)
    api_base: Optional[str] = Field(default=None, description="API base URL for remote embeddings")
    api_key: Optional[str] = Field(default=None, description="API key for remote embeddings")
    dimension: int = Field(default=DEFAULT_EMBEDDING_DIM)


class IngestConfig(BaseModel):
    """Ingestion configuration."""

    workspace: Path = Field(default=DEFAULT_WORKSPACE)
    patterns: list[str] = Field(
        default=["MEMORY.md", "memory/*.md", "AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md"]
    )
    exclude_patterns: list[str] = Field(
        default=["backups/**", "drafts/**", "**/.archive/**"],
        description="Glob patterns relative to workspace to exclude from indexing.",
    )
    chunk_size: int = Field(default=DEFAULT_CHUNK_SIZE)
    chunk_overlap: int = Field(default=DEFAULT_CHUNK_OVERLAP)


class Config(BaseModel):
    """Top-level configuration."""

    db_path: Path = Field(default=DEFAULT_DB_PATH)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    decay: DecayConfig = Field(default_factory=DecayConfig)
