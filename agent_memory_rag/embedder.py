"""Embedding providers. Abstracts over local (fastembed) and OpenAI-compatible API."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Sequence

import httpx


class Embedder(ABC):
    """Abstract embedding interface."""

    dimension: int

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""
        raise NotImplementedError

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


class LocalEmbedder(Embedder):
    """Local embeddings via fastembed (ONNX, no torch required).

    Default model is multilingual MiniLM: ~80MB, supports ~50 languages
    including English and Indonesian. Dimension 384.
    """

    def __init__(self, model_name: str, dimension: int):
        # Lazy import: heavy dependency
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name)
        self.model_name = model_name
        self.dimension = dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        # fastembed returns a generator of numpy arrays
        return [v.tolist() for v in self._model.embed(list(texts))]


class APIEmbedder(Embedder):
    """OpenAI-compatible /v1/embeddings endpoint (e.g. OpenAI, Sumopod, 9router-routed)."""

    def __init__(self, api_base: str, api_key: str, model: str, dimension: int):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.dimension = dimension
        self._client = httpx.Client(timeout=30.0)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = self._client.post(
            f"{self.api_base}/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={"model": self.model, "input": list(texts)},
        )
        resp.raise_for_status()
        data = resp.json()
        if "data" not in data:
            raise RuntimeError(f"Unexpected embedding response: {data}")
        # Sort by index to be safe (most providers return in order, but spec is by index)
        items = sorted(data["data"], key=lambda x: x.get("index", 0))
        return [it["embedding"] for it in items]


def build_embedder(config) -> Embedder:
    """Factory: build the right embedder from a Config.embedding."""
    cfg = config.embedding
    if cfg.provider == "local":
        return LocalEmbedder(cfg.model, cfg.dimension)
    if cfg.provider == "api":
        if not cfg.api_base or not cfg.api_key:
            raise ValueError("api provider requires api_base and api_key")
        return APIEmbedder(cfg.api_base, cfg.api_key, cfg.model, cfg.dimension)
    raise ValueError(f"unknown embedding provider: {cfg.provider}")


def batched(iterable: Iterable, batch_size: int):
    """Yield successive batch_size-sized lists from iterable."""
    batch: list = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
