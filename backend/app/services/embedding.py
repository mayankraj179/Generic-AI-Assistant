from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Protocol


class EmbeddingProviderError(Exception):
    """Raised when an embedding provider is misconfigured or a call fails.

    Callers (IngestionService, RetrievalService) must treat the message as
    internal detail — never forward it verbatim into an HTTP response.
    Mirrors the pattern ModelProviderError already establishes for the chat
    path (app/orchestration/model_provider.py).
    """


class EmbeddingProvider(Protocol):
    """Provider-neutral embedding boundary.

    Document and query embedding are kept as separate operations rather than
    one embed() method that guesses which it's doing: some models (including
    the one this framework uses — see gemini_embedding.py) are asymmetric,
    trained to embed "things to be searched" (documents) differently from
    "search intent" (queries) via a task-type distinction the model itself
    requires. embed_documents() is the batch form used during ingestion,
    where the underlying SDK supports embedding many chunks in one call.

    ``model_id`` names the vector space this provider embeds into (e.g.
    "gemini:gemini-embedding-001"). It is stored on every chunk at ingestion
    and filtered on at search, so a query is only ever compared against
    chunks embedded by the same model — vectors from different models are
    meaningless to compare, even when their dimensions happen to match.
    """

    dim: int
    model_id: str

    async def embed_document(self, text: str) -> list[float]: ...

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


class HashEmbeddingProvider:
    """Deterministic local embedding provider used when no external model SDK
    is configured.

    Preserves the repository contract while keeping vector search fully
    local and reproducible for tests — this is the default test fake;
    production wiring uses GeminiEmbeddingProvider instead (see
    app/services/gemini_embedding.py). Symmetric by construction: it has no
    notion of task type, so document and query embedding share identical
    logic.
    """

    def __init__(self, dim: int = 1536) -> None:
        if dim <= 0:
            raise ValueError("embedding dimension must be positive")
        self.dim = dim
        self.model_id = f"hash:{dim}"

    async def embed_document(self, text: str) -> list[float]:
        return self._embed(text)

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        tokens = re.findall(r"[A-Za-z0-9]+", (text or "").lower())
        vector = [0.0] * self.dim
        if not tokens:
            return vector

        for token in tokens:
            token_hash = hashlib.sha256(token.encode()).digest()
            for index in range(self.dim):
                digest_part = hashlib.sha256(f"{token}:{index}".encode()).digest()
                value = int.from_bytes(digest_part[:8], byteorder="big", signed=False)
                signal = ((value % 10_000) / 10_000.0) * 2.0 - 1.0
                vector[index] += signal / max(1, len(tokens))
                # Add a small deterministic bias to keep shared terms correlated across queries.
                bias = (int.from_bytes(token_hash[:4], "big") % 17) / 100.0
                vector[index] += bias * (1.0 / (index + 1))

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]
