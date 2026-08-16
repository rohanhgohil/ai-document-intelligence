from __future__ import annotations

from dataclasses import dataclass

import faiss
import numpy as np

from app.services.ollama_client import OllamaClient


@dataclass
class RetrievedChunk:
    text: str
    score: float
    index: int


class LocalRAGRetriever:
    """In-memory FAISS index backed by Ollama embeddings."""

    def __init__(
        self,
        ollama: OllamaClient | None = None,
        batch_size: int = 8,
    ) -> None:
        self.ollama = ollama or OllamaClient()
        self.batch_size = batch_size
        self.index: faiss.Index | None = None
        self.chunks: list[str] = []

    def build(self, chunks: list[str]) -> None:
        if not chunks:
            raise ValueError("Cannot build a RAG index without chunks")

        all_vectors: list[list[float]] = []

        # Embed in small batches rather than sending the entire document
        # to Ollama in one request.
        for start in range(0, len(chunks), self.batch_size):
            batch = chunks[start:start + self.batch_size]

            print(
                f"[RAG] Embedding chunks "
                f"{start + 1}-{start + len(batch)} "
                f"of {len(chunks)}"
            )

            vectors = self.ollama.embed(batch)
            all_vectors.extend(vectors)

        vectors_np = np.asarray(all_vectors, dtype="float32")

        if vectors_np.ndim != 2 or vectors_np.shape[0] != len(chunks):
            raise ValueError(
                "Embedding response shape does not match chunks"
            )

        faiss.normalize_L2(vectors_np)

        self.index = faiss.IndexFlatIP(vectors_np.shape[1])
        self.index.add(vectors_np)
        self.chunks = chunks

    def search(
        self,
        query: str,
        top_k: int = 4,
    ) -> list[RetrievedChunk]:

        if self.index is None:
            raise RuntimeError("RAG index has not been built")

        print("[RAG] Embedding retrieval query...")

        query_vector = np.asarray(
            self.ollama.embed([query]),
            dtype="float32",
        )

        faiss.normalize_L2(query_vector)

        k = min(top_k, len(self.chunks))

        scores, indices = self.index.search(
            query_vector,
            k,
        )

        results: list[RetrievedChunk] = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            results.append(
                RetrievedChunk(
                    text=self.chunks[int(idx)],
                    score=float(score),
                    index=int(idx),
                )
            )

        return results