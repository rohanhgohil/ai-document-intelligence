from __future__ import annotations

import os
import time
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()


class OllamaClient:
    """HTTP client for local Ollama chat and embedding operations."""

    def __init__(self) -> None:
        self.base_url = os.getenv(
            "OLLAMA_BASE_URL",
            "http://localhost:11434",
        ).rstrip("/")

        self.llm_model = os.getenv(
            "OLLAMA_LLM_MODEL",
            "gemma3:4b",
        )

        self.embedding_model = os.getenv(
            "OLLAMA_EMBED_MODEL",
            "nomic-embed-text",
        )

        self.timeout = int(
            os.getenv("OLLAMA_TIMEOUT", "120")
        )

        self.keep_alive = os.getenv(
            "OLLAMA_KEEP_ALIVE",
            "30m",
        )

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        num_predict: int = 256,
    ) -> str:
        """
        Generate a JSON object from Ollama.

        Uses Ollama's JSON response format so the model is explicitly
        constrained to return valid JSON.
        """

        payload = {
            "model": self.llm_model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "stream": False,
            "format": "json",
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": 0,
                "num_predict": num_predict,
            },
        }

        start = time.perf_counter()

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )

            elapsed = time.perf_counter() - start

            response.raise_for_status()

            body: dict[str, Any] = response.json()

            content = (
                body.get("message", {})
                .get("content", "")
                .strip()
            )

            if not content:
                raise RuntimeError(
                    f"Ollama returned an empty response "
                    f"after {elapsed:.2f}s. "
                    f"Model={self.llm_model}"
                )

            print(
                f"[Ollama] model={self.llm_model} "
                f"time={elapsed:.2f}s "
                f"chars={len(content)}"
            )

            return content

        except requests.Timeout as exc:
            raise RuntimeError(
                f"Ollama timed out after {self.timeout}s "
                f"while using {self.llm_model}"
            ) from exc

    def embed(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        if not texts:
            return []

        payload = {
            "model": self.embedding_model,
            "input": texts,
            "keep_alive": self.keep_alive,
        }

        start = time.perf_counter()

        response = requests.post(
            f"{self.base_url}/api/embed",
            json=payload,
            timeout=self.timeout,
        )

        elapsed = time.perf_counter() - start

        response.raise_for_status()

        body: dict[str, Any] = response.json()

        embeddings = body.get("embeddings")

        if not embeddings:
            raise RuntimeError(
                f"Ollama returned no embeddings after "
                f"{elapsed:.2f}s"
            )

        print(
            f"[Ollama] embeddings={len(embeddings)} "
            f"time={elapsed:.2f}s"
        )

        return embeddings