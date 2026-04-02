from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
import math
import os
import time

import httpx
from dotenv import load_dotenv
from google import genai

load_dotenv()


@dataclass
class CacheEntry:
    prompt: str
    response: str
    embedding: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    hit_count: int = 0


class SimpleSemanticCache:
    def __init__(
        self,
        similarity_threshold: float = 0.92,
        max_entries: int = 1000,
        eviction_policy: str = "lru",
    ):
        self.entries: List[CacheEntry] = []
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self.eviction_policy = eviction_policy.lower()

        if self.eviction_policy not in {"lru", "fifo"}:
            raise ValueError("eviction_policy must be 'lru' or 'fifo'")

        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GOOGLE_API_KEY or GEMINI_API_KEY in .env")

        self.client = genai.Client(api_key=api_key)

        # local in-memory embedding cache so repeated questions do not
        # repeatedly hit the embedding API
        self.embedding_cache: Dict[str, List[float]] = {}

    def embed_prompt(
        self,
        text: str,
        max_retries: int = 5,
        base_delay: float = 2.0,
    ) -> List[float]:
        if text in self.embedding_cache:
            return self.embedding_cache[text]

        last_err = None

        for attempt in range(max_retries):
            try:
                result = self.client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=text,
                )
                embedding = list(result.embeddings[0].values)
                self.embedding_cache[text] = embedding
                return embedding

            except Exception as e:
                last_err = e
                msg = f"{type(e).__name__}: {e}"

                is_retryable = (
                    isinstance(e, httpx.ConnectTimeout)
                    or "ConnectTimeout" in msg
                    or "timed out" in msg.lower()
                    or "503" in msg
                    or "UNAVAILABLE" in msg
                )

                if is_retryable and attempt < max_retries - 1:
                    sleep_time = base_delay * (2 ** attempt)
                    print(
                        f"[EMBED RETRY] attempt={attempt + 1}/{max_retries} "
                        f"sleeping {sleep_time:.1f}s after error: {msg}"
                    )
                    time.sleep(sleep_time)
                    continue

                raise

        raise last_err

    def _cosine(self, a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _evict_if_needed(self) -> None:
        while len(self.entries) > self.max_entries:
            if self.eviction_policy == "fifo":
                self.entries.pop(0)
            else:
                # crude LRU: evict the entry with smallest hit_count first.
                # not perfect recency, but fine for now.
                victim_idx = min(range(len(self.entries)), key=lambda i: self.entries[i].hit_count)
                self.entries.pop(victim_idx)

    def lookup_top_k_with_embedding(
        self,
        query_embedding: List[float],
        k: int = 5,
    ) -> List[Tuple[CacheEntry, float]]:
        scored: List[Tuple[CacheEntry, float]] = []

        for entry in self.entries:
            sim = self._cosine(query_embedding, entry.embedding)
            if sim >= self.similarity_threshold:
                scored.append((entry, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def record_hit(self, entry: CacheEntry) -> None:
        entry.hit_count += 1

    def insert_with_embedding(
        self,
        prompt: str,
        response: str,
        embedding: List[float],
        metadata: Dict[str, Any],
    ) -> None:
        self.entries.append(
            CacheEntry(
                prompt=prompt,
                response=response,
                embedding=embedding,
                metadata=metadata,
                hit_count=0,
            )
        )
        self._evict_if_needed()

    def size(self) -> int:
        return len(self.entries)