from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import math
import os
import time
from itertools import count

from dotenv import load_dotenv
from google import genai

load_dotenv()


@dataclass
class CacheEntry:
    entry_id: int
    prompt: str
    response: str
    embedding: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    inserted_at: float = field(default_factory=time.time)
    last_accessed_at: float = field(default_factory=time.time)
    hit_count: int = 0


class SimpleSemanticCache:
    def __init__(
        self,
        similarity_threshold: float = 0.92,
        max_entries: int = 1000,
        eviction_policy: str = "lru",  # "lru" or "fifo"
    ):
        self.entries: List[CacheEntry] = []
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self.eviction_policy = eviction_policy.lower()
        self._id_counter = count(1)

        if self.eviction_policy not in {"lru", "fifo"}:
            raise ValueError("eviction_policy must be 'lru' or 'fifo'")

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GOOGLE_API_KEY in .env")

        self.client = genai.Client(api_key=api_key)

    def embed_prompt(self, text: str) -> List[float]:
        result = self.client.models.embed_content(
            model="gemini-embedding-001",
            contents=text,
        )
        return list(result.embeddings[0].values)

    def _cosine(self, a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _touch(self, entry: CacheEntry) -> None:
        entry.last_accessed_at = time.time()
        entry.hit_count += 1

    def _evict_if_needed(self) -> None:
        while len(self.entries) > self.max_entries:
            if self.eviction_policy == "fifo":
                victim = min(self.entries, key=lambda e: e.inserted_at)
            else:  # lru
                victim = min(self.entries, key=lambda e: e.last_accessed_at)

            self.entries = [e for e in self.entries if e.entry_id != victim.entry_id]

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
        self._touch(entry)

    def insert_with_embedding(
        self,
        prompt: str,
        response: str,
        embedding: List[float],
        metadata: Dict[str, Any],
    ) -> None:
        now = time.time()
        entry = CacheEntry(
            entry_id=next(self._id_counter),
            prompt=prompt,
            response=response,
            embedding=embedding,
            metadata=metadata,
            inserted_at=now,
            last_accessed_at=now,
            hit_count=0,
        )
        self.entries.append(entry)
        self._evict_if_needed()

    def size(self) -> int:
        return len(self.entries)