from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import math
import os

from dotenv import load_dotenv
from google import genai

load_dotenv()


@dataclass
class CacheEntry:
    prompt: str
    response: str
    embedding: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class SimpleSemanticCache:
    def __init__(self, similarity_threshold: float = 0.92):
        self.entries: List[CacheEntry] = []
        self.similarity_threshold = similarity_threshold

        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GOOGLE_API_KEY or GEMINI_API_KEY in .env")

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

    def lookup_with_embedding(self, query_embedding: List[float]) -> Optional[CacheEntry]:
        best = None
        best_sim = -1.0

        for entry in self.entries:
            sim = self._cosine(query_embedding, entry.embedding)
            if sim > best_sim:
                best_sim = sim
                best = entry

        if best is not None and best_sim >= self.similarity_threshold:
            return best
        return None

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
            )
        )