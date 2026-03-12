from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json
import math

from google.adk import Agent
from google.adk.apps.app import App
from google.adk.agents.context_cache_config import ContextCacheConfig


# ----------------------------
# 1. ADK app with context cache
# ----------------------------
root_agent = Agent(
    name="validity_agent",
    model="gemini-2.0-flash",
    instruction="Answer the user's question using the provided context."
)

app = App(
    name="validity-cache-app",
    root_agent=root_agent,
    context_cache_config=ContextCacheConfig(
        min_tokens=2048,
        ttl_seconds=600,
        cache_intervals=5,
    ),
)


# ----------------------------
# 2. Your response cache entry
# ----------------------------
@dataclass
class CacheEntry:
    prompt: str
    response: str
    embedding: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


# ----------------------------
# 3. Tiny local semantic cache
# ----------------------------
class SimpleSemanticCache:
    def __init__(self, similarity_threshold: float = 0.92):
        self.entries: List[CacheEntry] = []
        self.similarity_threshold = similarity_threshold

    def _embed(self, text: str) -> List[float]:
        # starter placeholder: replace with real embeddings later
        counts = [0.0] * 26
        for ch in text.lower():
            if "a" <= ch <= "z":
                counts[ord(ch) - ord("a")] += 1.0
        norm = math.sqrt(sum(x * x for x in counts)) or 1.0
        return [x / norm for x in counts]

    def _cosine(self, a: List[float], b: List[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    def lookup(self, prompt: str) -> Optional[CacheEntry]:
        q = self._embed(prompt)
        best = None
        best_sim = -1.0
        for entry in self.entries:
            sim = self._cosine(q, entry.embedding)
            if sim > best_sim:
                best_sim = sim
                best = entry
        if best is not None and best_sim >= self.similarity_threshold:
            return best
        return None

    def insert(self, prompt: str, response: str, metadata: Dict[str, Any]) -> None:
        self.entries.append(
            CacheEntry(
                prompt=prompt,
                response=response,
                embedding=self._embed(prompt),
                metadata=metadata,
            )
        )


# ----------------------------
# 4. Validity checks
# ----------------------------
def dialogue_valid(current_sig: Dict[str, Any], cached_sig: Dict[str, Any]) -> bool:
    return current_sig == cached_sig


def document_valid(current_versions: Dict[str, int], cached_versions: Dict[str, int]) -> bool:
    for doc_id, cached_ver in cached_versions.items():
        if current_versions.get(doc_id) != cached_ver:
            return False
    return True


# ----------------------------
# 5. Fresh generation through ADK
# ----------------------------
def generate_fresh_with_agent(prompt: str, long_context: str) -> str:
    """
    Pseudocode shape: send long_context + prompt to your ADK app/session runner.
    ADK will handle context caching under the hood if the request is large enough.
    Replace this with your actual ADK invocation code.
    """
    full_input = f"Context:\n{long_context}\n\nQuestion:\n{prompt}"

    # Placeholder until you wire in your actual ADK runner:
    return f"[FRESH GEMINI ANSWER] {full_input[:200]}..."


# ----------------------------
# 6. End-to-end request handler
# ----------------------------
response_cache = SimpleSemanticCache(similarity_threshold=0.92)


def handle_request(
    prompt: str,
    long_context: str,
    dialogue_signature: Dict[str, Any],
    document_versions: Dict[str, int],
) -> str:
    candidate = response_cache.lookup(prompt)

    if candidate is not None:
        cached_dialogue = candidate.metadata.get("dialogue_signature", {})
        cached_versions = candidate.metadata.get("document_versions", {})

        if dialogue_valid(dialogue_signature, cached_dialogue) and document_valid(document_versions, cached_versions):
            print("RESPONSE CACHE HIT")
            return candidate.response

    print("RESPONSE CACHE MISS OR INVALID -> FRESH GEMINI CALL")
    fresh_response = generate_fresh_with_agent(prompt, long_context)

    response_cache.insert(
        prompt=prompt,
        response=fresh_response,
        metadata={
            "dialogue_signature": dialogue_signature,
            "document_versions": document_versions,
        },
    )
    return fresh_response


# ----------------------------
# 7. Example usage
# ----------------------------
if __name__ == "__main__":
    doc_text = "Hotel database snapshot v1: Boston Plaza is downtown and costs $220 per night."
    prompt = "What hotel in downtown Boston is available?"
    dialogue_sig = {"domain": "travel", "city": "Boston"}
    doc_versions = {"hotel_doc": 1}

    r1 = handle_request(prompt, doc_text, dialogue_sig, doc_versions)
    print(r1)

    # Same state -> response cache hit
    r2 = handle_request(prompt, doc_text, dialogue_sig, doc_versions)
    print(r2)

    # Document changed -> response cache invalid, fresh generation again
    doc_text_v2 = "Hotel database snapshot v2: Boston Plaza is sold out. Harbor Inn costs $210."
    doc_versions_v2 = {"hotel_doc": 2}

    r3 = handle_request(prompt, doc_text_v2, dialogue_sig, doc_versions_v2)
    print(r3)