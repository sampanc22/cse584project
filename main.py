from cache import SimpleSemanticCache
from validity import dialogue_valid, document_valid
from signatures import make_dialogue_signature
from adk_runtime import generate_fresh_with_agent
from document_registry import DocumentRegistry
import warnings

warnings.filterwarnings(
    "ignore",
    message=".*GeminiContextCacheManager.*",
    category=UserWarning,
)

warnings.filterwarnings(
    "ignore",
    message=".*FeatureName.AGENT_CONFIG.*",
    category=UserWarning,
)

response_cache = SimpleSemanticCache(
    similarity_threshold=0.92,
    max_entries=1000,
    eviction_policy="lru",
)
document_registry = DocumentRegistry()


def handle_request(
    prompt: str,
    long_context: str,
    dialogue_signature: dict,
    mode: str,
    history: list[str] | None = None,
    document_title: str | None = None,
    top_k: int = 5,
    dialogue_strictness: str = "strict",
    verbose: bool = True,
) -> str:
    # Step 1: embed prompt once
    query_embedding = response_cache.embed_prompt(prompt)

    # Step 2: retrieve top-k semantic matches
    candidates = response_cache.lookup_top_k_with_embedding(
        query_embedding=query_embedding,
        k=top_k,
    )

    # Step 3: derive current validity metadata
    if mode == "document":
        if not document_title:
            raise ValueError("document_title is required in document mode")
        current_document_versions = document_registry.get_document_versions(
            document_title,
            long_context,
        )
    elif mode == "dialogue":
        current_document_versions = {}
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    if verbose:
        print("\nCURRENT DOCUMENT VERSIONS:", current_document_versions)
        print("CACHE SIZE:", response_cache.size())
        print("TOP-K CANDIDATES FOUND:", len(candidates))

    # Step 4: validity filtering over top-k
    for rank, (candidate, sim) in enumerate(candidates, start=1):
        cached_dialogue = candidate.metadata.get("dialogue_signature", {})
        cached_versions = candidate.metadata.get("document_versions", {})

        if mode == "document":
            is_valid = document_valid(current_document_versions, cached_versions)
        else:  # dialogue
            is_valid = dialogue_valid(
                dialogue_signature,
                cached_dialogue,
                strictness=dialogue_strictness,
            )

        if verbose:
            print(
                f"CANDIDATE rank={rank} sim={sim:.4f} "
                f"valid={is_valid} hits={candidate.hit_count}"
            )

        if is_valid:
            response_cache.record_hit(candidate)
            if verbose:
                print("\nRESPONSE CACHE HIT")
            return candidate.response

    # Step 5: recompute if no valid cached response
    if verbose:
        print("\nRESPONSE CACHE MISS OR INVALID -> FRESH GEMINI CALL")

    fresh_response = generate_fresh_with_agent(
        prompt=prompt,
        long_context=long_context,
        history=history,
        mode=mode,
    )

    # Step 6: store response with derived metadata
    response_cache.insert_with_embedding(
        prompt=prompt,
        response=fresh_response,
        embedding=query_embedding,
        metadata={
            "dialogue_signature": dialogue_signature,
            "document_versions": current_document_versions,
            "mode": mode,
        },
    )

    return fresh_response


if __name__ == "__main__":
    history = []
    prompt = "Which NFL team represented the AFC at Super Bowl 50?"
    prompt2 = "Which NFL team represented the NFC at Super Bowl 50?"
    dialogue_sig = make_dialogue_signature(prompt, history)

    print("===== DOCUMENT MODE TEST (SQuAD STYLE) =====")
    doc_title = "Super_Bowl_50"

    doc_text_v1 = (
        "Super Bowl 50 was an American football game to determine the champion "
        "of the National Football League (NFL) for the 2015 season. "
        "The American Football Conference (AFC) champion Denver Broncos defeated "
        "the National Football Conference (NFC) champion Carolina Panthers 24–10 "
        "to earn their third Super Bowl title. The game was played on February 7, 2016, "
        "at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California."
    )

    doc_text_v2 = (
        "Super Bowl 50 was an American football game to determine the champion "
        "of the National Football League (NFL) for the 2015 season. "
        "The American Football Conference (AFC) champion Kansas City Chiefs defeated "
        "the National Football Conference (NFC) champion Carolina Panthers 24–10 "
        "to earn their third Super Bowl title. The game was played on February 7, 2016, "
        "at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California."
    )

    print("\n--- FIRST RESPONSE ---")
    r1 = handle_request(
        prompt=prompt,
        long_context=doc_text_v1,
        dialogue_signature=dialogue_sig,
        document_title=doc_title,
        mode="document",
        history=history,
        top_k=5,
        verbose=True,
    )
    print(r1)

    print("\n--- SECOND RESPONSE ---")
    r2 = handle_request(
        prompt=prompt,
        long_context=doc_text_v1,
        dialogue_signature=dialogue_sig,
        document_title=doc_title,
        mode="document",
        history=history,
        top_k=5,
        verbose=True,
    )
    print(r2)

    print("\n--- THIRD RESPONSE ---")
    r3 = handle_request(
        prompt=prompt,
        long_context=doc_text_v2,
        dialogue_signature=dialogue_sig,
        document_title=doc_title,
        mode="document",
        history=history,
        top_k=5,
        verbose=True,
    )
    print(r3)

    print("\n--- FOURTH RESPONSE ---")
    r4 = handle_request(
        prompt=prompt2,
        long_context=doc_text_v2,
        dialogue_signature=dialogue_sig,
        document_title=doc_title,
        mode="document",
        history=history,
        top_k=5,
        verbose=True,
    )
    print(r4)

    print("\n==============================\n")

    # # Example dialogue mode test
    # dialogue_history_v1 = [
    #     "I need a hotel in Boston.",
    #     "Somewhere downtown would be nice.",
    # ]
    # prompt_dialogue = "What hotel should I book?"
    # dialogue_sig_v1 = make_dialogue_signature(prompt_dialogue, dialogue_history_v1)

    # context_dialogue = (
    #     "Available hotels: Boston Plaza is downtown and costs $220 per night."
    # )

    # print("===== DIALOGUE MODE TEST =====")

    # print("\n--- FOURTH RESPONSE ---")
    # r4 = handle_request(
    #     prompt=prompt_dialogue,
    #     long_context=context_dialogue,
    #     dialogue_signature=dialogue_sig_v1,
    #     mode="dialogue",
    #     history=dialogue_history_v1,
    #     top_k=5,
    #     dialogue_strictness="strict",
    #     verbose=True,
    # )
    # print(r4)

    # print("\n--- FIFTH RESPONSE ---")
    # r5 = handle_request(
    #     prompt=prompt_dialogue,
    #     long_context=context_dialogue,
    #     dialogue_signature=dialogue_sig_v1,
    #     mode="dialogue",
    #     history=dialogue_history_v1,
    #     top_k=5,
    #     dialogue_strictness="strict",
    #     verbose=True,
    # )
    # print(r5)

    # dialogue_history_v2 = [
    #     "I need a hotel in Boston.",
    #     "Actually, make that something cheap instead.",
    # ]
    # dialogue_sig_v2 = make_dialogue_signature(prompt_dialogue, dialogue_history_v2)

    # context_dialogue_v2 = (
    #     "Available hotels: Boston Plaza is downtown and costs $220 per night. "
    #     "Budget Stay is cheap and costs $90 per night."
    # )

    # print("\n--- SIXTH RESPONSE ---")
    # r6 = handle_request(
    #     prompt=prompt_dialogue,
    #     long_context=context_dialogue_v2,
    #     dialogue_signature=dialogue_sig_v2,
    #     mode="dialogue",
    #     history=dialogue_history_v2,
    #     top_k=5,
    #     dialogue_strictness="strict",
    #     verbose=True,
    # )
    # print(r6)