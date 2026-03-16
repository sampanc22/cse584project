from cache import SimpleSemanticCache
from validity import dialogue_valid, document_valid
from signatures import make_dialogue_signature
from adk_runtime import generate_fresh_with_agent

response_cache = SimpleSemanticCache(similarity_threshold=0.92)

def handle_request(
    prompt: str,
    long_context: str,
    dialogue_signature: dict,
    document_versions: dict,
    mode: str,
    history: list[str] | None = None,
) -> str:
    # Step 1: embed prompt once
    query_embedding = response_cache.embed_prompt(prompt)

    # Step 2: semantic retrieval only
    candidate = response_cache.lookup_with_embedding(query_embedding)

    # Step 3: validity filtering
    if candidate is not None:
        cached_dialogue = candidate.metadata.get("dialogue_signature", {})
        cached_versions = candidate.metadata.get("document_versions", {})

        if mode == "document":
            is_valid = document_valid(document_versions, cached_versions)
        elif mode == "dialogue":
            is_valid = dialogue_valid(dialogue_signature, cached_dialogue)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        if is_valid:
            print("RESPONSE CACHE HIT")
            return candidate.response

    # Step 4: recompute if no valid cached response
    print("RESPONSE CACHE MISS OR INVALID -> FRESH GEMINI CALL")
    fresh_response = generate_fresh_with_agent(
        prompt=prompt,
        long_context=long_context,
        history=history,
        mode=mode,
    )

    # Step 5: store response with the already-computed embedding
    response_cache.insert_with_embedding(
        prompt=prompt,
        response=fresh_response,
        embedding=query_embedding,
        metadata={
            "dialogue_signature": dialogue_signature,
            "document_versions": document_versions,
        },
    )

    return fresh_response


if __name__ == "__main__":
    history = [
        "I need a hotel in Boston.",
        "Somewhere downtown would be nice.",
    ]
    prompt = "What hotel in downtown Boston should I book?"
    dialogue_sig = make_dialogue_signature(prompt, history)

    print("===== DOCUMENT MODE TEST =====")
    doc_text_v1 = "Hotel database snapshot v1: Boston Plaza is downtown and costs $220 per night."
    doc_versions_v1 = {"hotel_doc": 1}

    r1 = handle_request(
        prompt=prompt,
        long_context=doc_text_v1,
        dialogue_signature=dialogue_sig,
        document_versions=doc_versions_v1,
        mode="document",
        history=history,
    )
    print("\nFIRST RESPONSE:\n", r1)

    r2 = handle_request(
        prompt=prompt,
        long_context=doc_text_v1,
        dialogue_signature=dialogue_sig,
        document_versions=doc_versions_v1,
        mode="document",
        history=history,
    )
    print("\nSECOND RESPONSE:\n", r2)

    doc_text_v2 = "Hotel database snapshot v2: Boston Plaza is sold out. Harbor Inn is downtown and costs $210."
    doc_versions_v2 = {"hotel_doc": 2}

    r3 = handle_request(
        prompt=prompt,
        long_context=doc_text_v2,
        dialogue_signature=dialogue_sig,
        document_versions=doc_versions_v2,
        mode="document",
        history=history,
    )
    print("\nTHIRD RESPONSE:\n", r3)

    # print("\n===== DIALOGUE MODE TEST =====")
    # dialogue_history_v1 = [
    #     "I need a hotel in Boston.",
    #     "Somewhere downtown would be nice.",
    # ]
    # prompt_dialogue = "What hotel should I book?"
    # dialogue_sig_v1 = make_dialogue_signature(prompt_dialogue, dialogue_history_v1)

    # context_dialogue = "Available hotels: Boston Plaza is downtown and costs $220 per night."

    # r4 = handle_request(
    #     prompt=prompt_dialogue,
    #     long_context=context_dialogue,
    #     dialogue_signature=dialogue_sig_v1,
    #     document_versions={},
    #     mode="dialogue",
    #     history=dialogue_history_v1,
    # )
    # print("\nFOURTH RESPONSE:\n", r4)

    # r5 = handle_request(
    #     prompt=prompt_dialogue,
    #     long_context=context_dialogue,
    #     dialogue_signature=dialogue_sig_v1,
    #     document_versions={},
    #     mode="dialogue",
    #     history=dialogue_history_v1,
    # )
    # print("\nFIFTH RESPONSE:\n", r5)

    # dialogue_history_v2 = [
    #     "I need a hotel in Boston.",
    #     "Actually, make that something cheap instead.",
    # ]
    # dialogue_sig_v2 = make_dialogue_signature(prompt_dialogue, dialogue_history_v2)

    # context_dialogue_v2 = (
    #     "Available hotels: Boston Plaza is downtown and costs $220 per night. "
    #     "Budget Stay is cheap and costs $90 per night."
    # )

    # r6 = handle_request(
    #     prompt=prompt_dialogue,
    #     long_context=context_dialogue_v2,
    #     dialogue_signature=dialogue_sig_v2,
    #     document_versions={},
    #     mode="dialogue",
    #     history=dialogue_history_v2,
    # )
    # print("\nSIXTH RESPONSE:\n", r6)