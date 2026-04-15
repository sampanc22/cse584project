from typing import Any, Dict


def dialogue_valid(
    current_sig: Dict[str, Any],
    cached_sig: Dict[str, Any],
    strictness: str = "strict",
) -> bool:
    if strictness == "strict":
        return current_sig == cached_sig

    current_domains = set(current_sig.get("domains", []))
    cached_domains = set(cached_sig.get("domains", []))

    current_intents = set(current_sig.get("intents", []))
    cached_intents = set(cached_sig.get("intents", []))

    current_constraints = current_sig.get("constraints", {})
    cached_constraints = cached_sig.get("constraints", {})

    current_requested = set(current_sig.get("requested_info", []))
    cached_requested = set(cached_sig.get("requested_info", []))

    if strictness == "intent_domain":
        return (
            current_domains == cached_domains
            and current_intents == cached_intents
        )

    if strictness == "slot_relaxed":
        if current_domains != cached_domains:
            return False
        if current_intents != cached_intents:
            return False
        if current_requested != cached_requested:
            return False

        for key, cached_val in cached_constraints.items():
            if current_constraints.get(key) != cached_val:
                return False

        return True

    raise ValueError(f"Unsupported dialogue strictness: {strictness}")


def document_valid(
    current_versions: Dict[str, int],
    cached_versions: Dict[str, int],
) -> bool:
    return current_versions == cached_versions