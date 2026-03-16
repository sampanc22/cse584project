from typing import Any, Dict


def dialogue_valid(current_sig: Dict[str, Any], cached_sig: Dict[str, Any]) -> bool:
    return current_sig == cached_sig


def document_valid(current_versions: Dict[str, int], cached_versions: Dict[str, int]) -> bool:
    for doc_id, cached_ver in cached_versions.items():
        if current_versions.get(doc_id) != cached_ver:
            return False
    return True