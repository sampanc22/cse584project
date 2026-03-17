import hashlib
from typing import Dict


class DocumentRegistry:
    def __init__(self):
        # title -> {"document_id": str, "versions": {context_hash: version_int}}
        self._documents: Dict[str, Dict] = {}

    def _normalize_title(self, title: str) -> str:
        return title.strip().lower().replace(" ", "_")

    def _hash_context(self, context: str) -> str:
        return hashlib.sha256(context.strip().encode("utf-8")).hexdigest()

    def get_document_versions(self, title: str, context: str) -> Dict[str, int]:
        """
        Returns a document_versions dict in the format your cache already expects:
            {document_id: version}

        Policy:
        - Same title => same logical document_id
        - New context under same title => new version number
        - Previously seen context under same title => existing version number
        """
        norm_title = self._normalize_title(title)
        context_hash = self._hash_context(context)

        if norm_title not in self._documents:
            self._documents[norm_title] = {
                "document_id": norm_title,
                "versions": {context_hash: 1},
            }
            return {norm_title: 1}

        doc_info = self._documents[norm_title]
        versions = doc_info["versions"]

        if context_hash not in versions:
            versions[context_hash] = len(versions) + 1

        return {doc_info["document_id"]: versions[context_hash]}