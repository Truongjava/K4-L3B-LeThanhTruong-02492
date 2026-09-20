from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    An in-memory vector store for text chunks.

    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._store: list[dict[str, Any]] = []
        self._next_index = 0

    def _make_record(self, doc: Document) -> dict[str, Any]:
        """Normalise one Document into the record shape the store holds."""
        # Copy rather than alias: the caller's dict stays theirs to mutate.
        metadata = dict(doc.metadata)
        # delete_document() matches on this key, so guarantee it exists even when
        # a caller only set Document.id -- otherwise every delete is a silent
        # no-op that returns False.
        metadata.setdefault("doc_id", doc.id)
        self._next_index += 1
        return {
            "id": f"{doc.id}#{self._next_index}",
            "content": doc.content,
            "metadata": metadata,
            "embedding": self._embedding_fn(doc.content),
        }

    def _search_records(
        self, query: str, records: list[dict[str, Any]], top_k: int
    ) -> list[dict[str, Any]]:
        """Rank an arbitrary record set against the query.

        Both search() and search_with_filter() route through here, differing only
        in which records they pass in. Sharing one path means the two cannot
        disagree on scoring, so an empty filter provably changes nothing.

        The stored embedding is dropped from the results -- a 1536-dimension
        vector is pure noise once it has been scored.
        """
        query_embedding = self._embedding_fn(query)
        scored = [
            {
                "id": record["id"],
                "content": record["content"],
                "metadata": record["metadata"],
                # Stored vectors are already unit length, so the dot product is
                # the cosine similarity.
                "score": _dot(query_embedding, record["embedding"]),
            }
            for record in records
        ]
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        One Document becomes one record -- chunking is the caller's job, at the
        layer above this store.
        """
        for doc in docs:
            self._store.append(self._make_record(doc))

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.
        """
        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        Filter first, then search. Taking top_k and discarding non-matching rows
        afterwards can leave nothing at all: the k slots were already spent on
        documents that were never eligible.
        """
        if not metadata_filter:
            return self._search_records(query, self._store, top_k)

        candidates = [
            record
            for record in self._store
            if all(
                record["metadata"].get(key) == value
                for key, value in metadata_filter.items()
            )
        ]
        return self._search_records(query, candidates, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        remaining = [
            record for record in self._store if record["metadata"].get("doc_id") != doc_id
        ]
        removed = len(self._store) - len(remaining)
        self._store = remaining
        return removed > 0
