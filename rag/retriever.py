from __future__ import annotations

from dataclasses import dataclass

from config.settings import get_settings
from rag.vectordb import get_vectorstore


@dataclass(frozen=True)
class Evidence:
    content: str
    citation: str
    score: float


def retrieve(query: str) -> list[Evidence]:
    """Return scored local evidence, never calls a remote search provider."""
    store = get_vectorstore()
    results = store.similarity_search_with_relevance_scores(query, k=get_settings().rag["top_k"])
    return [
        Evidence(
            content=document.page_content,
            citation=f"{document.metadata.get('source', 'local document')}, page {document.metadata.get('page', 0) + 1}",
            score=round(float(score), 3),
        )
        for document, score in results
    ]
