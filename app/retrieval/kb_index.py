"""Knowledge-base retriever.

Build the index once per process (it is a pure function of the KB directory) and
serve `search()` from memory. Every hit carries its document path, heading and
chunk id so a downstream claim can be cited — and so a citation can be checked
against the real corpus rather than trusted.
"""

from __future__ import annotations

import re
from functools import lru_cache

from app.data.loader import load_kb_documents
from app.models import RetrievedChunk
from app.retrieval.bm25 import BM25Index
from app.retrieval.chunker import Chunk, chunk_documents

# Literal error identifiers such as ERR_CONNECTION_TIMEOUT or SCHEMA_MISMATCH.
_ERROR_CODE = re.compile(r"\b[A-Z][A-Z0-9]{2,}(?:_[A-Z0-9]+)+\b")

# A chunk containing an error code quoted verbatim in the ticket is strong
# evidence regardless of surrounding wording, so it gets a bounded bonus.
ERROR_CODE_BONUS = 0.25


class KnowledgeBaseIndex:
    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        self._by_id = {chunk.chunk_id: chunk for chunk in chunks}
        self._index = BM25Index(
            [f"{c.heading}\n{c.heading}\n{c.text}" for c in chunks]
        )

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def documents(self) -> list[str]:
        return sorted({chunk.document for chunk in self._chunks})

    def get(self, chunk_id: str) -> Chunk | None:
        return self._by_id.get(chunk_id)

    def search(self, query: str, *, top_k: int = 5) -> list[RetrievedChunk]:
        if not query or not query.strip():
            return []
        codes = set(_ERROR_CODE.findall(query))
        hits = self._index.search(query, top_k=top_k)

        results: list[RetrievedChunk] = []
        for hit in hits:
            chunk = self._chunks[hit.index]
            normalised = hit.normalised_score
            if codes and any(code in chunk.text for code in codes):
                normalised = min(1.0, normalised + ERROR_CODE_BONUS)
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    document=chunk.document,
                    heading=chunk.heading,
                    text=chunk.text,
                    score=hit.score,
                    normalised_score=round(normalised, 6),
                )
            )
        # The bonus can reorder hits; re-sort so the caller always sees the
        # best-normalised candidate first, with a stable tie-break.
        results.sort(key=lambda r: (-r.normalised_score, -r.score, r.chunk_id))
        return results


@lru_cache(maxsize=1)
def get_kb_index() -> KnowledgeBaseIndex:
    """Process-wide singleton. Built once; safe to call per request."""
    return KnowledgeBaseIndex(chunk_documents(load_kb_documents()))


def format_context(chunks: list[RetrievedChunk], *, max_chars: int = 5000) -> str:
    """Render retrieved chunks as delimited, clearly-untrusted evidence."""
    blocks: list[str] = []
    used = 0
    for chunk in chunks:
        block = (
            f'<kb_chunk id="{chunk.chunk_id}" document="{chunk.document}" '
            f'heading="{chunk.heading}" relevance="{chunk.normalised_score:.2f}">\n'
            f"{chunk.text}\n</kb_chunk>"
        )
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks) if blocks else "(no relevant knowledge-base passages found)"
