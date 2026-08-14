"""Markdown chunking for the knowledge base.

Follows the strategy DATA_SCHEMA.md recommends: split on `---` section rules,
keep heading hierarchy as metadata, and keep tables intact so an error-code row
stays attached to the heading that explains it.

Chunk ids are derived from a sorted document walk and a per-document counter,
so the same corpus always yields the same ids — retrieval results are citable
and comparable across runs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.data.loader import KBDocument

# A section chunk beyond this many characters is split on paragraph breaks so a
# single long section cannot dominate a retrieval prompt.
MAX_CHUNK_CHARS = 1400
MIN_CHUNK_CHARS = 40

_RULE = re.compile(r"^-{3,}\s*$")
_HEADING = re.compile(r"^(#{1,4})\s+(.*)$")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document: str
    heading: str
    text: str


def _flush(
    buffer: list[str],
    *,
    document: str,
    heading: str,
    counter: list[int],
    out: list[Chunk],
) -> None:
    body = "\n".join(buffer).strip()
    if len(body) < MIN_CHUNK_CHARS:
        return
    for piece in _split_long(body):
        out.append(
            Chunk(
                chunk_id=f"{document}#{counter[0]:03d}",
                document=document,
                heading=heading or "(document root)",
                text=piece,
            )
        )
        counter[0] += 1


def _split_long(body: str) -> list[str]:
    """Split an oversized section on blank lines, never mid-table."""
    if len(body) <= MAX_CHUNK_CHARS:
        return [body]
    pieces: list[str] = []
    current: list[str] = []
    size = 0
    for paragraph in body.split("\n\n"):
        addition = len(paragraph) + 2
        if size + addition > MAX_CHUNK_CHARS and current:
            pieces.append("\n\n".join(current).strip())
            current, size = [], 0
        current.append(paragraph)
        size += addition
    if current:
        pieces.append("\n\n".join(current).strip())
    return [p for p in pieces if p]


def chunk_document(document: KBDocument) -> list[Chunk]:
    """Split one markdown document into retrieval chunks."""
    out: list[Chunk] = []
    counter = [0]
    trail: dict[int, str] = {}
    buffer: list[str] = []
    heading = document.title

    def current_heading() -> str:
        parts = [trail[level] for level in sorted(trail) if trail.get(level)]
        return " > ".join(parts) if parts else document.title

    for line in document.text.splitlines():
        if _RULE.match(line):
            _flush(buffer, document=document.path, heading=heading, counter=counter, out=out)
            buffer = []
            continue

        match = _HEADING.match(line)
        if match:
            level = len(match.group(1))
            # A new heading at this level or above closes the previous chunk.
            if buffer:
                _flush(
                    buffer, document=document.path, heading=heading, counter=counter, out=out
                )
                buffer = []
            for deeper in [lvl for lvl in trail if lvl > level]:
                trail.pop(deeper, None)
            trail[level] = match.group(2).strip()
            heading = current_heading()
            buffer.append(line)
            continue

        buffer.append(line)

    _flush(buffer, document=document.path, heading=heading, counter=counter, out=out)
    return out


def chunk_documents(documents: tuple[KBDocument, ...]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document))
    return chunks
