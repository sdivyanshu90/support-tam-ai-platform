"""Okapi BM25 over an in-memory corpus.

Why not a vector database: the knowledge base is 12 markdown files (~130 chunks).
An in-process lexical index is exact, has no cold-start or network dependency,
adds zero deployment surface, and — decisively for this problem — matches the
literal error codes (`ERR_CONNECTION_TIMEOUT`, `SCHEMA_MISMATCH`) that tickets
quote verbatim, which is where most known-issue matches actually come from.

Alongside the raw score each hit gets a `normalised_score` in [0, 1]: the share
of the query's total inverse-document-frequency mass that the chunk matched.
Raw BM25 is unbounded and corpus-dependent, so a threshold on it would not
survive a KB edit; the normalised form is comparable across corpora and is what
the known-issue gate is configured against.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

K1 = 1.5
B = 0.75

_TOKEN = re.compile(r"[a-z0-9_]+")
_STOPWORDS = frozenset(
    """a an and are as at be been but by can could did do does for from had has have
    he her his how i if in into is it its me my no nor not of on or our so than that
    the their them then there these they this to too us was we were what when where
    which who will with would you your please hello hi thanks team""".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, with underscore-joined codes kept *and* split.

    `ERR_CONNECTION_TIMEOUT` yields the whole token plus `err`, `connection`,
    `timeout`, so an exact code match scores highly while a partial description
    ("connection timeout") still retrieves the same chunk.
    """
    tokens: list[str] = []
    for raw in _TOKEN.findall(text.lower()):
        if "_" in raw:
            tokens.append(raw)
            tokens.extend(part for part in raw.split("_") if part)
        else:
            tokens.append(raw)
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


@dataclass(frozen=True)
class Scored:
    index: int
    score: float
    normalised_score: float


class BM25Index:
    """A frozen BM25 index. Build once, query many times."""

    def __init__(self, documents: list[str]) -> None:
        self._tokens: list[list[str]] = [tokenize(doc) for doc in documents]
        self._lengths: list[int] = [len(t) for t in self._tokens]
        self._n = len(documents)
        self._avg_len = (sum(self._lengths) / self._n) if self._n else 0.0

        self._freqs: list[dict[str, int]] = []
        doc_freq: dict[str, int] = {}
        for tokens in self._tokens:
            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            self._freqs.append(counts)
            for token in counts:
                doc_freq[token] = doc_freq.get(token, 0) + 1

        # BM25+ style idf: always positive, so a term in most documents still
        # contributes a little rather than penalising the match.
        self._idf = {
            token: math.log(1.0 + (self._n - df + 0.5) / (df + 0.5))
            for token, df in doc_freq.items()
        }

    def __len__(self) -> int:
        return self._n

    def search(self, query: str, *, top_k: int = 5) -> list[Scored]:
        """Return the best `top_k` documents, highest score first.

        Ties break on ascending document index, so a tie never reorders between
        runs.
        """
        query_tokens = [t for t in dict.fromkeys(tokenize(query))]
        known = [t for t in query_tokens if t in self._idf]
        if not known or self._n == 0:
            return []

        total_idf = sum(self._idf[t] for t in known)
        results: list[Scored] = []
        for index in range(self._n):
            counts = self._freqs[index]
            length = self._lengths[index] or 1
            norm = K1 * (1 - B + B * length / (self._avg_len or 1))
            score = 0.0
            matched_idf = 0.0
            for token in known:
                freq = counts.get(token, 0)
                if not freq:
                    continue
                saturation = (freq * (K1 + 1)) / (freq + norm)
                score += self._idf[token] * saturation
                # saturation / (K1 + 1) is in [0, 1): the share of this term's
                # information the document actually delivers.
                matched_idf += self._idf[token] * (saturation / (K1 + 1))
            if score <= 0.0:
                continue
            results.append(
                Scored(
                    index=index,
                    score=round(score, 6),
                    normalised_score=round(matched_idf / total_idf, 6) if total_idf else 0.0,
                )
            )

        results.sort(key=lambda s: (-s.score, s.index))
        return results[:top_k]
