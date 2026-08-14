"""Verbatim quote verification.

The account brief promises that every evidence quote is a real span of a real
ticket. That promise is worth only as much as the check behind it, so quotes are
verified mechanically rather than trusted:

1. **Exact match** — the quote is already a substring of the source. Accept.
2. **Deterministic repair** — the quote matches after whitespace normalisation
   (a model re-wrapping a line, collapsing a newline into a space). We locate the
   corresponding span and return *the source's own text* for it, so the published
   quote is genuinely verbatim even though the model's copy was not.
3. **Reject** — anything else. Paraphrases, merged fragments and inventions are
   dropped, and the risk they were supporting is dropped with them.

Step 2 never invents: it can only ever return a span that exists in the source.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Wrapping characters a model tends to add around a copied span.
_TRIM = " \t\r\n\"'`“”‘’«»…"
_ELLIPSIS = re.compile(r"^\s*(?:\.\.\.|…)\s*|\s*(?:\.\.\.|…)\s*$")
_WS = re.compile(r"\s+")

# A quote must be long enough to actually evidence something. "Hi team," is a
# verbatim substring of most tickets and evidence of nothing.
MIN_QUOTE_CHARS = 20
MAX_QUOTE_CHARS = 400


@dataclass(frozen=True)
class QuoteVerdict:
    verified: bool
    quote: str
    method: str  # "exact" | "whitespace_repair" | "rejected"
    reason: str = ""


def _clean(quote: str) -> str:
    return _ELLIPSIS.sub("", quote or "").strip(_TRIM)


def _normalise_with_map(text: str) -> tuple[str, list[int]]:
    """Whitespace-collapsed text plus original index for each output character."""
    out: list[str] = []
    index_map: list[int] = []
    previous_space = False
    for i, char in enumerate(text):
        if char.isspace():
            if previous_space or not out:
                continue
            out.append(" ")
            index_map.append(i)
            previous_space = True
        else:
            out.append(char)
            index_map.append(i)
            previous_space = False
    while out and out[-1] == " ":
        out.pop()
        index_map.pop()
    return "".join(out), index_map


def verify_quote(quote: str, source: str) -> QuoteVerdict:
    """Check a model-supplied quote against its source text."""
    cleaned = _clean(quote)
    if not cleaned:
        return QuoteVerdict(False, "", "rejected", "empty quote")
    if len(cleaned) < MIN_QUOTE_CHARS:
        return QuoteVerdict(
            False, cleaned, "rejected", f"quote shorter than {MIN_QUOTE_CHARS} characters"
        )
    if len(cleaned) > MAX_QUOTE_CHARS:
        return QuoteVerdict(
            False, cleaned, "rejected", f"quote longer than {MAX_QUOTE_CHARS} characters"
        )
    if not source:
        return QuoteVerdict(False, cleaned, "rejected", "no source text to verify against")

    if cleaned in source:
        return QuoteVerdict(True, cleaned, "exact")

    norm_source, index_map = _normalise_with_map(source)
    norm_quote, _ = _normalise_with_map(cleaned)
    if norm_quote and norm_quote in norm_source:
        start = norm_source.index(norm_quote)
        end = start + len(norm_quote) - 1
        repaired = source[index_map[start] : index_map[end] + 1]
        # Guard the invariant: only ever publish text taken from the source.
        if repaired and repaired in source:
            return QuoteVerdict(True, repaired, "whitespace_repair")

    return QuoteVerdict(False, cleaned, "rejected", "not a verbatim substring of the source")


def ticket_source_text(ticket: dict) -> str:
    """The text a ticket quote may legitimately come from."""
    return f"{ticket.get('subject', '')}\n{ticket.get('body', '')}"
