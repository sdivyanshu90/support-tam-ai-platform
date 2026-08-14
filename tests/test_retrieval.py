"""Chunking and BM25 retrieval."""

from __future__ import annotations

from app.data.loader import KBDocument, load_kb_documents
from app.retrieval.bm25 import BM25Index, tokenize
from app.retrieval.chunker import chunk_document, chunk_documents
from app.retrieval.kb_index import KnowledgeBaseIndex, format_context

SAMPLE = KBDocument(
    path="knowledge-base/test/sample.md",
    title="Sample",
    text=(
        "# Sample Product\n\n"
        "Intro paragraph that is comfortably longer than the minimum chunk size.\n\n"
        "---\n\n"
        "## Errors\n\n"
        "| Error | Cause |\n|---|---|\n| ERR_TEST_FAILURE | something went wrong here |\n\n"
        "---\n\n"
        "## Setup\n\n"
        "Steps to configure the product correctly for a production deployment.\n"
    ),
)


def test_tokenizer_keeps_error_codes_whole_and_split():
    tokens = tokenize("Error: ERR_CONNECTION_TIMEOUT after 30s")
    assert "err_connection_timeout" in tokens
    assert "connection" in tokens and "timeout" in tokens


def test_tokenizer_drops_stopwords_and_single_chars():
    assert tokenize("the a an of I") == []


def test_chunking_splits_on_rules_and_tracks_headings():
    chunks = chunk_document(SAMPLE)
    assert len(chunks) >= 3
    headings = " ".join(c.heading for c in chunks)
    assert "Errors" in headings and "Setup" in headings


def test_chunk_ids_are_stable_across_runs():
    first = [c.chunk_id for c in chunk_document(SAMPLE)]
    second = [c.chunk_id for c in chunk_document(SAMPLE)]
    assert first == second
    assert all(c.startswith("knowledge-base/test/sample.md#") for c in first)


def test_table_row_stays_with_its_heading():
    chunks = chunk_document(SAMPLE)
    error_chunk = next(c for c in chunks if "ERR_TEST_FAILURE" in c.text)
    assert "Errors" in error_chunk.heading


def test_bm25_ranking_is_deterministic():
    index = BM25Index(["alpha beta gamma", "beta gamma delta", "totally unrelated words"])
    first = [(s.index, s.score) for s in index.search("beta gamma", top_k=3)]
    second = [(s.index, s.score) for s in index.search("beta gamma", top_k=3)]
    assert first == second


def test_bm25_normalised_score_is_bounded():
    index = BM25Index(["alpha beta gamma", "beta gamma delta"])
    for hit in index.search("alpha beta gamma", top_k=2):
        assert 0.0 <= hit.normalised_score <= 1.0


def test_bm25_returns_nothing_for_unknown_terms():
    index = BM25Index(["alpha beta gamma"])
    assert index.search("xylophone quokka", top_k=3) == []


def test_real_kb_retrieves_error_code_reference(kb_index: KnowledgeBaseIndex):
    hits = kb_index.search("ERR_CONNECTION_TIMEOUT after 30s pipeline failing", top_k=3)
    assert hits
    assert any("ERR_CONNECTION_TIMEOUT" in hit.text for hit in hits)
    assert hits[0].normalised_score > 0.3


def test_error_code_match_boosts_relevance(kb_index: KnowledgeBaseIndex):
    with_code = kb_index.search("SAML_ASSERTION_EXPIRED", top_k=1)
    assert with_code and with_code[0].normalised_score >= 0.25


def test_off_topic_query_retrieves_nothing(kb_index: KnowledgeBaseIndex):
    """The 'no relevant KB results' path must actually be reachable."""
    assert kb_index.search("quokka platypus armadillo wombat", top_k=5) == []


def test_incidental_vocabulary_overlap_still_scores(kb_index: KnowledgeBaseIndex):
    """A documented limitation of lexical retrieval, pinned by a test.

    "sourdough starter" scores well against the billing document purely because
    "Starter" is a plan tier. Retrieval score alone therefore cannot decide a
    known-issue match — which is why `TriageService` additionally requires the
    model to cite a retrieved chunk id and quote it verbatim. See DESIGN.md.
    """
    hits = kb_index.search("sourdough bread hydration ratio starter", top_k=3)
    assert hits, "the collision is real; this test documents it rather than hiding it"
    assert any("starter" in hit.text.lower() for hit in hits)


def test_empty_query_is_handled(kb_index: KnowledgeBaseIndex):
    assert kb_index.search("", top_k=5) == []
    assert kb_index.search("   ", top_k=5) == []


def test_every_chunk_cites_a_real_document(kb_index: KnowledgeBaseIndex):
    real_paths = {d.path for d in load_kb_documents()}
    assert set(kb_index.documents) <= real_paths


def test_format_context_marks_untrusted_content(kb_index: KnowledgeBaseIndex):
    hits = kb_index.search("SSO SAML configuration", top_k=2)
    rendered = format_context(hits)
    assert "<kb_chunk" in rendered and "</kb_chunk>" in rendered


def test_format_context_handles_no_hits():
    assert "no relevant" in format_context([]).lower()


def test_format_context_respects_char_budget(kb_index: KnowledgeBaseIndex):
    hits = kb_index.search("SSO SAML configuration error", top_k=5)
    assert len(format_context(hits, max_chars=300)) <= 400


def test_chunk_documents_covers_whole_corpus():
    chunks = chunk_documents(load_kb_documents())
    assert len(chunks) > 50
    assert all(chunk.text.strip() for chunk in chunks)
