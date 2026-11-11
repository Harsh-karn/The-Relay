"""Unit tests for research paper ingestion and corpus validation."""

import pytest
from the_relay.retrieval.ingest import PaperCorpus, PAPERS_SPEC


def test_papers_specification_complete():
    """Verify all 3 papers required by the assignment brief are defined."""
    assert len(PAPERS_SPEC) == 3
    ids = [p["id"] for p in PAPERS_SPEC]
    assert "1706.03762v7" in ids
    assert "2410.00037v1" in ids
    assert "2507.07955v2" in ids


def test_corpus_loads_and_parses():
    """Verify that all three papers load from disk cache with valid text content."""
    corpus = PaperCorpus()
    docs = corpus.load_corpus()

    assert len(docs) == 3
    for spec in PAPERS_SPEC:
        pid = spec["id"]
        assert pid in docs
        doc = docs[pid]
        assert doc.title == spec["title"]
        assert doc.total_pages > 0
        assert len(doc.pages) == doc.total_pages
        # Verify text is non-empty
        full = doc.full_text()
        assert len(full) > 1000
        assert "[Page 1]" in full


def test_specific_paper_contents():
    """Verify key distinctive phrases from each paper exist in extracted text."""
    corpus = PaperCorpus()
    docs = corpus.load_corpus()

    # Paper 1: Transformer / Scaled Dot-Product
    assert "Scaled Dot-Product Attention" in docs["1706.03762v7"].full_text()

    # Paper 2: Moshi / Mimi audio codec
    assert "Mimi" in docs["2410.00037v1"].full_text() or "audio" in docs["2410.00037v1"].full_text().lower()

    # Paper 3: Dynamic Chunking / Hierarchical
    assert "Dynamic Chunking" in docs["2507.07955v2"].full_text() or "chunking" in docs["2507.07955v2"].full_text().lower()

