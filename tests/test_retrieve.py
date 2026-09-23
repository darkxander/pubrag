import numpy as np
import pytest

from pubrag.embed import HashingEmbedder, l2_normalize
from pubrag.retrieve import HybridRetriever, tokenize

from .conftest import make_chunk, unit_vectors


def test_tokenize_lowercases_and_drops_punctuation():
    assert tokenize("SHANK3, 22q13 deletion!") == ["shank3", "22q13", "deletion"]


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError):
        HybridRetriever([make_chunk("1", "a")], unit_vectors(2), HashingEmbedder())


def test_empty_index_returns_no_hits():
    retriever = HybridRetriever([], np.zeros((0, 256), dtype=np.float32), HashingEmbedder())
    assert retriever.search("anything") == []


def test_normalisation_makes_dot_product_a_cosine():
    matrix = l2_normalize(np.array([[3.0, 4.0]], dtype=np.float32))
    assert np.isclose(np.linalg.norm(matrix[0]), 1.0)


def test_bm25_branch_finds_exact_gene_symbol(documents, embedder):
    """Le lexical doit rattraper un symbole de gène, là où le dense peut échouer."""
    from pubrag.index import build_index

    chunks, embeddings = build_index(documents, embedder, target_tokens=60, overlap_tokens=10)
    retriever = HybridRetriever(chunks, embeddings, embedder)
    hits = retriever.search("MARCH6", top_k=3)
    assert "31664039" in {h.chunk.pmid for h in hits}


def test_search_respects_top_k(retriever):
    assert len(retriever.search("autism", top_k=2)) <= 2


def test_hits_are_sorted_by_fused_score(retriever):
    scores = [h.score for h in retriever.search("SHANK3", top_k=5)]
    assert scores == sorted(scores, reverse=True)
