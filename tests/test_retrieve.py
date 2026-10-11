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


def test_relief_is_computed_and_shared_by_all_hits(retriever):
    hits = retriever.search("SHANK3", top_k=3)
    assert hits
    assert all(h.relief == hits[0].relief for h in hits)
    assert hits[0].relief >= 0.0


def test_per_document_cap_limits_monopolisation(embedder):
    """Un long article ne doit pas occuper tout le contexte."""
    from pubrag.chunk import Chunk

    long_doc = [
        Chunk(f"111:{i}", "111", "Long", "C", i, f"shank synapse passage {i}") for i in range(10)
    ]
    short_doc = [Chunk("222:0", "222", "Short", "C", 0, "shank synapse other study")]
    chunks = long_doc + short_doc
    retriever = HybridRetriever(
        chunks, embedder.encode_documents([c.text for c in chunks]), embedder
    )

    capped = retriever.search("shank synapse", top_k=6, max_per_doc=2)
    per_doc = {}
    for h in capped:
        per_doc[h.chunk.pmid] = per_doc.get(h.chunk.pmid, 0) + 1
    assert per_doc.get("111", 0) <= 2
    assert "222" in per_doc, "le second article doit trouver sa place"


def test_cap_can_be_disabled(embedder):
    from pubrag.chunk import Chunk

    chunks = [Chunk(f"111:{i}", "111", "T", "C", i, f"shank passage {i}") for i in range(8)]
    retriever = HybridRetriever(
        chunks, embedder.encode_documents([c.text for c in chunks]), embedder
    )
    assert len(retriever.search("shank", top_k=5, max_per_doc=None)) == 5


def test_tokenize_handles_french_accents():
    """Régression : l'ancien motif découpait « sévérité » en « s », « v », « rit »."""
    tokens = tokenize("sévérité de la déficience intellectuelle")
    assert "severite" in tokens and "deficience" in tokens
    assert not any(len(t) == 1 for t in tokens)


def test_tokenize_drops_french_and_english_stopwords():
    tokens = tokenize("Quel est le lien entre les mutations SHANK et the autism?")
    assert tokens == ["lien", "mutations", "shank", "autism"]
