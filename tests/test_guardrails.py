from pubrag.guardrails import check_answer, check_retrieval, extract_citations


def test_extract_citations_handles_spacing():
    assert extract_citations("Vrai [PMID:123] et [PMID: 456].") == {"123", "456"}


def test_retrieval_refuses_when_nothing_retrieved():
    verdict = check_retrieval([], min_top_score=0.3)
    assert not verdict.ok


def test_retrieval_refuses_below_threshold(hits):
    verdict = check_retrieval(hits, min_top_score=0.99)
    assert not verdict.ok
    assert "seuil" in verdict.reasons[0]


def test_retrieval_accepts_above_threshold(hits):
    assert check_retrieval(hits, min_top_score=-1.0).ok


def test_answer_without_citation_is_flagged(hits):
    verdict = check_answer("SHANK3 est impliqué dans l'autisme.", hits, require_citations=True)
    assert not verdict.ok
    assert "aucune citation" in verdict.reasons[0]


def test_answer_citing_absent_pmid_is_flagged(hits):
    """Le contrôle décisif : le modèle cite un article qu'on ne lui a pas fourni."""
    verdict = check_answer("Affirmation [PMID:99999999].", hits)
    assert not verdict.ok
    assert "99999999" in verdict.unsupported_pmids


def test_answer_citing_supplied_pmid_passes(hits):
    pmid = hits[0].chunk.pmid
    verdict = check_answer(f"Affirmation étayée [PMID:{pmid}].", hits)
    assert verdict.ok
    assert verdict.cited_pmids == {pmid}
