"""Tests de l'API et de l'interface web, sans index réel ni clé API."""

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from pubrag.api import create_app  # noqa: E402
from pubrag.generate import MissingAPIKeyError  # noqa: E402
from pubrag.pipeline import Pipeline  # noqa: E402

from .conftest import FakeGenerator, make_config  # noqa: E402


class NoKeyGenerator:
    def answer(self, question, hits):
        raise MissingAPIKeyError("ANTHROPIC_API_KEY absent.")


def client_for(retriever, embedder, generator, min_relief=-1.0) -> TestClient:
    pipeline = Pipeline(make_config(min_relief), embedder, retriever, generator)
    return TestClient(create_app(pipeline))


@pytest.fixture
def client(retriever, embedder):
    with client_for(retriever, embedder, FakeGenerator("Réponse étayée [PMID:{pmid}].")) as c:
        yield c


def test_index_serves_the_web_interface(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "pubrag" in response.text and "/search" in response.text


def test_head_on_index_is_allowed(client):
    """Les sondes de disponibilité envoient HEAD /."""
    assert client.head("/").status_code == 200


def test_search_returns_hits_and_guardrail_without_calling_the_model(client):
    body = client.post("/search", json={"question": "SHANK3 intellectual disability"}).json()
    assert body["hits"], "des passages doivent être retrouvés"
    first = body["hits"][0]
    assert first["rank"] == 1
    assert first["url"].startswith("https://pubmed.ncbi.nlm.nih.gov/")
    assert {"ok", "relief", "min_relief", "reasons"} <= body["guardrail"].keys()


def test_ask_returns_sourced_answer_and_cited_pmids(client):
    body = client.post("/ask", json={"question": "SHANK3"}).json()
    assert not body["refused"]
    assert body["warnings"] == []
    assert body["cited_pmids"] == [body["hits"][0]["pmid"]]


def test_ask_without_api_key_is_503_not_500(retriever, embedder):
    """Clé absente : fonction indisponible, pas panne. L'interface s'en sert pour
    afficher quand même les passages."""
    with client_for(retriever, embedder, NoKeyGenerator()) as c:
        response = c.post("/ask", json={"question": "SHANK3"})
    assert response.status_code == 503
    assert "ANTHROPIC_API_KEY" in response.json()["detail"]


def test_ask_refuses_out_of_corpus_without_api_key(retriever, embedder):
    """Le refus amont n'appelle pas le modèle : il marche même sans clé."""
    with client_for(retriever, embedder, NoKeyGenerator(), min_relief=0.99) as c:
        body = c.post("/ask", json={"question": "capitale du Portugal"}).json()
    assert body["refused"]
    assert not body["guardrail"]["ok"]


def test_too_short_question_is_rejected(client):
    assert client.post("/search", json={"question": "x"}).status_code == 422


def test_eval_returns_summary_and_one_result_per_question(client):
    from pubrag.evaluation import load_questions

    body = client.get("/eval").json()
    assert len(body["results"]) == len(load_questions())
    assert {"recall_global", "out_of_corpus_total", "min_relief"} <= body["summary"].keys()


def test_stats_lists_indexed_publications(client, retriever):
    body = client.get("/stats").json()
    assert body["passages"] == len(retriever.chunks)
    assert {p["pmid"] for p in body["publications"]} == {c.pmid for c in retriever.chunks}
