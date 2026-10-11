from pubrag.guardrails import REFUSAL
from pubrag.pipeline import Pipeline

from .conftest import FakeGenerator, make_config


def test_pipeline_refuses_without_calling_the_model(retriever, embedder):
    generator = FakeGenerator("réponse [PMID:{pmid}]")
    pipeline = Pipeline(make_config(0.99), embedder, retriever, generator)

    answer = pipeline.ask("quelle est la capitale du Portugal")

    assert answer.refused
    assert answer.text == REFUSAL
    assert generator.calls == 0, "le modèle ne doit pas être appelé si le corpus ne couvre pas"


def test_pipeline_returns_sourced_answer(retriever, embedder):
    pipeline = Pipeline(
        make_config(-1.0), embedder, retriever, FakeGenerator("réponse [PMID:{pmid}]")
    )
    answer = pipeline.ask("SHANK3")
    assert not answer.refused
    assert answer.warnings == []
    assert answer.sources


def test_pipeline_warns_on_uncited_answer(retriever, embedder):
    pipeline = Pipeline(
        make_config(-1.0), embedder, retriever, FakeGenerator("réponse sans source")
    )
    answer = pipeline.ask("SHANK3")
    assert not answer.refused
    assert answer.warnings
