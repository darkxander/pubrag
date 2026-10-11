from pubrag.config import Config
from pubrag.guardrails import REFUSAL
from pubrag.pipeline import Pipeline


def make_config(min_top_score: float) -> Config:
    return Config(
        pmids=[],
        target_tokens=60,
        overlap_tokens=10,
        top_k=3,
        candidates_per_branch=10,
        rrf_k=60,
        min_top_score=min_top_score,
        require_citations=True,
        model="claude-sonnet-5",
        max_tokens=500,
        temperature=0.0,
    )


class FakeGenerator:
    """Générateur factice : aucun appel réseau, réponse contrôlée."""

    def __init__(self, template: str) -> None:
        self.template = template
        self.calls = 0

    def answer(self, question, hits):
        self.calls += 1
        return self.template.format(pmid=hits[0].chunk.pmid)


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
