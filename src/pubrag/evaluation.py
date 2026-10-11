"""Évaluation de la recherche, partagée par le script `eval/run_eval.py` et l'API.

On évalue la recherche seule, pas la génération : le rappel est la borne
supérieure de la qualité du système. Si la bonne source n'est pas retrouvée,
aucun modèle de langage ne pourra produire une réponse correcte et sourcée.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from .config import ROOT
from .guardrails import check_retrieval

QUESTIONS_PATH = ROOT / "eval" / "questions.yaml"
OUT_OF_CORPUS = "out_of_corpus"


@dataclass
class QuestionResult:
    question: str
    regime: str
    expected_pmids: list[str]
    retrieved_pmids: list[str]
    recall: float | None  # None pour les questions hors corpus
    relief: float
    refused_upstream: bool


@dataclass
class Summary:
    k: int
    min_relief: float
    recall_by_regime: dict[str, float]
    count_by_regime: dict[str, int]
    recall_global: float | None
    out_of_corpus_total: int
    out_of_corpus_refused: int

    def to_dict(self) -> dict:
        return asdict(self)


def load_questions(path: Path | None = None) -> list[dict]:
    return yaml.safe_load((path or QUESTIONS_PATH).read_text("utf-8"))["questions"]


def evaluate(pipeline, questions: list[dict]) -> tuple[list[QuestionResult], Summary]:
    results: list[QuestionResult] = []
    by_regime: dict[str, list[float]] = defaultdict(list)

    for item in questions:
        hits = pipeline.retrieve(item["q"])
        retrieved = list(dict.fromkeys(h.chunk.pmid for h in hits))
        expected = [str(p) for p in item.get("expected_pmids", [])]
        regime = item["regime"]
        refused = not check_retrieval(hits, pipeline.config.min_relief).ok
        relief = hits[0].relief if hits else 0.0

        recall = None
        if regime != OUT_OF_CORPUS and expected:
            recall = len(set(expected) & set(retrieved)) / len(expected)
            by_regime[regime].append(recall)

        results.append(
            QuestionResult(item["q"], regime, expected, retrieved, recall, relief, refused)
        )

    all_scores = [s for scores in by_regime.values() for s in scores]
    out = [r for r in results if r.regime == OUT_OF_CORPUS]
    summary = Summary(
        k=pipeline.config.top_k,
        min_relief=pipeline.config.min_relief,
        recall_by_regime={k: sum(v) / len(v) for k, v in sorted(by_regime.items())},
        count_by_regime={k: len(v) for k, v in sorted(by_regime.items())},
        recall_global=sum(all_scores) / len(all_scores) if all_scores else None,
        out_of_corpus_total=len(out),
        out_of_corpus_refused=sum(r.refused_upstream for r in out),
    )
    return results, summary
