"""Garde-fous : refuser plutôt que d'inventer.

Trois contrôles, appliqués à des moments différents.

1. *Avant* la génération — si le meilleur passage retrouvé a un cosinus trop
   faible, la question sort du corpus. On refuse sans appeler le modèle : cela
   évite une réponse plausible mais sans fondement, et économise l'appel.
2. *Après* la génération — la réponse doit citer au moins un PMID.
3. *Après* la génération — chaque PMID cité doit faire partie des passages
   effectivement transmis. C'est le contrôle décisif : il détecte le cas où le
   modèle cite un article réel de son pré-entraînement plutôt que la source
   qu'on lui a fournie.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

CITATION = re.compile(r"\[PMID:\s*(\d+)\]")

REFUSAL = (
    "Je ne trouve pas d'élément suffisamment proche de cette question dans le corpus "
    "indexé. Ce corpus ne couvre que les publications d'Alexandre Mathieu en génétique "
    "humaine ; je préfère ne pas répondre plutôt que d'extrapoler."
)


@dataclass
class Verdict:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    cited_pmids: set[str] = field(default_factory=set)
    unsupported_pmids: set[str] = field(default_factory=set)


def extract_citations(answer: str) -> set[str]:
    return set(CITATION.findall(answer))


def check_retrieval(hits, min_top_score: float) -> Verdict:
    """Contrôle amont : la question est-elle couverte par le corpus ?"""
    if not hits:
        return Verdict(ok=False, reasons=["aucun passage retrouvé"])
    best = max(hit.dense_score for hit in hits)
    if best < min_top_score:
        return Verdict(
            ok=False,
            reasons=[f"meilleur cosinus {best:.3f} < seuil {min_top_score:.3f}"],
        )
    return Verdict(ok=True)


def check_answer(answer: str, hits, require_citations: bool = True) -> Verdict:
    """Contrôle aval : la réponse est-elle étayée par les passages fournis ?"""
    cited = extract_citations(answer)
    available = {hit.chunk.pmid for hit in hits}
    unsupported = cited - available

    reasons: list[str] = []
    if require_citations and not cited:
        reasons.append("aucune citation [PMID:…] dans la réponse")
    if unsupported:
        reasons.append(
            "PMID cités mais absents du contexte fourni : " + ", ".join(sorted(unsupported))
        )

    return Verdict(
        ok=not reasons,
        reasons=reasons,
        cited_pmids=cited,
        unsupported_pmids=unsupported,
    )
