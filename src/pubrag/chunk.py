"""Découpage des documents en passages indexables.

Le découpage se fait sur les frontières de paragraphes puis de phrases, jamais
au milieu d'un mot. Un recouvrement est conservé entre passages consécutifs :
sans lui, une phrase coupée en deux devient inintelligible pour la recherche
comme pour le modèle.

La longueur est mesurée en tokens approchés (4 caractères ≈ 1 token). C'est
volontaire : compter les tokens exactement imposerait une dépendance au
tokenizer d'un modèle donné, pour un gain nul à cette échelle.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

CHARS_PER_TOKEN = 4
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    pmid: str
    title: str
    citation: str
    position: int
    text: str


def approx_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def split_units(text: str) -> list[str]:
    """Découpe en paragraphes, puis en phrases si un paragraphe est trop long."""
    units: list[str] = []
    for paragraph in (p.strip() for p in text.split("\n\n")):
        if not paragraph:
            continue
        units.extend(s.strip() for s in _SENTENCE_END.split(paragraph) if s.strip())
    return units


def chunk_text(text: str, target_tokens: int = 220, overlap_tokens: int = 40) -> list[str]:
    units = split_units(text)
    if not units:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for unit in units:
        unit_tokens = approx_tokens(unit)

        # Une unité seule plus longue que la cible part dans son propre passage :
        # la refuser ou la tronquer perdrait de l'information.
        if unit_tokens >= target_tokens:
            if current:
                chunks.append(" ".join(current))
                current, current_tokens = [], 0
            chunks.append(unit)
            continue

        if current_tokens + unit_tokens > target_tokens and current:
            chunks.append(" ".join(current))
            current, current_tokens = _tail(current, overlap_tokens)

        current.append(unit)
        current_tokens += unit_tokens

    if current:
        chunks.append(" ".join(current))
    return chunks


def _tail(units: list[str], overlap_tokens: int) -> tuple[list[str], int]:
    """Dernières unités d'un passage, à reporter en tête du suivant."""
    kept: list[str] = []
    total = 0
    for unit in reversed(units):
        unit_tokens = approx_tokens(unit)
        if total + unit_tokens > overlap_tokens:
            break
        kept.insert(0, unit)
        total += unit_tokens
    return kept, total


def chunk_document(document, target_tokens: int = 220, overlap_tokens: int = 40) -> list[Chunk]:
    """Découpe un `Document`. Le titre est préfixé à chaque passage.

    Ce préfixe donne à chaque passage un contexte minimal : sans lui, un
    passage isolé du corps d'un article ne dit pas de quelle étude il provient,
    ce qui dégrade nettement la recherche dense.
    """
    pieces = chunk_text(document.text, target_tokens, overlap_tokens)
    return [
        Chunk(
            chunk_id=f"{document.pmid}:{position}",
            pmid=document.pmid,
            title=document.title,
            citation=document.citation,
            position=position,
            text=f"{document.title}\n\n{piece}",
        )
        for position, piece in enumerate(pieces)
    ]
