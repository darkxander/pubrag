"""Recherche hybride : dense + BM25, fusionnées par Reciprocal Rank Fusion.

Pourquoi deux branches. La recherche dense retrouve les reformulations ("troubles
du spectre autistique" ≈ "ASD") mais rate souvent les identifiants exacts. BM25
fait l'inverse : il excelle sur "SHANK3", "MARCH6", "22q13", et ignore les
synonymes. Sur un corpus de génétique, saturé de symboles de gènes et de
coordonnées cytogénétiques, se priver de l'un des deux coûte cher.

Pourquoi un balayage exhaustif plutôt qu'un index vectoriel. Le corpus fait
quelques centaines de passages : un produit matriciel NumPy sur (N, 384) prend
moins d'une milliseconde, soit moins que la surcharge d'un appel à FAISS ou à
une base vectorielle. Le jour où le corpus dépasse ~100 000 passages, seule la
méthode `_dense_scores` est à remplacer — l'interface ne bouge pas.

Pourquoi un plafond de passages par article. Le corpus mélange des résumés
(2-4 passages) et des textes intégraux (34-93 passages). Sans plafond, un long
article occupe souvent tout le contexte : mesuré, 5 questions d'évaluation sur 9
recevaient leurs six passages d'un seul article ; avec un plafond de 2, aucune.
C'est un compromis, pas un correctif : le plafond ne change pas le rappel sur ce
jeu, il élargit le contexte au prix de la profondeur sur un article donné. Sans
évaluation de la génération, rien ne dit encore lequel des deux sert le mieux
les réponses.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from .chunk import Chunk
from .embed import Embedder

_TOKEN = re.compile(r"[a-z0-9]+")

# Mots-outils français et anglais. Sans cette liste, une question posée en
# français contre un corpus anglais est classée par BM25 sur « est », « la »,
# « le » : rares dans un texte anglais, donc d'IDF élevé, donc dominants — alors
# qu'ils ne portent aucune information.
_STOPWORDS = frozenset(
    """
    le la les un une des de du et est sont quel quelle quels quelles que qui quoi
    dans pour par sur avec ce cet cette ces au aux en il elle on ne pas plus ou se
    sa son ses leur leurs entre comment pourquoi ont été etre être fait sait y
    the an of and or in on at to for with by is are was were be been being what
    which who whom how why does do did this that these those it its from as about
    into than then there their they them can could would should has have had
    """.split()
)


def _strip_accents(text: str) -> str:
    # « sévérité » → « severite » : sans cette normalisation, le motif ASCII
    # découpait le mot en débris (« s », « v », « rit ») qui polluaient BM25.
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def tokenize(text: str) -> list[str]:
    tokens = _TOKEN.findall(_strip_accents(text.lower()))
    return [t for t in tokens if len(t) > 1 and t not in _STOPWORDS]


def _capped_order(order, chunks: list[Chunk], limit: int, max_per_doc: int | None) -> list[int]:
    """Premiers `limit` indices d'un classement, au plus `max_per_doc` par article.

    Appliquer le plafond *pendant* la constitution du vivier, et non seulement à
    la fin, empêche un long article de remplir à lui seul les candidats — sans
    quoi le plafond final n'aurait plus rien d'autre à proposer.
    """
    picked: list[int] = []
    per_doc: dict[str, int] = {}
    for index in order:
        index = int(index)
        pmid = chunks[index].pmid
        if max_per_doc is not None and per_doc.get(pmid, 0) >= max_per_doc:
            continue
        per_doc[pmid] = per_doc.get(pmid, 0) + 1
        picked.append(index)
        if len(picked) == limit:
            break
    return picked


@dataclass
class Hit:
    chunk: Chunk
    score: float  # score de fusion (RRF), sans unité
    dense_score: float  # cosinus, dans [-1, 1]
    bm25_rank: int | None
    relief: float = 0.0  # meilleur cosinus − cosinus médian ; sert au garde-fou amont


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray, embedder: Embedder) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(f"{len(chunks)} passages pour {len(embeddings)} vecteurs")
        self.chunks = chunks
        self.embeddings = embeddings
        self.embedder = embedder
        self._bm25 = BM25Okapi([tokenize(c.text) for c in chunks]) if chunks else None

    def _dense_scores(self, query: str) -> np.ndarray:
        # Vecteurs normalisés en amont : le produit scalaire est un cosinus.
        return self.embeddings @ self.embedder.encode_query(query)

    def search(
        self,
        query: str,
        top_k: int = 6,
        candidates: int = 20,
        rrf_k: int = 60,
        max_per_doc: int | None = 2,
    ) -> list[Hit]:
        if not self.chunks:
            return []

        dense = self._dense_scores(query)
        dense_order = _capped_order(np.argsort(-dense), self.chunks, candidates, max_per_doc)

        # « Relief » : écart entre le meilleur passage et le passage médian. Une
        # question hors sujet est uniformément moyenne avec tout le corpus ; une
        # question couverte présente un pic net. Mesuré, ce signal sépare bien
        # mieux les deux cas que le cosinus brut — voir guardrails.py.
        relief = float(dense.max() - np.median(dense))

        bm25_order: list[int] = []
        if self._bm25 is not None:
            lexical = np.asarray(self._bm25.get_scores(tokenize(query)))
            positive = [i for i in np.argsort(-lexical) if lexical[i] > 0]
            bm25_order = _capped_order(positive, self.chunks, candidates, max_per_doc)

        # Reciprocal Rank Fusion : 1/(k + rang). On fusionne des rangs, pas des
        # scores — cosinus et BM25 n'ont ni la même échelle ni la même
        # distribution, les additionner directement n'aurait pas de sens.
        fused: dict[int, float] = {}
        for rank, index in enumerate(dense_order):
            fused[int(index)] = fused.get(int(index), 0.0) + 1.0 / (rrf_k + rank + 1)
        for rank, index in enumerate(bm25_order):
            fused[int(index)] = fused.get(int(index), 0.0) + 1.0 / (rrf_k + rank + 1)

        bm25_rank_of = {int(index): rank for rank, index in enumerate(bm25_order)}

        hits: list[Hit] = []
        per_doc: dict[str, int] = {}
        for index, score in sorted(fused.items(), key=lambda item: -item[1]):
            pmid = self.chunks[index].pmid
            if max_per_doc is not None and per_doc.get(pmid, 0) >= max_per_doc:
                continue
            per_doc[pmid] = per_doc.get(pmid, 0) + 1
            hits.append(
                Hit(
                    chunk=self.chunks[index],
                    score=score,
                    dense_score=float(dense[index]),
                    bm25_rank=bm25_rank_of.get(index),
                    relief=relief,
                )
            )
            if len(hits) == top_k:
                break
        return hits
