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
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from .chunk import Chunk
from .embed import Embedder

_TOKEN = __import__("re").compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass
class Hit:
    chunk: Chunk
    score: float          # score de fusion (RRF), sans unité
    dense_score: float    # cosinus, dans [-1, 1] — c'est lui qui sert de garde-fou
    bm25_rank: int | None


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

    def search(self, query: str, top_k: int = 6, candidates: int = 20, rrf_k: int = 60) -> list[Hit]:
        if not self.chunks:
            return []

        dense = self._dense_scores(query)
        dense_order = np.argsort(-dense)[:candidates]

        bm25_order: list[int] = []
        if self._bm25 is not None:
            lexical = np.asarray(self._bm25.get_scores(tokenize(query)))
            bm25_order = [i for i in np.argsort(-lexical)[:candidates] if lexical[i] > 0]

        # Reciprocal Rank Fusion : 1/(k + rang). On fusionne des rangs, pas des
        # scores — cosinus et BM25 n'ont ni la même échelle ni la même
        # distribution, les additionner directement n'aurait pas de sens.
        fused: dict[int, float] = {}
        for rank, index in enumerate(dense_order):
            fused[int(index)] = fused.get(int(index), 0.0) + 1.0 / (rrf_k + rank + 1)
        for rank, index in enumerate(bm25_order):
            fused[int(index)] = fused.get(int(index), 0.0) + 1.0 / (rrf_k + rank + 1)

        bm25_rank_of = {int(index): rank for rank, index in enumerate(bm25_order)}
        ordered = sorted(fused.items(), key=lambda item: -item[1])[:top_k]

        return [
            Hit(
                chunk=self.chunks[index],
                score=score,
                dense_score=float(dense[index]),
                bm25_rank=bm25_rank_of.get(index),
            )
            for index, score in ordered
        ]
