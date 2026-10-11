"""Calcul des vecteurs de représentation.

Deux implémentations derrière la même interface :

- `SentenceTransformerEmbedder` — le modèle réel, `BAAI/bge-small-en-v1.5`.
  384 dimensions, tourne en local sur CPU, aucune clé API. Un corpus de treize
  articles n'a pas besoin d'un modèle plus lourd.
- `HashingEmbedder` — un encodeur déterministe sans dépendance ni réseau, qui
  permet aux tests de s'exécuter en CI sans télécharger 130 Mo de poids. Il ne
  produit aucune sémantique et n'est pas destiné à l'usage réel.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

MODEL_NAME = "BAAI/bge-small-en-v1.5"

# bge attend ce préfixe sur les requêtes, pas sur les documents.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    dimension: int

    def encode_documents(self, texts: list[str]) -> np.ndarray: ...
    def encode_query(self, text: str) -> np.ndarray: ...


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Normalise chaque ligne, pour que le produit scalaire soit un cosinus."""
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        from sentence_transformers import SentenceTransformer  # import différé : dépendance lourde

        self._model = SentenceTransformer(model_name)
        # Méthode renommée dans les versions récentes de sentence-transformers.
        dim = getattr(self._model, "get_embedding_dimension", None) or (
            self._model.get_sentence_embedding_dimension
        )
        self.dimension = int(dim())

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(texts, batch_size=32, show_progress_bar=False)
        return l2_normalize(np.asarray(vectors, dtype=np.float32))

    def encode_query(self, text: str) -> np.ndarray:
        vector = self._model.encode([QUERY_PREFIX + text], show_progress_bar=False)
        return l2_normalize(np.asarray(vector, dtype=np.float32))[0]


class HashingEmbedder:
    """Sac de mots projeté par hachage. Déterministe, hors ligne, pour les tests."""

    def __init__(self, dimension: int = 256) -> None:
        self.dimension = dimension

    def _encode_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        return vector

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return l2_normalize(np.stack([self._encode_one(t) for t in texts]))

    def encode_query(self, text: str) -> np.ndarray:
        return l2_normalize(self._encode_one(text)[None, :])[0]


def get_embedder(offline: bool = False) -> Embedder:
    if offline:
        return HashingEmbedder()
    return SentenceTransformerEmbedder()
