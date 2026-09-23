"""Construction et persistance de l'index.

Stockage en Parquet : colonnaire, compressé, typé, et lisible par pandas comme
par DuckDB ou Spark. Les vecteurs sont stockés en liste de float32 dans une
colonne — à cette échelle c'est plus simple qu'un format binaire séparé, et le
fichier reste inspectable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .chunk import Chunk, chunk_document
from .embed import Embedder
from .fetch import Document


def build_documents_frame(documents: list[Document]) -> pd.DataFrame:
    return pd.DataFrame([d.to_row() for d in documents])


def build_index(
    documents: list[Document],
    embedder: Embedder,
    target_tokens: int = 220,
    overlap_tokens: int = 40,
) -> tuple[list[Chunk], np.ndarray]:
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, target_tokens, overlap_tokens))
    if not chunks:
        return [], np.zeros((0, embedder.dimension), dtype=np.float32)
    embeddings = embedder.encode_documents([c.text for c in chunks])
    return chunks, embeddings


def save_index(path: Path, chunks: list[Chunk], embeddings: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "chunk_id": [c.chunk_id for c in chunks],
            "pmid": [c.pmid for c in chunks],
            "title": [c.title for c in chunks],
            "citation": [c.citation for c in chunks],
            "position": [c.position for c in chunks],
            "text": [c.text for c in chunks],
            "embedding": [row.astype(np.float32).tolist() for row in embeddings],
        }
    )
    frame.to_parquet(path, index=False, compression="zstd")


def load_index(path: Path) -> tuple[list[Chunk], np.ndarray]:
    frame = pd.read_parquet(path)
    chunks = [
        Chunk(
            chunk_id=row.chunk_id,
            pmid=row.pmid,
            title=row.title,
            citation=row.citation,
            position=int(row.position),
            text=row.text,
        )
        for row in frame.itertuples()
    ]
    embeddings = np.asarray([np.asarray(v, dtype=np.float32) for v in frame.embedding])
    return chunks, embeddings
