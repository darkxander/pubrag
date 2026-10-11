"""Orchestration : ingestion, puis chaîne question → réponse vérifiée."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .config import Config
from .embed import Embedder, get_embedder
from .fetch import NCBIClient
from .guardrails import REFUSAL, Verdict, check_answer, check_retrieval
from .index import build_documents_frame, build_index, load_index, save_index
from .retrieve import Hit, HybridRetriever


@dataclass
class Answer:
    question: str
    text: str
    hits: list[Hit] = field(default_factory=list)
    refused: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def sources(self) -> list[str]:
        seen: dict[str, str] = {}
        for hit in self.hits:
            seen.setdefault(hit.chunk.pmid, hit.chunk.citation)
        return list(seen.values())


def ingest(config: Config, offline: bool = False) -> tuple[int, int]:
    """Récupère le corpus, le découpe, l'encode et écrit l'index. Renvoie (documents, passages)."""
    with NCBIClient() as client:
        documents = client.fetch_documents(config.pmids)
    if not documents:
        raise RuntimeError("Aucun document récupéré — vérifiez les PMID et la connexion réseau.")

    config.documents_path.parent.mkdir(parents=True, exist_ok=True)
    build_documents_frame(documents).to_parquet(config.documents_path, index=False)

    embedder = get_embedder(offline=offline)
    chunks, embeddings = build_index(
        documents, embedder, config.target_tokens, config.overlap_tokens
    )
    save_index(config.index_path, chunks, embeddings)
    return len(documents), len(chunks)


class Pipeline:
    def __init__(
        self, config: Config, embedder: Embedder, retriever: HybridRetriever, generator=None
    ):
        self.config = config
        self.embedder = embedder
        self.retriever = retriever
        self._generator = generator

    @classmethod
    def from_index(cls, config: Config, offline: bool = False, generator=None) -> Pipeline:
        if not Path(config.index_path).exists():
            raise FileNotFoundError(
                f"Index absent ({config.index_path}). Lancez d'abord : pubrag ingest"
            )
        chunks, embeddings = load_index(config.index_path)
        embedder = get_embedder(offline=offline)
        return cls(config, embedder, HybridRetriever(chunks, embeddings, embedder), generator)

    @property
    def generator(self):
        if self._generator is None:
            from .generate import Generator

            self._generator = Generator(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
        return self._generator

    def retrieve(self, question: str) -> list[Hit]:
        return self.retriever.search(
            question,
            top_k=self.config.top_k,
            candidates=self.config.candidates_per_branch,
            rrf_k=self.config.rrf_k,
            max_per_doc=self.config.max_per_doc,
        )

    def ask(self, question: str) -> Answer:
        hits = self.retrieve(question)

        upstream: Verdict = check_retrieval(hits, self.config.min_relief)
        if not upstream.ok:
            return Answer(question, REFUSAL, hits, refused=True, warnings=upstream.reasons)

        text = self.generator.answer(question, hits)

        downstream: Verdict = check_answer(text, hits, self.config.require_citations)
        return Answer(question, text, hits, refused=False, warnings=downstream.reasons)
