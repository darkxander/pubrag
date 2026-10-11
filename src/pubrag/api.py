"""API HTTP et interface web — `pubrag serve`, ou `uvicorn pubrag.api:app`.

L'interface (`/`) est une page HTML autonome, sans dépendance externe ni étape
de compilation. Elle s'appuie sur quatre points d'accès :

- `POST /search` — recherche seule. **Fonctionne sans clé API.**
- `POST /ask`    — recherche + réponse rédigée par Claude. Exige une clé ; en son
                   absence, renvoie 503 et l'interface affiche quand même les
                   passages retrouvés.
- `GET  /eval`   — le harnais d'évaluation, exécuté sur l'index chargé.
- `GET  /stats`  — composition du corpus.

L'index est chargé une fois au démarrage : recharger le modèle d'encodage à
chaque requête coûterait plusieurs secondes.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .config import load_config
from .evaluation import evaluate, load_questions
from .generate import MissingAPIKeyError
from .guardrails import check_retrieval, extract_citations
from .pipeline import Pipeline
from .retrieve import Hit

STATIC_DIR = Path(__file__).parent / "static"
PUBMED_URL = "https://pubmed.ncbi.nlm.nih.gov/{pmid}/"


class Question(BaseModel):
    question: str = Field(min_length=2, max_length=2000)


class HitOut(BaseModel):
    rank: int
    pmid: str
    title: str
    citation: str
    text: str
    dense_score: float
    bm25_rank: int | None  # rang lexical, à partir de 1 ; None si absent du vivier BM25
    rrf_score: float
    url: str


class Guardrail(BaseModel):
    ok: bool
    relief: float
    min_relief: float
    reasons: list[str]


class SearchResponse(BaseModel):
    question: str
    guardrail: Guardrail
    hits: list[HitOut]


class AskResponse(SearchResponse):
    answer: str
    refused: bool
    warnings: list[str]
    cited_pmids: list[str]


def _hits_out(hits: list[Hit]) -> list[HitOut]:
    return [
        HitOut(
            rank=i,
            pmid=h.chunk.pmid,
            title=h.chunk.title,
            citation=h.chunk.citation,
            text=h.chunk.text,
            dense_score=round(h.dense_score, 4),
            bm25_rank=None if h.bm25_rank is None else h.bm25_rank + 1,
            rrf_score=round(h.score, 5),
            url=PUBMED_URL.format(pmid=h.chunk.pmid),
        )
        for i, h in enumerate(hits, start=1)
    ]


def _guardrail(pipeline: Pipeline, hits: list[Hit]) -> Guardrail:
    verdict = check_retrieval(hits, pipeline.config.min_relief)
    return Guardrail(
        ok=verdict.ok,
        relief=round(hits[0].relief, 4) if hits else 0.0,
        min_relief=pipeline.config.min_relief,
        reasons=verdict.reasons,
    )


def create_app(pipeline: Pipeline | None = None) -> FastAPI:
    """Fabrique l'application. Un pipeline peut être injecté — c'est ce que font
    les tests, avec un encodeur de hachage et un générateur factice."""
    state: dict[str, Pipeline] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state["pipeline"] = pipeline or Pipeline.from_index(load_config())
        yield
        state.clear()

    app = FastAPI(title="pubrag", version="0.2.0", lifespan=lifespan)

    def current() -> Pipeline:
        if "pipeline" not in state:
            raise HTTPException(status_code=503, detail="Index non chargé")
        return state["pipeline"]

    @app.api_route(
        "/", methods=["GET", "HEAD"], response_class=HTMLResponse, include_in_schema=False
    )
    def index() -> str:
        return (STATIC_DIR / "index.html").read_text(encoding="utf-8")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/search", response_model=SearchResponse)
    def search(payload: Question) -> SearchResponse:
        p = current()
        hits = p.retrieve(payload.question)
        return SearchResponse(
            question=payload.question, guardrail=_guardrail(p, hits), hits=_hits_out(hits)
        )

    @app.post("/ask", response_model=AskResponse)
    def ask(payload: Question) -> AskResponse:
        p = current()
        try:
            answer = p.ask(payload.question)
        except MissingAPIKeyError as exc:
            # 503 et non 500 : rien n'est cassé, la fonction n'est pas configurée.
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return AskResponse(
            question=payload.question,
            guardrail=_guardrail(p, answer.hits),
            hits=_hits_out(answer.hits),
            answer=answer.text,
            refused=answer.refused,
            warnings=answer.warnings,
            cited_pmids=sorted(extract_citations(answer.text)),
        )

    @app.get("/eval")
    def run_eval() -> dict:
        results, summary = evaluate(current(), load_questions())
        return {"summary": summary.to_dict(), "results": [r.__dict__ for r in results]}

    @app.get("/stats")
    def stats() -> dict:
        p = current()
        chunks = p.retriever.chunks
        meta: dict[str, dict] = {}
        if p.config.documents_path.exists():
            frame = pd.read_parquet(p.config.documents_path)
            meta = {str(row["pmid"]): row for row in frame.to_dict("records")}

        publications: dict[str, dict] = {}
        for chunk in chunks:
            entry = publications.setdefault(
                chunk.pmid,
                {
                    "pmid": chunk.pmid,
                    "title": chunk.title,
                    "citation": chunk.citation,
                    "passages": 0,
                    "url": PUBMED_URL.format(pmid=chunk.pmid),
                },
            )
            entry["passages"] += 1
        for pmid, entry in publications.items():
            row = meta.get(pmid, {})
            entry["year"] = str(row.get("year", "") or "")
            entry["journal"] = str(row.get("journal", "") or "")
            entry["source"] = str(row.get("source", "") or "")

        ordered = sorted(publications.values(), key=lambda e: e["year"], reverse=True)
        return {
            "publications": ordered,
            "passages": len(chunks),
            "full_text": sum(e["source"] == "pmc_oa" for e in ordered),
            "model": p.config.model,
            "top_k": p.config.top_k,
            "max_per_doc": p.config.max_per_doc,
        }

    return app


app = create_app()
