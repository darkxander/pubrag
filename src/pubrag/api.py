"""API HTTP minimale — `uvicorn pubrag.api:app`.

Expose la même chaîne que la CLI. L'index est chargé une fois au démarrage :
recharger le modèle d'encodage à chaque requête coûterait plusieurs secondes.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import load_config
from .pipeline import Pipeline

_state: dict[str, Pipeline] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["pipeline"] = Pipeline.from_index(load_config())
    yield
    _state.clear()


app = FastAPI(title="pubrag", version="0.1.0", lifespan=lifespan)


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


class Source(BaseModel):
    pmid: str
    citation: str
    dense_score: float


class AnswerResponse(BaseModel):
    answer: str
    refused: bool
    warnings: list[str]
    sources: list[Source]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AnswerResponse)
def ask(payload: Question) -> AnswerResponse:
    pipeline = _state.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Index non chargé")

    answer = pipeline.ask(payload.question)
    return AnswerResponse(
        answer=answer.text,
        refused=answer.refused,
        warnings=answer.warnings,
        sources=[
            Source(pmid=h.chunk.pmid, citation=h.chunk.citation, dense_score=h.dense_score)
            for h in answer.hits
        ],
    )
