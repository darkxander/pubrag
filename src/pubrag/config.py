"""Chargement de la configuration et des chemins du projet."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "corpus.yaml"
DATA_DIR = ROOT / "data"


@dataclass(frozen=True)
class Config:
    pmids: list[str]
    target_tokens: int
    overlap_tokens: int
    top_k: int
    candidates_per_branch: int
    rrf_k: int
    max_per_doc: int
    min_relief: float
    require_citations: bool
    model: str
    max_tokens: int
    temperature: float

    @property
    def documents_path(self) -> Path:
        return DATA_DIR / "documents.parquet"

    @property
    def index_path(self) -> Path:
        return DATA_DIR / "index.parquet"


def load_config(path: Path | None = None) -> Config:
    raw = yaml.safe_load((path or DEFAULT_CONFIG).read_text(encoding="utf-8"))
    chunking = raw.get("chunking", {})
    retrieval = raw.get("retrieval", {})
    guardrails = raw.get("guardrails", {})
    generation = raw.get("generation", {})
    return Config(
        pmids=[str(p) for p in raw["pmids"]],
        target_tokens=int(chunking.get("target_tokens", 220)),
        overlap_tokens=int(chunking.get("overlap_tokens", 40)),
        top_k=int(retrieval.get("top_k", 6)),
        candidates_per_branch=int(retrieval.get("candidates_per_branch", 20)),
        rrf_k=int(retrieval.get("rrf_k", 60)),
        max_per_doc=int(retrieval.get("max_per_doc", 2)),
        min_relief=float(guardrails.get("min_relief", 0.065)),
        require_citations=bool(guardrails.get("require_citations", True)),
        model=str(generation.get("model", "claude-sonnet-5")),
        max_tokens=int(generation.get("max_tokens", 1200)),
        temperature=float(generation.get("temperature", 0.0)),
    )
