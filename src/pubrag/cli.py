"""Interface en ligne de commande."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import load_config
from .pipeline import Pipeline, ingest

app = typer.Typer(add_completion=False, help="RAG sur un corpus de publications PubMed.")
console = Console()

ConfigOption = typer.Option(None, "--config", "-c", help="Chemin du fichier de configuration.")
OfflineOption = typer.Option(False, "--offline", help="Encodeur de test, sans téléchargement.")


@app.command()
def ingest_cmd(config: Path = ConfigOption, offline: bool = OfflineOption) -> None:
    """Récupère le corpus depuis PubMed, le découpe, l'encode et écrit l'index."""
    cfg = load_config(config)
    with console.status("Récupération depuis le NCBI…"):
        documents, chunks = ingest(cfg, offline=offline)
    console.print(f"[green]✓[/green] {documents} documents, {chunks} passages → {cfg.index_path}")


@app.command()
def ask(
    question: str = typer.Argument(..., help="La question posée au corpus."),
    config: Path = ConfigOption,
    offline: bool = OfflineOption,
    show_sources: bool = typer.Option(True, "--sources/--no-sources"),
) -> None:
    """Pose une question et affiche la réponse sourcée."""
    pipeline = Pipeline.from_index(load_config(config), offline=offline)
    answer = pipeline.ask(question)

    style = "yellow" if answer.refused else "cyan"
    console.print(Panel(answer.text, title=question, border_style=style))

    if show_sources and answer.hits:
        table = Table("PMID", "Cosinus", "BM25", "Source", show_lines=False)
        for hit in answer.hits:
            table.add_row(
                hit.chunk.pmid,
                f"{hit.dense_score:.3f}",
                "—" if hit.bm25_rank is None else f"#{hit.bm25_rank + 1}",
                hit.chunk.title[:60],
            )
        console.print(table)

    for warning in answer.warnings:
        console.print(f"[yellow]⚠ garde-fou :[/yellow] {warning}")


@app.command()
def search(
    query: str = typer.Argument(..., help="Requête de recherche."),
    config: Path = ConfigOption,
    offline: bool = OfflineOption,
) -> None:
    """Affiche les passages retrouvés, sans appeler le modèle de langage."""
    pipeline = Pipeline.from_index(load_config(config), offline=offline)
    for hit in pipeline.retrieve(query):
        console.print(
            f"[bold]{hit.chunk.pmid}[/bold] cos={hit.dense_score:.3f} "
            f"rrf={hit.score:.4f}\n{hit.chunk.text[:260]}…\n"
        )


@app.command()
def stats(config: Path = ConfigOption) -> None:
    """Résume le contenu de l'index."""
    import pandas as pd

    cfg = load_config(config)
    if not cfg.index_path.exists():
        console.print("[red]Index absent.[/red] Lancez : pubrag ingest")
        raise typer.Exit(1)

    frame = pd.read_parquet(cfg.index_path)
    documents = pd.read_parquet(cfg.documents_path) if cfg.documents_path.exists() else None

    console.print(f"Passages : [bold]{len(frame)}[/bold] sur {frame.pmid.nunique()} publications")
    console.print(f"Longueur médiane : {int(frame.text.str.len().median())} caractères")
    if documents is not None:
        counts = documents.source.value_counts().to_dict()
        console.print(f"Sources : {counts}  (pmc_oa = texte intégral, abstract = résumé seul)")


app.command(name="ingest")(ingest_cmd)

if __name__ == "__main__":
    app()
