"""Évaluation de la recherche : rappel@k par régime de question.

Usage :
    python eval/run_eval.py            # modèle réel
    python eval/run_eval.py --offline  # encodeur de test, sans téléchargement

La logique vit dans `pubrag.evaluation`, partagée avec l'interface web.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pubrag.config import load_config  # noqa: E402
from pubrag.evaluation import OUT_OF_CORPUS, evaluate, load_questions  # noqa: E402
from pubrag.pipeline import Pipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    pipeline = Pipeline.from_index(load_config(), offline=args.offline)
    results, summary = evaluate(pipeline, load_questions())

    rows = []
    for r in results:
        if r.regime == OUT_OF_CORPUS:
            verdict = "refus amont" if r.refused_upstream else "→ modèle (refus attendu en aval)"
        else:
            verdict = f"rappel@{summary.k} = {r.recall:.2f}"
        rows.append((r.question[:52], r.regime, verdict))

    width = max(len(r[0]) for r in rows)
    print(f"\n{'Question'.ljust(width)}  {'Régime'.ljust(13)}  Résultat")
    print("-" * (width + 40))
    for question, regime, verdict in rows:
        print(f"{question.ljust(width)}  {regime.ljust(13)}  {verdict}")

    print("\nSynthèse")
    for regime, mean in summary.recall_by_regime.items():
        n = summary.count_by_regime[regime]
        print(f"  {regime.ljust(13)} rappel@{summary.k} moyen = {mean:.2f}  (n={n})")
    if summary.recall_global is not None:
        print(f"  {'global'.ljust(13)} rappel@{summary.k} moyen = {summary.recall_global:.2f}")
    if summary.out_of_corpus_total:
        rest = summary.out_of_corpus_total - summary.out_of_corpus_refused
        tail = (
            f"  — {rest} transmise(s) au modèle, qui doit refuser faute d'extrait" if rest else ""
        )
        print(
            f"  hors corpus   refusés en amont : "
            f"{summary.out_of_corpus_refused}/{summary.out_of_corpus_total}{tail}"
        )

    # Code de sortie non nul si le rappel s'effondre : utilisable en CI.
    ok = summary.recall_global is None or summary.recall_global >= 0.8
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
