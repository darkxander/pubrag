"""Évaluation de la recherche : rappel@k par régime de question.

On évalue la recherche seule, pas la génération. C'est délibéré : le rappel est
la borne supérieure de la qualité du système. Si le bon article n'est pas
retrouvé, aucun modèle de langage ne pourra produire une réponse correcte et
sourcée — il ne pourra qu'inventer.

Usage :
    python eval/run_eval.py            # modèle réel
    python eval/run_eval.py --offline  # encodeur de test, sans téléchargement
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pubrag.config import load_config  # noqa: E402
from pubrag.guardrails import check_retrieval  # noqa: E402
from pubrag.pipeline import Pipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--k", type=int, default=6)
    args = parser.parse_args()

    config = load_config()
    pipeline = Pipeline.from_index(config, offline=args.offline)
    questions = yaml.safe_load(Path(__file__).with_name("questions.yaml").read_text("utf-8"))[
        "questions"
    ]

    by_regime: dict[str, list[float]] = defaultdict(list)
    refusals_correct = 0
    refusals_expected = 0
    rows = []

    for item in questions:
        hits = pipeline.retrieve(item["q"])
        retrieved = {h.chunk.pmid for h in hits}
        expected = set(item["expected_pmids"])
        regime = item["regime"]

        if regime == "out_of_corpus":
            refusals_expected += 1
            refused = not check_retrieval(hits, config.min_relief).ok
            refusals_correct += int(refused)
            rows.append(
                (
                    item["q"][:52],
                    regime,
                    "refus amont" if refused else "→ modèle (refus attendu en aval)",
                )
            )
            continue

        recall = len(expected & retrieved) / len(expected)
        by_regime[regime].append(recall)
        rows.append((item["q"][:52], regime, f"rappel@{args.k} = {recall:.2f}"))

    width = max(len(r[0]) for r in rows)
    print(f"\n{'Question'.ljust(width)}  {'Régime'.ljust(13)}  Résultat")
    print("-" * (width + 40))
    for question, regime, result in rows:
        print(f"{question.ljust(width)}  {regime.ljust(13)}  {result}")

    print("\nSynthèse")
    all_scores = []
    for regime, scores in sorted(by_regime.items()):
        all_scores.extend(scores)
        mean = sum(scores) / len(scores)
        print(f"  {regime.ljust(13)} rappel@{args.k} moyen = {mean:.2f}  (n={len(scores)})")
    if all_scores:
        overall = sum(all_scores) / len(all_scores)
        print(f"  {'global'.ljust(13)} rappel@{args.k} moyen = {overall:.2f}")
    if refusals_expected:
        print(
            f"  hors corpus   refusés en amont : {refusals_correct}/{refusals_expected}"
            f"  — les autres sont transmis au modèle, qui doit refuser faute d'extrait"
        )

    # Code de sortie non nul si le rappel s'effondre : utilisable en CI.
    return 0 if not all_scores or sum(all_scores) / len(all_scores) >= 0.8 else 1


if __name__ == "__main__":
    raise SystemExit(main())
