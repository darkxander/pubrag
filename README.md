# pubrag

Système de questions-réponses augmenté par la recherche (RAG) sur un corpus de
publications scientifiques en génétique humaine, avec citation obligatoire des
sources et refus explicite hors périmètre.

Le corpus est constitué de mes treize publications indexées dans PubMed —
279 passages, dont quatre articles en texte intégral. Rien n'est stocké dans ce
dépôt : tout est récupéré à l'exécution depuis les API publiques du NCBI, avec
filtrage des licences.

```
$ pubrag ask "Quel est le lien entre les mutations SHANK et la sévérité de la déficience intellectuelle ?"

╭─ Quel est le lien entre les mutations SHANK et la sévérité… ─────────────────╮
│ Les mutations des trois gènes SHANK suivent un gradient de sévérité :        │
│ SHANK1 est associé à un autisme sans déficience intellectuelle, SHANK2 à un  │
│ autisme avec déficience légère à modérée, et SHANK3 à une déficience         │
│ modérée à profonde [PMID:25188300].                                          │
╰──────────────────────────────────────────────────────────────────────────────╯
  PMID       Cosinus   BM25   Source
  25188300   0.741     #1     Meta-analysis of SHANK Mutations in Autism Spect…
```

Hors corpus, le système refuse — sans appeler le modèle de langage :

```
$ pubrag ask "Quelle est la capitale du Portugal ?"
⚠ garde-fou : meilleur cosinus 0.061 < seuil 0.280
```

---

## Ce que ce projet cherche à montrer

Un RAG qui fonctionne sur une démonstration est facile. Un RAG dont on peut
répondre de la sortie devant un clinicien l'est moins. Les quatre décisions qui
comptent ici :

### 1. Recherche hybride, parce qu'un corpus de génétique le demande

La recherche dense retrouve les reformulations — « troubles du spectre
autistique » et « ASD » se ressemblent dans l'espace vectoriel. Elle rate en
revanche les identifiants exacts : demander `MARCH6` à un modèle d'embedding
donne des résultats médiocres, parce que ce symbole ne porte presque aucun
signal sémantique.

BM25 fait exactement l'inverse. Sur un corpus saturé de symboles de gènes
(`SHANK3`, `CNTN6`, `ELMOD3`) et de coordonnées cytogénétiques (`22q13`,
`11q24.2-25`), se priver du lexical coûte cher.

Les deux branches sont fusionnées par **Reciprocal Rank Fusion** : on fusionne
des *rangs*, pas des scores. Un cosinus et un score BM25 n'ont ni la même
échelle ni la même distribution ; les additionner n'aurait aucun sens.

### 2. Un balayage exhaustif, et c'est un choix

Pas de base vectorielle. Le corpus fait quelques centaines de passages : un
produit matriciel NumPy sur `(N, 384)` prend moins d'une milliseconde — moins
que la surcharge d'un appel réseau à un service vectoriel.

Ajouter FAISS ou Chroma ici serait du cargo cult : de la complexité
opérationnelle pour une perte de performance. Le jour où le corpus dépasse
~100 000 passages, seule la méthode `HybridRetriever._dense_scores` est à
remplacer ; l'interface ne bouge pas.

### 3. Des garde-fous à deux moments distincts

| Moment | Contrôle | Effet |
|---|---|---|
| **Avant** la génération | Le meilleur cosinus dépasse-t-il le seuil ? | Refus, **sans appeler le modèle** |
| **Après** la génération | La réponse cite-t-elle au moins un PMID ? | Signalement |
| **Après** la génération | Les PMID cités font-ils partie des passages fournis ? | Signalement |

Le troisième contrôle est le plus important. Ces articles sont publics et
connus : un modèle peut très bien citer `[PMID:25188300]` de mémoire plutôt que
depuis le contexte fourni. Comparer les PMID cités aux PMID effectivement
transmis est le seul moyen de distinguer une réponse *sourcée* d'une réponse
*plausible*.

### 4. Le filtrage de licence n'est pas un détail

Un article peut être librement **consultable** sur PMC sans être librement
**redistribuable**. Indexer son texte intégral dans une base vectorielle est une
copie. Le code ne récupère donc le texte intégral que lorsque PMC déclare une
licence Creative Commons, et se rabat sur le résumé sinon — voir
`_extract_license_and_body` dans `fetch.py`.

Détail qui a son importance : l'ancien service `oa.fcgi`, que reprennent la
plupart des tutoriels, a été **retiré par le NCBI** et renvoie aujourd'hui une
page 404. La licence est donc lue dans le bloc `<permissions>` du XML de
l'article — source faisant foi, et une requête de moins.

Effet mesuré sur le corpus : les articles sous licence permissive passent de
~2 passages (résumé seul) à 24-84 passages (texte intégral).

---

## Évaluation

On évalue **la recherche**, pas la génération. C'est délibéré : le rappel est la
borne supérieure de la qualité du système. Si le bon article n'est pas retrouvé,
aucun modèle ne produira une réponse correcte — il ne pourra qu'inventer.

Le jeu d'évaluation (`eval/questions.yaml`) couvre trois régimes séparés :
reformulation sémantique, identifiant lexical exact, et hors corpus.

```bash
python eval/run_eval.py              # modèle réel
python eval/run_eval.py --offline    # encodeur de hachage, sans téléchargement
```

Le script renvoie un code de sortie non nul si le rappel moyen passe sous 0,80 —
il est donc utilisable comme garde-fou en intégration continue.

### Résultat de référence, encodeur de hachage

Mesuré sur le corpus réel : 13 publications, 279 passages, dont 4 articles en
texte intégral sous licence Creative Commons.

| Régime | rappel@6 | n |
|---|---|---|
| lexical (`MARCH6`, `22q13`, `CNTN6`, `ELMOD3`) | **1,00** | 4 |
| sémantique (reformulations) | **0,20** | 5 |
| hors corpus | refus correct 1/1 | 1 |

Ces chiffres sont la **borne inférieure**, et ils sont instructifs. L'encodeur de
hachage ne produit aucune sémantique : la branche dense est ici un placebo. Que
le rappel lexical reste malgré tout à 1,00 montre que BM25 porte seul toute sa
moitié du travail — c'est précisément l'argument en faveur de la recherche
hybride. Et que le sémantique s'effondre à 0,20 montre que le harnais
d'évaluation discrimine réellement, au lieu de valider n'importe quoi.

> Relancer `python eval/run_eval.py` après `uv pip install -e '.[embed]'` pour
> mesurer les chiffres avec `bge-small-en-v1.5`. C'est l'écart entre les deux
> colonnes qui quantifie l'apport de la recherche dense.

---

## Installation

```bash
uv venv --python 3.12
uv pip install -e '.[embed,api]'
cp .env.example .env      # renseigner ANTHROPIC_API_KEY et NCBI_EMAIL
```

## Utilisation

```bash
pubrag ingest                       # récupère PubMed, découpe, encode, indexe
pubrag stats                        # résumé de l'index
pubrag search "expansions MARCH6"   # passages retrouvés, sans appel au modèle
pubrag ask "…"                      # réponse sourcée
```

En API :

```bash
uvicorn pubrag.api:app --reload
curl -X POST localhost:8000/ask -H 'Content-Type: application/json' \
     -d '{"question":"Que sait-on de CNTN6 ?"}'
```

En conteneur :

```bash
docker build -t pubrag . && docker run -p 8000:8000 --env-file .env pubrag
```

## Tests

```bash
uv run pytest
```

Les tests s'exécutent **hors ligne et sans clé API** : ils utilisent
`HashingEmbedder`, un encodeur déterministe qui évite de télécharger 130 Mo de
poids en intégration continue, et un générateur factice à la place de l'appel à
Claude.

---

## Architecture

```
question
   │
   ├─► recherche dense   (bge-small-en-v1.5, 384 dim, cosinus) ──┐
   │                                                              ├─► RRF ─► top-k
   └─► recherche lexicale (BM25 Okapi) ──────────────────────────┘           │
                                                                             ▼
                                                        garde-fou amont : seuil de cosinus
                                                                             │
                                                              ┌──────────────┴─────────────┐
                                                        sous le seuil                au-dessus
                                                              │                           │
                                                           refus                    Claude Sonnet
                                                     (aucun appel LLM)          (citations imposées)
                                                                                            │
                                                                        garde-fou aval : les PMID cités
                                                                        étaient-ils dans le contexte ?
```

| Module | Rôle |
|---|---|
| `fetch.py` | NCBI E-utilities + PMC, avec filtrage de licence et limitation de débit |
| `chunk.py` | Découpage sur frontières de phrases, avec recouvrement |
| `embed.py` | Encodeurs interchangeables : `bge-small` local, ou hachage déterministe pour les tests |
| `index.py` | Construction et persistance en Parquet |
| `retrieve.py` | Recherche hybride dense + BM25, fusion RRF |
| `guardrails.py` | Seuil amont, vérification des citations en aval |
| `generate.py` | Appel à Claude sous contrainte de citation |
| `pipeline.py` | Orchestration |
| `cli.py` / `api.py` | Interfaces ligne de commande et HTTP |

---

## Choix techniques, en bref

| Décision | Motif |
|---|---|
| `bge-small-en-v1.5` en local | 384 dimensions, tourne sur CPU, aucune clé API pour l'encodage. Un corpus de treize articles n'a pas besoin de plus |
| Stockage Parquet | Colonnaire, compressé, typé, lisible par pandas comme par DuckDB |
| Température 0 | Sur une tâche factuelle sourcée, la variabilité est un défaut |
| Titre préfixé à chaque passage | Un passage isolé ne dit pas de quelle étude il provient ; le préfixe améliore nettement la recherche dense |
| Tokens approchés (4 car.) | Compter exactement imposerait une dépendance à un tokenizer donné, pour un gain nul à cette échelle |

## Limites connues

- Le corpus est petit : treize publications. Les mesures de rappel sont
  indicatives, pas statistiquement robustes.
- La plupart des articles ne sont disponibles qu'en résumé — le texte intégral
  n'est indexé que lorsque la licence PMC l'autorise. La couverture factuelle
  s'en ressent.
- Pas de réordonnancement par cross-encoder. Ce serait le premier ajout utile si
  le corpus grandissait.
- L'évaluation porte sur la recherche seule. Évaluer la fidélité de la
  génération demanderait un jeu de réponses de référence annotées.

## Licence

MIT pour le code. Les publications interrogées restent la propriété de leurs
éditeurs respectifs ; ce dépôt n'en redistribue aucun texte.
