# pubrag

[![CI](https://github.com/darkxander/pubrag/actions/workflows/ci.yml/badge.svg)](https://github.com/darkxander/pubrag/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)
![Licence](https://img.shields.io/badge/licence-MIT-green)

Système de questions-réponses augmenté par la recherche (RAG) sur un corpus de
publications scientifiques en génétique humaine, avec citation obligatoire des
sources et refus explicite hors périmètre.

Le corpus est constitué de mes treize publications indexées dans PubMed —
279 passages, dont quatre articles en texte intégral. Rien n'est stocké dans ce
dépôt : tout est récupéré à l'exécution depuis les API publiques du NCBI, avec
filtrage des licences.

Les passages retrouvés pour une question posée en français :

```
$ pubrag search "Quel est le lien entre les mutations SHANK et la sévérité de la déficience intellectuelle ?"
  25188300  cos=0.616  bm25=#3   Meta-analysis of SHANK Mutations in Autism Spectrum
  26334118  cos=0.557  bm25=#5   11q24.2-25 micro-rearrangements in autism spectrum d
  29263841  cos=0.562  bm25=#6   A framework to identify contributing genes in patien
  26334118  cos=0.565  bm25=#7   11q24.2-25 micro-rearrangements in autism spectrum d
  27166760  cos=0.678  bm25=—    CNTN6 mutations are risk factors for abnormal audito
  27166760  cos=0.662  bm25=#1   CNTN6 mutations are risk factors for abnormal audito
```

La bonne source arrive en tête alors qu'elle n'a pas le meilleur cosinus — c'est
la fusion des deux recherches qui la fait remonter. Hors corpus, le système
refuse sans appeler le modèle de langage :

```
$ pubrag ask "Quelle est la capitale du Portugal ?"
╭──────────────────────── Quelle est la capitale du Portugal ? ────────────────────────╮
│ Je ne trouve pas d'élément suffisamment proche de cette question dans le corpus      │
│ indexé. Ce corpus ne couvre que les publications d'Alexandre Mathieu en génétique    │
│ humaine ; je préfère ne pas répondre plutôt que d'extrapoler.                        │
╰──────────────────────────────────────────────────────────────────────────────────────╯
⚠ garde-fou : relief 0.061 < seuil 0.065 : question hors corpus
```

---

## Ce que ce projet cherche à montrer

Un RAG qui fonctionne sur une démonstration est facile. Un RAG dont on peut
répondre de la sortie devant un clinicien l'est moins. Le cœur du projet n'est
pas l'architecture, c'est **l'évaluation, et ce qu'elle a révélé** — voir plus
bas. Les décisions de conception qui en découlent :

### 1. Recherche hybride, parce qu'un corpus de génétique le demande

La recherche dense retrouve les reformulations — « troubles du spectre
autistique » et « ASD » se ressemblent dans l'espace vectoriel. Elle rate en
revanche les identifiants exacts : un symbole comme `MARCH6` porte presque aucun
signal sémantique. BM25 fait exactement l'inverse. Sur un corpus saturé de
symboles de gènes (`SHANK3`, `CNTN6`, `ELMOD3`) et de coordonnées cytogénétiques
(`22q13`), se priver du lexical coûte cher.

Les deux branches sont fusionnées par **Reciprocal Rank Fusion** : on fusionne
des *rangs*, pas des scores. Un cosinus et un score BM25 n'ont ni la même
échelle ni la même distribution ; les additionner n'aurait aucun sens.

### 2. Un plafond de passages par article

Le corpus mélange des résumés (2-3 passages) et des textes intégraux (24-84
passages). Sans plafond, un long article occupe tout le contexte. Le plafond (2
par article) est appliqué **pendant la constitution des viviers de candidats**,
pas seulement à la fin — c'est ce qui le rend efficace (voir Évaluation).

### 3. Un balayage exhaustif, et c'est un choix

Pas de base vectorielle. Le corpus fait 279 passages : un produit matriciel
NumPy sur `(279, 384)` prend moins d'une milliseconde — moins que la surcharge
d'un appel réseau à un service vectoriel. Ajouter FAISS ou Chroma serait de la
complexité sans gain. Au-delà de ~100 000 passages, seule la méthode
`HybridRetriever._dense_scores` est à remplacer.

### 4. Des garde-fous à deux moments distincts

| Moment | Contrôle | Effet |
|---|---|---|
| **Avant** la génération | La question présente-t-elle un « relief » sur le corpus ? | Refus, **sans appeler le modèle** |
| **Après** la génération | La réponse cite-t-elle au moins un PMID ? | Signalement |
| **Après** la génération | Les PMID cités font-ils partie des passages fournis ? | Signalement |

Le contrôle amont est une **économie d'appels, pas le filet de sécurité** —
l'évaluation montre pourquoi. Le vrai filet, ce sont la consigne donnée au
modèle (« si les extraits ne permettent pas de répondre, dis-le ») et le
troisième contrôle : ces articles sont publics, un modèle peut citer
`[PMID:25188300]` de mémoire plutôt que depuis le contexte. Comparer les PMID
cités aux PMID transmis est le seul moyen de distinguer une réponse *sourcée*
d'une réponse *plausible*.

### 5. Le filtrage de licence n'est pas un détail

Un article peut être librement **consultable** sur PMC sans être librement
**redistribuable**. L'indexer en texte intégral serait une copie. Le code ne
récupère le texte intégral que sous licence Creative Commons, et se rabat sur le
résumé sinon — voir `_extract_license_and_body` dans `fetch.py`.

L'ancien service `oa.fcgi`, que reprennent la plupart des tutoriels, a été
**retiré par le NCBI** et renvoie une page 404. La licence est donc lue dans le
bloc `<permissions>` du XML de l'article : source faisant foi, et une requête de
moins. Effet mesuré : les articles sous licence permissive passent de ~2
passages (résumé) à 24-84 (texte intégral).

---

## Évaluation

On évalue **la recherche**, pas la génération : le rappel est la borne
supérieure de la qualité du système. Si la bonne source n'est pas retrouvée,
aucun modèle ne produira une réponse correcte — il ne pourra qu'inventer.

Le jeu (`eval/questions.yaml`) comporte 15 questions en trois régimes :
reformulation sémantique (5), identifiant exact (4), et hors corpus (6), dont un
piège volontaire — une question de génétique sur un sujet absent du corpus.

```bash
python eval/run_eval.py              # modèle réel
python eval/run_eval.py --offline    # encodeur de hachage, sans téléchargement
```

### Résultats mesurés

| Configuration | lexical (n=4) | sémantique (n=5) | **global** | refus amont (n=6) |
|---|---|---|---|---|
| Encodeur de hachage — référence sans sémantique | 1,00 | 0,40 | 0,67 | 0/6 |
| `bge-small-en-v1.5`, version initiale | 1,00 | 0,80 | 0,89 | 0/6 |
| `bge-small-en-v1.5`, après corrections | **1,00** | **1,00** | **1,00** | **4/6** |

La première ligne sert de contrôle : la branche dense y est un placebo. Que le
lexical reste à 1,00 montre que BM25 porte seul sa moitié du travail ; l'écart
avec les lignes suivantes chiffre l'apport de la recherche dense.

### Ce que l'évaluation a révélé

La version initiale « marchait » à la démonstration. Le harnais y a trouvé trois
problèmes, aucun visible à l'œil nu.

**1. Aucun seuil de cosinus ne peut servir de garde-fou.** Les questions hors
corpus obtiennent un meilleur cosinus de 0,41 à 0,51 ; les questions valides, de
0,51 à 0,79. Les plages se chevauchent : « MARCH6 », valide, score 0,512 —
« comment fonctionne un moteur diesel », 0,514. Le score BM25 brut ne vaut pas
mieux : « la recette de la paella » obtient le score lexical **le plus élevé**
du jeu, parce que les mots-outils français entrent en collision avec le
vocabulaire génétique (« *de novo* »).

Le signal retenu est le **relief** — écart entre le meilleur passage et le
passage médian : une question hors sujet est uniformément moyenne avec tout le
corpus, une question couverte présente un pic. Il sépare presque tout, sans être
parfait (MARCH6 : 0,068 ; diesel : 0,083). Le seuil est donc réglé **sous la
plus basse valeur valide observée**, parce que les coûts d'erreur sont
asymétriques : un faux accepté coûte un appel au modèle, qui refusera lui-même ;
un faux refus perd une réponse juste. Résultat : 4 questions hors corpus sur 6
bloquées sans appel, les 2 autres transmises au modèle.

**2. Un seul article monopolisait le contexte.** Pour la question sur SHANK, les
six passages venaient tous de l'article CNTN6, et la méta-analyse SHANK — la
bonne source — était absente. Un plafond par article appliqué en fin de chaîne
ne suffisait pas : l'article CNTN6 remplissait déjà les 40 premières places du
vivier dense. Le plafond est désormais appliqué pendant la constitution du
vivier.

**3. Le tokeniseur cassait le français.** Le motif `[a-z0-9]+` découpait
« sévérité » en `s`, `v`, `rit`. Ces débris et les mots-outils (« est », « la »,
« le ») — rares dans un corpus anglais, donc d'IDF élevé — dominaient le
classement BM25 et noyaient « shank ». Correction : normalisation des accents
et liste de mots-outils français et anglais, couvertes par des tests de
non-régression.

### Ce que ces chiffres ne disent pas

- **Le jeu d'évaluation a aussi servi à diagnostiquer.** Les corrections sont des
  correctifs de principe — un bug de tokenisation, une règle de plafonnement —
  plutôt que des constantes ajustées, à une exception près : le seuil de relief,
  fixé explicitement à partir des valeurs observées. Le 1,00 est donc
  **optimiste** ; une estimation non biaisée demanderait un jeu de questions
  tenu à l'écart.
- **Le seuil de relief dépend du modèle d'encodage** : avec l'encodeur de
  hachage, le même seuil ne bloque aucune question hors corpus. Changer de
  modèle impose de le recalibrer.
- **La génération n'est pas évaluée.** En particulier, le refus par le modèle des
  deux questions qu'il reçoit hors corpus n'a pas encore été vérifié.

---

## Installation

```bash
uv venv --python 3.12
uv pip install -e '.[embed,api,dev]'
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

37 tests, exécutés **hors ligne et sans clé API** : ils utilisent
`HashingEmbedder`, un encodeur déterministe qui évite de télécharger 130 Mo de
poids en intégration continue, et un générateur factice à la place de Claude.
La CI exécute aussi `ruff check` et `ruff format --check`.

---

## Architecture

```
question
   │
   ├─► recherche dense    (bge-small-en-v1.5, 384 dim) ─┐  vivier plafonné
   │                                                     ├─► à 2 passages ─► RRF ─► top-k
   └─► recherche lexicale (BM25, mots-outils retirés) ──┘  par article             │
                                                                                    ▼
                                                              garde-fou amont : relief du corpus
                                                                                    │
                                                            ┌───────────────────────┴──────────┐
                                                      sous le seuil                       au-dessus
                                                            │                                  │
                                                         refus                           Claude Sonnet
                                                   (aucun appel LLM)                (citations imposées)
                                                                                               │
                                                                          garde-fou aval : les PMID cités
                                                                          étaient-ils dans le contexte ?
```

| Module | Rôle |
|---|---|
| `fetch.py` | NCBI E-utilities + PMC, filtrage de licence, limitation de débit |
| `chunk.py` | Découpage sur frontières de phrases, avec recouvrement |
| `embed.py` | Encodeurs interchangeables : `bge-small` local, ou hachage pour les tests |
| `index.py` | Construction et persistance en Parquet |
| `retrieve.py` | Recherche hybride, plafond par article, fusion RRF, calcul du relief |
| `guardrails.py` | Garde-fou amont, vérification des citations en aval |
| `generate.py` | Appel à Claude sous contrainte de citation |
| `pipeline.py` | Orchestration |
| `cli.py` / `api.py` | Interfaces ligne de commande et HTTP |

## Limites connues

- **Corpus petit** : treize publications, quinze questions d'évaluation. Les
  mesures sont indicatives, pas statistiquement robustes.
- **Modèle d'encodage anglais, questions en français.** Cela fonctionne ici,
  mais un modèle multilingue (`multilingual-e5-small`, `bge-m3`) serait la
  première évolution, d'autant que le relief est à recalibrer à chaque modèle.
- **Texte intégral pour 4 articles sur 13** seulement : les autres n'ont pas de
  licence permissive sur PMC et ne sont indexés qu'en résumé.
- **Pas de réordonnancement par cross-encoder.** Ce serait le premier ajout
  pour améliorer la précision au-delà du rappel.

## Licence

MIT pour le code. Les publications interrogées restent la propriété de leurs
éditeurs respectifs ; ce dépôt n'en redistribue aucun texte.
