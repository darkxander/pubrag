# pubrag

[![CI](https://github.com/darkxander/pubrag/actions/workflows/ci.yml/badge.svg)](https://github.com/darkxander/pubrag/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)
![Licence](https://img.shields.io/badge/licence-MIT-green)

Système de questions-réponses augmenté par la recherche (RAG) sur un corpus de
publications scientifiques en génétique humaine, avec citation obligatoire des
sources et refus explicite hors périmètre. Une interface web permet d'interroger
le corpus et d'inspecter l'évaluation.

Le corpus est constitué de mes treize publications indexées dans PubMed —
431 passages, dont huit articles en texte intégral. Rien n'est stocké dans ce
dépôt : tout est récupéré à l'exécution depuis les API publiques du NCBI, avec
filtrage des licences.

```
$ pubrag search "Quel est le lien entre les mutations SHANK et la sévérité de la déficience intellectuelle ?"
  29263841  cos=0.588  bm25=#5   A framework to identify contributing genes in patien
  26334118  cos=0.565  bm25=#11  11q24.2-25 micro-rearrangements in autism spectrum d
  26334118  cos=0.557  bm25=#10  11q24.2-25 micro-rearrangements in autism spectrum d
  33414449  cos=0.518  bm25=#9   Decreased phenol sulfotransferase activities associa
  25188300  cos=0.673  bm25=—    Meta-analysis of SHANK Mutations in Autism Spectrum
  25188300  cos=0.652  bm25=#1   Meta-analysis of SHANK Mutations in Autism Spectrum
```

Hors corpus, le système refuse sans appeler le modèle de langage :

```
$ pubrag ask "Quelle est la capitale du Portugal ?"
╭──────────────────────── Quelle est la capitale du Portugal ? ────────────────────────╮
│ Je ne trouve pas d'élément suffisamment proche de cette question dans le corpus      │
│ indexé. Ce corpus ne couvre que les publications d'Alexandre Mathieu en génétique    │
│ humaine ; je préfère ne pas répondre plutôt que d'extrapoler.                        │
╰──────────────────────────────────────────────────────────────────────────────────────╯
⚠ garde-fou : relief 0.070 < seuil 0.090 : question hors corpus
```

---

## Ce que ce projet cherche à montrer

Un RAG qui fonctionne sur une démonstration est facile. Un RAG dont on peut
répondre de la sortie devant un clinicien l'est moins. Le cœur du projet n'est
pas l'architecture, c'est **l'évaluation, et ce qu'elle a révélé** — y compris
ce qu'elle *ne pouvait pas* révéler (voir plus bas).

### 1. Recherche hybride, parce qu'un corpus de génétique le demande

La recherche dense retrouve les reformulations — « troubles du spectre
autistique » et « ASD » se ressemblent dans l'espace vectoriel. Elle rate en
revanche les identifiants exacts : un symbole comme `MARCH6` porte peu de signal
sémantique. BM25 fait l'inverse. Sur un corpus saturé de symboles de gènes
(`SHANK3`, `CNTN6`, `ELMOD3`) et de coordonnées cytogénétiques (`22q13`), se
priver du lexical coûte cher.

Les deux branches sont fusionnées par **Reciprocal Rank Fusion** : on fusionne
des *rangs*, pas des scores, parce qu'un cosinus et un score BM25 n'ont ni la
même échelle ni la même distribution. La contrepartie est visible dans l'exemple
ci-dessus : la méta-analyse SHANK a le meilleur cosinus mais n'arrive que
cinquième, RRF favorisant les passages présents dans les deux classements. Les
six passages étant transmis ensemble au modèle, l'ordre importe peu ici ; il
compterait avec un contexte plus court — c'est le rôle d'un réordonnancement.

### 2. Un balayage exhaustif, et c'est un choix

Pas de base vectorielle. Le corpus fait 431 passages : un produit matriciel
NumPy sur `(431, 384)` prend moins d'une milliseconde, moins que la surcharge
d'un appel réseau à un service vectoriel. Au-delà de ~100 000 passages, seule la
méthode `HybridRetriever._dense_scores` est à remplacer.

### 3. Des garde-fous à deux moments distincts

| Moment | Contrôle | Effet |
|---|---|---|
| **Avant** la génération | La question présente-t-elle un « relief » sur le corpus ? | Refus, **sans appeler le modèle** |
| **Après** la génération | La réponse cite-t-elle au moins un PMID ? | Signalement |
| **Après** la génération | Les PMID cités font-ils partie des passages fournis ? | Signalement |

Le contrôle amont est une **économie d'appels, pas le filet de sécurité**. Le
vrai filet, ce sont la consigne donnée au modèle (« si les extraits ne
permettent pas de répondre, dis-le ») et le troisième contrôle : ces articles
sont publics, un modèle peut citer `[PMID:25188300]` de mémoire plutôt que
depuis le contexte. Comparer les PMID cités aux PMID transmis est le seul moyen
de distinguer une réponse *sourcée* d'une réponse *plausible*.

### 4. Le filtrage de licence n'est pas un détail

Un article peut être librement **consultable** sur PMC sans être librement
**redistribuable**. Le code ne récupère le texte intégral que sous licence
Creative Commons, et se rabat sur le résumé sinon. La licence est lue dans le
bloc `<permissions>` du XML de l'article : l'ancien service `oa.fcgi`, que
reprennent la plupart des tutoriels, a été **retiré par le NCBI**.

Les sections sans contenu scientifique sont exclues : information
supplémentaire, déclarations d'intérêts, contributions, financements. Pour la
méta-analyse SHANK, les seules légendes supplémentaires représentaient 57
paragraphes sur 96.

---

## Évaluation

On évalue **la recherche**, pas la génération : le rappel est la borne
supérieure de la qualité du système. Le jeu (`eval/questions.yaml`) comporte 15
questions : reformulation sémantique (5), identifiant exact (4), et hors corpus
(6) — dont un piège volontaire, une question de génétique sur un sujet absent.

```bash
python eval/run_eval.py              # modèle réel
python eval/run_eval.py --offline    # encodeur de hachage, sans téléchargement
```

| Configuration | lexical (n=4) | sémantique (n=5) | **global** | refus amont (n=6) |
|---|---|---|---|---|
| Encodeur de hachage — référence sans sémantique | 1,00 | 0,40 | 0,67 | 0/6 |
| `bge-small-en-v1.5` | **1,00** | **1,00** | **1,00** | **6/6** |

La première ligne sert de contrôle : la branche dense y est un placebo. Que le
lexical reste à 1,00 montre que BM25 porte seul sa moitié du travail ; l'écart
avec la seconde chiffre l'apport de la recherche dense.

### Ce que l'évaluation a révélé — et ce qu'elle a manqué

**1. Un seuil de cosinus ne peut pas servir de garde-fou.** La question piège
« Explain how CRISPR gene editing works », hors corpus, obtient un meilleur
cosinus de 0,645 ; la question valide « MARCH6 », 0,584. Le signal retenu est le
**relief** — écart entre le meilleur passage et le passage médian : une question
hors sujet est uniformément moyenne avec tout le corpus, une question couverte
présente un pic. Sur les 15 questions, il sépare les deux groupes (valides de
0,107 à 0,189, hors corpus de 0,058 à 0,081). Le seuil, 0,090, est placé dans
cet écart mais plus près du groupe hors corpus : un faux refus perd une bonne
réponse, un faux accepté ne coûte qu'un appel au modèle, qui refusera lui-même.

**2. Le tokeniseur cassait le français.** Le motif `[a-z0-9]+` découpait
« sévérité » en `s`, `v`, `rit`. Ces débris et les mots-outils (« est », « la »,
« le ») — rares dans un corpus anglais, donc d'IDF élevé — dominaient le
classement BM25. Correction : normalisation des accents et liste de mots-outils,
couvertes par des tests de non-régression.

**3. Le bug le plus grave, l'évaluation ne pouvait pas le voir.** En lisant les
passages dans l'interface web, un extrait attribué à la méta-analyse SHANK
parlait d'« essais à zéro événement » et d'« historique de pré-publication » —
le vocabulaire d'un tout autre article. Cause : l'analyseur lisait les
identifiants avec `node.iter("ArticleId")`, qui parcourt aussi la
**bibliographie**. Chaque article recevait le PMCID de sa *dernière référence
citée* — les huit PMCID vérifiables étaient faux — et l'on indexait le texte
intégral d'autres articles sous ses PMID. Sous celui de l'article CNTN6, par
exemple, c'était le texte de la méta-analyse SHANK.

Corrigé, le corpus est passé de 4 à **8 textes intégraux**, cette fois les bons,
et de 279 à 431 passages. Un test de non-régression reproduit le cas, et
échouerait avec l'ancien code.

La leçon tient en une phrase : **un rappel de 1,00 peut cacher un corpus faux.**
Le rappel vérifie que le bon PMID est retrouvé, pas que le texte placé sous ce
PMID est le bon. Le bug avait d'ailleurs faussé deux conclusions antérieures,
retirées depuis : un prétendu chevauchement des reliefs, et un plafonnement par
article présenté comme indispensable au rappel.

**4. Le plafond de passages par article est un compromis, pas un correctif.**
Sans plafond, 5 questions sur 9 reçoivent leurs six passages d'un seul article ;
avec un plafond de 2, aucune. Le rappel est identique dans les deux cas. Le
plafond élargit le contexte au prix de la profondeur sur un article donné — et
seule une évaluation de la génération dirait lequel sert le mieux les réponses.

### Ce que ces chiffres ne disent pas

- **Le jeu d'évaluation a aussi servi à diagnostiquer et à calibrer.** Le 1,00 et
  le 6/6 sont donc **optimistes** ; une estimation non biaisée demanderait des
  questions tenues à l'écart.
- **Le rappel est mesuré au niveau de l'article, pas du passage.** Il ne dit pas
  si le passage retrouvé est pertinent — c'est précisément ce qui a laissé passer
  le bug n°3. L'étape suivante serait un jugement de pertinence par passage.
- **Le seuil de relief dépend du modèle d'encodage et du corpus** : avec
  l'encodeur de hachage, le même seuil ne bloque rien, et la correction du corpus
  l'a déplacé. Il est à recalibrer à chaque changement.
- **La génération n'est pas évaluée**, et le refus par le modèle d'une question
  hors corpus que le filtre laisserait passer n'est pas encore vérifié.

---

## Interface web

```bash
pubrag serve        # http://127.0.0.1:8000
```

Une page autonome — sans bibliothèque ni CDN, utilisable hors ligne, en thème
clair ou sombre, sur ordinateur comme sur téléphone :

- **Interroger** — le verdict du garde-fou amont, la réponse avec ses citations
  vérifiées, et les passages transmis au modèle avec leurs scores (cosinus, rang
  BM25, score de fusion) et les termes de la question surlignés. **Fonctionne
  sans clé API** : seule la réponse rédigée en demande une ; en son absence, la
  page affiche quand même les passages retrouvés.
- **Évaluation** — les résultats du harnais, et le relief de chaque question
  tracé face au seuil de refus.
- **Corpus** — les publications indexées, texte intégral ou résumé, et leur
  nombre de passages.

C'est cette interface qui a permis de trouver le bug n°3 : lire les passages a
montré ce que les métriques ne montraient pas.

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
pubrag serve                        # interface web et API
```

API (documentation interactive sur `/docs`) :

| Point d'accès | Rôle | Clé API |
|---|---|---|
| `POST /search` | Passages retrouvés et verdict du garde-fou | non |
| `POST /ask` | Réponse sourcée ; `503` si aucune clé n'est configurée | oui |
| `GET /eval` | Harnais d'évaluation sur l'index chargé | non |
| `GET /stats` | Composition du corpus | non |

En conteneur :

```bash
docker build -t pubrag . && docker run -p 8000:8000 --env-file .env pubrag
```

## Tests

```bash
uv run pytest
```

48 tests, exécutés **hors ligne et sans clé API** : ils utilisent
`HashingEmbedder`, un encodeur déterministe qui évite de télécharger 130 Mo de
poids en intégration continue, et un générateur factice à la place de Claude.
Ils couvrent aussi l'API et l'interface. La CI exécute en plus `ruff check` et
`ruff format --check`.

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
| `fetch.py` | NCBI E-utilities + PMC, filtrage de licence et des sections hors contenu |
| `chunk.py` | Découpage sur frontières de phrases, avec recouvrement |
| `embed.py` | Encodeurs interchangeables : `bge-small` local, ou hachage pour les tests |
| `index.py` | Construction et persistance en Parquet |
| `retrieve.py` | Recherche hybride, plafond par article, fusion RRF, calcul du relief |
| `guardrails.py` | Garde-fou amont, vérification des citations en aval |
| `generate.py` | Appel à Claude sous contrainte de citation |
| `evaluation.py` | Harnais d'évaluation, partagé par le script et l'interface |
| `pipeline.py` | Orchestration |
| `cli.py` / `api.py` | Ligne de commande ; API HTTP et interface web (`static/index.html`) |

## Limites connues

- **Corpus petit** : treize publications, quinze questions d'évaluation. Les
  mesures sont indicatives, pas statistiquement robustes.
- **Modèle d'encodage anglais, questions en français.** Cela fonctionne ici, mais
  un modèle multilingue (`multilingual-e5-small`, `bge-m3`) serait la première
  évolution — le seuil de relief serait alors à recalibrer.
- **Texte intégral pour 8 articles sur 13** : les autres n'ont pas de licence
  permissive sur PMC et ne sont indexés qu'en résumé.
- **Pas de réordonnancement par cross-encoder**, ce qui laisse l'ordre final à la
  seule fusion RRF.

## Licence

MIT pour le code. Les publications interrogées restent la propriété de leurs
éditeurs respectifs ; ce dépôt n'en redistribue aucun texte.
