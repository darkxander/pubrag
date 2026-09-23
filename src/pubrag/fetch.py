"""Récupération des publications depuis les API publiques du NCBI.

Deux sources, dans cet ordre :

1. **PubMed E-utilities** — métadonnées et résumé. Toujours disponible.
2. **PMC** — texte intégral, mais *uniquement* pour les articles dont la
   licence l'autorise.

Le point important est le filtrage de licence dans `_extract_license_and_body`.
Un article peut être librement *consultable* sur PMC sans être librement
*redistribuable* : indexer son texte intégral dans une base vectorielle serait
une copie. On ne récupère donc le texte intégral que sous licence Creative
Commons ou équivalent, et on se rabat sur le résumé sinon.

La licence est lue dans le bloc `<permissions>` du XML de l'article lui-même.
L'ancien service `oa.fcgi`, souvent cité dans les tutoriels, a été retiré par le
NCBI et renvoie aujourd'hui une page 404 ; le XML est de toute façon la source
faisant foi, et cela économise une requête.

Aucun texte n'est versionné dans le dépôt : tout est récupéré à l'exécution.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import asdict, dataclass
from xml.etree import ElementTree as ET

import httpx

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# Le NCBI limite à 3 requêtes/seconde sans clé API.
_MIN_INTERVAL = 0.4

# Licences dont les termes permettent de réutiliser le texte. On teste à la fois
# l'URL de la licence (creativecommons.org/licenses/by/4.0) et sa mention en
# clair ("CC BY-NC-SA"), les deux formes se rencontrant dans les fichiers PMC.
_REDISTRIBUTABLE = re.compile(
    r"creativecommons\.org/(licenses/(by|by-sa|by-nc|by-nc-sa)|publicdomain/zero)"
    r"|CC[ -]?(BY|BY-SA|BY-NC|BY-NC-SA|0)\b",
    re.IGNORECASE,
)


@dataclass
class Document:
    pmid: str
    title: str
    abstract: str
    journal: str
    year: str
    authors: str
    doi: str
    pmcid: str
    full_text: str
    source: str  # "abstract" ou "pmc_oa"

    @property
    def text(self) -> str:
        """Texte à indexer : le texte intégral s'il est utilisable, sinon le résumé."""
        return self.full_text or self.abstract

    @property
    def citation(self) -> str:
        return f"{self.authors} ({self.year}) {self.journal}. PMID:{self.pmid}"

    def to_row(self) -> dict:
        return asdict(self)


class NCBIClient:
    """Client minimal pour les E-utilities, avec limitation de débit."""

    def __init__(self, email: str | None = None, timeout: float = 30.0) -> None:
        self.email = email or os.environ.get("NCBI_EMAIL", "")
        self._client = httpx.Client(timeout=timeout, follow_redirects=True)
        self._last_call = 0.0

    def __enter__(self) -> NCBIClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - elapsed)
        self._last_call = time.monotonic()

    def _get(self, url: str, params: dict) -> httpx.Response:
        self._throttle()
        if self.email:
            params = {**params, "email": self.email, "tool": "pubrag"}
        response = self._client.get(url, params=params)
        response.raise_for_status()
        return response

    def fetch_documents(self, pmids: list[str]) -> list[Document]:
        """Récupère métadonnées, résumé et — si la licence l'autorise — texte intégral."""
        if not pmids:
            return []
        xml = self._get(
            f"{EUTILS}/efetch.fcgi",
            {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"},
        ).text
        documents = [_parse_pubmed_article(node) for node in ET.fromstring(xml).iter("PubmedArticle")]

        for document in documents:
            if document.pmcid:
                text = self._fetch_pmc_body_if_licensed(document.pmcid)
                if text:
                    document.full_text = text
                    document.source = "pmc_oa"
        return documents

    def _fetch_pmc_body_if_licensed(self, pmcid: str) -> str:
        """Renvoie le corps de l'article, ou une chaîne vide si la licence l'interdit."""
        try:
            xml = self._get(
                f"{EUTILS}/efetch.fcgi",
                {"db": "pmc", "id": pmcid.replace("PMC", ""), "retmode": "xml"},
            ).text
        except httpx.HTTPError:
            return ""
        licence, body = _extract_license_and_body(ET.fromstring(xml))
        return body if _REDISTRIBUTABLE.search(licence) else ""


def _extract_license_and_body(root: ET.Element) -> tuple[str, str]:
    """Extrait la licence déclarée et le corps de l'article d'un XML PMC.

    La licence est cherchée dans `<permissions>` : d'abord dans les attributs
    `xlink:href` du nœud `<license>`, puis dans son texte. Une absence de bloc
    `permissions` est traitée comme une absence de licence, donc un refus.
    """
    licence_parts: list[str] = []
    permissions = root.find(".//permissions")
    if permissions is not None:
        for node in permissions.iter("license"):
            licence_parts.extend(v for k, v in node.attrib.items() if k.endswith("href"))
            licence_parts.append("".join(node.itertext()))

    body_node = root.find(".//body")
    paragraphs: list[str] = []
    if body_node is not None:
        paragraphs = [
            " ".join("".join(p.itertext()).split())
            for p in body_node.iter("p")
        ]

    return " ".join(licence_parts), "\n\n".join(p for p in paragraphs if len(p) > 40)


def _text_of(node: ET.Element | None) -> str:
    return "".join(node.itertext()).strip() if node is not None else ""


def _parse_pubmed_article(node: ET.Element) -> Document:
    pmid = _text_of(node.find(".//PMID"))

    # Un résumé peut être structuré (Background / Methods / Results…).
    parts: list[str] = []
    for chunk in node.iter("AbstractText"):
        label = chunk.get("Label")
        body = "".join(chunk.itertext()).strip()
        parts.append(f"{label}: {body}" if label else body)

    authors = [
        f"{_text_of(a.find('LastName'))} {_text_of(a.find('Initials'))}".strip()
        for a in node.iter("Author")
        if a.find("LastName") is not None
    ]
    author_line = authors[0] + " et al." if len(authors) > 3 else ", ".join(authors)

    ids = {i.get("IdType"): _text_of(i) for i in node.iter("ArticleId")}
    year = _text_of(node.find(".//PubDate/Year")) or _text_of(node.find(".//PubDate/MedlineDate"))[:4]

    return Document(
        pmid=pmid,
        title=_text_of(node.find(".//ArticleTitle")),
        abstract="\n\n".join(parts),
        journal=_text_of(node.find(".//Journal/ISOAbbreviation")) or _text_of(node.find(".//Journal/Title")),
        year=year,
        authors=author_line,
        doi=ids.get("doi", ""),
        pmcid=ids.get("pmc", ""),
        full_text="",
        source="abstract",
    )
