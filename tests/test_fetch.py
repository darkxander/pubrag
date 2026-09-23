from xml.etree import ElementTree as ET

from pubrag.fetch import _REDISTRIBUTABLE, _extract_license_and_body, _parse_pubmed_article

SAMPLE = """
<PubmedArticle><MedlineCitation>
  <PMID>25188300</PMID>
  <Article>
    <Journal><ISOAbbreviation>PLoS Genet</ISOAbbreviation>
      <JournalIssue><PubDate><Year>2014</Year></PubDate></JournalIssue></Journal>
    <ArticleTitle>Meta-analysis of SHANK mutations</ArticleTitle>
    <Abstract>
      <AbstractText Label="BACKGROUND">Scaffolding proteins.</AbstractText>
      <AbstractText Label="RESULTS">A gradient of severity.</AbstractText>
    </Abstract>
    <AuthorList>
      <Author><LastName>Leblond</LastName><Initials>CS</Initials></Author>
      <Author><LastName>Nava</LastName><Initials>C</Initials></Author>
      <Author><LastName>Polge</LastName><Initials>A</Initials></Author>
      <Author><LastName>Mathieu</LastName><Initials>A</Initials></Author>
    </AuthorList>
  </Article>
</MedlineCitation>
<PubmedData><ArticleIdList>
  <ArticleId IdType="doi">10.1371/journal.pgen.1004580</ArticleId>
  <ArticleId IdType="pmc">PMC4154644</ArticleId>
</ArticleIdList></PubmedData></PubmedArticle>
"""


def test_parse_extracts_core_metadata():
    document = _parse_pubmed_article(ET.fromstring(SAMPLE))
    assert document.pmid == "25188300"
    assert document.journal == "PLoS Genet"
    assert document.year == "2014"
    assert document.doi.endswith("1004580")
    assert document.pmcid == "PMC4154644"


def test_structured_abstract_keeps_its_labels():
    document = _parse_pubmed_article(ET.fromstring(SAMPLE))
    assert "BACKGROUND: Scaffolding proteins." in document.abstract
    assert "RESULTS: A gradient of severity." in document.abstract


def test_long_author_list_is_abbreviated():
    assert _parse_pubmed_article(ET.fromstring(SAMPLE)).authors == "Leblond CS et al."


def test_text_falls_back_to_abstract_without_full_text():
    document = _parse_pubmed_article(ET.fromstring(SAMPLE))
    assert document.text == document.abstract


def test_licence_filter_accepts_creative_commons():
    assert _REDISTRIBUTABLE.search("CC BY")
    assert _REDISTRIBUTABLE.search("CC-BY-NC-SA")
    assert _REDISTRIBUTABLE.search("http://creativecommons.org/licenses/by/4.0/")
    assert _REDISTRIBUTABLE.search("https://creativecommons.org/publicdomain/zero/1.0/")


def test_licence_filter_rejects_restrictive_terms():
    assert not _REDISTRIBUTABLE.search("NO-CC CODE")
    assert not _REDISTRIBUTABLE.search("PUBLISHER SPECIFIC")
    assert not _REDISTRIBUTABLE.search("All rights reserved")


PMC_ARTICLE = """
<article>
  <front><article-meta><permissions>
    <license xmlns:xlink="http://www.w3.org/1999/xlink"
             xlink:href="http://creativecommons.org/licenses/by/4.0/">
      <license-p>This is an open-access article.</license-p>
    </license>
  </permissions></article-meta></front>
  <body>
    <p>Short.</p>
    <p>A paragraph long enough to be kept by the extractor, describing the method in detail.</p>
  </body>
</article>
"""

PMC_RESTRICTED = PMC_ARTICLE.replace(
    "http://creativecommons.org/licenses/by/4.0/", "https://example.org/publisher-terms"
).replace("This is an open-access article.", "All rights reserved.")


def test_extract_reads_license_href_and_body():
    licence, body = _extract_license_and_body(ET.fromstring(PMC_ARTICLE))
    assert _REDISTRIBUTABLE.search(licence)
    assert "describing the method in detail" in body
    assert "Short." not in body, "les paragraphes trop courts sont écartés"


def test_extract_on_restrictive_license_is_not_redistributable():
    licence, body = _extract_license_and_body(ET.fromstring(PMC_RESTRICTED))
    assert body, "le corps est bien extrait…"
    assert not _REDISTRIBUTABLE.search(licence), "…mais la licence doit bloquer son usage"


def test_missing_permissions_block_means_refusal():
    xml = "<article><body><p>" + "x" * 80 + "</p></body></article>"
    licence, _ = _extract_license_and_body(ET.fromstring(xml))
    assert not _REDISTRIBUTABLE.search(licence)
