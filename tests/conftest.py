import numpy as np
import pytest

from pubrag.chunk import Chunk
from pubrag.embed import HashingEmbedder
from pubrag.fetch import Document
from pubrag.retrieve import HybridRetriever


@pytest.fixture
def documents() -> list[Document]:
    return [
        Document(
            pmid="25188300",
            title="Meta-analysis of SHANK mutations in autism spectrum disorders",
            abstract=(
                "SHANK1, SHANK2 and SHANK3 encode postsynaptic scaffolding proteins. "
                "We report a gradient of severity in cognitive impairment.\n\n"
                "Mutations in SHANK3 were associated with moderate to profound "
                "intellectual disability in the cohort we analysed."
            ),
            journal="PLoS Genet",
            year="2014",
            authors="Leblond CS et al.",
            doi="10.1371/journal.pgen.1004580",
            pmcid="PMC4154644",
            full_text="",
            source="abstract",
        ),
        Document(
            pmid="31664039",
            title=(
                "Unstable TTTTA/TTTCA expansions in MARCH6 cause familial adult myoclonic epilepsy"
            ),
            abstract=(
                "We identified intronic pentanucleotide repeat expansions in MARCH6. "
                "The TTTCA motif segregated with familial adult myoclonic epilepsy type 3."
            ),
            journal="Nat Commun",
            year="2019",
            authors="Florian RT et al.",
            doi="10.1038/s41467-019-12763-9",
            pmcid="PMC6820781",
            full_text="",
            source="abstract",
        ),
    ]


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder(dimension=256)


@pytest.fixture
def retriever(documents, embedder) -> HybridRetriever:
    from pubrag.index import build_index

    chunks, embeddings = build_index(documents, embedder, target_tokens=60, overlap_tokens=10)
    return HybridRetriever(chunks, embeddings, embedder)


@pytest.fixture
def hits(retriever):
    return retriever.search("SHANK3 intellectual disability", top_k=3)


def make_chunk(pmid: str, text: str) -> Chunk:
    return Chunk(chunk_id=f"{pmid}:0", pmid=pmid, title="T", citation="C", position=0, text=text)


def unit_vectors(n: int, d: int = 8) -> np.ndarray:
    rng = np.random.default_rng(0)
    m = rng.normal(size=(n, d)).astype(np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)
