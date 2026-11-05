"""Find the policy clauses that matter for a claim.

Retrieval is hybrid: an embedding search (finds text with a similar MEANING) and a keyword search
(finds text that shares the same WORDS) each produce a ranked list, and the two lists are merged.
"""
import re
from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from claims_rag.ingestion.index import open_vector_store
from claims_rag.schemas import Claim

TOP_K = 15  # how many chunks to keep after the two searches are merged
CANDIDATES_PER_SEARCH = 30  # how many chunks each search hands to the merge step
RRF_CONSTANT = 60  # the standard constant in Reciprocal Rank Fusion; it softens the lead of the top ranks
EMBEDDING_WEIGHT = 0.7  # how much the embedding search counts when the two ranked lists are merged...
KEYWORD_WEIGHT = 0.3  # ...and how much the keyword search counts (the two weights add up to 1)
LIMITS_CHUNK_ID = "16.1"  # the plan limits table, always included so the LLM can see the limits


@dataclass
class RetrievalResult:
    query: str
    chunks: list[Document]  # best match first, with the limits chunk at the end if it was not already in

    @property
    def context_clause_ids(self) -> set[str]:
        """Every clause ID inside the chunks that go to the LLM. Citations are checked against this set."""
        clause_ids = set()
        for chunk in self.chunks:
            clause_ids.update(chunk.metadata["clause_ids"].split(","))
        return clause_ids


def build_query(claim: Claim) -> str:
    """Make the search text from the free-text claim fields only.

    Age, amounts, dates and names are left out: they say nothing about which clauses matter, and they add noise.
    """
    return "\n".join([claim.claim_type, claim.diagnosis, claim.procedure, claim.treatment_notes])


def split_into_words(text: str) -> list[str]:
    """Lowercase words and numbers, for the keyword search.

    BM25Retriever's default just splits on spaces, which would treat "Rent." and "rent" as different words.
    """
    return re.findall(r"[a-z0-9]+", text.lower())


def load_all_chunks(vector_store: Chroma) -> list[Document]:
    """Read every chunk back from the database, so the keyword search sees exactly the text that was embedded."""
    stored = vector_store.get()
    chunks = []
    for text, metadata in zip(stored["documents"], stored["metadatas"]):
        chunks.append(Document(page_content=text, metadata=metadata))
    return chunks


def retrieve(claim: Claim, vector_store: Chroma | None = None, top_k: int = TOP_K) -> RetrievalResult:
    """Search the policy for one claim and return the best chunks.

    Step 1, embedding search: turn the query into a vector and take the nearest chunk vectors.
            (Plain Python: compute the cosine distance from the query vector to every chunk vector.)
    Step 2, keyword search (BM25): score each chunk by how many of the query's words it contains,
            counting rare words more than common ones.
    Step 3, merge with Reciprocal Rank Fusion: a chunk earns 1/(60 + rank) from each list it appears in,
            and the earned points are added up. (Plain Python: a dict of chunk_id -> total points.)
            Only ranks are used, because distances and BM25 scores are on different scales.
    Step 4: keep the top chunks, then make sure the limits chunk (16.1) is included.
    """
    if vector_store is None:
        vector_store = open_vector_store()
    query = build_query(claim)

    embedding_retriever = vector_store.as_retriever(search_kwargs={"k": CANDIDATES_PER_SEARCH})
    keyword_retriever = BM25Retriever.from_documents(
        load_all_chunks(vector_store), k=CANDIDATES_PER_SEARCH, preprocess_func=split_into_words
    )
    hybrid_retriever = EnsembleRetriever(
        retrievers=[embedding_retriever, keyword_retriever],
        weights=[EMBEDDING_WEIGHT, KEYWORD_WEIGHT],  # same order as the retrievers above
        c=RRF_CONSTANT,
        id_key="chunk_id",  # chunks with the same chunk_id are the same chunk, whichever search found them
    )
    best_chunks = hybrid_retriever.invoke(query)[:top_k]

    # The limits table is needed for almost every payable amount, so add it if the searches did not find it.
    limits_already_included = False
    for chunk in best_chunks:
        if chunk.metadata["chunk_id"] == LIMITS_CHUNK_ID:
            limits_already_included = True
    if not limits_already_included:
        limits_record = vector_store.get(ids=[LIMITS_CHUNK_ID])  # fetched by ID, not by searching
        limits_chunk = Document(page_content=limits_record["documents"][0], metadata=limits_record["metadatas"][0])
        best_chunks.append(limits_chunk)

    return RetrievalResult(query=query, chunks=best_chunks)
