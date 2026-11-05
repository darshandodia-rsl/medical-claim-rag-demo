"""Build and open the vector database (Chroma) that holds the policy chunks."""
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from claims_rag.ingestion.chunking import build_chunks
from claims_rag.config import CHROMA_DIR, EMBED_MODEL

COLLECTION_NAME = "policy_clauses"


def create_embedding_model() -> HuggingFaceEmbeddings:
    """Load the local model that turns text into vectors.

    Vectors are normalized to length 1, so comparing them by cosine distance is straightforward.
    (Plain-Python equivalent: SentenceTransformer(EMBED_MODEL).encode(text).)
    """
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL, encode_kwargs={"normalize_embeddings": True})


def open_vector_store() -> Chroma:
    """Open the Chroma database stored in CHROMA_DIR, using cosine distance (smaller = more similar)."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=create_embedding_model(),
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )


def add_section_title(chunk: Document) -> Document:
    """Put the section title in front of the chunk text, so the embedding also carries the topic.

    The title-prefixed text is both embedded and stored, so retrieved chunks start with their section title.
    """
    text_with_title = chunk.metadata["section_title"] + "\n" + chunk.page_content
    return Document(page_content=text_with_title, metadata=chunk.metadata)


def build_index() -> int:
    """Chunk the policy, embed every chunk and store it. Rebuilds from scratch. Returns the number of chunks."""
    # Delete the old collection first, so re-running never leaves stale chunks behind.
    vector_store = open_vector_store()
    vector_store.delete_collection()
    vector_store = open_vector_store()

    chunks = [add_section_title(chunk) for chunk in build_chunks()]
    chunk_ids = [chunk.metadata["chunk_id"] for chunk in chunks]

    # add_documents embeds each chunk's text and stores the vector together with its metadata.
    # Giving every chunk its own ID lets us fetch the limits chunk (16.1) directly later.
    vector_store.add_documents(chunks, ids=chunk_ids)
    return len(chunks)
