"""
RAG knowledge base for schema-grounded extraction.

The knowledge base embeds schema templates and few-shot examples
(one .txt file per document type) into a local FAISS vector store.
Before each extraction, the relevant context is retrieved and injected
into the extraction prompt — analogous to the Policy & Compliance Agent
in the original multi-agent architecture.
"""

from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

TEMPLATES_DIR = Path(__file__).parent / "templates"
_CHUNK_SIZE = 1_000
_CHUNK_OVERLAP = 100


def build_knowledge_base(embeddings: AzureOpenAIEmbeddings) -> FAISS:
    """
    Load all template files, chunk them, embed with the provided embeddings
    model, and return a compiled FAISS vector store.

    Call once at startup; the result is cached in config.get_vector_store().
    """
    documents: list[Document] = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHUNK_SIZE,
        chunk_overlap=_CHUNK_OVERLAP,
    )

    template_files = list(TEMPLATES_DIR.glob("*.txt"))
    if not template_files:
        raise RuntimeError(
            f"No schema template files found in {TEMPLATES_DIR}. "
            "Expected medical.txt, financial.txt, generic.txt."
        )

    for template_file in template_files:
        doc_type = template_file.stem  # "medical", "financial", "generic"
        content = template_file.read_text(encoding="utf-8")

        for chunk in splitter.split_text(content):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={"document_type": doc_type, "source": template_file.name},
                )
            )

    return FAISS.from_documents(documents, embeddings)


def retrieve_schema_context(
    vector_store: FAISS,
    document_type: str,
    query: str,
    k: int = 3,
) -> str:
    """
    Retrieve the top-k most relevant schema chunks for the given document type.

    Filters by document_type metadata first; falls back to an unfiltered
    search if no results are returned for that type.

    Security note: the returned context is injected into the extraction prompt
    inside XML delimiters so the LLM treats it as data, not as instructions.
    """
    try:
        results = vector_store.similarity_search(
            query,
            k=k,
            filter={"document_type": document_type},
        )
    except Exception:
        results = []

    if not results:
        # Fallback: unfiltered search across all document types
        results = vector_store.similarity_search(query, k=k)

    return "\n\n---\n\n".join(doc.page_content for doc in results)
