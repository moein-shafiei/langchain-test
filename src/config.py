"""
Application configuration.

Loads environment variables from a .env file and exposes cached
factory functions for the two LLM clients (router + extractor),
the embeddings model, and the lazily-built FAISS vector store.

All Azure credentials are read exclusively from environment variables —
never hardcoded — following OWASP A02 (Cryptographic Failures) and
A05 (Security Misconfiguration) guidelines.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

# ⚠️  Never hardcode credentials here. Use a .env file (see .env.example).

if TYPE_CHECKING:
    from langchain_community.vectorstores import FAISS

load_dotenv()

# ── Azure OpenAI ───────────────────────────────────────────────────────────────
_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "")
_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
_ROUTER_DEPLOYMENT = os.getenv("AZURE_ROUTER_DEPLOYMENT", "gpt-4o-mini")
_EXTRACTION_DEPLOYMENT = os.getenv("AZURE_EXTRACTION_DEPLOYMENT", "gpt-4o")
_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")

# ── Extraction tuning ──────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.99"))
MAX_EXTRACTION_ATTEMPTS: int = int(os.getenv("MAX_EXTRACTION_ATTEMPTS", "2"))
CHUNK_THRESHOLD_PAGES: int = int(os.getenv("CHUNK_THRESHOLD_PAGES", "15"))

# ── Paths ──────────────────────────────────────────────────────────────────────
CHECKPOINT_DB_PATH: str = os.getenv("CHECKPOINT_DB_PATH", "checkpoints.db")
OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "output")


def _require_azure_credentials() -> None:
    """Raise a clear error if the required Azure credentials are absent."""
    missing = [
        name
        for name, value in [
            ("AZURE_OPENAI_ENDPOINT", _ENDPOINT),
            ("AZURE_OPENAI_API_KEY", _API_KEY),
        ]
        if not value
    ]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}. "
            "Copy .env.example to .env and fill in your Azure credentials."
        )


@lru_cache(maxsize=1)
def get_router_model() -> AzureChatOpenAI:
    """
    Lightweight model used by the Router Agent for document classification.
    Cached — only one instance is created per process.
    """
    _require_azure_credentials()
    return AzureChatOpenAI(
        azure_endpoint=_ENDPOINT,
        api_key=_API_KEY,
        api_version=_API_VERSION,
        azure_deployment=_ROUTER_DEPLOYMENT,
        temperature=0,
        max_retries=6,  # built-in exponential backoff + jitter for 429s / 5xx
    )


@lru_cache(maxsize=1)
def get_extraction_model() -> AzureChatOpenAI:
    """
    Full-capability model used by extraction agents.
    Cached — only one instance is created per process.
    """
    _require_azure_credentials()
    return AzureChatOpenAI(
        azure_endpoint=_ENDPOINT,
        api_key=_API_KEY,
        api_version=_API_VERSION,
        azure_deployment=_EXTRACTION_DEPLOYMENT,
        temperature=0,
        max_retries=6,
    )


@lru_cache(maxsize=1)
def get_embeddings() -> AzureOpenAIEmbeddings:
    """
    Embeddings model used to build and query the RAG knowledge base.
    Cached — only one instance is created per process.
    """
    _require_azure_credentials()
    return AzureOpenAIEmbeddings(
        azure_endpoint=_ENDPOINT,
        api_key=_API_KEY,
        api_version=_API_VERSION,
        azure_deployment=_EMBEDDING_DEPLOYMENT,
    )


# Module-level cache for the FAISS vector store (built lazily on first call).
_vector_store: "FAISS | None" = None


def get_vector_store() -> "FAISS":
    """
    Lazily build and cache the FAISS schema knowledge base.

    Building involves embedding all template files; this costs a few API calls
    once at startup and is then served from memory for the lifetime of the process.
    """
    global _vector_store
    if _vector_store is None:
        from rag.knowledge_base import build_knowledge_base

        _vector_store = build_knowledge_base(get_embeddings())
    return _vector_store
