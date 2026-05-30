"""
PDF parsing tool.

Uses pdfplumber as the primary parser for machine-generated PDFs.
For documents exceeding CHUNK_THRESHOLD_PAGES, the full text is also
split into overlapping chunks so that extraction nodes can retrieve
only the most relevant sections (sliding-window token management).
"""

from __future__ import annotations

import os
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter


def parse_pdf(pdf_path: str, chunk_threshold_pages: int = 15) -> dict:
    """
    Parse a machine-generated PDF and return raw text, tables, and metadata.

    Returns a dict with keys:
        raw_text   – full concatenated text of all pages
        tables     – list of {"page": int, "data": list[list]} dicts
        page_count – total number of pages
        chunks     – list[str] | None  (populated only for large PDFs)
    """
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    raw_text_parts: list[str] = []
    tables: list[dict] = []

    with pdfplumber.open(pdf_path) as pdf:
        page_count = len(pdf.pages)

        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            raw_text_parts.append(text)

            # Extract tables using pdfplumber's line/rect strategy
            for table in page.extract_tables():
                if table:
                    tables.append({"page": page_num, "data": table})

    raw_text = "\n".join(raw_text_parts).strip()

    result: dict = {
        "raw_text": raw_text,
        "tables": tables,
        "page_count": page_count,
        "chunks": None,
    }

    # For large documents, pre-compute overlapping text chunks.
    # Extraction nodes will select the most relevant subset rather than
    # stuffing the entire document into the context window.
    if page_count > chunk_threshold_pages:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=2_000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
        )
        result["chunks"] = splitter.split_text(raw_text)

    return result
