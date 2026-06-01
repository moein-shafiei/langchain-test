"""
LangGraph state definition.

ExtractionState is the single source of truth passed between every node
in the extraction graph. At each super-step boundary LangGraph serialises
this object to the SQLite checkpointer, enabling:

  - Fault recovery  — rehydrate from last valid checkpoint on crash.
  - Time-travel debugging — inspect state at any past step.
  - HITL resumption  — the graph pauses here; a human corrects the result
                        and the graph continues from this snapshot.
"""

from __future__ import annotations

from typing import Any, Optional, TypedDict


class ExtractionState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────────────────
    pdf_path: str
    thread_id: str

    # ── PDF parsing (populated by parse_document node) ─────────────────────────
    raw_text: str
    tables: list[dict]        # [{"page": int, "data": list[list]}]
    page_count: int
    chunks: Optional[list[str]]  # Pre-computed chunks for large PDFs (>15 pages)

    # ── Classification (populated by classify_document node) ───────────────────
    document_type: str        # "medical" | "financial" | "generic"
    classification_confidence: float

    # ── RAG context (populated by retrieve_schema_context node) ────────────────
    schema_context: str       # Top-k relevant schema / few-shot chunks

    # ── Extraction state (updated by extract_* and validate_extraction nodes) ──
    extraction_result: Optional[dict[str, Any]]
    extraction_attempts: int          # Incremented on every extraction call
    validation_errors: list[str]      # Non-empty → self-correction or HITL
    extraction_confidence: float      # Mirrored from extraction_result

    # ── Custom-field extraction (optional, provided by caller) ────────────────
    # When set, classification and RAG are skipped.  The pipeline extracts
    # only the fields listed here; keys are field names, values are natural-
    # language descriptions (used in the extraction prompt).
    # Example: {"company_name": "Legal name of the company", "revenue": "Total revenue in USD"}
    custom_fields: Optional[dict[str, Any]]

    # ── Human-in-the-loop ──────────────────────────────────────────────────────
    human_review_required: bool
    human_review_payload: Optional[dict[str, Any]]

    # ── Output ─────────────────────────────────────────────────────────────────
    output_path: Optional[str]
