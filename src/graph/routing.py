"""
Conditional edge / routing functions for the extraction graph.

route_to_extractor        – maps document_type → extraction node name.
route_after_validation    – self-correction loop, HITL escalation, or success path.
"""

from __future__ import annotations

from config import CONFIDENCE_THRESHOLD, MAX_EXTRACTION_ATTEMPTS
from .state import ExtractionState


def route_to_extractor(state: ExtractionState) -> str:
    """
    Route from retrieve_schema_context to the correct extraction node
    based on the document type determined by the Router Agent.
    """
    doc_type = state.get("document_type", "generic")

    if doc_type == "medical":
        return "extract_medical"
    if doc_type == "financial":
        return "extract_financial"
    return "extract_generic"


def route_after_validation(state: ExtractionState) -> str:
    """
    Decide what happens after validate_extraction:

    1. Valid output AND confidence ≥ threshold  →  write_output  (success)
    2. Invalid AND attempts still remaining     →  back to the right extractor
                                                   (self-correction loop)
    3. Invalid AND attempts exhausted           →  human_review_queue
    4. Confidence too low (even if valid)       →  human_review_queue

    The self-correction loop is capped at MAX_EXTRACTION_ATTEMPTS iterations
    to prevent runaway token consumption.
    """
    has_errors = bool(state.get("validation_errors"))
    attempts = state.get("extraction_attempts", 0)
    confidence = state.get("extraction_confidence", 0.0)

    # Happy path
    if not has_errors and confidence >= CONFIDENCE_THRESHOLD:
        return "write_output"

    # Self-correction: still have budget, route back to the same extractor
    if attempts < MAX_EXTRACTION_ATTEMPTS:
        return route_to_extractor(state)

    # Exhausted retries or confidence below threshold → escalate
    return "human_review_queue"
