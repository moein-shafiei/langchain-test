"""
Graph assembly and compilation.

build_graph() constructs the full StateGraph, wires all nodes and
conditional edges, and compiles it with a SQLite checkpointer for
durable state persistence.

Architecture summary
────────────────────
  START
    │
    ▼
  parse_document
    │
    ▼
  classify_document           ← Router Agent (GPT-4o-mini)
    │
    ▼
  retrieve_schema_context     ← RAG / Policy Agent (FAISS)
    │
    ├──medical──▶  extract_medical   ──┐
    ├──financial─▶ extract_financial──┤  ← Extraction Agents (GPT-4o)
    └──generic───▶ extract_generic  ──┘
                                        │
                                        ▼
                                  validate_extraction
                                        │
                    ┌───────────────────┼─────────────────────┐
                    ▼                   ▼                      ▼
              write_output       extract_*             human_review_queue
              (success)        (self-correction,          (HITL pause)
                                max 2 attempts)               │
                                                              ▼
                                                        write_output

Resiliency
──────────
  • Per-node RetryPolicy on all extraction nodes (exponential backoff + jitter,
    max 3 attempts).  Handles transient LLM provider errors and 429s.
  • error_handler on all extraction nodes: circuit-breaker pattern — after all
    retries are exhausted the state is routed directly to human_review_queue.
  • Self-correction loop capped at MAX_EXTRACTION_ATTEMPTS via routing.py.
  • SQLite checkpointer: state snapshot saved at every super-step boundary.
    Fault recovery = re-run with the same thread_id; session is rehydrated
    from the last valid checkpoint.
"""

from __future__ import annotations

import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, RetryPolicy

import config
from .nodes import (
    classify_document_node,
    extract_financial_node,
    extract_generic_node,
    extract_medical_node,
    human_review_queue_node,
    parse_document_node,
    retrieve_schema_context_node,
    validate_extraction_node,
    write_output_node,
)
from .routing import route_after_validation, route_to_extractor
from .state import ExtractionState

# ── Resiliency: per-node retry policy ─────────────────────────────────────────
# Applied to all three extraction nodes.
# Handles: transient network errors, HTTP 429 (rate limit), 5xx server errors.
# Does NOT retry: ValueError / TypeError (logic errors → go straight to HITL).
_EXTRACTION_RETRY_POLICY = RetryPolicy(
    max_attempts=3,
    initial_interval=0.5,   # seconds
    backoff_factor=2.0,      # doubles each attempt → 0.5s, 1s, 2s (+ jitter)
    max_interval=30.0,
    jitter=True,
)


def _extraction_error_handler(
    state: ExtractionState, error: Exception
) -> Command:
    """
    Circuit-breaker callback invoked after all retry attempts are exhausted.

    Routes the graph directly to human_review_queue, bypassing the normal
    validate_extraction → routing flow, so the operator can inspect and
    correct the result manually.
    """
    return Command(
        update={
            "validation_errors": [
                f"Extraction node failed after all retries: {type(error).__name__}: {error}"
            ],
            # Force route_after_validation to escalate even if attempts < MAX
            "extraction_attempts": config.MAX_EXTRACTION_ATTEMPTS + 1,
        },
        goto="human_review_queue",
    )


# ── Graph assembly ─────────────────────────────────────────────────────────────

def build_graph(db_path: str | None = None) -> StateGraph:
    """
    Build and compile the PDF extraction StateGraph.

    Args:
        db_path: Path to the SQLite checkpoint database.
                 Defaults to config.CHECKPOINT_DB_PATH ("checkpoints.db").

    Returns:
        A compiled LangGraph StateGraph ready to call .invoke() / .stream() on.

    Persistence swap: to upgrade to PostgreSQL for production, replace
        SqliteSaver(conn)
    with
        from langgraph.checkpoint.postgres import PostgresSaver
        PostgresSaver.from_conn_string(os.environ["DATABASE_URL"])
    No other code changes are required.
    """
    resolved_db_path = db_path or config.CHECKPOINT_DB_PATH

    builder = StateGraph(ExtractionState)

    # ── Register nodes ─────────────────────────────────────────────────────────
    builder.add_node("parse_document", parse_document_node)
    builder.add_node("classify_document", classify_document_node)
    builder.add_node("retrieve_schema_context", retrieve_schema_context_node)

    # Extraction nodes carry retry + circuit-breaker resiliency
    builder.add_node(
        "extract_medical",
        extract_medical_node,
        retry_policy=_EXTRACTION_RETRY_POLICY,
        error_handler=_extraction_error_handler,
    )
    builder.add_node(
        "extract_financial",
        extract_financial_node,
        retry_policy=_EXTRACTION_RETRY_POLICY,
        error_handler=_extraction_error_handler,
    )
    builder.add_node(
        "extract_generic",
        extract_generic_node,
        retry_policy=_EXTRACTION_RETRY_POLICY,
        error_handler=_extraction_error_handler,
    )

    builder.add_node("validate_extraction", validate_extraction_node)
    builder.add_node("human_review_queue", human_review_queue_node)
    builder.add_node("write_output", write_output_node)

    # ── Static edges ───────────────────────────────────────────────────────────
    builder.add_edge(START, "parse_document")
    builder.add_edge("parse_document", "classify_document")
    builder.add_edge("classify_document", "retrieve_schema_context")

    # All extractors funnel into validation
    builder.add_edge("extract_medical", "validate_extraction")
    builder.add_edge("extract_financial", "validate_extraction")
    builder.add_edge("extract_generic", "validate_extraction")

    # HITL resume → output
    builder.add_edge("human_review_queue", "write_output")
    builder.add_edge("write_output", END)

    # ── Conditional edges ──────────────────────────────────────────────────────
    # 1. retrieve_schema_context → correct extraction node (Router dispatch)
    builder.add_conditional_edges(
        "retrieve_schema_context",
        route_to_extractor,
        {
            "extract_medical": "extract_medical",
            "extract_financial": "extract_financial",
            "extract_generic": "extract_generic",
        },
    )

    # 2. validate_extraction → success / self-correction loop / HITL
    builder.add_conditional_edges(
        "validate_extraction",
        route_after_validation,
        {
            "write_output": "write_output",
            "extract_medical": "extract_medical",
            "extract_financial": "extract_financial",
            "extract_generic": "extract_generic",
            "human_review_queue": "human_review_queue",
        },
    )

    # ── Compile with SQLite checkpointer ──────────────────────────────────────
    conn = sqlite3.connect(resolved_db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    return builder.compile(checkpointer=checkpointer)
