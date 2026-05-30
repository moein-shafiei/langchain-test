"""
Multi-Agent PDF Document Extraction — CLI entrypoint.

Usage
─────
  # Extract a PDF (generates a new thread ID automatically)
  python main.py --pdf path/to/document.pdf

  # Re-run with an existing thread ID (state rehydration / fault recovery)
  python main.py --pdf path/to/document.pdf --thread-id <uuid>

  # Resume a HITL-paused workflow with a human-corrected JSON
  python main.py --resume path/to/corrected.json --thread-id <uuid>

Human-in-the-Loop workflow
───────────────────────────
  When extraction confidence is too low or validation fails after all retries,
  the graph pauses and prints the interrupt payload to stdout.  The operator:

    1. Saves the corrected extraction result to a JSON file.
    2. Runs:  python main.py --resume corrected.json --thread-id <uuid>

  The graph resumes from exactly where it left off (no re-extraction needed).
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from typing import Any

from graph.graph import build_graph


# ── Core extraction ────────────────────────────────────────────────────────────

def run_extraction(pdf_path: str, thread_id: str | None = None) -> str | None:
    """
    Run the full extraction pipeline for a single PDF.

    Returns the path to the output JSON file on success, or None if the
    graph was interrupted for human review.
    """
    graph = build_graph()
    thread_id = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state: dict[str, Any] = {
        "pdf_path": pdf_path,
        "thread_id": thread_id,
    }

    print(f"[INFO] Starting extraction")
    print(f"       PDF    : {pdf_path}")
    print(f"       Thread : {thread_id}")

    result = graph.invoke(initial_state, config)

    # ── Check for HITL interrupt ───────────────────────────────────────────────
    # When interrupt() is called inside a node, graph.invoke() returns the
    # current state dict with an extra "__interrupt__" key.
    interrupts = result.get("__interrupt__")
    if interrupts:
        print("\n[HUMAN REVIEW REQUIRED]")
        print("─" * 60)
        print("The graph was paused because extraction requires human review.")
        print(f"Thread ID for resumption: {thread_id}\n")

        for i, interrupt_obj in enumerate(interrupts, start=1):
            payload = getattr(interrupt_obj, "value", interrupt_obj)
            print(f"Interrupt {i} payload:")
            print(json.dumps(payload, indent=2, default=str))

        print("\nTo resume:")
        print("  1. Save your corrected extraction to a JSON file.")
        print(f"  2. Run: python main.py --resume <path/to/corrected.json> --thread-id {thread_id}")
        return None

    output_path = result.get("output_path")
    if output_path:
        doc_type = result.get("document_type", "unknown")
        confidence = result.get("extraction_confidence", 0.0)
        hitl = result.get("human_review_required", False)
        print(f"\n[SUCCESS] Extraction complete")
        print(f"          Document type  : {doc_type}")
        print(f"          Confidence     : {confidence:.2f}")
        print(f"          Human reviewed : {hitl}")
        print(f"          Output         : {output_path}")
    else:
        print("[WARNING] Graph finished but no output_path in state.", file=sys.stderr)

    return output_path


# ── HITL resumption ────────────────────────────────────────────────────────────

def resume_with_correction(thread_id: str, corrected_json_path: str) -> str | None:
    """
    Resume a HITL-paused workflow.

    Reads the corrected extraction result from ``corrected_json_path`` and
    passes it as the ``Command(resume=...)`` value to LangGraph so the graph
    can continue from the human_review_queue node.
    """
    from langgraph.types import Command

    graph = build_graph()
    config = {"configurable": {"thread_id": thread_id}}

    with open(corrected_json_path, "r", encoding="utf-8") as fh:
        corrected_result: dict[str, Any] = json.load(fh)

    print(f"[INFO] Resuming thread : {thread_id}")
    print(f"       Correction file : {corrected_json_path}")

    result = graph.invoke(Command(resume=corrected_result), config)

    output_path = result.get("output_path")
    if output_path:
        print(f"\n[SUCCESS] Resumed extraction complete")
        print(f"          Output : {output_path}")
    else:
        print("[WARNING] Graph finished after resume but no output_path in state.", file=sys.stderr)

    return output_path


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-Agent PDF Document Extraction (LangGraph)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--pdf", metavar="PATH", help="Path to the PDF file to extract.")
    parser.add_argument(
        "--thread-id",
        metavar="UUID",
        default=None,
        help="Thread ID for state rehydration, fault recovery, or HITL resumption.",
    )
    parser.add_argument(
        "--resume",
        metavar="PATH",
        default=None,
        help="Path to a corrected JSON file to resume a HITL-paused workflow.",
    )

    args = parser.parse_args()

    if args.resume:
        if not args.thread_id:
            parser.error("--thread-id is required when using --resume.")
        resume_with_correction(args.thread_id, args.resume)
    elif args.pdf:
        run_extraction(args.pdf, args.thread_id)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
