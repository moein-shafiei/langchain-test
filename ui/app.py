"""
Streamlit UI for the Multi-Agent PDF Document Extraction pipeline.

Run from the project root:
    streamlit run ui/app.py

Tabs
────
  Extract  – upload a PDF and run the extraction pipeline
  Review   – human-in-the-loop correction when the pipeline pauses
  History  – browse all past extractions from the output/ directory
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st

# ── Bootstrap: add src/ to path so all existing imports resolve ────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import config  # noqa: E402 — must come after sys.path modification
from main import resume_with_correction, run_extraction  # noqa: E402

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PDF Extractor",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Session state defaults ─────────────────────────────────────────────────────
_DEFAULTS: dict[str, Any] = {
    "thread_id": None,
    "output_path": None,
    "result": None,        # dict loaded from the output JSON file
    "interrupted": False,
    "interrupt_payload": None,
    "pdf_name": None,
    "active_tab": 0,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _confidence_badge(score: float) -> str:
    if score >= 0.8:
        return f"🟢 {score:.0%}"
    if score >= 0.5:
        return f"🟡 {score:.0%}"
    return f"🔴 {score:.0%}"


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def render_result(extraction: dict) -> None:
    """Render a type-specific extraction result dict inside the current container."""
    doc_type = extraction.get("document_type", "generic")
    confidence = extraction.get("extraction_confidence", 0.0)
    notes = extraction.get("extraction_notes", "")

    # ── Header strip ──────────────────────────────────────────────────────────
    col_type, col_conf, col_hitl = st.columns([2, 2, 2])
    col_type.metric("Document type", doc_type.capitalize())
    col_conf.metric("Confidence", _confidence_badge(confidence))
    col_hitl.metric("Human reviewed", "Yes" if extraction.get("human_review_required") else "No")

    if notes:
        st.info(f"ℹ️ {notes}")

    st.divider()

    # ── Financial ─────────────────────────────────────────────────────────────
    if doc_type == "financial":
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Company", extraction.get("company_name") or "—")
        m2.metric("Period", extraction.get("reporting_period") or "—")
        m3.metric("Currency", extraction.get("currency") or "—")

        rev = extraction.get("revenue")
        exp = extraction.get("total_expenses")
        net = extraction.get("net_income")
        assets = extraction.get("total_assets")

        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Revenue", f"{rev:,.2f}" if rev is not None else "—")
        f2.metric("Total Expenses", f"{exp:,.2f}" if exp is not None else "—")
        f3.metric(
            "Net Income",
            f"{net:,.2f}" if net is not None else "—",
            delta=None if net is None else ("profit" if net >= 0 else "loss"),
            delta_color="normal" if (net or 0) >= 0 else "inverse",
        )
        f4.metric("Total Assets", f"{assets:,.2f}" if assets is not None else "—")

        items = extraction.get("key_line_items", [])
        if items:
            st.subheader("Key line items")
            st.dataframe(
                items,
                column_config={
                    "name":   st.column_config.TextColumn("Line Item"),
                    "amount": st.column_config.NumberColumn("Amount", format="%.2f"),
                    "period": st.column_config.TextColumn("Period"),
                },
                hide_index=True,
            )

    # ── Medical ───────────────────────────────────────────────────────────────
    elif doc_type == "medical":
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Patient", extraction.get("patient_name") or "—")
        c2.metric("DOB", extraction.get("date_of_birth") or "—")
        c3.metric("Visit date", extraction.get("visit_date") or "—")
        c4.metric("Facility", extraction.get("facility") or "—")

        st.metric("Provider", extraction.get("provider_name") or "—")

        codes = extraction.get("diagnosis_codes", [])
        descs = extraction.get("diagnosis_descriptions", [])
        if codes or descs:
            st.subheader("Diagnoses")
            paired = [
                {"Code": c, "Description": d}
                for c, d in zip(
                    codes + [""] * max(0, len(descs) - len(codes)),
                    descs + [""] * max(0, len(codes) - len(descs)),
                )
            ]
            st.dataframe(paired, hide_index=True)

        meds = extraction.get("medications", [])
        if meds:
            st.subheader("Medications")
            st.dataframe(meds, hide_index=True)

    # ── Custom fields ─────────────────────────────────────────────────────────
    elif doc_type == "custom":
        # Show every key that isn't a bookkeeping field as a simple table
        _SKIP = {"document_type", "extraction_confidence", "extraction_notes",
                 "human_review_required"}
        rows = [
            {"Field": k, "Value": ("—" if v is None else str(v))}
            for k, v in extraction.items()
            if k not in _SKIP
        ]
        if rows:
            st.dataframe(rows, hide_index=True)
        else:
            st.info("No fields were extracted.")

    # ── Generic ───────────────────────────────────────────────────────────────
    else:
        if extraction.get("title"):
            st.subheader(extraction["title"])
        if extraction.get("summary"):
            st.write(extraction["summary"])

        entities = extraction.get("key_entities", [])
        dates = extraction.get("dates_mentioned", [])
        fields = extraction.get("extracted_fields", {})

        if entities:
            st.subheader("Key entities")
            st.dataframe(entities, hide_index=True)

        if dates:
            st.subheader("Dates mentioned")
            st.write("  •  ".join(dates))

        if fields:
            st.subheader("Extracted fields")
            rows = [{"Field": k, "Value": str(v)} for k, v in fields.items()]
            st.dataframe(rows, hide_index=True)

    # ── Raw JSON expander ──────────────────────────────────────────────────────
    with st.expander("Raw JSON"):
        st.json(extraction)


# ══════════════════════════════════════════════════════════════════════════════
# Layout
# ══════════════════════════════════════════════════════════════════════════════

st.title("📄 PDF Document Extractor")
st.caption("Multi-agent extraction pipeline · LangGraph + Azure OpenAI")

tab_extract, tab_review, tab_history = st.tabs(["Extract", "⚠️ Review (HITL)", "History"])


# ──────────────────────────────────────────────────────────────────────────────
# Tab 1 — Extract
# ──────────────────────────────────────────────────────────────────────────────
with tab_extract:
    uploaded = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help="Machine-generated (text-layer) PDFs only. Scanned PDFs are not supported.",
    )

    st.markdown("**Custom fields to extract** *(optional)*")
    st.caption(
        "Provide a JSON object where each key is the field name and each value is "
        "a plain-English description. When filled in, the pipeline extracts **only** "
        "these fields and skips automatic document classification."
    )
    custom_fields_input = st.text_area(
        "Custom fields JSON",
        value="",
        height=140,
        placeholder='{\n  "company_name": "Legal name of the company",\n  "revenue": "Total revenue in USD",\n  "fiscal_year": "Fiscal year covered by the report"\n}',
        label_visibility="collapsed",
        key="custom_fields_textarea",
    )

    run_btn = st.button("Extract", type="primary", disabled=uploaded is None)

    if run_btn and uploaded is not None:
        # Parse custom fields if provided
        custom_fields: dict | None = None
        if custom_fields_input.strip():
            try:
                custom_fields = json.loads(custom_fields_input)
                if not isinstance(custom_fields, dict):
                    st.error("Custom fields must be a JSON object (key → description).")
                    st.stop()
                if not custom_fields:
                    custom_fields = None
            except json.JSONDecodeError as exc:
                st.error(f"Invalid JSON in custom fields: {exc}")
                st.stop()

        # Save upload to a named temp file (pdfplumber needs a real path)
        tmp_pdf = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        try:
            tmp_pdf.write(uploaded.read())
            tmp_pdf.flush()
            tmp_pdf.close()

            st.session_state.pdf_name = uploaded.name
            st.session_state.output_path = None
            st.session_state.result = None
            st.session_state.interrupted = False
            st.session_state.interrupt_payload = None

            with st.spinner("Running extraction pipeline…"):
                output_path, interrupt_payload = run_extraction(
                    tmp_pdf.name, custom_fields=custom_fields
                )

        finally:
            try:
                os.unlink(tmp_pdf.name)
            except OSError:
                pass

        if interrupt_payload:
            st.session_state.interrupted = True
            st.session_state.interrupt_payload = interrupt_payload
            st.session_state.thread_id = interrupt_payload.get("thread_id")
            st.warning(
                "⚠️ The pipeline paused for human review. "
                "Switch to the **Review (HITL)** tab to correct and resume.",
                icon="⚠️",
            )
        else:
            st.session_state.thread_id = None
            st.session_state.output_path = output_path
            if output_path and os.path.exists(output_path):
                data = _load_json(output_path)
                st.session_state.result = data.get("extraction_result", data)
            st.success("Extraction complete!")

    # Show the result if available
    if st.session_state.result:
        render_result(st.session_state.result)
    elif st.session_state.interrupted:
        st.warning(
            "A pipeline run is paused and awaiting review. "
            "Open the **Review (HITL)** tab."
        )


# ──────────────────────────────────────────────────────────────────────────────
# Tab 2 — Review (HITL)
# ──────────────────────────────────────────────────────────────────────────────
with tab_review:
    if not st.session_state.interrupted:
        st.info("No extraction is currently paused. Run an extraction first.")
    else:
        payload: dict = st.session_state.interrupt_payload or {}
        thread_id: str = payload.get("thread_id", st.session_state.thread_id or "")

        st.subheader("Human review required")
        st.caption(f"Thread ID: `{thread_id}`")

        # ── Diagnostics panel ─────────────────────────────────────────────────
        with st.expander("Diagnostics", expanded=True):
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Document type", payload.get("document_type") or "—")
            d2.metric("Classification confidence", _confidence_badge(payload.get("classification_confidence", 0.0)))
            d3.metric("Extraction confidence", _confidence_badge(payload.get("extraction_confidence", 0.0)))
            d4.metric("Attempts", str(payload.get("extraction_attempts", "—")))

            errors = payload.get("validation_errors", [])
            if errors:
                st.error("Validation errors:\n" + "\n".join(f"• {e}" for e in errors))

        st.divider()

        # ── Editable correction area ──────────────────────────────────────────
        st.markdown("**Edit the extraction result below, then submit:**")
        raw_result = payload.get("extraction_result") or {}
        edited_json = st.text_area(
            "Corrected extraction (JSON)",
            value=json.dumps(raw_result, indent=2),
            height=400,
            key="hitl_editor",
        )

        submit_btn = st.button("Submit correction", type="primary")

        if submit_btn:
            # Validate JSON before sending
            try:
                corrected: dict = json.loads(edited_json)
            except json.JSONDecodeError as exc:
                st.error(f"Invalid JSON: {exc}")
                st.stop()

            tmp_correction = tempfile.NamedTemporaryFile(
                suffix=".json", mode="w", encoding="utf-8", delete=False
            )
            try:
                json.dump(corrected, tmp_correction)
                tmp_correction.flush()
                tmp_correction.close()

                with st.spinner("Resuming pipeline…"):
                    output_path, new_interrupt = resume_with_correction(
                        thread_id, tmp_correction.name
                    )
            finally:
                try:
                    os.unlink(tmp_correction.name)
                except OSError:
                    pass

            if new_interrupt:
                # Rare: interrupted again
                st.session_state.interrupt_payload = new_interrupt
                st.warning("Pipeline interrupted again — please review the updated payload above.")
                st.rerun()
            else:
                st.session_state.interrupted = False
                st.session_state.interrupt_payload = None
                st.session_state.output_path = output_path
                if output_path and os.path.exists(output_path):
                    data = _load_json(output_path)
                    st.session_state.result = data.get("extraction_result", data)
                st.success("Correction accepted — extraction complete!")
                st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# Tab 3 — History
# ──────────────────────────────────────────────────────────────────────────────
with tab_history:
    output_dir = Path(_ROOT / config.OUTPUT_DIR)
    json_files = sorted(output_dir.glob("*_extracted.json"), reverse=True) if output_dir.exists() else []

    if not json_files:
        st.info("No extractions found in the output directory yet.")
    else:
        # Build summary table
        rows: list[dict] = []
        file_data: dict[str, dict] = {}
        for jf in json_files:
            try:
                data = _load_json(str(jf))
                er = data.get("extraction_result", {})
                rows.append(
                    {
                        "_path": str(jf),
                        "File": jf.name.replace("_extracted.json", ".pdf"),
                        "Type": er.get("document_type", "—").capitalize(),
                        "Confidence": er.get("extraction_confidence", 0.0),
                        "Extracted at": data.get("extracted_at", "—"),
                    }
                )
                file_data[jf.name] = data
            except Exception:
                pass

        # Summary dataframe — show all columns except the hidden path
        display_rows = [{k: v for k, v in r.items() if k != "_path"} for r in rows]
        st.dataframe(
            display_rows,
            column_config={
                "Confidence": st.column_config.ProgressColumn(
                    "Confidence", min_value=0.0, max_value=1.0, format="%.0%%"
                )
            },
            hide_index=True,
        )

        st.divider()

        # Per-file detail expanders
        for row, jf in zip(rows, json_files):
            with st.expander(f"📄 {row['File']}  ·  {row['Type']}  ·  {_confidence_badge(row['Confidence'])}"):
                data = file_data.get(jf.name, {})
                er = data.get("extraction_result", data)
                # Inject human_review_required from top-level state if absent
                render_result(er)
