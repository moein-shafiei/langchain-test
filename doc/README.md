# Multi-Agent PDF Document Extraction

A LangGraph-based multi-agent pipeline that extracts structured data from machine-generated PDFs. Documents are classified, schema-grounded via RAG, extracted with a type-specific Pydantic model, validated with automatic self-correction, and escalated to a human reviewer when needed. Every state transition is persisted to SQLite for fault recovery and time-travel debugging.

---

## Architecture

```
[PDF file]
    │
    ▼
parse_document          pdfplumber → raw text + tables; chunks for large PDFs
    │
    ▼
classify_document       Router Agent (GPT-4o-mini) → document_type + confidence
    │
    ▼
retrieve_schema_context RAG node — FAISS over schema templates (Policy Agent)
    │
    ├─ medical ──▶ extract_medical   ─┐
    ├─ financial ▶ extract_financial ─┤  Extraction Agents (GPT-4o + Pydantic)
    └─ generic ──▶ extract_generic  ─┘
                                       │
                                       ▼
                                 validate_extraction   Pydantic re-validation
                                       │
                    ┌──────────────────┼──────────────────┐
                    ▼                  ▼                   ▼
              write_output      extract_*           human_review_queue
              JSON file     (self-correction,       interrupt() → pause
                             max 2 attempts)              │
                                                          ▼
                                                    write_output
```

### Agents

| Agent | Model | Role |
|---|---|---|
| Router Agent | GPT-4o-mini | Classifies document into medical / financial / generic |
| RAG / Policy Agent | FAISS + embeddings | Retrieves schema templates and few-shot examples |
| Extraction Agent × 3 | GPT-4o | Structured extraction via `with_structured_output(PydanticModel)` |
| Validation Node | — | Pydantic re-validation; triggers self-correction or HITL |
| Human-in-the-Loop | — | `interrupt()` pause; operator corrects and resumes |

### Resiliency

| Feature | Mechanism |
|---|---|
| Transient LLM errors / 429s | `RetryPolicy(max_attempts=3, jitter=True)` on all extraction nodes |
| Circuit breaker | `error_handler` on extraction nodes — routes to HITL after all retries |
| Self-correction loop | Up to 2 re-extraction attempts with validation error injected into prompt |
| Fault recovery | `SqliteSaver` checkpoints state at every super-step; resume with same `thread_id` |
| Token management | Docs > 15 pages are pre-chunked; extraction uses ≤ 8 000 chars |
| Prompt injection defence | Document content wrapped in `<document_text>` XML delimiters |

---

## Project Structure

```
langchain-test/
├── run.py                   ← Root entry point (sets sys.path → src/, calls main())
├── requirements.txt
├── .env.example
│
├── src/
│   ├── main.py              ← CLI: --pdf, --thread-id, --resume
│   ├── config.py            ← Cached AzureChatOpenAI / embeddings / FAISS clients
│   │
│   ├── schemas/
│   │   ├── base.py          ← ClassificationResult, ExtractionResultBase
│   │   ├── medical.py       ← MedicalExtractionResult (patient, dx, meds, …)
│   │   ├── financial.py     ← FinancialExtractionResult (revenue, expenses, …)
│   │   └── generic.py       ← GenericExtractionResult (entities, dates, KV map)
│   │
│   ├── tools/
│   │   ├── pdf_parser.py    ← parse_pdf() — pdfplumber text + tables + chunking
│   │   └── output_writer.py ← write_json_output() — timestamped JSON file
│   │
│   ├── rag/
│   │   ├── knowledge_base.py   ← build_knowledge_base(), retrieve_schema_context()
│   │   └── templates/
│   │       ├── medical.txt     ← Schema description + 2 few-shot examples
│   │       ├── financial.txt   ← Schema description + 2 few-shot examples
│   │       └── generic.txt     ← Schema description + 2 few-shot examples
│   │
│   └── graph/
│       ├── state.py         ← ExtractionState TypedDict (single source of truth)
│       ├── nodes.py         ← All 7 node functions
│       ├── routing.py       ← route_to_extractor(), route_after_validation()
│       └── graph.py         ← StateGraph assembly + SqliteSaver compilation
│
└── doc/
    ├── architecture.md      ← This file
    ├── setup.md             ← Installation and configuration
    └── usage.md             ← CLI reference and workflows
```

---

## State

`ExtractionState` (TypedDict in `src/graph/state.py`) is the single source of truth serialised at every node transition:

| Field | Type | Set by |
|---|---|---|
| `pdf_path`, `thread_id` | `str` | Caller |
| `raw_text`, `tables`, `page_count`, `chunks` | various | `parse_document` |
| `document_type`, `classification_confidence` | `str`, `float` | `classify_document` |
| `schema_context` | `str` | `retrieve_schema_context` |
| `extraction_result`, `extraction_attempts`, `validation_errors`, `extraction_confidence` | various | `extract_*`, `validate_extraction` |
| `human_review_required`, `human_review_payload` | `bool`, `dict` | `human_review_queue` |
| `output_path` | `str` | `write_output` |

---

## Setup

### Prerequisites
- Python 3.11+
- An Azure OpenAI resource with three deployed models:
  - A fast chat model for routing (e.g. `gpt-4o-mini`)
  - A full chat model for extraction (e.g. `gpt-4o`)
  - An embeddings model for RAG (e.g. `text-embedding-3-small`)

### Install

```bash
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# Edit .env and fill in your Azure OpenAI credentials and deployment names
```

Key variables:

| Variable | Description |
|---|---|
| `AZURE_OPENAI_ENDPOINT` | Your Azure OpenAI resource endpoint URL |
| `AZURE_OPENAI_API_KEY` | API key (never commit this) |
| `AZURE_ROUTER_DEPLOYMENT` | Deployment name for the routing model |
| `AZURE_EXTRACTION_DEPLOYMENT` | Deployment name for the extraction model |
| `AZURE_EMBEDDING_DEPLOYMENT` | Deployment name for the embeddings model |
| `CONFIDENCE_THRESHOLD` | Min confidence to skip HITL (default `0.50`) |
| `MAX_EXTRACTION_ATTEMPTS` | Self-correction cap (default `2`) |
| `CHUNK_THRESHOLD_PAGES` | Pages above which PDF is chunked (default `15`) |

---

## Usage

### Extract a PDF

```bash
python run.py --pdf path/to/document.pdf
```

On success, a timestamped JSON file is written to `output/<filename>_extracted.json`.

### Resume from a checkpoint (fault recovery)

```bash
# Re-run with the same thread ID — state is rehydrated from checkpoints.db
python run.py --pdf path/to/document.pdf --thread-id <uuid>
```

### Human-in-the-Loop workflow

When extraction confidence is below the threshold or validation fails after all retries, the graph pauses and prints the interrupt payload:

```
[HUMAN REVIEW REQUIRED]
Thread ID for resumption: 3f7a...
Interrupt 1 payload:
{
  "document_type": "financial",
  "extraction_result": { ... },
  "validation_errors": [ "..." ],
  ...
}

To resume:
  1. Save your corrected extraction to a JSON file.
  2. Run: python run.py --resume corrected.json --thread-id 3f7a...
```

The graph resumes from exactly where it left off — no re-extraction needed.

### Output format

```json
{
  "source_pdf": "invoice.pdf",
  "extracted_at": "2026-05-29T14:22:01.123456+00:00",
  "extraction_result": {
    "document_type": "financial",
    "company_name": "Acme Corp",
    "reporting_period": "Year ended Dec 31, 2025",
    "currency": "USD",
    "revenue": 48500.0,
    ...
  }
}
```

---

## Extending the system

| Goal | Where to change |
|---|---|
| Add a new document type | Add schema in `src/schemas/`, template in `src/rag/templates/`, new node in `src/graph/nodes.py`, update routing in `src/graph/routing.py` and `src/graph/graph.py` |
| Switch to PostgreSQL | Replace `SqliteSaver` in `src/graph/graph.py` with `PostgresSaver` — no other changes needed |
| Add OCR for scanned PDFs | Replace `pdfplumber.open()` in `src/tools/pdf_parser.py` with `pymupdf4llm` + Tesseract |
| Batch multiple PDFs | Use LangGraph's `Send()` fan-out in `src/graph/graph.py` for map-reduce over a list of paths |
| HITL dashboard | Replace the CLI `--resume` flow with a webhook that calls `graph.invoke(Command(resume=…), config)` |
