"""
Output writer tool.

Serialises the extraction result to a JSON file alongside metadata
about the source PDF and the extraction run.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def write_json_output(
    result: dict[str, Any],
    pdf_path: str,
    output_dir: str = "output",
) -> str:
    """
    Write the extraction result to ``<output_dir>/<pdf_stem>_extracted.json``.

    Returns the absolute path to the written file.
    """
    os.makedirs(output_dir, exist_ok=True)

    pdf_stem = Path(pdf_path).stem
    output_path = os.path.join(output_dir, f"{pdf_stem}_extracted.json")

    payload = {
        "source_pdf": os.path.basename(pdf_path),
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "extraction_result": result,
    }

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)

    return os.path.abspath(output_path)
