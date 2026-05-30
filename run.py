#!/usr/bin/env python3
"""
Root-level entry point for the PDF extraction pipeline.

Adds src/ to Python's module search path so that all absolute imports
inside src/ (e.g. ``from config import …``, ``from schemas.medical import …``)
resolve correctly when the script is invoked from the project root.

Usage
─────
  python run.py --pdf path/to/document.pdf
  python run.py --pdf path/to/document.pdf --thread-id <uuid>
  python run.py --resume corrected.json --thread-id <uuid>
"""

import os
import sys

_src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, _src_path)

from main import main  # noqa: E402 — import after sys.path modification

if __name__ == "__main__":
    main()
