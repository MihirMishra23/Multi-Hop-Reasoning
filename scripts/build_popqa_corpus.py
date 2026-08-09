#!/usr/bin/env python3
"""Repository-local entry point for the PopQA Wikipedia corpus builder."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from data.popqa_corpus import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
