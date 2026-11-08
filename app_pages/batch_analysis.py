"""Batch Customer Analysis page entry point."""

import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from predictlab.app.pages.batch_analysis import render_batch_analysis_page  # noqa: E402

render_batch_analysis_page()
