"""Make the scripts under ``scripts/nn_eddy/`` importable by the tests that test them.

Those scripts are written to be RUN (``python scripts/nn_eddy/coarsen_stream.py ...``), and a
script's own directory goes on ``sys.path`` automatically when it is run that way. Under pytest
it does not, so every one of them fails to import at collection with
``ModuleNotFoundError: No module named 'nn_data'`` — a module they reach for as a sibling.

⚠️ **This went unnoticed because the suite never ran these files.** ``scripts/runs/run_suite.sbatch``
invokes three groups — ocean, ice, sharding — while the cost manifest defines five; the
``scripts`` group has no invocation, so its 11 files only ever ran when somebody typed pytest by
hand from the right directory. The manifest's own header records that this was discovered on
2026-08-08 and a group was created for them; the runner was not updated to run it. Found again
2026-08-10, from the other end: an edit to ``coarsen_stream.py`` could not be checked because its
test file would not even collect.

Both halves are needed. This file fixes the import; the runner change makes the fix matter.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for p in (_ROOT, _ROOT / "scripts" / "nn_eddy"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
