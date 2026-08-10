"""Put the repository ROOT on ``sys.path`` for the tests under ``scripts/tests/``.

The three test files here import their target by module name — ``core2_kpp_climate_run``,
``core2_lw_foundation_check``, ``write_namelist`` — which are scripts written to be RUN, not
imported. A script's own directory joins ``sys.path`` automatically when it is run directly;
under pytest it does not, so the import has to be arranged here.

⚠️ **This used to add ``scripts/nn_eddy/`` as well, and must not.** That directory held the NN-GM
research project, which moved to its own repository on 2026-08-10; the library no longer contains
it and no test here needs it. Adding a path that does not exist is silent — ``sys.path`` accepts
anything — so the coupling would have survived the split without a single error, and the next
person to wonder whether the split was complete would have found a live reference to the moved
directory.

⚠️ **The reason this file exists at all is worth keeping.** These tests had NEVER run in the
suite: every runner globbed ``fesom_jax/tests/`` only, so they passed only when somebody typed
pytest by hand from the right directory (found 2026-08-08, and again 2026-08-10 from the other
end, when an edit could not be checked because its test would not collect). The manifest defines
a ``scripts`` group and ``run_suite_parallel.sbatch`` runs it; both halves are needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
