"""Every ``--freeze`` knob in ``adjoint_horizon.py`` must survive being CALLED.

The knobs are monkeypatches over library attributes, so a wrapper's signature is a private copy
of the library's — and the library moves. On 2026-08-09 `--freeze convect` died 27 s into a GPU
job with ``TypeError: w() got an unexpected keyword argument 'freeze_increment'``: the library
had grown its own convection freeze (``mo_convect(..., freeze_increment=)``, driven by
``AdjointConfig.freeze_convection``) and every caller now passes that keyword, while the wrapper
still declared the old four positional arguments. The knob had been unusable ever since, and
nothing said so until a job spent a GPU slot to find out.

`check_fired` in the script guards the opposite failure — a knob that never fires — but a knob
that RAISES gets no such treatment, because it takes the whole run down with it. This file is
the cheap guard: install each knob and call the patched attribute the way the library calls it
TODAY, on the packaged pi mesh, with no model step. It costs seconds and it fails on the login
node instead of on an A100.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts" / "nn_eddy"))

import fesom_jax  # noqa: E402,F401
import jax.numpy as jnp  # noqa: E402

from fesom_jax import eos, gm, gm_redi, momentum, pgf, pp, tracer_adv  # noqa: E402
from fesom_jax.mesh import load_mesh  # noqa: E402

import adjoint_horizon as ah  # noqa: E402

PI_MESH_DIR = _ROOT / "fesom_jax" / "data" / "mesh_pi"

# knob -> (module, attribute). `icevisc` is left out: it needs --ice exact, which returns NaN
# by construction, and its wrapper takes (*a, **k) so it cannot drift the way `convect` did.
PATCHED = {
    "mix": (None, "tke.mixing_tke"),
    "tkestate": (None, "tke.mixing_tke"),
    "convect": (pp, "mo_convect"),
    "visc": (momentum, "impl_vert_visc"),
    "slope": (gm, "gm_diagnostics"),
    "k33": (gm_redi, "k33_augmentation"),
    "bv": (eos, "compute_pressure_bv"),
    "fct": (tracer_adv, "zalesak_limit"),
    "pgf": (pgf, "pressure_force_shchepetkin"),
}


@pytest.fixture(scope="module")
def mesh():
    return load_mesh(PI_MESH_DIR)


def test_every_knob_name_is_installable():
    """`patched` must accept every name it documents — a knob listed in the help but not wired
    up would be requested, silently do nothing, and be reported as 'not the culprit'."""
    for name in ah.FREEZE_HELP:
        with ah.patched([name]):
            pass


def test_wrapper_signatures_still_accept_what_the_library_passes():
    """The drift check, generalised beyond the one knob that broke.

    For each patched attribute, compare the wrapper's accepted parameters against the ORIGINAL
    function's. A wrapper is safe if it takes ``**kwargs`` or if it names every keyword the
    original does; anything else means the library can add a keyword the wrapper will reject.
    """
    bad = []
    for name, (mod, attr) in PATCHED.items():
        if mod is None:
            continue                      # tke.mixing_tke's wrapper is (*a, **k) by construction
        orig = getattr(mod, attr)
        orig_kw = {p.name for p in inspect.signature(orig).parameters.values()
                   if p.kind in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD)}
        with ah.patched([name]):
            wrapped = getattr(mod, attr)
            params = inspect.signature(wrapped).parameters.values()
            takes_var_kw = any(p.kind is p.VAR_KEYWORD for p in params)
            named = {p.name for p in params}
        if not takes_var_kw and not orig_kw <= named:
            bad.append(f"--freeze {name}: wrapper over {attr} takes {sorted(named)} but the "
                       f"library passes {sorted(orig_kw - named)}")
    assert not bad, "stale freeze wrapper(s):\n  " + "\n  ".join(bad)


def test_convect_wrapper_accepts_the_librarys_own_freeze_keyword(mesh):
    """The specific regression, exercised rather than inspected: call `mo_convect` exactly as
    `tke.py`, `kpp.py` and `pp.py` call it today, with the knob installed."""
    nl = int(mesh.nl)
    shape_n = (int(mesh.nod2D), nl)
    shape_e = (int(mesh.elem2D), nl)
    rng = np.random.default_rng(0)
    Kv = jnp.asarray(rng.uniform(1e-5, 1e-3, size=shape_n))
    Av = jnp.asarray(rng.uniform(1e-4, 1e-2, size=shape_e))
    bv = jnp.asarray(rng.normal(size=shape_n) * 1e-4)     # both signs => the switch fires

    with ah.patched(["convect"]):
        out = pp.mo_convect(mesh, Kv, Av, bv, freeze_increment=False)
    assert len(out) == 2 and out[0].shape == shape_n and out[1].shape == shape_e
    assert ah._FIRED.get("convect", 0) > 0, "the knob was installed but never fired"


def test_a_requested_knob_that_never_fires_is_refused():
    """The other half of the guard, and the reason it exists: a monkeypatch that silently misses
    its call site would report 'not the culprit' for a term never actually tested."""
    with pytest.raises(SystemExit):
        with ah.patched(["convect"]):
            pass                                   # nothing calls mo_convect inside the block
        ah.check_fired(["convect"])
