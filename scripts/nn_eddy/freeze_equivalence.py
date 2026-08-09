"""Do the TWO implementations of "freeze the mixing closure" actually agree?

This project has two, and until now nothing asserted they are the same thing:

* ``AdjointConfig(freeze_mixing=True)`` — the library flag, applied in ``step.py`` after the
  halo exchange. Used by ``seam_gate.py``.
* ``--freeze mix`` — ``adjoint_horizon.py``'s monkeypatch, which wraps ``tke.mixing_tke`` and
  freezes ``Kv``/``Av`` at its output, before the exchange. Used by every clean long-window
  record in ``adjoint_horizon_results.jsonl``.

Why it matters. The 5-day seam gradient came back non-reproducible (two identical jobs, 26x
apart, sign opposed to finite differences). The diagnosis on the table is that ``gm_nn`` is a
NEW state-dependent closure the flags do not cover. The competing explanation — the one this
script exists to kill or confirm — is duller and more likely to be true: that the library flag
does not freeze what the monkeypatch freezes, so the seam runs were simply less frozen than the
runs they were being compared against, and the network is being blamed for a plumbing gap.

Reading the two code paths says they are equivalent (``step.py:283`` — the closure emits
``Kv``/``Av`` post-``mo_convect``, so both freeze the same quantity; ``stop_gradient`` either
side of a pure data movement cuts the same upstream path). Reading is not measuring. Twice
today a result was carried onto a configuration it was never measured on.

Three claims, in order of what they rule out:

1. the flag is LIVE — freezing changes the gradient at all. A flag that silently did nothing
   would pass claim 2 trivially and would be the worst possible outcome;
2. the two implementations agree to round-off on d(loss)/d(state);
3. with mixing frozen BOTH ways, the ``gm_nn`` feature path still carries gradient — i.e.
   whatever the network contributes is genuinely NOT covered by either mixing freeze.

Cheap: one step on the CORE2 mesh, CPU. Run via a 30-minute compute job, not the login node.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fesom_jax  # noqa: E402,F401  (enables x64)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

_got = Path(fesom_jax.__file__).resolve()
if ROOT not in _got.parents:
    raise SystemExit(f"FATAL: fesom_jax imported from {_got}, not {ROOT}")

from fesom_jax import gm_nn, ssh, surface_forcing  # noqa: E402
from fesom_jax.adjoint_stab import AdjointConfig  # noqa: E402
from fesom_jax.ale import AleConfig  # noqa: E402
from fesom_jax.gm import GMConfig  # noqa: E402
from fesom_jax.ice import IceConfig  # noqa: E402
from fesom_jax.mesh import load_mesh  # noqa: E402
from fesom_jax.params import Params  # noqa: E402
from fesom_jax.phc_ic import phc_initial_state  # noqa: E402
from fesom_jax.step import step  # noqa: E402
from fesom_jax.tke import TkeConfig  # noqa: E402
from adjoint_horizon import patched  # noqa: E402  (the monkeypatch under test)

import dataclasses  # noqa: E402
import pickle  # noqa: E402

MESH_DIR = ROOT / "data" / "mesh_core2"
IC_DIR = ROOT / "data" / "ic_core2_dist864"
DT = 1800.0
SNAP = Path("/work/ba1583/a270088/nn_eddy/student_free_1958/snap_day150.pkl")


def main():
    mesh = load_mesh(MESH_DIR)
    op = ssh.build_ssh_operator(mesh, dt=DT)
    with open(SNAP, "rb") as f:
        st0 = jax.device_put(pickle.load(f))
    nn = gm_nn.init_gm_nn(jax.random.PRNGKey(17), zero_last=False, w_scale=0.5)
    ice, gmc, tkec, alec = (IceConfig(whichEVP=1, adjoint_mode="rheology"),
                            GMConfig(), TkeConfig(), AleConfig())

    # The TKE closure needs the ice-blended nodal wind stress, so `step` REFUSES a tke_cfg
    # without surface forcing (step.py:296) — and dropping tke_cfg to dodge that would delete
    # the very closure whose freeze is under test. Build one step of real forcing instead.
    sst0 = np.asarray(phc_initial_state(mesh, IC_DIR).T[:, 0])
    cf = surface_forcing.build_surface_forcing(mesh, 1958, sst_ic=sst0)
    dates = surface_forcing.dates_for_steps(1958, DT, 1, start_month=5, start_day=31)
    sfs = cf.stack(dates)
    sf1 = jax.tree.map(lambda x: x[0], sfs)

    p_nn = dataclasses.replace(Params.defaults(), gm_nn=nn)
    T0 = jnp.asarray(st0.T)

    def g(adj, params=p_nn):
        """d(one-step loss)/dT under one adjoint configuration.

        ⚠️ ``adj`` is CLOSED OVER, never passed as an argument to the differentiated
        function. ``AdjointConfig`` is a ``NamedTuple``, so JAX flattens it as a pytree and
        turns its boolean fields into TRACERS — which then poison the static branches that
        read them, surfacing far downstream as a ``TracerArrayConversionError`` on an index
        array inside the SSH solve. Cost me a job (26802455). The same applies to ``params``
        here: its config content must stay static even though its weights are differentiable
        elsewhere."""
        def loss(T):
            st = dataclasses.replace(st0, T=T)
            nxt = step(st, mesh, op, None, params, dt=DT, is_first_step=False,
                       step_forcing=sf1, forcing_static=cf.static,
                       ice_cfg=ice, gm_cfg=gmc, tke_cfg=tkec, ale_cfg=alec, adjoint_cfg=adj)
            return jnp.sum(nxt.T ** 2) + jnp.sum(nxt.uv ** 2)

        return np.asarray(jax.grad(loss)(T0))

    unfrozen = g(AdjointConfig())
    lib = g(AdjointConfig(freeze_mixing=True))
    with patched(["mix"]):
        mp = g(AdjointConfig())

    def rel(a, b):
        d = np.max(np.abs(a - b))
        return d, d / max(np.max(np.abs(b)), 1e-300)

    d_live, r_live = rel(lib, unfrozen)
    d_eq, r_eq = rel(lib, mp)
    print(f"[1] flag is LIVE      : max|frozen - unfrozen| = {d_live:.6e}  ({r_live:.3e} rel)")
    print(f"[2] two impls agree   : max|library - patch|   = {d_eq:.6e}  ({r_eq:.3e} rel)")

    # [3] with mixing frozen both ways, does the NN feature path still carry gradient?
    lib_nnfroz = g(AdjointConfig(freeze_mixing=True, freeze_nn_features=True))
    d_nn, r_nn = rel(lib_nnfroz, lib)
    print(f"[3] NN path uncovered : max|+nn_frozen - mixing_only| = {d_nn:.6e}  ({r_nn:.3e} rel)")

    ok1 = r_live > 1e-10
    ok2 = r_eq < 1e-10
    ok3 = r_nn > 1e-10
    print(f"\n[1] flag live            : {'PASS' if ok1 else 'FAIL — the flag does NOTHING'}")
    print(f"[2] implementations agree: {'PASS' if ok2 else 'FAIL — seam_gate was NOT frozen '
                                                          'the way the clean runs were'}")
    print(f"[3] NN path not covered  : {'PASS' if ok3 else 'FAIL — freezing the network changes '
                                                          'nothing, so it cannot be the cause'}")
    print("\nFREEZE_EQUIV_" + ("OK" if (ok1 and ok2 and ok3) else "MISMATCH"))
    return 0 if (ok1 and ok2 and ok3) else 1


if __name__ == "__main__":
    sys.exit(main())
