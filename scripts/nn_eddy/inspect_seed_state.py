"""Field-by-field comparison of two seed states, and of a seed against the model's own output.

Measurement, not inference. The 5-day seam gradient blows up in exactly one chunk — chunk 0 —
which runs the SAME compiled VJP as chunks 1-4 with the same switches. The only thing that
differs is its input state: chunks 1-4 take states the model produced during the forward sweep,
chunk 0 takes ``snap_day150.pkl`` unpickled from disk. So the question is entirely empirical:
what, field by field, is different about a state that came off disk?

Two comparisons, because either alone is ambiguous:

A. ``student_free_1958/snap_day150.pkl`` (the seed the broken runs use) against
   ``foundation_snaps/snap_step037440.pkl`` (the seed every CLEAN long-window run used). A field
   that is populated in one and zero/absent in the other is the immediate suspect.
B. each seed against the state the model produces after ONE step from it. Any field the model
   overwrites at every step but the snapshot stores as zero is a field the snapshot is lying
   about, and the first step from it is not a continuation of anything. This is the class of
   fault ``fesom2/RESTART_REPRODUCIBILITY.md`` found five instances of, its general lesson being
   that a solver's initial guess IS state — ``d_eta``, ``ssh_rhs_old``, ``hbar_old``,
   ``uv_rhsAB``, ``t_skin`` are all warm starts or multi-step history here.

Reported per field: exact-zero fraction, non-finite count, magnitude, and the relative change
after one model step. A field that is exactly zero in the seed and O(1) after one step is the
answer; a field whose one-step relative change is enormous is the next candidate.
"""

from __future__ import annotations

import dataclasses
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fesom_jax  # noqa: E402,F401
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

if ROOT not in Path(fesom_jax.__file__).resolve().parents:
    raise SystemExit(f"FATAL: fesom_jax from {fesom_jax.__file__}, not {ROOT}")

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

MESH_DIR = ROOT / "data" / "mesh_core2"
IC_DIR = ROOT / "data" / "ic_core2_dist864"
DT = 1800.0
BROKEN = Path("/work/ba1583/a270088/nn_eddy/student_free_1958/snap_day150.pkl")
CLEAN = Path("/work/ab0995/a270088/port_jax/foundation_snaps/snap_step037440.pkl")


def stats(a):
    a = np.asarray(a)
    fin = np.isfinite(a)
    return dict(n=a.size,
                zero_frac=float(np.mean(a == 0.0)),
                nonfinite=int((~fin).sum()),
                absmax=float(np.max(np.abs(a[fin]))) if fin.any() else float("nan"),
                rms=float(np.sqrt(np.mean(a[fin] ** 2))) if fin.any() else float("nan"))


def field_table(st, title):
    print(f"\n=== {title} ===")
    print(f"{'field':16s} {'zero%':>7s} {'nonfin':>7s} {'absmax':>12s} {'rms':>12s}")
    out = {}
    for f in dataclasses.fields(st):
        v = getattr(st, f.name)
        if not hasattr(v, "shape"):
            continue
        s = stats(v)
        out[f.name] = s
        flag = ""
        if s["zero_frac"] == 1.0:
            flag = "  <-- ALL ZERO"
        elif s["nonfinite"]:
            flag = "  <-- NON-FINITE"
        print(f"{f.name:16s} {100*s['zero_frac']:6.2f}% {s['nonfinite']:7d} "
              f"{s['absmax']:12.4e} {s['rms']:12.4e}{flag}")
    return out


def one_step(st0, mesh, op, cf, sf1, nn):
    p = dataclasses.replace(Params.defaults(), gm_nn=nn)
    return step(st0, mesh, op, None, p, dt=DT, is_first_step=False,
                step_forcing=sf1, forcing_static=cf.static,
                ice_cfg=IceConfig(whichEVP=1, adjoint_mode="rheology"), gm_cfg=GMConfig(),
                tke_cfg=TkeConfig(), ale_cfg=AleConfig(),
                adjoint_cfg=AdjointConfig(freeze_mixing=True))


def main():
    mesh = load_mesh(MESH_DIR)
    op = ssh.build_ssh_operator(mesh, dt=DT)
    sst0 = np.asarray(phc_initial_state(mesh, IC_DIR).T[:, 0])
    cf = surface_forcing.build_surface_forcing(mesh, 1958, sst_ic=sst0)
    sf1 = jax.tree.map(lambda x: x[0],
                       cf.stack(surface_forcing.dates_for_steps(1958, DT, 1,
                                                                start_month=5, start_day=31)))
    nn = gm_nn.init_gm_nn(jax.random.PRNGKey(17), zero_last=False, w_scale=0.5)

    tables = {}
    for tag, path in (("BROKEN seed (student day150)", BROKEN),
                      ("CLEAN seed (foundation step037440)", CLEAN)):
        with open(path, "rb") as f:
            st = jax.device_put(pickle.load(f))
        tables[tag] = field_table(st, f"{tag}  {path.name}")
        nxt = one_step(st, mesh, op, cf, sf1, nn)
        print(f"\n--- {tag}: relative change after ONE model step ---")
        print(f"{'field':16s} {'seed rms':>12s} {'after rms':>12s} {'rel change':>12s}")
        for f in dataclasses.fields(st):
            a, b = getattr(st, f.name), getattr(nxt, f.name)
            if not hasattr(a, "shape"):
                continue
            ra, rb = stats(a)["rms"], stats(b)["rms"]
            rel = abs(rb - ra) / max(ra, 1e-300)
            mark = "  <-- SEED WAS ZERO, MODEL FILLS IT" if ra == 0.0 and rb > 0 else ""
            if rel > 0.5 or mark:
                print(f"{f.name:16s} {ra:12.4e} {rb:12.4e} {rel:12.4e}{mark}")
        del st, nxt

    print("\n=== fields that differ STRUCTURALLY between the two seeds ===")
    a, b = tables["BROKEN seed (student day150)"], tables["CLEAN seed (foundation step037440)"]
    for k in sorted(set(a) | set(b)):
        sa, sb = a.get(k), b.get(k)
        if sa is None or sb is None:
            print(f"{k:16s} present in only one seed")
            continue
        if (sa["zero_frac"] == 1.0) != (sb["zero_frac"] == 1.0):
            print(f"{k:16s} all-zero in {'BROKEN' if sa['zero_frac'] == 1.0 else 'CLEAN'} only"
                  f"   (broken rms {sa['rms']:.3e}, clean rms {sb['rms']:.3e})  <-- SUSPECT")
        elif sa["nonfinite"] != sb["nonfinite"]:
            print(f"{k:16s} non-finite counts differ: {sa['nonfinite']} vs {sb['nonfinite']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
