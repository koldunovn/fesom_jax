"""Why does the sea-ice adjoint blow up after three steps?

The bisection (`ice_adjoint_probe.py`) established the shape of the failure and it is a
strange one:

    n=1  dL/dm = +3.507e-05     clean
    n=2  dL/dm = +5.040e-05     clean, growing sensibly
    n=3  dL/dm = +6.836e-05     clean, growing sensibly
    n=4  dL/dm = +1.160e+05     <- 1.7e9 in ONE step
    n=6  dL/dm = -8.915e+21
    n=8  dL/dm = +1.192e+39     <- steady ~3e8 per step from here

Three healthy steps, then a cliff, then a different and much faster regime. That is not
accumulation — accumulation looks like a straight line on a log axis from step one. A cliff
means something switches on. The forward is untroubled throughout (the loss moves by 6e-6
over all eight steps), so whatever it is, it is only visible to the reverse pass.

This script attacks it three ways.

**A. Which field, which step.** Walk the reverse pass back ONE step at a time and print the
cotangent norm of every state field at every step boundary. The field whose norm jumps by 1e9
between step 4 and step 3 is where the cliff lives; whether it is an ice field or an ocean
field decides everything downstream.

**B. Which component.** The ice step is six differentiable pieces in sequence — ocean-to-ice
coupling, the mEVP rheology, ice tracer advection, the cut-off clipping, thermodynamics, and
the ice-ocean stress blend. Freeze each one's state-dependence in the backward, in turn, and
see which removes the cliff. Same attribution method that found the ocean mixing closure.

**C. What changes in the forward at step 4.** A cliff in the adjoint with a smooth forward
means a *threshold* is crossed. The candidates are all visible in the forward state, so print
them per step: how close the deformation rate gets to its floor (the rheology's viscosity goes
as 1/(Delta+delta_min)^2), how many nodes sit near the 1% ice-concentration branch in the mass
term, and how many are near the cut-off bounds. If one of these first fires between step 3 and
step 4, that is the mechanism.
"""

from __future__ import annotations

import argparse
import dataclasses
import contextlib
import pickle
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fesom_jax  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

if Path(fesom_jax.__file__).resolve().parents[1] != _ROOT:
    raise SystemExit(f"fesom_jax resolved to {fesom_jax.__file__}, not {_ROOT}")

from fesom_jax import (ice_adv, ice_coupling, ice_evp, ice_mevp, ice_thermo,  # noqa: E402
                       ssh, surface_forcing)
from fesom_jax.adjoint_stab import freeze  # noqa: E402
from fesom_jax.calibrate import build_params  # noqa: E402
from fesom_jax.ice import IceConfig  # noqa: E402
from fesom_jax.integrate import _run_steps  # noqa: E402
from fesom_jax.mesh import load_mesh  # noqa: E402
from fesom_jax.params import Params  # noqa: E402
from fesom_jax.phc_ic import phc_initial_state  # noqa: E402
from fesom_jax.step import step  # noqa: E402
from horizon_scan import ALE_CFG, DT, GM_CFG, IC_DIR, MESH_DIR, TKE_CFG, seed_date  # noqa: E402

ICE_EXACT = IceConfig(whichEVP=1, adjoint_mode="exact")

# The six differentiable pieces of one ice step, in execution order (ice_step.py:92-160).
# Freezing one means: keep the value it computes, drop its dependence on the state in the
# backward. Same operator as the ocean mixing freeze, applied inside the ice.
COMPONENTS = {
    "ocean2ice":  (ice_coupling, "ocean2ice",
                   "the ocean state handed to the ice (SST, SSS, surface currents, SSH)"),
    "rheology":   (ice_mevp, "mevp_dynamics",
                   "the mEVP momentum solve: 120 stress-relaxation subcycles per step"),
    "iceadv":     (ice_adv, "fct_solve",
                   "flux-corrected transport of ice concentration, volume and snow"),
    "cutoff":     (ice_thermo, "cut_off",
                   "clipping concentration/volume back into physical bounds — a hard switch"),
    "thermo":     (ice_thermo, "thermodynamics",
                   "growth and melt, including the Newton solve for skin temperature"),
}


@contextlib.contextmanager
def frozen(name):
    """Freeze one ice component's state-dependence in the backward pass."""
    if name is None:
        yield
        return
    mod, attr, _ = COMPONENTS[name]
    orig = getattr(mod, attr)

    def wrapped(*a, **k):
        return freeze(orig(*a, **k))

    setattr(mod, attr, wrapped)
    try:
        yield
    finally:
        setattr(mod, attr, orig)


def field_norms(tree, top=6):
    """(field, L2 norm) for the largest-norm float fields, in descending order."""
    names = getattr(tree, "_fields", None)
    leaves = jax.tree.leaves(tree)
    labels = names if names and len(names) == len(leaves) else [f"leaf{i}" for i in range(len(leaves))]
    out = []
    for lab, leaf in zip(labels, leaves):
        a = np.asarray(leaf)
        if a.dtype.kind != "f":
            continue
        if not np.isfinite(a).all():
            out.append((lab, float("inf")))
            continue
        out.append((lab, float(np.sqrt((a.astype(np.float64) ** 2).sum()))))
    out.sort(key=lambda r: (-r[1] if np.isfinite(r[1]) else -np.inf))
    return out[:top]


def ice_thresholds(mesh, st, cfg):
    """The forward quantities whose thresholds could produce an adjoint cliff."""
    u, v = np.asarray(st.u_ice), np.asarray(st.v_ice)
    e11, e22, e12 = ice_evp.strain_rates(mesh, jnp.asarray(u), jnp.asarray(v))
    e11, e22, e12 = (np.asarray(x) for x in (e11, e22, e12))
    e1, e2 = e11 + e22, e11 - e22
    radicand = e1 * e1 + cfg.vale * (e2 * e2 + 4.0 * e12 * e12)
    delta = np.sqrt(np.maximum(radicand, 0.0))
    a = np.asarray(st.a_ice)
    icy = a > 1.0e-8
    # Only ELEMENTS carrying ice matter: an ice-free element has u_ice=v_ice=0, hence Delta=0,
    # and its stress is masked out anyway (ice_mevp.py: `jnp.where(st.ice_el, ...)`). Counting
    # them swamped every statistic with a constant 153392 and hid whatever was really moving.
    el_icy = np.asarray(jnp.any(jnp.asarray(icy)[jnp.asarray(mesh.elem_nodes)], axis=1))
    delta = np.where(el_icy[:, None] if delta.ndim == 2 else el_icy, delta, np.nan)
    d_icy = delta[np.isfinite(delta)]
    return dict(
        # the rheology: viscosity ~ 1/(Delta+delta_min), so its derivative ~ 1/(Delta+dmin)^2
        delta_min_seen=float(d_icy.min()) if d_icy.size else 0.0,
        n_delta_below_1e9=int((d_icy < 1e-9).sum()),
        n_delta_below_dmin=int((d_icy < cfg.delta_min).sum()),
        worst_visc_deriv=float(1.0 / (d_icy.min() + cfg.delta_min) ** 2)
                         if d_icy.size else 0.0,
        # the 1% concentration branch in the ice mass term (ice_mevp.py:94)
        n_a_near_001=int(((a > 0.005) & (a < 0.02)).sum()),
        # the cut-off bounds
        n_a_at_1=int((a >= 0.999).sum()),
        n_icy=int(icy.sum()),
        a_max=float(a.max()),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", required=True)
    ap.add_argument("--steps", type=int, default=8, help="window length in steps")
    ap.add_argument("--run-start-year", type=int, default=1958)
    args = ap.parse_args()
    N = args.steps

    print(f"[setup] backend={jax.default_backend()}", flush=True)
    mesh = load_mesh(MESH_DIR)
    op = ssh.build_ssh_operator(mesh, dt=DT)
    with open(args.snap, "rb") as f:
        st0 = jax.device_put(pickle.load(f))
    yr, mo, dy = seed_date(args.snap, args.run_start_year)
    sst0 = np.asarray(phc_initial_state(mesh, IC_DIR).T[:, 0])
    cf = surface_forcing.build_surface_forcing(mesh, yr, sst_ic=sst0)
    dates = surface_forcing.dates_for_steps(yr, DT, N, start_month=mo, start_day=dy)
    fs = cf.static
    kgm0 = float(Params.defaults().k_gm)
    wet0 = jnp.asarray(mesh.node_layer_mask[:, 0])
    inv_wet = 1.0 / float(jnp.sum(wet0))
    m0 = jnp.asarray(1.0, jnp.float64)
    # ice_dt tracks the ocean dt inside ice_surface_step; mirror that here for the diagnostics
    cfg_eff = ICE_EXACT._replace(ice_dt=ICE_EXACT.ice_ave_steps * DT)

    def one_step(state, sf, m):
        p = build_params({"k_gm": kgm0 * m, "redi_kmax": kgm0 * m})
        return step(state, mesh, op, None, p, dt=DT, is_first_step=False, step_forcing=sf,
                    forcing_static=fs, ice_cfg=ICE_EXACT, gm_cfg=GM_CFG, tke_cfg=TKE_CFG,
                    ale_cfg=ALE_CFG, remat_blocks=True)

    fwd1 = jax.jit(one_step)

    @jax.jit
    def bwd1(state, sf, m, cot):
        _, vjp = jax.vjp(lambda s, mm: one_step(s, sf, mm), state, m)
        return vjp(cot)

    # ---- C. forward thresholds, per step ---------------------------------------------
    print(f"\n=== C. forward: what crosses a threshold, step by step ===", flush=True)
    sfs = [cf.step_forcing(*d) for d in dates]
    states = [st0]
    s = st0
    hdr = ("step  min(Delta)   n(D<1e-9)  n(D<dmin)  worst 1/(D+dm)^2   n(a~0.01)  n(a>=1)  n_icy")
    print(hdr, flush=True)
    d0 = ice_thresholds(mesh, jax.device_get(st0), cfg_eff)
    print(f"   0  {d0['delta_min_seen']:.3e}  {d0['n_delta_below_1e9']:9d}  "
          f"{d0['n_delta_below_dmin']:9d}  {d0['worst_visc_deriv']:16.3e}  "
          f"{d0['n_a_near_001']:9d}  {d0['n_a_at_1']:7d}  {d0['n_icy']:6d}", flush=True)
    for k in range(N):
        s = fwd1(s, sfs[k], m0)
        states.append(s)
        d = ice_thresholds(mesh, jax.device_get(s), cfg_eff)
        print(f"  {k+1:2d}  {d['delta_min_seen']:.3e}  {d['n_delta_below_1e9']:9d}  "
              f"{d['n_delta_below_dmin']:9d}  {d['worst_visc_deriv']:16.3e}  "
              f"{d['n_a_near_001']:9d}  {d['n_a_at_1']:7d}  {d['n_icy']:6d}", flush=True)

    # ---- A. which field, which step ---------------------------------------------------
    print(f"\n=== A. backward one step at a time: cotangent norm by field ===", flush=True)
    seed = jax.tree.map(jnp.zeros_like, states[-1])
    seed = dataclasses.replace(seed, T=seed.T.at[:, 0].set(jnp.where(wet0, inv_wet, 0.0)))
    cot, g_m = seed, jnp.asarray(0.0, jnp.float64)
    for k in reversed(range(N)):
        cot, gm_k = bwd1(states[k], sfs[k], m0, cot)
        g_m = g_m + gm_k
        tot = float(jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in jax.tree.leaves(cot))))
        tops = field_norms(jax.device_get(cot))
        print(f"  after step {k:2d} -> |cot|={tot:.4e}  dL/dm={float(g_m):+.4e}", flush=True)
        print("        " + "  ".join(f"{n}={v:.2e}" for n, v in tops), flush=True)

    # ---- D. how many rheology sweeps can the backward afford? --------------------------
    # 0 sweeps = frozen rheology (measured clean). 120 = exact (measured useless). In between
    # costs 1.1795^K of amplification for the sensitivity of the last K relaxation sweeps.
    def grad_with(name):
        def loss(m):
            def body(c, sf):
                return one_step(c, sf, m), None
            fin = _run_steps(body, st0, xs=cf.stack(dates), checkpoint=True, segments=0)
            return jnp.sum(jnp.where(wet0, fin.T[:, 0], 0.0)) / jnp.sum(wet0)
        with frozen(name):
            return float(jax.jit(jax.grad(loss))(m0))

    print(f"\n=== D. differentiate only the last K mEVP sweeps ({N} steps) ===", flush=True)
    orig_dyn = ice_mevp.mevp_dynamics
    print(f"  {'K':>4s} {'dL/dm':>15s}  predicted amplification 1.1795^K", flush=True)
    for K in (0, 1, 2, 4, 8, 16, 32, 120):
        def patched(*a, _K=K, **k):
            k["adjoint_subcycles"] = _K
            return orig_dyn(*a, **k)
        ice_mevp.mevp_dynamics = patched
        try:
            g = grad_with(None)
        finally:
            ice_mevp.mevp_dynamics = orig_dyn
        print(f"  {K:>4d} {g:>15.6e}  x{1.1795 ** K:.3g}", flush=True)

    # ---- B. which component ------------------------------------------------------------
    print(f"\n=== B. freeze one ice component at a time, {N} steps ===", flush=True)
    base = grad_with(None)
    print(f"  {'nothing frozen':14s} dL/dm={base:+.6e}", flush=True)
    for name in COMPONENTS:
        g = grad_with(name)
        tamed = np.isfinite(g) and abs(g) < 1.0e-2
        print(f"  {name:14s} dL/dm={g:+.6e}  {'<-- TAMED' if tamed else ''}"
              f"   ({COMPONENTS[name][2]})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
