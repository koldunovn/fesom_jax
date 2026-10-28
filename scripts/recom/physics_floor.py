"""The physics floor of the pi REcoM pair (lane P, step P3): JAX ``configs/pi_recom.yaml`` against its physics-matched
Fortran twin (``recom_ref/pi_30d_jan_jaxphys``) after steps 1 and 2, both from the Fortran step-0 state.

The step-0 state is the Fortran cold start read from the step-1 dumps: T and S from ``recom_in`` (the IC; the ice
step does not touch the tracers), their AB2 history ``valuesold`` from ``tra_in``, the rest thicknesses, and the
seeded sea ice (FESOM2's cold-start rule, :func:`fesom_jax.ice.seed_ice`). Fortran fields after step N:

  forcing  ``recom_in(N)``: u_wind, v_wind, shortwave, press_air (the step-N forcing)
  ice      ``recom_in(N)``: a_ice, m_ice, m_snow (FESOM2 runs the ice step before the ocean step)
  uv, hnode_new, Kv   ``transport(N)`` (start of the tracer step: after the momentum, SSH and ALE substeps)
  T, S     ``step_end(N)``
  hbar     Σ_k hnode_new (both sides on the same depth, so the difference is the hbar difference)

Run on a compute node (XLA compiles): ``python scripts/recom/physics_floor.py [--dumps DIR] [--steps 2]``.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from fesom_jax import ice, ssh, surface_forcing  # noqa: E402
from fesom_jax import step as stepmod  # noqa: E402
from fesom_jax.io_dump import load_recom_dump  # noqa: E402
from fesom_jax.mesh import load_mesh  # noqa: E402
from fesom_jax.run_config import load_yaml  # noqa: E402
from fesom_jax.state import State  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DUMPS = Path("/work/ab0995/a270088/recom_ref/pi_30d_jan_jaxphys/dumps")


def _pad(a, nl):
    """A Fortran layer field (n, nl-1[, ...]) into the JAX level count nl (zero below)."""
    if a.shape[1] == nl:
        return a
    pad = [(0, 0)] * a.ndim
    pad[1] = (0, nl - a.shape[1])
    return np.pad(a, pad)


def step0_state(mesh, d1, ale_cfg):
    nl = int(mesh.nl)
    mask = np.asarray(mesh.node_layer_mask)
    st = jax.tree.map(np.asarray, State.rest(mesh, ale_cfg=ale_cfg))
    T = np.where(mask, _pad(d1["recom_in.tr1"], nl), 0.0)
    S = np.where(mask, _pad(d1["recom_in.tr2"], nl), 0.0)
    T_old = np.where(mask, _pad(d1["tra_in.tr1.valuesold"], nl), 0.0)
    S_old = np.where(mask, _pad(d1["tra_in.tr2.valuesold"], nl), 0.0)
    st = dataclasses.replace(st, T=T, S=S, T_old=T_old, S_old=S_old)
    st = ice.seed_ice(st, mesh, st.T[:, 0], xp=np)
    return st


def _stats(name, jx, fx, valid):
    """max|Δ| over the valid entries, max|ref| there, the relative value and where the max sits."""
    d = np.abs(np.where(valid, jx - fx, 0.0))
    ref = float(np.max(np.abs(np.where(valid, fx, 0.0))))
    i = np.unravel_index(int(np.argmax(d)), d.shape)
    dmax = float(d[i])
    return dict(field=name, maxdiff=dmax, maxref=ref, rel=dmax / ref if ref else float("nan"), at=tuple(int(x) for x in i))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", type=Path, default=DUMPS)
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "pi_recom.yaml")
    ap.add_argument("--steps", type=int, default=2)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    phys = cfg.physics_kwargs()
    dt = float(cfg.dt)
    mesh = load_mesh()
    nl = int(mesh.nl)
    d = {n: load_recom_dump(args.dumps, n)[0] for n in range(1, args.steps + 1)}
    d1 = d[1]

    # the mesh is FESOM2's pi on one rank: identity numbering, the same coordinates and level counts
    assert np.array_equal(d1["mesh.gid_nod"], np.arange(1, mesh.nod2D + 1))
    geo_f, geo_j = d1["mesh.geo_coord_nod2D"], np.asarray(mesh.geo_coord_nod2D)       # (nod2D, 2) radians
    dlon = np.angle(np.exp(1j * (geo_f[:, 0] - geo_j[:, 0])))                         # longitudes differ by 2 pi
    assert np.max(np.abs(dlon)) < 1e-12 and np.max(np.abs(geo_f[:, 1] - geo_j[:, 1])) < 1e-12
    nlev_n = d1["mesh.nlevels_nod2D"]
    nlev_e = d1["mesh.nlevels"]
    assert np.array_equal(nlev_n, np.asarray(mesh.nlevels_nod2D)), "pi node level counts differ"

    st = step0_state(mesh, d1, phys["ale_cfg"])
    print(f"[floor] rest hnode vs Fortran step-1 recom_in.hnode: "
          f"{np.max(np.abs(np.asarray(st.hnode)[:, :nl - 1] - d1['recom_in.hnode'])):.3e}")
    st = jax.tree.map(jnp.asarray, st)

    cf = surface_forcing.build_surface_forcing(mesh, cfg.forcing["start_year"], sst_ic=np.asarray(st.T[:, 0]),
                                               dt=dt, mslp=cfg.forcing_mslp(), **cfg.forcing_paths())
    dates = surface_forcing.dates_for_steps(cfg.forcing["start_year"], dt, args.steps)
    op = ssh.build_ssh_operator(mesh, dt=dt)
    zero = jnp.zeros((mesh.elem2D, 2))

    lay_n = np.arange(nl - 1)[None, :] < (nlev_n - 1)[:, None]          # wet layers (nod2D, nl-1)
    lev_n = np.arange(nl)[None, :] < nlev_n[:, None]                    # interfaces (nod2D, nl)
    lay_e = (np.arange(nl - 1)[None, :] < (nlev_e - 1)[:, None])[..., None]
    rows = []
    for n in range(1, args.steps + 1):
        sf = cf.step_forcing(*dates[n - 1])
        st = stepmod.step_jit(st, mesh, op, zero, dt=dt, is_first_step=(n == 1), step_forcing=sf,
                              forcing_static=cf.static, **phys)
        f = d[n]
        allnod = np.ones(mesh.nod2D, bool)
        out = [
            _stats("u_wind", np.asarray(sf.u_air), f["recom_in.u_wind"], allnod),
            _stats("v_wind", np.asarray(sf.v_air), f["recom_in.v_wind"], allnod),
            _stats("shortwave", np.asarray(sf.shortwave), f["recom_in.shortwave"], allnod),
            _stats("press_air", np.asarray(sf.press_air), f["recom_in.press_air"], allnod),
            _stats("a_ice", np.asarray(st.a_ice), f["recom_in.a_ice"], allnod),
            _stats("m_ice", np.asarray(st.m_ice), f["recom_in.m_ice"], allnod),
            _stats("m_snow", np.asarray(st.m_snow), f["recom_in.m_snow"], allnod),
            _stats("uv", np.asarray(st.uv)[:, :nl - 1], f["transport.uv"], lay_e),
            _stats("hnode", np.asarray(st.hnode)[:, :nl - 1], f["transport.hnode_new"], lay_n),
            _stats("hbar(sum hnode)", np.sum(np.where(lay_n, np.asarray(st.hnode)[:, :nl - 1], 0), axis=1),
                   np.sum(np.where(lay_n, f["transport.hnode_new"], 0), axis=1), allnod),
            _stats("w", np.asarray(st.w), f["transport.w"], lev_n),
            _stats("Kv", np.asarray(st.Kv), f["transport.Kv"], lev_n),
            _stats("T", np.asarray(st.T)[:, :nl - 1], f["step_end.tr1"], lay_n),
            _stats("S", np.asarray(st.S)[:, :nl - 1], f["step_end.tr2"], lay_n),
        ]
        for r in out:
            r["step"] = n
            rows.append(r)
        # where the tracer gap sits: per layer, how many cells, and the column of the largest one
        dT = np.where(lay_n, np.asarray(st.T)[:, :nl - 1] - f["step_end.tr1"], 0.0)
        dS = np.where(lay_n, np.asarray(st.S)[:, :nl - 1] - f["step_end.tr2"], 0.0)
        print(f"[floor] step {n}: max|dT| per layer 0..5 " + " ".join(f"{x:.1e}" for x in np.abs(dT).max(0)[:6])
              + f"; cells |dT|>1e-6: {int((np.abs(dT) > 1e-6).sum())}, >1e-3: {int((np.abs(dT) > 1e-3).sum())} "
              f"of {int(lay_n.sum())}; |dS|>1e-6: {int((np.abs(dS) > 1e-6).sum())}")
        k = int(np.argmax(np.abs(dT).max(1)))
        print(f"[floor] step {n}: column {k} (lat {np.degrees(float(mesh.geo_coord_nod2D[k, 1])):.1f}, "
              f"nlev {int(nlev_n[k])}, a_ice F {f['recom_in.a_ice'][k]:.3f}) T_F[0:4] "
              f"{np.round(f['step_end.tr1'][k, :4], 5)} dT {np.array2string(dT[k, :4], precision=2)}")
    print(f"\n{'step':>4} {'field':>16} {'max|dJ-F|':>11} {'max|F|':>11} {'rel':>10}  at")
    for r in rows:
        print(f"{r['step']:>4} {r['field']:>16} {r['maxdiff']:11.3e} {r['maxref']:11.3e} {r['rel']:10.2e}  {r['at']}")
    print("PHYSICS_FLOOR_DONE")


if __name__ == "__main__":
    main()
