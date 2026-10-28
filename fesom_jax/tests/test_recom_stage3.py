"""REcoM plan Task 12, gate G4a: full REcoM's stage 3 (:func:`fesom_jax.recom.transport.recom_stage3`) fed with the
Fortran step's own transport fields reproduces Fortran's tracer loop for the 31 BGC tracers.

From each dumped step of the Fortran reference (``docs/recom/FORTRAN_REFERENCE.md``): the transport of the tracer step
(``transport.*``: uv, the GM bolus velocity, w_e/w_i, the thicknesses, Kv, the Redi slopes and Ki), the tracers at the
start of the loop (``tra_in``: stage 1's output and the AB2 ``valuesold``), the step's final T and S (``tra_out`` of
tracers 1 and 2, S before its clip), stage 1's surface fluxes, ``GlodecayBenthos`` and benthos (``recom_out``) →
stage 3 → every BGC tracer's ``tra_out.values`` and the merged ``step_end.Benthos``. The JAX pi mesh is FESOM2's pi
mesh in the same node order (its coordinates differ from Fortran's by ~1e-14 rad, lane I).
"""

from __future__ import annotations

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import ale, gm_redi
from fesom_jax.adjoint_stab import AdjointConfig
from fesom_jax.gm import GMDiag
from fesom_jax.io_dump import load_recom_dump
from fesom_jax.mesh import load_mesh
from fesom_jax.recom import params as P
from fesom_jax.recom import tracers as T
from fesom_jax.recom import transport
from fesom_jax.recom.forcing import SurfaceFluxes

REF = Path(os.environ.get("FESOM_RECOM_REF", "/work/ab0995/a270088/recom_ref"))
REPO = Path(__file__).resolve().parents[2]
NML = REPO / "scripts" / "recom" / "fortran" / "namelists" / "namelist.recom"
STEPS = [("pi_30d_jan", s) for s in (1, 2, 17, 224, 960)] + [("pi_1d_jul", s) for s in (1, 17)]
TOL = 1e-11
HAVE_REF = (REF / "pi_30d_jan" / "dumps").is_dir()
need_ref = pytest.mark.skipif(not HAVE_REF, reason=f"REcoM Fortran reference dumps absent ({REF})")

TRANSPORT = ("uv", "fer_uv", "w_e", "w_i", "fer_w", "hnode", "hnode_new", "helem", "Kv", "Ki", "slope_tapered")
SFC = ("GloCO2flux_seaicemask", "GloO2flux_seaicemask", "AtmFeInput", "AtmNInput", "relax_alk", "GlodecayBenthos")


def _names():
    r = [f"transport.{n}" for n in TRANSPORT] + [f"recom_out.{n}" for n in SFC] + ["recom_out.Benthos",
                                                                                     "step_end.Benthos"]
    r += [f"tra_out.tr{i}.values" for i in (1, 2)]
    for tid in T.BGC_IDS:
        r += [f"tra_in.tr{tid}.values", f"tra_in.tr{tid}.valuesold", f"tra_out.tr{tid}.values"]
    return r


def _pad(a, nl, axis=1):
    a = np.asarray(a, np.float64)
    if a.shape[axis] == nl:
        return a
    pad = [(0, 0)] * a.ndim
    pad[axis] = (0, nl - a.shape[axis])
    return np.pad(a, pad)


def _stage3_inputs(run, step):
    from fesom_jax.tests import test_recom_sinking as TS
    f, meta = load_recom_dump(REF / run / "dumps", step, names=_names())
    fs, _ = TS._load(run, step)
    geom_sink = TS._geom(fs)
    sw, prm = TS._params()
    mesh = load_mesh()
    nl = mesh.nl
    tr = {k: _pad(f[f"transport.{k}"], nl) for k in ("hnode", "hnode_new", "w_e", "w_i", "fer_w", "Kv", "Ki")}
    uv = _pad(f["transport.uv"], nl) + _pad(f["transport.fer_uv"], nl)
    helem = _pad(f["transport.helem"], nl)
    hnode, hnode_new = jnp.asarray(tr["hnode"]), jnp.asarray(tr["hnode_new"])
    zbar3_live, Z3d_live = ale.live_geometry(mesh, hnode)
    zbar3_new, Z3d_new = ale.live_geometry(mesh, hnode_new)
    gm = GMDiag(fer_uv=jnp.zeros_like(jnp.asarray(uv)), slope_tapered=jnp.asarray(f["transport.slope_tapered"]),
                Ki=jnp.asarray(tr["Ki"]), fer_K=jnp.zeros_like(hnode), fer_C=jnp.zeros(mesh.nod2D))
    Kv_eff = jnp.asarray(tr["Kv"]) + gm_redi.k33_augmentation(mesh, gm.slope_tapered, gm.Ki, zbar3=zbar3_new,
                                                              Z3d=Z3d_new)
    X1 = np.stack([_pad(f[f"tra_in.tr{i}.values"], nl) for i in T.BGC_IDS], axis=-1)
    X0 = np.stack([_pad(f[f"tra_in.tr{i}.valuesold"], nl) for i in T.BGC_IDS], axis=-1)
    zero = jnp.zeros(mesh.nod2D)
    sfc = SurfaceFluxes(**{n: jnp.asarray(f[f"recom_out.{n}"]) for n in SFC[:-1]},
                        GloCO2flux=zero, GloO2flux=zero, GloPCO2surf=zero, GlodPCO2surf=zero, GloHplus=zero,
                        PistonVelocity=zero, alphaCO2=zero, x_co2atm=zero,
                        GlodecayBenthos=jnp.asarray(f["recom_out.GlodecayBenthos"]).T)
    kw = dict(uv_adv=jnp.asarray(uv), w_e_adv=jnp.asarray(tr["w_e"] + tr["fer_w"]), w_i=jnp.asarray(tr["w_i"]),
              helem=jnp.asarray(helem), hnode=hnode, hnode_new=hnode_new, zbar3_live=zbar3_live,
              Z3d_live=Z3d_live, Z3d_new=Z3d_new, gm_diag=gm, Kv_eff=Kv_eff,
              temp=jnp.asarray(_pad(f["tra_out.tr1.values"], nl)), salt=jnp.asarray(_pad(f["tra_out.tr2.values"], nl)),
              sfc=sfc, benthos=jnp.asarray(f["recom_out.Benthos"]).T, geom_sink=geom_sink, params=prm,
              switches=sw, dt=float(meta["dt"]))
    want = dict(X=np.stack([f[f"tra_out.tr{i}.values"] for i in T.BGC_IDS], axis=-1),
                benthos=np.asarray(f["step_end.Benthos"]).T, nlev=np.asarray(geom_sink.nlev))
    return mesh, jnp.asarray(X1), jnp.asarray(X0), kw, want


def g4a_errors(run, step) -> dict:
    """``{(tracer ID or "Benthos"): (rel. error, max |fortran|)}`` over wet cells."""
    mesh, X1, X0, kw, want = _stage3_inputs(run, step)
    ident = lambda f, kind: f                                         # noqa: E731
    closed = {k: kw.pop(k) for k in ("params", "switches", "dt", "geom_sink")}   # constants of the program
    X_new, _, b_new = jax.jit(lambda X1, X0, dyn: transport.recom_stage3(
        mesh, X1, X0, exch=ident, ckpt=lambda g: g, adj=AdjointConfig(), **dyn, **closed))(X1, X0, kw)
    nl1 = mesh.nl - 1
    wet = np.arange(nl1)[None, :] < want["nlev"][:, None]
    out = {}
    X_new = np.asarray(X_new)[:, :nl1]
    for s, tid in enumerate(T.BGC_IDS):
        a, b = X_new[..., s][wet], want["X"][..., s][wet]
        den = float(np.max(np.abs(b)))
        out[tid] = (float(np.max(np.abs(a - b))) / den if den > 0 else float(np.max(np.abs(a))), den)
    a, b = np.asarray(b_new), want["benthos"]
    den = float(np.max(np.abs(b)))
    out["Benthos"] = (float(np.max(np.abs(a - b))) / den if den > 0 else float(np.max(np.abs(a))), den)
    return out


@need_ref
@pytest.mark.parametrize("run,step", STEPS)
# Needs FESOM2 #944 (the Redi K33 weights) in gm_redi.k33_augmentation (8707dce): without it the BGC tracers miss by
# ~3e-4 rel in the implicit diffusion (advection, Redi and T* match at round-off — stage diagnosis 2026-10-25).
def test_g4a_stage3_vs_fortran(run, step):
    err = g4a_errors(run, step)
    worst = max(err.items(), key=lambda kv: kv[1][0])
    print(f"G4A {run} {step}: worst {worst[0]} {worst[1][0]:.2e} "
          f"(median {np.median([v[0] for v in err.values()]):.2e})")
    bad = {k: v for k, v in err.items() if v[0] > TOL}
    assert not bad, bad


def test_surface_bc_columns():
    """dt times the stage-1 flux in the bc_surface columns (1001 DIN, 1002 DIC, 1003 Alk, 1019 Fe, 1022 O2), 0 in every
    other column (1018 DSi included)."""
    n = 5
    vals = {name: jnp.full((n,), float(k + 1)) for k, name in enumerate(SurfaceFluxes._fields)}
    vals["GlodecayBenthos"] = jnp.zeros((n, 4))
    bc = np.asarray(transport.surface_bc(SurfaceFluxes(**vals), 2.0))
    assert bc.shape == (n, T.NTR)
    for s, tid in enumerate(T.BGC_IDS):
        src = T.BC_SURFACE_SOURCE.get(tid, "zero")
        want = 0.0 if src == "zero" else 2.0 * float(vals[src][0])
        np.testing.assert_array_equal(bc[:, s], want, err_msg=str(tid))
