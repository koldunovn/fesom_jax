"""Task G.6 gate — the Redi tracer terms (G7a vertical-explicit + G7b 5-branch +
K33), against the C Redi dump (T captured before/after each Redi piece all-node).

The dump (``redi_*.f64`` in ``data/gm_dump_core2/``, from the GM-ON CORE2 run with
``FESOM_REDI_DUMP_DIR``) holds, for the T tracer at step 1: ``T_old`` (the AB2
``valuesold`` the gradients read), ``T_pre`` (post-advection T, before Redi),
``T_g7a`` (after G7a), ``T_g7b`` (after G7b), ``tr_xy`` (element), ``tr_z`` (node).
``slope_tapered``/``Ki``/``hnode_new`` come from the GM dump (same step-1 state).
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import gm_redi, ops
from fesom_jax.io_dump import load_gm_dump
from fesom_jax.mesh import load_mesh

CORE2_MESH_DIR = Path(__file__).resolve().parents[2] / "data" / "mesh_core2"
GM_DUMP_DIR = Path(__file__).resolve().parents[2] / "data" / "gm_dump_core2"
DT = 500.0   # the CORE2 dump dt (fesom_phase1_dt)

pytestmark = pytest.mark.skipif(
    not (CORE2_MESH_DIR.is_dir() and (GM_DUMP_DIR / "redi_meta.txt").is_file()),
    reason=f"CORE2 mesh / Redi dump missing ({GM_DUMP_DIR}); run jax_gm_dump_core2.sh",
)


def _redi(name, shape):
    return np.fromfile(GM_DUMP_DIR / f"redi_{name}.f64", dtype="<f8").reshape(shape)


@pytest.fixture(scope="module")
def setup():
    mesh = load_mesh(CORE2_MESH_DIR)
    f, meta = load_gm_dump(GM_DUMP_DIR)
    N, E, nl = meta["N"], meta["E"], meta["nl"]
    redi = {
        "T_old": _redi("T_old", (N, nl)), "T_pre": _redi("T_pre", (N, nl)),
        "T_g7a": _redi("T_g7a", (N, nl)), "T_g7b": _redi("T_g7b", (N, nl)),
        "tr_xy": _redi("tr_xy", (E, nl - 1, 2)), "tr_z": _redi("tr_z", (N, nl)),
    }
    return mesh, f, redi


def test_tr_xy_matches_c_dump(setup):
    """tr_xy = ∇(T_old) per element."""
    mesh, f, redi = setup
    txy = np.asarray(gm_redi.tr_xy_elem(mesh, jnp.asarray(redi["T_old"])))
    nl = mesh.nl
    m = np.broadcast_to(np.asarray(mesh.elem_layer_mask)[:, : nl - 1, None],
                        redi["tr_xy"].shape)
    d = np.max(np.abs(txy[:, : nl - 1, :][m] - redi["tr_xy"][m]))
    print(f"\ntr_xy max|Δ|={d:.3e}")
    assert d < 1e-12


def test_g7a_matches_c_dump(setup):
    """T_pre + diff_ver_part_redi_expl == T_g7a (the vertical-explicit Redi delta)."""
    mesh, f, redi = setup
    delta = gm_redi.diff_ver_part_redi_expl(
        mesh, jnp.asarray(redi["T_old"]), jnp.asarray(f["slope_tapered"]),
        jnp.asarray(f["Ki"]), jnp.asarray(f["hnode_new"]), dt=DT)
    got = redi["T_pre"] + np.asarray(delta)
    ref = redi["T_g7a"]
    m = np.asarray(mesh.node_layer_mask)
    d = np.max(np.abs(got[m] - ref[m]))
    # the Redi delta itself (how much G7a moved T):
    moved = np.max(np.abs(redi["T_g7a"][m] - redi["T_pre"][m]))
    print(f"\nG7a: T max|Δ|={d:.3e}  (G7a moved T by up to {moved:.3e})")
    assert d < 1e-12
    assert moved > 1e-9       # G7a is genuinely active (not a trivial gate)


def test_g7b_matches_c_dump(setup):
    """T_g7a + diff_part_hor_redi == T_g7b — the horizontal Redi edge flux (the
    5-branch → 3-case edge loop). The hardest GM kernel."""
    mesh, f, redi = setup
    helem = ops.gather_nodes_to_elem(
        jnp.asarray(f["hnode"]), mesh.elem_nodes).mean(axis=1)
    helem = jnp.where(mesh.elem_layer_mask, helem, 0.0)
    delta = gm_redi.diff_part_hor_redi(
        mesh, jnp.asarray(redi["T_old"]), jnp.asarray(f["slope_tapered"]),
        jnp.asarray(f["Ki"]), jnp.asarray(f["hnode"]), jnp.asarray(f["hnode_new"]),
        helem, dt=DT)
    got = redi["T_g7a"] + np.asarray(delta)
    ref = redi["T_g7b"]
    m = np.asarray(mesh.node_layer_mask)
    d = np.max(np.abs(got[m] - ref[m]))
    moved = np.max(np.abs(redi["T_g7b"][m] - redi["T_g7a"][m]))
    print(f"\nG7b: T max|Δ|={d:.3e}  (G7b moved T by up to {moved:.3e})")
    assert d < 1e-11          # scatter class (edge→node antisymmetric scatter)
    assert moved > 1e-9       # G7b genuinely active


def test_k33_augmentation_sane(setup):
    """K33_aug = the isoneutral vertical-diffusivity augmentation (Kv += K33_aug).
    No K33-isolated dump (it lives in the diffusion) → sanity here (finite, active,
    physical magnitude); the tight gate is the G.7 assembled post-step T/S."""
    import jax
    mesh, f, redi = setup
    st = jnp.asarray(f["slope_tapered"]); Ki = jnp.asarray(f["Ki"])
    k33 = np.asarray(gm_redi.k33_augmentation(mesh, st, Ki))
    m = np.asarray(mesh.node_layer_mask)
    assert np.all(np.isfinite(k33))
    assert np.all(k33[~m] == 0.0)        # 0 off the interior interfaces
    assert np.max(k33[m]) > 0.0          # active where slopes present
    # K33 ~ Ki·slope² : physical (positive, bounded — not blowing past ~Ki).
    print(f"\nK33_aug: max={np.max(k33[m]):.3e}  (Ki up to ~1000, slope² small)")
    assert np.max(k33[m]) < 1e4
    g = np.asarray(jax.grad(lambda s: jnp.sum(gm_redi.k33_augmentation(mesh, s, Ki)))(st))
    assert np.all(np.isfinite(g))        # AD-finite
