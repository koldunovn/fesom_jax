"""Phase 9b JT.0 — TKE scaffolding gates (cfg seam, State.tke, Params, reader).

JT.0 lands the **config seam + the prognostic field + the dump reader** with NO behavior
change: ``tke_cfg=None`` is byte-identical (the KPP/PP dispatch is untouched), and the
column core / driver are stubs (JT.1 / JT.2). These gates assert exactly that contract:

* :class:`~fesom_jax.tke.TkeConfig` raises on the un-ported combinations (the C
  ``fesom_tke.c:247-253`` init-abort parity: IDEMIX / Langmuir / Dirichlet / mxl_choice≠2);
* :class:`~fesom_jax.params.Params` carries the 4 trainable TKE leaves at the right defaults;
* ``State.tke`` exists, is ``[nod2D, nl]`` float64, and IC = 0 (cold start);
* the 3-way step dispatch raises loudly on the pi path (no surface forcing) and when BOTH
  ``kpp_cfg`` and ``tke_cfg`` are set (one scheme per process);
* ``tke_cfg=None`` runs a normal pi step and leaves ``state.tke`` at 0 (the dead branch);
* :func:`~fesom_jax.io_dump.load_tke_dump` round-trips the 16-rank cdump oracle.

The controlled-replay column gate (cdump inputs → the JAX column core) lands in JT.1.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import ic
from fesom_jax import step as stepmod
from fesom_jax import forcing, ssh
from fesom_jax.kpp import KppConfig
from fesom_jax.mesh import DEFAULT_PI_MESH_DIR, load_mesh
from fesom_jax.params import Params
from fesom_jax.state import State
from fesom_jax.tke import TkeConfig, mixing_tke
from fesom_jax.config import TKE_ALPHA, TKE_C_EPS, TKE_C_K, TKE_CD

ROOT = Path(__file__).resolve().parents[2]
TKE_CDUMP = Path("/work/ab0995/a270088/port/tke/cdump/dump")
DT = 100.0

_pi_skip = pytest.mark.skipif(
    not DEFAULT_PI_MESH_DIR.is_dir(),
    reason=f"pi mesh export missing: {DEFAULT_PI_MESH_DIR}")


# --------------------------------------------------------------------------
# 1. TkeConfig validation — the C fesom_tke.c:247-253 abort parity
# --------------------------------------------------------------------------
def test_tkeconfig_default_valid():
    """The reference config (the defaults) validates and returns self."""
    cfg = TkeConfig()
    assert cfg.validate() is cfg
    assert cfg.mxl_choice == 2 and cfg.only_tke and not cfg.use_dirichlet
    assert cfg.mxl_min == 1.0e-8 and cfg.tke_min == 1.0e-6 and cfg.kappaM_max == 100.0


@pytest.mark.parametrize("kwargs", [
    {"only_tke": False},        # IDEMIX coupling (.not.only_tke) — gate-only
    {"l_lc": True},             # Langmuir (tke_dolangmuir) — gate-only
    {"use_dirichlet": True},    # Dirichlet surface+bottom BCs — Neumann is executed
    {"mxl_choice": 1},          # only Blanke–Delecluse choice 2 is ported
    {"mxl_choice": 3},
])
def test_tkeconfig_unported_combos_raise(kwargs):
    """Each un-ported structural switch raises at validate() (fail loudly, not run
    un-ported physics) — the C abort parity."""
    with pytest.raises(ValueError, match="classical-TKE reference config"):
        TkeConfig(**kwargs).validate()


def test_tkeconfig_is_hashable_static():
    """A NamedTuple ⇒ hashable (usable as a jit static_argname / dict key), no leaves."""
    assert hash(TkeConfig()) == hash(TkeConfig())
    assert TkeConfig(with_diags=True) != TkeConfig(with_diags=False)


# --------------------------------------------------------------------------
# 2. Params — the 4 trainable TKE leaves (the PRIMARY ML-hook seam)
# --------------------------------------------------------------------------
def test_params_tke_defaults():
    """The 4 TKE constants default to the namelist.cvmix reference values, as float64
    scalar arrays (the k_gm default_factory pattern)."""
    for p in (Params(k_ver=jnp.asarray(1e-5), a_ver=jnp.asarray(1e-4)), Params.defaults()):
        assert float(p.tke_c_k) == TKE_C_K == 0.1
        assert float(p.tke_c_eps) == TKE_C_EPS == 0.7
        assert float(p.tke_cd) == TKE_CD == 3.75
        assert float(p.tke_alpha) == TKE_ALPHA == 30.0
        for leaf in (p.tke_c_k, p.tke_c_eps, p.tke_cd, p.tke_alpha):
            assert leaf.dtype == jnp.float64 and leaf.shape == ()


def test_params_tke_leaves_registered():
    """The TKE leaves are pytree DATA leaves (so jax.grad returns them) — the
    register_dataclass data_fields were updated alongside the dataclass."""
    import jax
    leaves = jax.tree_util.tree_leaves(Params.defaults())
    assert len(leaves) == 8  # k_ver, a_ver, k_gm, redi_kmax, + 4 tke
    # round-trips through tree flatten/unflatten unchanged
    flat, treedef = jax.tree_util.tree_flatten(Params.defaults())
    p2 = jax.tree_util.tree_unflatten(treedef, flat)
    assert float(p2.tke_cd) == 3.75


# --------------------------------------------------------------------------
# 3. State.tke — the one prognostic mixing field (interface-indexed, IC=0)
# --------------------------------------------------------------------------
@_pi_skip
def test_state_tke_field_zero_ic():
    mesh = load_mesh()
    st = State.zeros(mesh)
    assert st.tke.shape == (mesh.nod2D, mesh.nl)
    assert st.tke.dtype == jnp.float64
    assert np.all(np.asarray(st.tke) == 0.0)          # cold start
    # rest + the pi IC inherit the 0 default (cold start, no extra wiring)
    assert np.all(np.asarray(State.rest(mesh).tke) == 0.0)
    assert np.all(np.asarray(ic.initial_state(mesh).tke) == 0.0)


# --------------------------------------------------------------------------
# 4. The 3-way step dispatch — raises loudly; tke_cfg=None is the dead branch
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def pi_step():
    if not DEFAULT_PI_MESH_DIR.is_dir():
        pytest.skip(f"pi mesh export missing: {DEFAULT_PI_MESH_DIR}")
    mesh = load_mesh()
    op = ssh.build_ssh_operator(mesh, dt=DT)
    stress = forcing.surface_stress(mesh)
    st = ic.initial_state(mesh)
    return mesh, op, stress, st


def test_dispatch_pi_path_raises(pi_step):
    """TKE on the pi path (no CORE2 forcing ⇒ stress_node_surf is None) raises — TKE is
    a forced-path feature (it needs the surface |stress|), the KPP precedent."""
    mesh, op, stress, st = pi_step
    with pytest.raises(ValueError, match="requires CORE2 surface forcing"):
        stepmod.step(st, mesh, op, stress, dt=DT, is_first_step=True, tke_cfg=TkeConfig())


def test_dispatch_both_cfgs_raise(pi_step):
    """Setting BOTH kpp_cfg and tke_cfg raises (the C runs exactly one scheme/process)."""
    mesh, op, stress, st = pi_step
    with pytest.raises(ValueError, match="both set"):
        stepmod.step(st, mesh, op, stress, dt=DT, is_first_step=True,
                     tke_cfg=TkeConfig(), kpp_cfg=KppConfig())


def test_tke_cfg_none_leaves_tke_zero(pi_step):
    """tke_cfg=None ⇒ the dead branch: a normal pi step never touches state.tke (it stays
    0) and the step otherwise runs exactly as before (byte-identity is the full suite)."""
    mesh, op, stress, st = pi_step
    nxt = stepmod.step(st, mesh, op, stress, dt=DT, is_first_step=True)  # tke_cfg=None
    assert np.all(np.asarray(nxt.tke) == 0.0)
    # the field is preserved across the step (carry threads it untouched)
    assert nxt.tke.shape == st.tke.shape


def test_mixing_tke_is_stub():
    """The driver is a stub until JT.2 — calling it fails loudly (so a premature valid
    tke_cfg wiring can't silently run nothing)."""
    with pytest.raises(NotImplementedError, match="JT.2"):
        mixing_tke(None, None, None, None, None, None, TkeConfig(), Params.defaults())


# --------------------------------------------------------------------------
# 5. The dump reader — the 16-rank cdump oracle round-trips (the JT.1 gate's input)
# --------------------------------------------------------------------------
@pytest.mark.skipif(not TKE_CDUMP.is_dir(),
                    reason=f"TKE cdump oracle missing: {TKE_CDUMP}")
def test_load_tke_dump_roundtrip():
    """load_tke_dump merges the 16 owned-row ranks by gid into a dense global field with
    no gaps (strict mode), for a node input, a node output, a node diag, the wired Kv and
    the element Av — the controlled-replay gate (JT.1) reads exactly these."""
    from fesom_jax.io_dump import load_tke_dump, TKE_TAGS
    tags = ["dztrr", "tkeold", "tke", "tkeav", "tkekv", "ttot", "normstress", "kv", "av"]
    fields, meta = load_tke_dump(TKE_CDUMP, tags, step=1)
    for tag in tags:
        arr = fields[tag]
        assert not np.isnan(arr).any(), f"{tag}: unfilled gids after merge"
        assert arr.shape[1] == (1 if tag == "normstress" else 48)
        assert meta[tag]["nranks"] == 16
    # node vs element global sizes (the CORE2 mesh)
    assert fields["tke"].shape[0] == 126858          # nod2D
    assert fields["av"].shape[0] == 244659           # elem2D
    # all 20 tags are catalogued
    assert len(TKE_TAGS) == 20


@pytest.mark.skipif(not TKE_CDUMP.is_dir(),
                    reason=f"TKE cdump oracle missing: {TKE_CDUMP}")
def test_load_tke_dump_all_three_steps():
    """All 3 cdump steps are present (the replay gate reads s1–s3)."""
    from fesom_jax.io_dump import load_tke_dump
    for step in (1, 2, 3):
        f, _ = load_tke_dump(TKE_CDUMP, ["tke"], step=step)
        assert f["tke"].shape == (126858, 48) and not np.isnan(f["tke"]).any()
