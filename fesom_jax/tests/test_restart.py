"""A1 gate: device-count-portable sharded restart (:mod:`fesom_jax.zarr_output`).

Round-trips the **FULL** prognostic :class:`~fesom_jax.state.State` (every leaf — incl. the
history/carry slots ``T_old``/``S_old``/``uv_rhsAB``/``sigma*``/``tke``/…) through a gid-keyed
Zarr restart and reloads it onto a **different device count**, asserting bit-exact identity:

  * full-State round-trip at the SAME P (write → ``reconstruct_global`` == the original global);
  * **save P=4 → load P=2 == identity** AND **save P=2 → load P=4 == identity** (down- & up-size);
  * metadata (``step`` / ``calendar_date`` / ``dt_stage``) preserved;
  * **no leaf silently dropped** — every State leaf has its own on-disk dataset, and a per-field
    distinct-signature round-trip catches any dropped/zeroed leaf or mis-scattered entity.

Runs on CPU fake-devices + the small ``pi`` mesh + a **halo-free block partition**
(:func:`fesom_jax.partit.synth_block_partition`): restart gather/scatters entities by ownership,
**partition-independently**, and never exchanges halos — so no real ``dist_<NP>`` files are needed.
The multi-device cases SKIP below 4 fake-devices. Emits ``RESTART_PORTABLE_OK``.

Standalone (4 fake devices):
  XLA_FLAGS=--xla_force_host_platform_device_count=4 PY -m pytest fesom_jax/tests/test_restart.py -x
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("zarr")

from fesom_jax import halo, partit, shard_mesh
from fesom_jax import integrate_sharded as ish
from fesom_jax import zarr_output as zo
from fesom_jax.mesh import load_mesh
from fesom_jax.state import State

ROOT = Path(__file__).resolve().parents[2]
# the pi mesh SHIPS inside the package, so this gate runs anywhere (incl. CI) —
# it used to point at the repo-root data/ symlink, which only exists on Levante.
from fesom_jax.mesh import DEFAULT_PI_MESH_DIR as PI_MESH
NDEV = len(jax.devices())

avail = pytest.mark.skipif(not PI_MESH.is_dir(), reason="pi mesh missing")
need4 = pytest.mark.skipif(
    NDEV < 4,
    reason="needs >=4 fake-devices (XLA_FLAGS=--xla_force_host_platform_device_count=4)")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _signature_state(mesh) -> State:
    """A State whose EVERY leaf carries a distinct, gid-encoding signature so the
    round-trip catches a dropped leaf (its value would be zero/pad), a swapped leaf
    (field offset differs), and a mis-scattered entity (the leading-dim value encodes
    the gid). Trailing axes get a small per-index pattern so they're checked too."""
    st = State.zeros(mesh)
    out = {}
    for i, f in enumerate(dataclasses.fields(State)):
        a = np.asarray(getattr(st, f.name))
        lead = a.shape[0]                                  # nod2D- or elem2D-leading
        gid = np.arange(lead, dtype=np.float64)
        val = (i + 1) * 1e7 + gid                          # field offset + entity gid
        val = np.broadcast_to(val.reshape((lead,) + (1,) * (a.ndim - 1)), a.shape).copy()
        if a.ndim > 1:
            tail = np.arange(int(np.prod(a.shape[1:])), dtype=np.float64).reshape(a.shape[1:])
            val = val + 1e-3 * tail[None, ...]
        out[f.name] = jnp.asarray(val)
    return dataclasses.replace(st, **out)


def _to_folded_sharded(global_state, mesh, npes):
    """Global host State → folded ``[P*Lmax, …]`` sharded device State (what a sharded run
    holds) + its ShardedMesh + block partition."""
    part = partit.synth_block_partition(mesh.nod2D, mesh.elem2D, mesh.edge2D, npes)
    sm = shard_mesh.build_sharded_mesh(mesh, part)
    sp = shard_mesh.partition_state(global_state, part)    # [P, Lmax, …] host numpy
    fs, spec = ish.folded_state(sp)                        # [P*Lmax, …] + all-'p' spec
    jmesh = halo.device_mesh("p", devices=jax.devices()[:npes])
    placed = ish._to_global_sharded(fs, spec, jmesh)       # per-shard host→device copy
    return placed, sm, part


def _unfold_host(folded_state, npes):
    """Folded sharded device State → ``[P, Lmax, …]`` host numpy dict."""
    uf = ish.unfold_state(folded_state, npes)
    return {f.name: np.asarray(getattr(uf, f.name)) for f in dataclasses.fields(State)}


def _roundtrip(tmp_path, mesh, global0, p_save, p_load):
    """Save at ``p_save``, reload at ``p_load``, assert the loaded ``[P_load, Lmax, …]``
    State is bit-identical to partitioning the ORIGINAL global directly onto ``p_load``
    (interior + halo + pad lanes). Returns the restart metadata."""
    saved, sm_s, part_s = _to_folded_sharded(global0, mesh, p_save)
    d = tmp_path / f"restart_{p_save}to{p_load}"
    zo.write_restart(d, saved, sm_s, part_s,
                     step=1234, calendar_date="1958-03-15", dt_stage=1800.0)
    part_l = partit.synth_block_partition(mesh.nod2D, mesh.elem2D, mesh.edge2D, p_load)
    loaded, meta = zo.read_restart(d, mesh, part_l)
    ref = shard_mesh.partition_state(global0, part_l)      # [P_load, Lmax, …] host numpy
    got = _unfold_host(loaded, p_load)
    for f in dataclasses.fields(State):
        a = np.asarray(getattr(ref, f.name))
        b = got[f.name]
        assert b.shape == a.shape, f"{f.name}: shape {b.shape} != {a.shape}"
        assert np.array_equal(a, b), (
            f"{f.name}: P{p_save}->P{p_load} not identity "
            f"(max|Δ|={np.max(np.abs(a - b)) if a.size else 0:.3e})")
    return meta


# --------------------------------------------------------------------------
# 1. Full-State round-trip at the SAME P (write → reconstruct_global == global)
# --------------------------------------------------------------------------
@avail
def test_full_state_roundtrip_same_P(tmp_path):
    mesh = load_mesh(PI_MESH)
    g0 = _signature_state(mesh)
    p = min(2, NDEV)
    saved, sm, part = _to_folded_sharded(g0, mesh, p)
    d = tmp_path / "rt_same"
    zo.write_restart(d, saved, sm, part, step=7, calendar_date="1958-01-02", dt_stage=1800.0)
    for f in dataclasses.fields(State):
        g = zo.reconstruct_global(d, f.name)
        orig = np.asarray(getattr(g0, f.name))
        assert g.shape == orig.shape, f"{f.name}: {g.shape} != {orig.shape}"
        assert np.array_equal(g, orig), f"{f.name}: reconstruct_global != original"


# --------------------------------------------------------------------------
# 2. No leaf silently dropped — every State leaf has its own dataset on disk
# --------------------------------------------------------------------------
@avail
def test_no_leaf_dropped(tmp_path):
    import zarr
    mesh = load_mesh(PI_MESH)
    g0 = _signature_state(mesh)
    saved, sm, part = _to_folded_sharded(g0, mesh, min(2, NDEV))
    d = tmp_path / "rt_drop"
    zo.write_restart(d, saved, sm, part, step=0, calendar_date="1958-01-01", dt_stage=1800.0)
    on_disk = set(zarr.open_group(str(d), mode="r").array_keys())
    for f in dataclasses.fields(State):
        assert f.name in on_disk, f"State leaf {f.name!r} missing from restart store"
    # the field list is derived from the dataclass (pytree-flatten-equivalent), not a glob
    assert set(zo._all_state_fields()) == {f.name for f in dataclasses.fields(State)}


# --------------------------------------------------------------------------
# 3. Device-count portability — down-size (P=4 → P=2) and up-size (P=2 → P=4)
# --------------------------------------------------------------------------
@avail
@need4
def test_restart_portable_downsize(tmp_path):
    mesh = load_mesh(PI_MESH)
    meta = _roundtrip(tmp_path, mesh, _signature_state(mesh), p_save=4, p_load=2)
    assert int(meta["step"]) == 1234
    assert str(meta["calendar_date"]) == "1958-03-15"
    assert float(meta["dt_stage"]) == 1800.0


@avail
@need4
def test_restart_portable_upsize(tmp_path):
    mesh = load_mesh(PI_MESH)
    _roundtrip(tmp_path, mesh, _signature_state(mesh), p_save=2, p_load=4)


# --------------------------------------------------------------------------
# 4. Aggregate gate — prints RESTART_PORTABLE_OK for the acceptance log
# --------------------------------------------------------------------------
@avail
@need4
def test_restart_portable_ok(tmp_path):
    mesh = load_mesh(PI_MESH)
    g0 = _signature_state(mesh)
    _roundtrip(tmp_path, mesh, g0, 4, 2)
    _roundtrip(tmp_path, mesh, g0, 2, 4)
    _roundtrip(tmp_path, mesh, g0, 4, 1)                   # also P=4 → single-device
    print("RESTART_PORTABLE_OK")


# --------------------------------------------------------------------------
# 5. Partition-INDEPENDENCE of the canonical (default) restart + the folded path
# --------------------------------------------------------------------------
@avail
def test_restart_folded_layout_roundtrip(tmp_path):
    """The 'folded' layout (the multi-node / output-OOM path) still round-trips the full State and
    carries the gid/owned maps; `reconstruct_global` auto-detects it."""
    import zarr
    mesh = load_mesh(PI_MESH)
    g0 = _signature_state(mesh)
    saved, sm, part = _to_folded_sharded(g0, mesh, min(2, NDEV))
    d = tmp_path / "rt_folded"
    zo.write_restart(d, saved, sm, part, step=7, calendar_date="1958-01-02", dt_stage=1800.0,
                     layout="folded")
    root = zarr.open_group(str(d), mode="r")
    assert "gid_nod" in root and root.attrs.get("layout") != "canonical_global"
    for f in dataclasses.fields(State):
        g = zo.reconstruct_global(d, f.name)
        assert np.array_equal(g, np.asarray(getattr(g0, f.name))), f"{f.name}: folded roundtrip"


@avail
@need4
def test_restart_canonical_partition_independent(tmp_path):
    """The DEFAULT (canonical 'global') restart is partition-INDEPENDENT: the same global State saved
    from P=4 vs P=2 yields BYTE-identical on-disk arrays (the FESOM3 ``dist_2 ≡ dist_8`` property), it
    carries NO folded lane axis / gid maps, and it still reloads bit-exactly onto a different P."""
    import zarr
    mesh = load_mesh(PI_MESH)
    g0 = _signature_state(mesh)
    s4, sm4, p4 = _to_folded_sharded(g0, mesh, 4)
    s2, sm2, p2 = _to_folded_sharded(g0, mesh, 2)
    d4, d2 = tmp_path / "ci_p4", tmp_path / "ci_p2"
    zo.write_restart(d4, s4, sm4, p4, step=1, calendar_date="1958-01-01", dt_stage=1800.0)  # default global
    zo.write_restart(d2, s2, sm2, p2, step=1, calendar_date="1958-01-01", dt_stage=1800.0)
    r4 = zarr.open_group(str(d4), mode="r"); r2 = zarr.open_group(str(d2), mode="r")
    assert r4.attrs["layout"] == "canonical_global" and "gid_nod" not in r4
    for f in dataclasses.fields(State):
        assert np.array_equal(np.asarray(r4[f.name]), np.asarray(r2[f.name])), \
            f"{f.name}: canonical restart differs across source partition (not partition-independent)"
    # the canonical restart reloads bit-exactly onto a DIFFERENT device count
    loaded, _ = zo.read_restart(d4, mesh, p2)
    ref = shard_mesh.partition_state(g0, p2)
    got = _unfold_host(loaded, 2)
    for f in dataclasses.fields(State):
        assert np.array_equal(np.asarray(getattr(ref, f.name)), got[f.name]), f"{f.name}: P4->P2 reload"
    print("RESTART_PARTITION_INDEPENDENT_OK")
