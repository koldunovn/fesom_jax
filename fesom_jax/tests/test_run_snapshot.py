"""The step-0 snapshot of ``run_from_config(snapshot_out=...)`` (idemix-jax host patch 0006) is the
state the run integrates.

On a cold start the driver edits the partitioned state after ``partition_state``: upstream's
``FESOM_IC_PERTURB_T`` (d7e4e5d run.py) multiplies ``T`` and ``T_old`` by ``1 + eps N(0, 1)``. The
snapshot ``snap_<start>.npz`` must hold that state, as it does after a restart (where it reads the
folded state the run starts from), not the caller's ``state0``. Without the perturbation it is
``state0`` bitwise, as before.

The snapshots are canonical global NODE fields, so ``snapshot_fields`` must name node-leading State
leaves that are set in the run: an element field (``uv``) or an unset leaf (``carry_T`` in double
precision) is refused with ``ValueError`` once the start state is built, before the runner is
called, on a cold start and after a restart.

pi mesh, a zero forcing stack, no model step: the forced runner is replaced by a stub that records
the state it is handed (the first chunk's integrated initial state) and stops the run.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from fesom_jax import ic, partit, run, shard_mesh, surface_forcing
from fesom_jax.mesh import load_mesh
from fesom_jax.run_config import RunConfig

DT = 100.0
N_STEPS = 2


class _Stop(Exception):
    pass


@pytest.fixture(scope="module")
def pi():
    mesh = load_mesh()
    part = partit.synth_serial(mesh.nod2D, mesh.elem2D, mesh.edge2D)
    n = int(mesh.nod2D)
    static = surface_forcing._static_from_mesh(mesh, np.zeros(n), np.zeros(n))
    stack = surface_forcing.StepForcing(*(np.zeros((N_STEPS, n)) for _ in range(10)))
    return dict(mesh=mesh, part=part, sm=shard_mesh.build_sharded_mesh(mesh, part),
                state0=ic.initial_state(mesh),
                forcing=types.SimpleNamespace(static=static, point_slope=False), stack=stack)


def _start(pi, tmp_path, monkeypatch, eps):
    """``(snapshot npz, global T / T_old of the state the forced runner was handed)``."""
    if eps is None:
        monkeypatch.delenv("FESOM_IC_PERTURB_T", raising=False)
    else:
        monkeypatch.setenv("FESOM_IC_PERTURB_T", str(eps))
    seen = {}

    def runner(sm, state_p, *a, state_is_folded=False, **k):
        seen["state"], seen["folded"] = state_p, state_is_folded
        raise _Stop

    monkeypatch.setattr(run, "run_steps_sharded_forced", runner)
    cfg = RunConfig(dt=DT, n_steps=N_STEPS)
    with pytest.raises(_Stop):
        run.run_from_config(cfg, mesh=pi["mesh"], part=pi["part"], sm=pi["sm"],
                            forcing=pi["forcing"], forcing_stack=pi["stack"],
                            state0=pi["state0"], snapshot_out=str(tmp_path), snapshot_every=1)
    assert seen["folded"] is False                         # a cold start: host [P, Lmax]
    gl = {}
    for name in ("T", "T_old"):
        sp = np.asarray(getattr(seen["state"], name))      # [P, Lmax, nl]
        g = np.zeros((int(pi["mesh"].nod2D),) + sp.shape[2:], sp.dtype)
        for d in range(pi["part"].npes):
            own = np.asarray(pi["part"].myList_nod2D[d])[: int(pi["part"].myDim_nod2D[d])]
            g[own] = sp[d, : own.size]
        gl[name] = g
    return np.load(tmp_path / "snap_0000000.npz"), gl


def test_cold_start_snapshot_is_the_perturbed_integrated_state(pi, tmp_path, monkeypatch):
    snap, integrated = _start(pi, tmp_path, monkeypatch, eps=1e-3)
    T0 = np.asarray(pi["state0"].T)
    assert not np.array_equal(integrated["T"], T0)          # the perturbation is on
    np.testing.assert_array_equal(snap["T"], integrated["T"])
    assert int(snap["step"]) == 0


def test_cold_start_snapshot_without_perturbation_is_state0_bitwise(pi, tmp_path, monkeypatch):
    snap, integrated = _start(pi, tmp_path, monkeypatch, eps=None)
    for name in run.SNAPSHOT_FIELDS:
        np.testing.assert_array_equal(snap[name], np.asarray(getattr(pi["state0"], name)),
                                      err_msg=name)
    np.testing.assert_array_equal(snap["T"], integrated["T"])


def _folded_restart(pi, monkeypatch):
    """A restart start without a store: ``zarr_output.read_restart`` returns ``state0`` folded
    ``[P*Lmax, ...]`` (what it returns for a restart of ``state0``) at step 0."""
    from fesom_jax.integrate_sharded import _fold
    from fesom_jax.state import State
    sp = shard_mesh.partition_state(pi["state0"], pi["part"])
    folded = State(**{n: None if getattr(sp, n) is None else _fold(np.asarray(getattr(sp, n)))
                      for n in run._STATE_FIELDS})
    monkeypatch.setattr(run.zarr_output, "read_restart",
                        lambda *a, **k: (folded, {"step": 0}))
    return RunConfig(dt=DT, n_steps=N_STEPS, restart_in="restart_stub")


@pytest.mark.parametrize("start", ["cold", "restart"])
@pytest.mark.parametrize("fields,bad", [(("T", "uv"), r"uv \("),
                                        (("T", "carry_T"), r"carry_T \(None in this run\)"),
                                        (("uv", "helem", "eta_n"), r"uv \(.*helem \(")])
def test_non_node_snapshot_fields_are_refused_before_the_runner(pi, tmp_path, monkeypatch, start,
                                                                fields, bad):
    monkeypatch.delenv("FESOM_IC_PERTURB_T", raising=False)
    called = []

    def runner(*a, **k):
        called.append(True)
        raise _Stop

    monkeypatch.setattr(run, "run_steps_sharded_forced", runner)
    if start == "cold":
        cfg, kw = RunConfig(dt=DT, n_steps=N_STEPS), dict(state0=pi["state0"])
    else:
        cfg, kw = _folded_restart(pi, monkeypatch), {}
    with pytest.raises(ValueError, match="snapshot_fields: .*refused: .*" + bad) as e:
        run.run_from_config(cfg, mesh=pi["mesh"], part=pi["part"], sm=pi["sm"],
                            forcing=pi["forcing"], forcing_stack=pi["stack"],
                            snapshot_out=str(tmp_path), snapshot_every=1,
                            snapshot_fields=fields, **kw)
    refused = str(e.value).split("refused: ")[1]
    named = {item.split(" (")[0] for item in refused.split(", ")}
    assert named == set(fields) - {"T", "eta_n"}                         # node fields pass
    assert not called and not list(tmp_path.glob("snap_*.npz"))


def test_restart_start_default_fields_pass_and_snapshot_the_restart_state(pi, tmp_path,
                                                                        monkeypatch):
    monkeypatch.delenv("FESOM_IC_PERTURB_T", raising=False)
    seen = {}

    def runner(sm, state_p, *a, state_is_folded=False, **k):
        seen["folded"] = state_is_folded
        raise _Stop

    monkeypatch.setattr(run, "run_steps_sharded_forced", runner)
    with pytest.raises(_Stop):
        run.run_from_config(_folded_restart(pi, monkeypatch), mesh=pi["mesh"], part=pi["part"],
                            sm=pi["sm"], forcing=pi["forcing"], forcing_stack=pi["stack"],
                            snapshot_out=str(tmp_path), snapshot_every=1)
    assert seen["folded"] is True
    snap = np.load(tmp_path / "snap_0000000.npz")
    for name in run.SNAPSHOT_FIELDS:
        np.testing.assert_array_equal(snap[name], np.asarray(getattr(pi["state0"], name)),
                                      err_msg=name)


def test_a_restart_leaf_is_placed_by_its_field_not_by_coinciding_lane_counts(pi, monkeypatch):
    """On a restart start the leaves are folded lanes; where ``Lmax_nod == Lmax_elem`` the lanes
    cannot tell nodes from elements, so the field is placed by :meth:`State.zeros` on the mesh:
    the default node fields pass and an element field is still refused."""
    from types import SimpleNamespace
    P, L = 1, 7
    monkeypatch.setattr(run.shard_mesh, "local_sizes",
                        lambda part: (P, {"nod": L, "elem": L, "edge": L}))
    names = tuple(run.SNAPSHOT_FIELDS) + ("uv",)
    state_p = SimpleNamespace(**{n: np.zeros((P * L, 2)) for n in names})
    run._check_snapshot_fields(run.SNAPSHOT_FIELDS, state_p, pi["part"], folded=True,
                               mesh=pi["mesh"])
    with pytest.raises(ValueError, match=r"refused: uv \(leading dim") as e:
        run._check_snapshot_fields(("T", "uv"), state_p, pi["part"], folded=True,
                                   mesh=pi["mesh"])
    assert "T (" not in str(e.value)
    # without the mesh the lane rule alone has to refuse every field here
    with pytest.raises(ValueError, match=r"refused: T \(not node-leading"):
        run._check_snapshot_fields(("T",), state_p, pi["part"], folded=True)
