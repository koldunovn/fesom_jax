"""The IDEMIX refusals where upstream's newer features meet IDEMIX (idemix-jax host patch 0004, port
onto d7e4e5d): each unsupported combination raises ``ValueError`` before any physics runs.

* ice-shelf cavities, by config: ``step`` (``cavity_cfg``), ``RunConfig.validate`` (``cavity``) and
  ``run_steps_sharded_forced`` (``cavity_cfg``);
* ice-shelf cavities, by mesh (``ulevels_nod2D > 1`` with ``cavity: null``, which upstream accepts):
  ``RunConfig.idemix_inputs``, ``build_lateral_operator``, the run entry and the column step
  (``mixing_idemix``, and ``budget_terms`` and ``lateral_stability``, patches 0006 and 0011) on a
  concrete mesh;
* the coupled ocean (``init_ocean``), the sharded chunk program (IDEMIX ``step_kw``), and several
  JAX processes at the run entry;
* ``Params.idemix`` (a pytree of five constants) and ``Params.idemix_field`` (per-node arrays,
  patch 0010) as one log-normal scalar control.

The checks only test whether an IDEMIX argument is set, so the refusals that fire before
``idemix_jax`` is imported run without it, on a stand-in config object; the others skip without
``idemix_jax`` (``pytest.importorskip`` inside the test).
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax.cavity import CavityConfig
from fesom_jax.ice import IceConfig
from fesom_jax.run_config import RunConfig

DT = 100.0


class _IdemixCfg:
    """Stand-in for ``fesom_jax.idemix.IdemixConfig`` (which needs ``idemix_jax``): the guards
    below look only at whether an IDEMIX config is set."""
    lateral = False

    def validate(self):
        pass


@pytest.fixture(scope="module")
def pi():
    from fesom_jax import forcing, ic, ssh
    from fesom_jax.mesh import load_mesh
    mesh = load_mesh()
    return dict(mesh=mesh, st=ic.initial_state(mesh), op=ssh.build_ssh_operator(mesh, dt=DT),
                stress=forcing.surface_stress(mesh))


def _cavity_mesh(mesh, nodes=(5, 17, 40)):
    """``mesh`` with ice-shelf cavities: the top wet level of ``nodes`` moved down by one
    (``ulevels_nod2D = 2``), every column keeping >= 1 wet layer."""
    u = np.asarray(mesh.ulevels_nod2D).copy()
    nl = np.asarray(mesh.nlevels_nod2D)
    nodes = [n for n in nodes if nl[n] - 2 >= 1]
    assert len(nodes) == 3
    u[nodes] = 2
    return dataclasses.replace(mesh, ulevels_nod2D=jnp.asarray(u, dtype=jnp.asarray(mesh.ulevels_nod2D).dtype))


# --- cavities by config ------------------------------------------------------------------------
def test_step_refuses_idemix_with_a_cavity_config(pi):
    from fesom_jax.params import Params
    from fesom_jax.step import step
    with pytest.raises(ValueError, match="idemix_cfg with ice-shelf cavities"):
        step(pi["st"], pi["mesh"], pi["op"], pi["stress"], Params.defaults(), dt=DT,
             idemix_cfg=_IdemixCfg(), idemix_forcing=object(), cavity_cfg=CavityConfig())


def test_run_config_refuses_idemix_with_a_cavity_config():
    with pytest.raises(ValueError, match="idemix and cavity are both set"):
        RunConfig(ale=None, ice=IceConfig(), cavity=CavityConfig(), idemix={}).validate()
    RunConfig(ale=None, ice=IceConfig(), cavity=CavityConfig()).validate()   # cavity alone passes


def test_run_config_accepts_idemix_alone():
    pytest.importorskip("idemix_jax")
    RunConfig(idemix={}).validate()


def test_forced_runner_refuses_idemix_with_a_cavity_config():
    from fesom_jax import integrate_sharded as R
    with pytest.raises(ValueError, match="ice-shelf cavities"):
        R.run_steps_sharded_forced(None, None, None, None, None, None, None, dt=DT, npes=1,
                                   idemix_cfg=_IdemixCfg(), cavity_cfg=CavityConfig())


# --- the coupled ocean, the sharded chunk program -------------------------------------------------
def test_coupled_ocean_refuses_idemix():
    pytest.importorskip("idemix_jax")          # init_ocean validates the config first
    from fesom_jax.coupled import ocean
    with pytest.raises(ValueError, match="idemix is not wired into the coupled ocean"):
        ocean.init_ocean(RunConfig(ice=IceConfig(), idemix={}))


@pytest.mark.parametrize("key", ["idemix_cfg", "idemix_forcing", "idemix_lateral"])
def test_sharded_chunk_program_refuses_idemix_step_keywords(key):
    from fesom_jax import integrate_sharded as R
    with pytest.raises(ValueError, match="IDEMIX in the sharded chunk program"):
        R.sharded_chunk_program(None, None, None, None, npes=1, dt=DT, to_params=None,
                                step_kw={key: _IdemixCfg() if key == "idemix_cfg" else object()})
    # without IDEMIX keywords the program reaches upstream's own checks (here: our None inputs)
    with pytest.raises(Exception) as e:
        R.sharded_chunk_program(None, None, None, None, npes=1, dt=DT, to_params=None, step_kw={})
    assert "IDEMIX" not in str(e.value)


# --- cavities by mesh ----------------------------------------------------------------------------
def test_cavity_mesh_check(pi):
    pytest.importorskip("idemix_jax")
    from fesom_jax import idemix as I
    I.check_no_cavity_mesh(pi["mesh"])                       # pi: ulevels_nod2D == 1 everywhere
    with pytest.raises(ValueError, match=r"3 node\(s\) have ulevels_nod2D > 1"):
        I.check_no_cavity_mesh(_cavity_mesh(pi["mesh"]))


def test_cavity_mesh_is_refused_on_every_concrete_idemix_path(pi, monkeypatch):
    """With ``cavity: null`` (upstream accepts a cavity mesh so): the run config's inputs, the
    lateral operator, the run entry, the column step, the budget re-evaluation and the lateral
    preflight all refuse the mesh, before anything is read or built; a traced mesh (a jit
    argument) cannot be read and is left to those checks."""
    pytest.importorskip("idemix_jax")
    from idemix_jax import IdemixForcing, IdemixParams
    from fesom_jax import idemix as I
    from fesom_jax import partit, run
    bad = _cavity_mesh(pi["mesh"])
    msg = r"ice-shelf cavities: 3 node\(s\) have ulevels_nod2D > 1"

    def stop(*a, **k):
        raise AssertionError("past the cavity-mesh guard")

    monkeypatch.setattr("idemix_jax.load_idemix_forcing", stop)
    monkeypatch.setattr(I, "partition_idemix_inputs", stop)
    for lateral in (False, True):
        cfg = RunConfig.from_dict({"idemix": {"lateral": lateral}})
        with pytest.raises(ValueError, match=msg):
            cfg.idemix_inputs(bad)
    with pytest.raises(ValueError, match=msg):
        I.build_lateral_operator(bad)
    n, nl = int(bad.nod2D), int(bad.nl)
    f = IdemixForcing(fsrf=jnp.full(n, 1e-6), fbot=jnp.full(n, 1e-6))
    part = partit.synth_serial(bad.nod2D, bad.elem2D, bad.edge2D)
    with pytest.raises(ValueError, match=msg):
        run.run_from_config(RunConfig.from_dict({"idemix": {"lateral": False}}), mesh=bad,
                            part=part, idemix_forcing=f)
    z = jnp.zeros((n, nl))
    params = jax.tree.map(jnp.asarray, IdemixParams.fesom_operational())
    with pytest.raises(ValueError, match=msg):
        I.mixing_idemix(bad, z, z, z, f, params, I.IdemixConfig(lateral=False), None, dt=DT)

    class S:
        pass
    s0, s1 = S(), S()
    s0.iwe, s0.hnode, s1.iwe, s1.bvfreq = z, z, z, z
    lat = I.build_lateral_operator(pi["mesh"])
    for lateral in (False, True):                       # the budget re-evaluation (patch 0006)
        with pytest.raises(ValueError, match=msg):
            I.budget_terms(bad, s0, s1, f, params, I.IdemixConfig(lateral=lateral),
                           lat if lateral else None, dt=DT)
    with pytest.raises(ValueError, match=msg):         # the lateral preflight (patch 0011)
        I.lateral_stability(bad, z, z, z, f, params, I.IdemixConfig(lateral=True), lat, dt=DT)
    with pytest.raises(ValueError, match=msg):         # a jit that closes over the mesh
        jax.jit(lambda e: I.mixing_idemix(bad, e, z, z, f, params, I.IdemixConfig(lateral=False),
                                          None, dt=DT).iwe_new)(z)


def test_cavity_mesh_is_refused_by_the_column_step_on_every_path(pi):
    """Patch 0014 moved the column-step guards of ``mixing_idemix``, ``budget_terms`` (0006) and
    ``lateral_stability`` (0011) into ``_column_step``, which the step, the budget re-evaluation
    and the lateral stability preflight all pass, under every metric."""
    pytest.importorskip("idemix_jax")
    from idemix_jax import IdemixForcing, IdemixParams
    from fesom_jax import idemix as I
    good, bad = pi["mesh"], _cavity_mesh(pi["mesh"])
    msg = r"ice-shelf cavities: 3 node\(s\) have ulevels_nod2D > 1"
    n, nl = int(bad.nod2D), int(bad.nl)
    f = IdemixForcing(fsrf=jnp.full(n, 1e-6), fbot=jnp.full(n, 1e-6))
    z = jnp.zeros((n, nl))
    params = jax.tree.map(jnp.asarray, IdemixParams.fesom_operational())
    lat = I.build_lateral_operator(good)

    class S:
        pass
    s0, s1 = S(), S()
    s0.iwe, s0.hnode, s1.iwe, s1.bvfreq = z, z, z, z
    for metric in ("volume", "mixed", "prism"):
        for lateral in (False, True):
            cfg = I.IdemixConfig(lateral=lateral, metric=metric)
            op = lat if lateral else None
            with pytest.raises(ValueError, match=msg):
                I.mixing_idemix(bad, z, z, z, f, params, cfg, op, dt=DT)
            with pytest.raises(ValueError, match=msg):
                I.budget_terms(bad, s0, s1, f, params, cfg, op, dt=DT)
            if lateral:
                with pytest.raises(ValueError, match=msg):
                    I.lateral_stability(bad, z, z, z, f, params, cfg, op, dt=DT)


# --- several JAX processes -----------------------------------------------------------------------
def test_run_entry_refuses_idemix_under_several_processes(pi, monkeypatch):
    pytest.importorskip("idemix_jax")          # run_from_config validates the idemix block first
    from fesom_jax import partit, run
    mesh = pi["mesh"]
    part = partit.synth_serial(mesh.nod2D, mesh.elem2D, mesh.edge2D)
    monkeypatch.setattr(jax, "process_count", lambda: 2)
    with pytest.raises(ValueError, match="idemix with 2 JAX processes is not supported"):
        run.run_from_config(RunConfig.from_dict({"idemix": {"lateral": False}}), mesh=mesh,
                            part=part)


# --- controls --------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["idemix", "idemix_field"])
def test_idemix_params_are_not_a_scalar_control(name):
    # upstream's log-normal scalar controls (fesom_jax.controls, 4c921ca) would replace the
    # IDEMIX pytree (Params.idemix, patch 0004) or the per-node field (Params.idemix_field,
    # patch 0010) by one 0-d array; the name is refused at construction instead
    from fesom_jax.controls import LogNormalControl
    with pytest.raises(ValueError, match="not a scalar Params field"):
        LogNormalControl(name, 1.0, 0.5)
