"""Phase 1 of docs/plans/20261105-recom-parameter-estimation.md: how far does the REcoM gradient reach?

Physics reads no BGC tracer, so the day-scale limit of the physics adjoint (docs/ADJOINT_HORIZON.md) need not apply to
REcoM parameters. This measures it. For each window of D days from a pi restart (``--restart``, ``--start-day``; by
default the 30-day spectral restart, a young ecosystem), with a cost taken on the LAST day only (so every window tests sensitivity across its whole length):

- ``jax.grad`` with respect to the parameters against central differences at two step sizes;
- the noise of J: the same forward repeated. The GPU forward is not bit-reproducible, and the noise bar of a central
  difference is noise / (2 h p), printed beside it. ``--xla_gpu_deterministic_ops`` is impractical (ADJOINT_HORIZON.md);
  a CPU run gives clean differences at a higher cost;
- the reverse field at the start of the window, raw ``dJ/dx0`` and normalised by the reference ``x0 * dJ/dx0`` (the
  plan's normalisation, x = xbar * xt), per tracer with the location of the raw peak. Raw growth that collapses when
  scaled is a magnitude effect at a tracer near zero; growth that survives is a real amplification;
- forward and gradient wall time.

Costs: ``chl`` area-mean surface Chl of the four groups, ``din`` area-mean surface DIN, ``npp`` area-integrated NPP.
The time loop is a two-level checkpointed scan (about 2 sqrt(N) stored states). Windows must end inside 1958 (<= 330
days), the year of the forcing stack. Appends one JSON line per (window, cost) to OUT/horizon.jsonl.

    sbatch scripts/albedo/run_bgc_horizon.sbatch --days 1 5 30 --cost chl
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fesom_jax import step as stepmod  # noqa: E402
from fesom_jax import surface_forcing  # noqa: E402
from fesom_jax.params import Params  # noqa: E402
from fesom_jax.recom import wiring  # noqa: E402
from fesom_jax.tests import test_recom_step_fork as TF  # noqa: E402

NAMES = ("expon_phy", "graz_max", "rho_N", "deg_Chl", "cdomcoeff", "k_Fe_d")   # fast and slow, light, loss, iron
RUN = "pi_1d_spectral_jaxphys"
STEPS_PER_DAY = 32
CHL = ("PhyChl", "DiaChl", "CoccoChl", "PhaeoChl")


def split(n: int) -> tuple[int, int]:
    """``(segments, length)`` with segments * length = n and length the divisor of n nearest sqrt(n)."""
    divs = [d for d in range(1, n + 1) if n % d == 0]
    length = min(divs, key=lambda d: abs(d - n ** 0.5))
    return n // length, length


def make_cost(fx, kind: str, n: int):
    mesh, dt, lay = fx["mesh"], fx["cfg"].dt, fx["lay"]
    zero = jnp.zeros((mesh.elem2D, 2))
    _, p0 = fx["cfg"].recom.load()
    a2 = mesh.areasvol[:, 0]
    k_npp = wiring.DIAG2D.index("npp")
    chl = [lay.names.index(c) for c in CHL]
    din = lay.names.index("DIN")
    n_seg, n_len = split(n)

    def metric(st):
        if kind == "npp":
            return jnp.sum(st.bgc_diag[:, k_npp] * a2)
        x = jnp.sum(st.bgc[:, 0, jnp.array(chl)], axis=-1) if kind == "chl" else st.bgc[:, 0, din]
        return jnp.sum(x * a2) / jnp.sum(a2)

    def cost(theta, bgc0, st0, stack):
        params = dataclasses.replace(Params.defaults(), recom=p0._replace(**dict(zip(NAMES, theta))))

        def one(c, xs):
            st, acc = c
            sf, w = xs
            st = stepmod.step(st, mesh, fx["op"], zero, params, dt=dt, is_first_step=False, step_forcing=sf,
                              forcing_static=fx["fs"], **fx["phys"])
            return (st, acc + w * metric(st)), None

        def segment(c, xs):
            return jax.lax.scan(jax.checkpoint(one), c, xs)[0], None

        w = jnp.where(jnp.arange(n) >= n - STEPS_PER_DAY, 1.0 / STEPS_PER_DAY, 0.0)
        xs = jax.tree.map(lambda a: a.reshape((n_seg, n_len) + a.shape[1:]), (stack, w))
        (_, j), _ = jax.lax.scan(jax.checkpoint(segment), (dataclasses.replace(st0, bgc=bgc0), jnp.zeros(())), xs)
        return j

    return cost, jnp.array([getattr(p0, k) for k in NAMES], jnp.float64)


def tangent_check(fx, st0, cf, day):
    """Is the physics out of the linearisation? Linearise one step with respect to (parameters, BGC state): the
    tangent of every State field, exact zero meaning no dependence, and one forward against one tangent evaluation."""
    mesh, dt = fx["mesh"], fx["cfg"].dt
    zero = jnp.zeros((mesh.elem2D, 2))
    _, p0 = fx["cfg"].recom.load()
    sf = jax.tree.map(lambda x: x[0], cf.stack(surface_forcing.dates_for_steps(1958, dt, 1, start_day=day)))
    theta = jnp.array([getattr(p0, k) for k in NAMES], jnp.float64)

    def f(th, bgc):
        params = dataclasses.replace(Params.defaults(), recom=p0._replace(**dict(zip(NAMES, th))))
        return stepmod.step(dataclasses.replace(st0, bgc=bgc), mesh, fx["op"], zero, params, dt=dt, is_first_step=False,
                            step_forcing=sf, forcing_static=fx["fs"], **fx["phys"])

    fj = jax.jit(f)
    jax.block_until_ready(fj(theta, st0.bgc))
    _, t_f = timed(fj, theta, st0.bgc)
    _, lin = jax.linearize(f, theta, st0.bgc)
    linj = jax.jit(lin)
    th_t, x_t = 0.01 * theta, 0.01 * st0.bgc
    out = jax.block_until_ready(linj(th_t, x_t))
    _, t_l = timed(linj, th_t, x_t)
    n_eqn = len(jax.make_jaxpr(lin)(th_t, x_t).eqns)
    live, dead = [], []
    for path, v in jax.tree_util.tree_flatten_with_path(out)[0]:
        name = jax.tree_util.keystr(path)
        if v.dtype == jax.dtypes.float0:                 # integer fields have no tangent
            continue
        (live if float(jnp.max(jnp.abs(jnp.asarray(v, jnp.float64)))) > 0 else dead).append(name)
    print(f"tangent check: forward {t_f:.3f} s, tangent {t_l:.3f} s, {n_eqn} top-level eqns\n  nonzero: {live}\n"
          f"  exactly zero: {dead}", flush=True)


def tracer_report(names, bgc0, gx, top=6):
    """Per tracer: raw and reference-normalised reverse-field norms, the raw peak and the tracer's value there."""
    x, g = np.asarray(bgc0), np.asarray(gx)
    rows = []
    for i, nm in enumerate(names):
        gi, xi = g[..., i], x[..., i]
        k = np.unravel_index(int(np.argmax(np.abs(gi))), gi.shape)
        rows.append(dict(tracer=nm, raw=float(np.linalg.norm(gi)), scaled=float(np.linalg.norm(xi * gi)),
                         raw_max=float(np.abs(gi[k])), node=int(k[0]), level=int(k[1]), x_at_max=float(xi[k])))
    for r in sorted(rows, key=lambda r: -r["raw"])[:top]:
        print(f"    raw {r['raw']:.3e} scaled {r['scaled']:.3e}  {r['tracer']:>9}: peak {r['raw_max']:.3e} at node "
              f"{r['node']} level {r['level']}, x there {r['x_at_max']:.3e}", flush=True)
    return rows


def timed(fn, *a):
    t = time.time()
    r = jax.block_until_ready(fn(*a))
    return r, time.time() - t


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--days", type=int, nargs="+", default=[1, 5, 30, 90])
    ap.add_argument("--cost", nargs="+", choices=("chl", "din", "npp"), default=["chl"])
    ap.add_argument("--h", type=float, nargs="+", default=[1e-2, 3e-3], help="relative FD steps")
    ap.add_argument("--repeats", type=int, default=4, help="repeats of the forward, for the noise of J")
    ap.add_argument("--restart", default=TF.RESTART, help="restart to start from")
    ap.add_argument("--start-day", type=int, default=None, help="day of 1958 of the start (default: from the restart)")
    ap.add_argument("--no-fd", action="store_true")
    ap.add_argument("--tangent-check", action="store_true", help="one-step physics-tangent check first")
    ap.add_argument("--out", type=Path,
                    default=Path("/albedo/work/projects/p_specbic/sloza/fesom_jax_runs/bgc_gradient_horizon"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    fx = TF._setup(RUN, TF.RUN_CFG)
    st0, cf, day = TF.warm_start(fx, a.restart)
    if day <= 1:
        raise SystemExit(f"no warm restart at {a.restart}: the windows need a grown ecosystem")
    day = a.start_day or day
    print(f"start day {day} of 1958 from {a.restart}, backend {jax.default_backend()}, bgc {st0.bgc.shape}", flush=True)
    b0 = np.asarray(st0.bgc)
    print("surface means: " + "  ".join(f"{n} {b0[:, 0, i].mean():.3g}" for i, n in enumerate(fx["lay"].names)), flush=True)
    if a.tangent_check:
        tangent_check(fx, st0, cf, day)
    for d in a.days:
        if day + d > 361:
            raise SystemExit(f"window of {d} days from day {day} leaves 1958")
        n = d * STEPS_PER_DAY
        stack = cf.stack(surface_forcing.dates_for_steps(1958, fx["cfg"].dt, n, start_day=day))
        for kind in a.cost:
            cost, theta = make_cost(fx, kind, n)
            fwd = jax.jit(cost)
            grad = jax.jit(jax.grad(cost, argnums=(0, 1)))
            t = time.time()
            fwd_c = fwd.lower(theta, st0.bgc, st0, stack).compile()
            grad_c = grad.lower(theta, st0.bgc, st0, stack).compile()
            t_compile = time.time() - t
            j0, t_fwd = timed(fwd_c, theta, st0.bgc, st0, stack)
            (g, gx), t_grad = timed(grad_c, theta, st0.bgc, st0, stack)
            g = np.asarray(g)
            reps = [float(fwd_c(theta, st0.bgc, st0, stack)) for _ in range(a.repeats)]
            noise = float(np.std(reps + [float(j0)], ddof=1)) if a.repeats else 0.0
            rec = dict(days=d, steps=n, cost=kind, backend=jax.default_backend(), J=float(j0), names=NAMES,
                       theta=[float(x) for x in theta], grad=g.tolist(), finite=bool(np.all(np.isfinite(g))),
                       x0_norm=float(jnp.sqrt(jnp.sum(gx * gx))), x0_max=float(jnp.max(jnp.abs(gx))),
                       x0_scaled_norm=float(jnp.sqrt(jnp.sum((st0.bgc * gx) ** 2))), noise=noise, reps=reps,
                       t_compile=t_compile, t_fwd=t_fwd, t_grad=t_grad, split=split(n), fd={})
            if not a.no_fd:
                for h in a.h:
                    fd = []
                    for i in range(len(NAMES)):
                        e = jnp.zeros(len(NAMES)).at[i].set(h * theta[i])
                        fd.append((float(fwd_c(theta + e, st0.bgc, st0, stack))
                                   - float(fwd_c(theta - e, st0.bgc, st0, stack))) / (2 * float(e[i])))
                    rec["fd"][str(h)] = fd
            print(f"{d:4d} d {kind}: J {rec['J']:.6e}  noise {noise:.2e}  |dJ/dx0| {rec['x0_norm']:.3e}  "
                  f"|x0 dJ/dx0| {rec['x0_scaled_norm']:.3e}  fwd {t_fwd:.1f}s  grad {t_grad:.1f}s  "
                  f"compile {t_compile:.0f}s", flush=True)
            rec["tracers"] = tracer_report(fx["lay"].names, st0.bgc, gx)
            with open(a.out / "horizon.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            for i, k in enumerate(NAMES):
                fds = "  ".join(f"FD({h}) {rec['fd'][h][i]: .6e} +- {noise / (2 * float(h) * float(theta[i])):.1e}"
                                for h in rec["fd"])
                print(f"    {k:>10}: dJ/dlog p {g[i] * float(theta[i]): .4e}  adjoint {g[i]: .6e}  {fds}", flush=True)


if __name__ == "__main__":
    main()
