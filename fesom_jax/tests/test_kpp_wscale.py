"""Phase 6C Tasks K.1 + K.2 gates — KPP init tables + the wscale velocity scales.

Controlled-replay against the C KPP reference dump (``jax_kpp_dump_core2.sh`` →
``data/kpp_dump_core2/``):

* **K.1** — :func:`kpp.build_wscale_tables` reproduces the C ``kpp_init`` wm/ws
  ``(nni+2, nnj+2)`` lookup tables (and the 4 derived scalars in :class:`kpp.KppConfig`
  bit-match the dump) to ~1e-13.
* **K.2** — :func:`kpp.wscale` fed the C-dumped ``(zehat, ustar)`` sweep reproduces the
  C ``wm``/``ws`` to ~1e-12 (the bilinear lookup + the stable-analytic branch + the
  beyond-table ``ufrac`` extrapolation), both with the C-dumped tables (pure algebra)
  and with the K.1-built tables (the composed chain). Plus AD-finiteness (the velocity
  scales feed every downstream KPP kernel; the gradient must be finite incl. the
  ``ustar=0`` zero-wind column).
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import kpp
from fesom_jax.io_dump import load_kpp_init, load_kpp_wscale_sweep

KPP_DUMP_DIR = Path(__file__).resolve().parents[2] / "data" / "kpp_dump_core2"

pytestmark = pytest.mark.skipif(
    not (KPP_DUMP_DIR / "kpp_init_rank0.txt").is_file(),
    reason=f"KPP reference dump missing ({KPP_DUMP_DIR}); run jax_kpp_dump_core2.sh",
)


@pytest.fixture(scope="module")
def setup():
    cfg = kpp.KppConfig()
    init = load_kpp_init(KPP_DUMP_DIR)
    sweep = load_kpp_wscale_sweep(KPP_DUMP_DIR)
    wmt, wst = kpp.build_wscale_tables(cfg)
    return cfg, init, sweep, (np.asarray(wmt), np.asarray(wst))


# ---------------------------------------------------------------------------
# K.1 — derived scalars + the wm/ws lookup tables
# ---------------------------------------------------------------------------
def test_init_scalars_match_c_dump(setup):
    """Vtc/cg/deltaz/deltau in KppConfig BIT-match the C ``kpp_init`` dump (they are
    ported verbatim from ``fesom_kpp.c:130-138`` and computed with the same libm
    sqrt/pow)."""
    cfg, init, _, _ = setup
    for name, got in [("Vtc", cfg.vtc), ("cg", cfg.cg),
                      ("deltaz", cfg.deltaz), ("deltau", cfg.deltau)]:
        assert got == init[name], f"{name}: KppConfig {got!r} != C dump {init[name]!r}"


def test_wscale_tables_match_c_dump(setup):
    """build_wscale_tables reproduces the C wm/ws lookup tables (constant data)."""
    cfg, init, _, (wmt, wst) = setup
    assert wmt.shape == (cfg.nni + 2, cfg.nnj + 2)
    dwm = np.max(np.abs(wmt - init["wmt"]))
    dws = np.max(np.abs(wst - init["wst"]))
    print(f"\nwmt max|Δ|={dwm:.3e}  wst max|Δ|={dws:.3e}  "
          f"(scale {np.abs(init['wmt']).max():.3e})")
    assert dwm < 1e-13 and dws < 1e-13


# ---------------------------------------------------------------------------
# K.2 — wscale velocity scales (controlled replay over the C sweep)
# ---------------------------------------------------------------------------
def test_wscale_sweep_matches_c_dump(setup):
    """wscale over the C-dumped (zehat, ustar) grid matches the C wm/ws ~1e-12 — pure
    algebra (C tables) AND the composed chain (K.1 tables)."""
    cfg, init, sw, (wmt, wst) = setup
    zehat, us = jnp.asarray(sw["zehat"]), jnp.asarray(sw["ustar"])
    # pure-algebra: feed the C-dumped tables ⇒ isolates the bilinear/branch
    wm_a, ws_a = kpp.wscale(cfg, jnp.asarray(init["wmt"]), jnp.asarray(init["wst"]),
                            zehat, us)
    dwm_a = float(jnp.max(jnp.abs(wm_a - sw["wm"])))
    dws_a = float(jnp.max(jnp.abs(ws_a - sw["ws"])))
    # composed: K.1-built tables
    wm_b, ws_b = kpp.wscale(cfg, jnp.asarray(wmt), jnp.asarray(wst), zehat, us)
    dwm_b = float(jnp.max(jnp.abs(wm_b - sw["wm"])))
    dws_b = float(jnp.max(jnp.abs(ws_b - sw["ws"])))
    stable = np.asarray(sw["zehat"] > 0.0)
    print(f"\nwscale pure   wm={dwm_a:.3e} ws={dws_a:.3e} | composed wm={dwm_b:.3e} ws={dws_b:.3e}")
    print(f"  stable-region (zehat>0) max|Δ wm|={float(np.abs(np.asarray(wm_a)-sw['wm'])[stable].max()):.3e}")
    assert dwm_a < 1e-12 and dws_a < 1e-12
    assert dwm_b < 1e-12 and dws_b < 1e-12


def test_wscale_ad_finite(setup):
    """d(Σwm+Σws)/d(zehat) and /d(ustar) finite everywhere, incl. the ustar=0 column
    (the zero-wind lane the safe denom + stop-grad index must keep clean)."""
    cfg, init, sw, (wmt, wst) = setup
    zehat, us = jnp.asarray(sw["zehat"]), jnp.asarray(sw["ustar"])

    def loss(zh, u):
        wm, ws = kpp.wscale(cfg, jnp.asarray(wmt), jnp.asarray(wst), zh, u)
        return jnp.sum(wm) + jnp.sum(ws)

    gz, gu = jax.grad(loss, argnums=(0, 1))(zehat, us)
    assert bool(jnp.all(jnp.isfinite(gz)))
    assert bool(jnp.all(jnp.isfinite(gu)))
    # the ustar=0 column is in the sweep (j=0) — the zero-wind lane
    assert float(np.min(np.asarray(sw["ustar"]))) == 0.0
    assert bool(jnp.all(jnp.isfinite(gu[:, 0])))
