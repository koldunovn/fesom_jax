"""Gate G1 input grid + mocsy driver run (docs/plans/20261025-recom-lane-mocsy.md, M2/M3).

Writes the G1 input grid, runs the standalone Fortran driver
(``scripts/recom/fortran/mocsy_driver.F90``, built by ``build_mocsy_driver.sh``) on it, and
stores inputs and outputs as one ``g1.npz`` in the output directory (default
``/work/ab0995/a270088/recom_ref/mocsy_g1``). Login node, a few seconds.

The grid (fixed seed): realistic ocean ranges plus edges -- cold/fresh polar water at
REcoM's clamps (T 2 degC, S 21), water outside the clamps (T < 2, S < 21: mocsy itself
accepts it), warm salty surface, deep high-pressure points, low and high Alk/DIC ratios,
REcoM's floors (``tiny*1e-3`` for DIC/Alk/Si/DIN, recom_sms_init.F90:330-339), zero and
large wind, full ice cover, zero sea-level pressure (``l_mslp`` off), negative depths (zstar
surface above z=0), and the mocsy DIC guard (``dic <= 0`` -> the 1e20 fill, vars.F90:397).

Usage: python make_mocsy_g1.py [--out DIR] [--driver PATH]
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np

NAMES_IN = ["temp", "sal", "alk", "dic", "sil", "phos", "patm", "depth", "lat", "lon",
            "wind", "fice", "xco2", "o2"]
NAMES_OUT = (
    # interior vars_sprac (recom_carbonate_chemistry.F90:240)
    ["ph", "pco2", "fco2", "co2", "hco3", "co3", "omega_a", "omega_c", "kspc", "betad",
     "rho_sw", "p", "tempis", "niter", "niter_betad2", "replica_ok"]
    # surface pistonvel + flxco2 + o2flux (recom_forcing.F90:195-278)
    + ["kw660", "co2flux", "co2ex", "dpco2", "s_ph", "s_pco2", "s_fco2", "s_co2", "s_hco3",
       "s_co3", "s_omega_a", "s_omega_c", "s_betad", "s_rho_sw", "s_p", "s_tempis", "s_k0",
       "o2ex", "s_niter", "s_niter_betad2", "s_replica_ok"]
    # the interior call's equilibrium constants (constants.F90)
    + ["K0", "K1", "K2", "Kb", "Kw", "Ks", "Kf", "Kspc", "Kspa", "K1p", "K2p", "K3p", "Ksi",
       "St", "Ft", "Bt"]
)
NOUT = 70            # the driver's padded column count
TINY3 = 2.23e-16 * 1e-3   # REcoM tiny (recom_modules.F90:490) * 1e-3, the mol/m3 floor

DEFAULT_OUT = Path("/work/ab0995/a270088/recom_ref/mocsy_g1")
DEFAULT_DRIVER = DEFAULT_OUT / "build" / "mocsy_driver.x"


def make_grid(seed: int = 20261025) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Return ``(inputs, group)``: per-field float64 arrays and a group label per point."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, np.ndarray]] = []
    groups: list[str] = []

    def add(name, n, **ranges):
        base = dict(
            temp=(2.0, 30.0), sal=(32.0, 37.5), alk=(2.15, 2.45), dic=(1.85, 2.30),
            sil=(0.0, 0.15), phos=(0.0, 3.0e-3), patm=(0.97, 1.04), depth=(0.0, 5500.0),
            lat=(-80.0, 85.0), lon=(-180.0, 180.0), wind=(0.0, 20.0), fice=(0.0, 1.0),
            xco2=(270.0, 420.0), o2=(0.15, 0.35),
        )
        base.update(ranges)
        r = {}
        for k in NAMES_IN:
            lo_hi = base[k]
            if np.isscalar(lo_hi):
                r[k] = np.full(n, float(lo_hi))
            else:
                r[k] = rng.uniform(lo_hi[0], lo_hi[1], n)
        rows.append(r)
        groups.extend([name] * n)

    add("ocean", 2500)
    add("surface", 300, depth=(0.0, 20.0))
    add("polar_clamp", 200, temp=2.0, sal=(21.0, 34.5), lat=(60.0, 89.0), fice=(0.5, 1.0))
    add("below_clamps", 80, temp=(-1.9, 2.0), sal=(5.0, 21.0))
    add("warm_salty", 200, temp=(28.0, 40.0), sal=(37.0, 43.0), depth=(0.0, 100.0),
        lat=(-30.0, 30.0))
    add("deep", 300, temp=(0.5, 4.0), depth=(4000.0, 6500.0), dic=(2.1, 2.4),
        alk=(2.3, 2.45), sil=(0.08, 0.2), phos=(1.5e-3, 3.5e-3))
    add("low_alk_dic", 150, dic=(2.2, 2.6), alk=(1.6, 2.2))
    add("high_alk_dic", 150, dic=(1.2, 1.9), alk=(2.4, 2.8))
    add("tiny_si_po4", 100, sil=TINY3, phos=TINY3 / 16.0)
    add("tiny_dic", 40, dic=TINY3)
    add("tiny_alk", 40, alk=TINY3)
    add("tiny_both", 40, dic=TINY3, alk=TINY3, sil=TINY3, phos=TINY3 / 16.0)
    add("small_dic_alk", 60, dic=(1e-4, 0.5), alk=(1e-4, 0.5))
    add("wind_ice_edges", 60, wind=(0.0, 35.0), fice=1.0)
    add("calm", 30, wind=0.0, fice=0.0)
    add("patm_zero", 30, patm=0.0)
    add("negative_depth", 40, depth=(-4.0, 0.0))
    add("dic_guard", 4, dic=0.0)

    inputs = {k: np.concatenate([r[k] for r in rows]) for k in NAMES_IN}
    return inputs, np.array(groups)


def write_input(path: Path, inputs: dict[str, np.ndarray]) -> None:
    arr = np.stack([inputs[k] for k in NAMES_IN], axis=1).astype("<f8")  # (n, nin)
    with open(path, "wb") as f:
        np.array([arr.shape[0], arr.shape[1]], dtype="<i4").tofile(f)
        arr.tofile(f)


def read_output(path: Path) -> dict[str, np.ndarray]:
    with open(path, "rb") as f:
        n, nout = np.fromfile(f, dtype="<i4", count=2)
        arr = np.fromfile(f, dtype="<f8").reshape(n, nout)
    if nout != NOUT:
        raise ValueError(f"driver wrote {nout} columns, expected {NOUT}")
    return {k: arr[:, i].copy() for i, k in enumerate(NAMES_OUT)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--driver", type=Path, default=DEFAULT_DRIVER)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    inputs, group = make_grid()
    fin, fout = args.out / "g1_input.bin", args.out / "g1_output.bin"
    write_input(fin, inputs)
    subprocess.run([str(args.driver), str(fin), str(fout)], check=True)
    outputs = read_output(fout)
    np.savez(args.out / "g1.npz", group=group,
             **{f"in_{k}": v for k, v in inputs.items()},
             **{f"out_{k}": v for k, v in outputs.items()})
    n = len(group)
    print(f"G1: {n} points -> {args.out / 'g1.npz'}")
    for key in ("niter", "s_niter", "niter_betad2", "s_niter_betad2"):
        v = outputs[key]
        print(f"  {key:15s} min {v.min():.0f}  max {v.max():.0f}")
    print(f"  replica_ok: interior {int(outputs['replica_ok'].sum())}/{n}, "
          f"surface {int(outputs['s_replica_ok'].sum())}/{n}")


if __name__ == "__main__":
    main()
