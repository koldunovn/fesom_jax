"""Set up one run of the production-gap bisect (docs/plans/20261103-production-gap-bisect.md).

The production warm-1998 run kdind15w and the physics-matched Fortran run kdind15wjp differ in the executable and in
eight physics settings. A bisect run takes the production 1998 namelists (``tmp_run_kdind15w``; ``namelist.config``
from its 1998-99 backup, 12 months) and switches the named FACTORS to the matched run's values; the executable is
picked by ``--exe``. The run directory gets frozen copies of ``fesom.x``/``libfesom.so``, the namelists, the optics
tables, the clock and a job script; output goes to ``tmp_recom-fesom_output/fesom2.6_bisect_<name>``.

    python scripts/recom/fortran/make_bisect_run.py NAME --exe EXE [FACTOR ...] [--all-matched]
    # EXE: prod, r692alb, r692rel, or the stage-2 code variants e1base, ballast, ten, wind
    cd /albedo/work/user/sloza/SPECTRAL/tmp_run_bisect/NAME && sbatch job_bisect

FACTORS (production -> matched value):
  gmredi    namelist.oce Fer_GM, Redi .true. -> .false.; namelist.tra K_hor 3000. -> 0.
  mix       namelist.oce mix_scheme 'KPP' -> 'cvmix_TKE' (+ namelist.cvmix of the matched run)
  momix     namelist.tra use_momix .true. -> .false.
  adv       namelist.tra tracer advection weights 1., 1. -> 0., 1.
  evp       namelist.ice whichEVP 0 -> 1 (mEVP)
  icediff   namelist.ice ice_diff 0.0 -> 10.0
  visc      namelist.dyn opt_visc 5 -> 7
  pcell     namelist.config use_partial_cell .true. -> .false.
``--all-matched`` switches all eight (the matched run's namelists, as a check of this script: they must equal
``tmp_run_kdind15wjp``'s up to ResultPath and comments).
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

SPECTRAL = Path("/albedo/work/user/sloza/SPECTRAL")
PROD = SPECTRAL / "tmp_run_kdind15w"
MATCHED = SPECTRAL / "tmp_run_kdind15wjp"
OUT = Path("/albedo/work/projects/p_specbic/sloza/tmp_recom-fesom_output")
RESTART = Path("/albedo/work/projects/p_specbic/sloza/restart_1997_4pft")
EXE = {   # name -> (fesom.x, libfesom.so, what it is)
    # the frozen copy of B0: recom_tra_diags_spectral itself keeps being rebuilt (e4d41f52 on 4 Nov), B0 is the build
    # proven bit-identical to kdind15w
    "prod": (SPECTRAL / "tmp_run_bisect/prod_repro/fesom.x", SPECTRAL / "tmp_run_bisect/prod_repro/libfesom.so",
             "the production build recom_tra_diags_spectral at 21e0d02a, B0's frozen copy "
             "(albedo flags, '-g -O3 -march=core-avx2')"),
    "r692alb": (SPECTRAL / "recom_spectral_r692/install_alb/bin/fesom.x",
                SPECTRAL / "recom_spectral_r692/install_alb/lib64/libfesom.so",
                "recom_spectral_r692 _recom692_dumps 63283bc9 with the production build's flags (build_alb)"),
    "r692rel": (MATCHED / "fesom.x", MATCHED / "libfesom.so",
                "the matched run's executable (recom_spectral_r692 build_rel, 63283bc9; '-O3 -xHost')"),
    # stage 2, code split: worktrees recom_bisect_<v> (local branches bisect_<v>) on e1e62886, production flags
    **{v: (SPECTRAL / f"recom_bisect_{v}/install_alb/bin/fesom.x",
           SPECTRAL / f"recom_bisect_{v}/install_alb/lib64/libfesom.so", d) for v, d in {
        "e1base": "e1e62886 alone (the code of kdind15w), production flags: the control of the stage-2 builds",
        "ballast": "e1e62886 + 1cc2d0d3 (the ballasting fix), production flags",
        "ten": "e1e62886 + the ten other REcoM commits of r692 (all but 1cc2d0d3), production flags",
        "wind": "e1e62886 + 63283bc9 (first forcing window's winds rotated), production flags",
    }.items()},
}
FACTORS = {   # factor -> [(file, key, matched value)]
    "gmredi": [("namelist.oce", "Fer_GM", ".false."), ("namelist.oce", "Redi", ".false."),
               ("namelist.tra", "K_hor", "0.")],
    "mix": [("namelist.oce", "mix_scheme", "'cvmix_TKE'")],
    "momix": [("namelist.tra", "use_momix", ".false.")],
    "adv": [],                                                    # special: the tracer table
    "evp": [("namelist.ice", "whichEVP", "1")],
    "icediff": [("namelist.ice", "ice_diff", "10.0")],
    "visc": [("namelist.dyn", "opt_visc", "7")],
    "pcell": [("namelist.config", "use_partial_cell", ".false.")],
}
JOB = """#!/bin/bash
#SBATCH --account=clidyn.clidyn
#SBATCH --job-name=bis_{name}
#SBATCH --partition=mpp
#SBATCH -N 9
#SBATCH --cpus-per-task=1
#SBATCH --time={time}
#SBATCH --qos=12h
#SBATCH --tasks-per-node 96
#SBATCH -o slurm-%j.out
#SBATCH --hint=nomultithread

# production-gap bisect run {name}: {desc}
module purge
source /albedo/home/cvoelker/fesom2/env/albedo/shell
export OMP_NUM_THREADS=1
ulimit -s unlimited
unset FESOM_RECOM_DUMP_DIR FESOM_RECOM_DUMP_STEPS   # the r692 builds carry the fesom-jax dump hooks; keep them off

RESULTPATH="{result}"
JOBPATH="{jobpath}"
test -e $RESULTPATH || mkdir -p $RESULTPATH
test -e $RESULTPATH/fesom.clock || cp fesom.clock.init $RESULTPATH/fesom.clock
for c in oce ice bio; do
    test -e $RESULTPATH/fesom.1997.$c.restart || ln -s {restart}/fesom.1997.$c.restart $RESULTPATH/
done
cp fesom.x namelist.* *.dat ${{RESULTPATH}}/
cd ${{RESULTPATH}}
export LD_LIBRARY_PATH=${{JOBPATH}}:$LD_LIBRARY_PATH
date
srun --mpi=pmi2 ./fesom.x > fesom2.0.out
date
"""


def set_key(text: str, key: str, value: str, fname: str) -> str:
    pat = re.compile(rf"^(\s*){key}(\s*)=(\s*)[^!\n]*?(\s*)(!.*)?$", re.M)
    n = len(pat.findall(text))
    if n != 1:
        raise SystemExit(f"{fname}: {key} found {n} times")
    return pat.sub(lambda m: f"{m.group(1)}{key}{m.group(2)}={m.group(3)}{value}"
                             + (f" {m.group(5)}" if m.group(5) else ""), text)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("name")
    ap.add_argument("factors", nargs="*", choices=list(FACTORS), metavar="FACTOR")
    ap.add_argument("--exe", required=True, choices=list(EXE))
    ap.add_argument("--all-matched", action="store_true")
    ap.add_argument("--months", type=int, default=12)
    ap.add_argument("--time", default="04:00:00")
    ap.add_argument("--root", type=Path, default=SPECTRAL / "tmp_run_bisect")
    a = ap.parse_args()
    factors = list(FACTORS) if a.all_matched else list(dict.fromkeys(a.factors))
    run = a.root / a.name
    result = OUT / f"fesom2.6_bisect_{a.name}"
    if run.exists():
        raise SystemExit(f"{run} exists")
    run.mkdir(parents=True)
    texts = {f.name: f.read_text() for f in PROD.glob("namelist.*") if not f.name.startswith("namelist_")}
    texts["namelist.config"] = (PROD / "namelist_config_backup_20261029_yr1998-99.txt").read_text()
    t = texts["namelist.config"]
    t = set_key(t, "run_length", str(a.months), "namelist.config")
    t = set_key(t, "ResultPath", f"'{result}/'", "namelist.config")
    texts["namelist.config"] = t
    for fac in factors:
        for fname, key, val in FACTORS[fac]:
            texts[fname] = set_key(texts[fname], key, val, fname)
        if fac == "adv":
            t, n = re.subn(r"^(\s*\d+\s*,\s*'MFCT',\s*'QR4C',\s*'FCT ',\s*)1\.(,\s*1\.,)", r"\g<1>0.\2",
                           texts["namelist.tra"], flags=re.M)
            assert n == 39, n
            texts["namelist.tra"] = t
        if fac == "mix":
            texts["namelist.cvmix"] = (MATCHED / "namelist.cvmix").read_text()
    for fname, t in texts.items():
        (run / fname).write_text(t)
    for f in PROD.glob("*.dat"):
        shutil.copy(f, run / f.name)
    shutil.copy(PROD / "fesom.clock.init", run / "fesom.clock.init")
    x, so, what = EXE[a.exe]
    shutil.copy(x, run / "fesom.x")
    shutil.copy(so, run / "libfesom.so")
    desc = f"exe {a.exe}; matched factors: {', '.join(factors) or 'none (production namelists)'}"
    (run / "job_bisect").write_text(JOB.format(name=a.name, time=a.time, desc=desc, result=result,
                                               jobpath=f"{run}/", restart=RESTART))
    (run / "README.bisect").write_text(
        f"{a.name}: production-gap bisect run (fesom_jax_recom docs/plans/20261103-production-gap-bisect.md),\n"
        f"made by scripts/recom/fortran/make_bisect_run.py.\nExecutable ({a.exe}, frozen copies): {what}\n"
        f"Namelists: tmp_run_kdind15w (1998 config backup, {a.months} months) with these factors at the matched "
        f"run's values: {', '.join(factors) or 'none'}\nOutput: {result}\n")
    print(f"{run}: {desc} -> {result}")


if __name__ == "__main__":
    main()
