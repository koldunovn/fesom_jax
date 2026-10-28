# Lane F: the Fortran FESOM2 + REcoM reference (parent plan Tasks 3-4)

Parent plan: `docs/plans/20261025-recom-port.md` (read its Overview, Context and Tasks 3-4 first).
Branch `recom/fortran-ref` off `recom/port`, worktree `/work/ab0995/a270088/fesom_jax_dev/wt_recom_fref`.
Fortran tree outside the repo: `/work/ab0995/a270088/fesom2_recom`. Run and dump output:
`/work/ab0995/a270088/recom_ref/`. Time box: 8 hours wall. Effort: high.

## Goal

A double-precision FESOM2 + REcoM build, pi-mesh runs in the configuration the JAX port mirrors, and
full-field reference dumps that the JAX gates (G2-G4, GP, G5) read through `fesom_jax/io_dump.py`.

## Fixed choices (do not change without asking the parent session)

- FESOM2 at **e96a2b13** (upstream main of 2026-10-20) and its pinned REcoM submodule **692f804**
  (`git -C src/recom rev-parse HEAD` must print 692f8045...). Clone from `https://github.com/FESOM/fesom2`;
  the read-only checkout `/work/ab0995/a270088/fesom2_sp_zstar` (same commit) may be read for reference,
  never modified.
- DP build, `RECOM_COUPLED=ON`, Intel env `env/levante.dkrz.de/shell`, FESOM2's own CMake flags.
- REcoM namelists from `config/bin_2p3z2d` (`namelist.recom`, `namelist.tra`: 31 tracers 1001-1030 + 1037,
  advection weights `0., 1.`), with the pi input files of `test/input/recom` (Albani dust on the pi mesh,
  the 5° IC files) instead of the Albedo paths.
- Mesh pi (`test/meshes/pi`). Prefer ONE MPI rank so dumps need no gather; if FESOM2 cannot run on one
  rank, use the smallest existing partition and gather to global numbering via `myList_nod2D`.
- Forcing **JRA55-do** (`/pool/data/AWICM/FESOM2/FORCING/JRA55-do-v1.4.0`, `config/namelist.forcing.JRA` as
  the template) with **`l_mslp = .true.`** (`psl`); never the test's CORE forcing.
- Physics: zstar, mEVP (`whichEVP 1`, 120 subcycles), `use_wsplit`, `step_per_day 32` (dt 2700), no
  cavities, GM/Redi and mixing as in `config/bin_2p3z2d` unless pi cannot run them (then state it).
  Write the full namelist set into the report so the parent session can build `configs/pi_recom.yaml`.

## Steps

- [x] F1 clone + submodule + build; record commits, compiler, flags in `docs/recom/FORTRAN_REFERENCE.md`
- [x] F2 run setup (scripts in `scripts/recom/fortran/`: `build_fesom2_recom.sh`, `run_pi_recom.sbatch`);
      a 1-day smoke run from 1958-01-01 (cold start, BGC from the IC files); check that the log shows 31
      tracers and sane global REcoM diagnostics (no NaN, CO2 flux of plausible sign and size)
- [x] F3 instrumentation in our clone only, behind `FESOM_RECOM_DUMP_DIR` (no-op otherwise; steps from
      `FESOM_RECOM_DUMP_STEPS`), modelled on `origin/recom4-dumps` (single-column hooks) but FULL fields,
      global node numbering. Dump: every `recom()` input and output (incl. `press_air`, `a_ice`,
      shortwave, winds, `GlodecayBenthos`, `GloHplus`, `Alk_surf`, the surface flux arrays, the carbonate
      cache arrays, REcoM's 3-D diagnostics); per BGC tracer in the tracer loop, in loop order: values
      after advection+Redi, `dtr_bf`, ballasting densities `rho_particle1/2` and the scaling factors,
      `vert_sink`, `str_bf`, `Benthos` after the tracer, values after T*, values after the implicit
      diffusion; the transport fields of the step (uv, w_e, w_i, hnode, hnode_new, helem, Kv, the GM
      bolus velocity, the Redi slopes/Ki); the SLP momentum term of `oce_ale_vel_rhs.F90:202`
- [x] F4 dump format: reuse an existing `io_dump.py` record format if one fits (read `io_dump.py` first);
      otherwise a simple self-describing one (name, dtype, shape header + raw little-endian f64). Add
      `load_recom_dump(dirpath, step)` to `fesom_jax/io_dump.py` + `fesom_jax/tests/test_recom_dump_io.py`
      (shapes, tracer IDs, one round trip; skips cleanly when the directory is absent)
- [x] F5 runs (all dumps to `/work/ab0995/a270088/recom_ref/<run>/dumps`):
      (a) 30 days from 1958-01-01, dumps at steps 1, 2, 17, 224, 225, 960 (cold start, mid-day, the first
      7-day refresh, the 30-day refresh); (b) 1 day from 1958-07-01 (cold start), dumps at steps 1 and 17;
      (c) the same 30 days as (a) split into 15 d + restart + 15 d, dumps at the first two steps of the
      second segment. Check that (a) without the env variable is bitwise (a) with it (no-op check, on the
      final restart or the output)
- [x] F6 report in `docs/recom/FORTRAN_REFERENCE.md`: rebuild recipe, every run (commit, namelists,
      job id, wall time), the dump catalogue (field names, shapes, units), anything surprising (e.g. a
      switch the bin_2p3z2d namelist sets that the parent plan does not list)
- [x] F7 commit on `recom/fortran-ref` (scripts, docs, io_dump + test); the Fortran clone stays outside the
      repo. Lesson in `docs/PORTING_LESSONS.md`.

## Rules

- Jobs: `-A ab0995 -p compute --time=30:00` (debug QOS), `#SBATCH --hint=nomultithread`; no GPU. Wait on
  Slurm with a background shell `until` loop on `sacct` (exits on any terminal state), not by polling.
- Never delete, `rm`, `scancel`, overwrite another run's directory, push, or touch any other worktree or
  branch. English commit messages, files staged one by one (never `git add <dir>`).
- If a fixed choice cannot be met (e.g. pi cannot run with these physics), stop that step and report why
  instead of improvising.

Done 2026-10-25: jobs 28417235/6, 28417240 (smoke + no-op), 28417314/5 (a), 28417316 (b), 28417317/8 (c); ~2.0
node-hours. 30-day jobs ran with `--time=01:00:00` (2.3 s/step on one rank, 37 min). Details and findings:
`docs/recom/FORTRAN_REFERENCE.md`.

## Deliverable (final report to the parent session)

Commits on `recom/fortran-ref`; the run/dump directories; the namelist set used; job ids and cost (node
hours); open problems.
