# The Fortran spectral fork on current FESOM2 (`fesom2.8_recom_spectral`)

Written 2026-11-02 on Albedo, from a read-only inventory of the Fortran checkouts. Nothing below is done yet.

## Goal

A Fortran FESOM2-REcoM spectral model that **corresponds to the REcoM spectral of fesom-jax**: the same REcoM code
that fesom-jax was ported from and gated against, on the same FESOM2 that fesom-jax follows. A Fortran vs fesom-jax
comparison then differs only in fesom-jax's stated deviations, not in the biology code version or the ocean and
sea-ice model underneath (`docs/recom/FORTRAN_JAX_DIFFERENCES.md`). It also removes the need for the
`legacy_k33_weights`, `legacy_fct_bounds`, `legacy_sw_depth` switches and the fork's own level files in the twin
gates, and it is the base for the Fortran bisect that comes next (lane W).

## Decisions (user, 2026-11-02)

- **REcoM = the fork's own in-tree code** (`src/int_recom/` of `sveta-loza/fesom2`, not the upstream REcoM
  submodule), in the version fesom-jax corresponds to. fesom-jax's spectral port was gated against
  `fesom2.6_recom_tra_diags_spectral_recom692` (`9aaf96f7`) = the fork's `e1e62886` + 11 upstream REcoM commits
  (`1cc2d0d3` ballasting fix, the benthos/diag "deduplicate" refactors, allocation and NaN fixes), not against the
  production head `21e0d02a`. So the REcoM source is:
  - `9aaf96f7` (the `_recom692` branch), plus
  - `4d96c3aa` (optional linear phytoplankton mortality; namelist `mort_phy = 0`, `mort_light = .false.` by default,
    i.e. off; not ported to fesom-jax, so it must stay off in matched runs), plus
  - `21e0d02a` (`RECOM_CO2LIM_SPECTRAL`, `&spectral`, default `.false.`; fesom-jax's `RecomConfig.spectral_co2lim`
    is the same switch, default `True`, so matched runs set them equal), plus
  - the env-gated fesom-jax dump hooks of `_recom692_dumps` (`80f520c2`, `867cc6e3`, `d9e613f0`), inert unless the
    dump environment variables are set, so the twin gates run on this build directly. (`63283bc9`, the cold-start
    wind rotation, is upstream already as `66658357`/#919.)
  Production runs (`21e0d02a` without the 11 commits) will therefore differ from the new branch by those commits
  (FORTRAN_JAX_DIFFERENCES.md "Biology" item 1); that difference is wanted, it is the fesom-jax one.
- **FESOM2 = current `upstream/main`**: `0aaa2d73` at the 2026-11-02 fetch (Merge #1243; tags 2.8.1-2.8.3 are on it).
  It contains every fix fesom-jax takes as its default physics (#944 Redi K33 weights, #1116 relative shortwave
  depth, `18346243` FCT bounds, #919 cold-start wind rotation; the `legacy_*` switches select the state before them).
  The exact base commit is fixed at step 0 and recorded here.
- **Branch `fesom2.8_recom_spectral`** in `sveta-loza/fesom2`. `fesom2.6_recom_tra_diags_spectral` and its builds and
  `tmp_run_*` directories are not touched. Pushing needs the user (no GitHub credentials for the Fortran fork on Albedo).

## Inventory: why this is a port, not a `git rebase`

The fork is 86 commits on FESOM2 `cda35117` (2025-10-09), +5183/-138 lines in 19 files: 6 in `src/int_recom/`
(`recom_modules` +2767, `recom_sms` +1052, `recom_forcing` +795, `recom_init`, `recom_main`, `recom_sinking`), 8
host files (`fesom_module`, `gen_surface_forcing`, `io_meandata`, `gen_modules_read_NetCDF`, `oce_ale_tracer`,
`oce_setup_step`, `gen_model_setup`, `src/CMakeLists.txt`), plus `config/bin_4p3z2d/namelist.recom` and build
files. Upstream has since moved 365 first-parent merges (2034 commits, 663 files). Four of those changes hit the fork
directly:

1. **REcoM left the tree.** `23a3f2c0` deleted `src/int_recom/` and added REcoM as the submodule `src/recom`
   (`RECOM-Regulated-Ecosystem-Model/REcoM`, pinned `259e70e3`, 2026-10-26). That REcoM is refactored to a standalone
   library with an **array interface**: the host now calls `recom(f%ice%data(1)%values, f%mesh%nl, ...)`,
   `recom_init(f%mesh%nl, f%mesh%ulevels_nod2d, ...)`, `ver_sinking_recom(tr_num, mesh%nl, ...)`, and uses
   `recom_glovar`, `recom_config`, `recom_sinking`, `recom_declarations, only: ...`. The fork's REcoM still takes the
   FESOM derived types (`call recom(f%ice, f%dynamics, f%tracers, f%partit, f%mesh)`).
2. **REcoM forcing moved to the host.** Upstream reads it in `gen_surface_forcing.F90::sbc_ini_recom` /
   `sbc_do_recom`, called from `gen_forcing_init` / `gen_forcing_couple`; in the fork it is `REcoM_Forcing` inside
   `int_recom/recom_forcing.F90`, which is also where the OASIM spectral forcing was added.
3. **The `associate_*_def.h` / `_ass.h` alias headers are gone** (`e797b8ed`, #1128). The fork's REcoM includes
   them 60 times (`recom_sinking` 24, `recom_extra` 12, `recom_main` 8, `recom_init`/`atbox`/`forcing`/`sms` 4 each).
   `mod_parsup` no longer exists. `MOD_ICE` (+431/-...), `MOD_MESH`, `MOD_PARTIT`, `MOD_TRACER`, `MOD_DYN` changed.
4. **mocsy moved** into the REcoM submodule (`mocsy/`); the fork has it under `int_recom/recom/`.

Replaying 86 commits would repeat the conflict with (1)-(3) at every commit. Instead: branch off upstream, bring the
fork's REcoM over once, and adapt the host interface in a few reviewable commits. The fork's history stays on the
old branch, and the new branch's first commit names the source commit `21e0d02a`.

## Steps

### 0. Branch and worktree

- `git fetch upstream`; `git worktree add /albedo/work/user/sloza/SPECTRAL/fesom2.8_recom_spectral -b
  fesom2.8_recom_spectral upstream/main` from the existing checkout (same object store, no new clone). Record the
  base commit here.

### 1. Vanilla upstream builds and runs on Albedo

- Build upstream as is, with its REcoM submodule (`git submodule update --init src/recom`), double precision, the
  Albedo environment the production builds use (`env/albedo`, Intel). Run pi with `bin_2p3z2d` for a few steps.
- This separates toolchain/namelist problems of the new FESOM2 from problems of the port. It also gives the upstream
  namelist set (`config/`) that steps 4-5 start from.

**Steps 0-1 done 2026-11-02.** Base commit **`0aaa2d73`** (upstream/main at the fetch). Worktree
`/albedo/work/user/sloza/SPECTRAL/fesom2.8_recom_spectral`, branch `fesom2.8_recom_spectral`, submodule `src/recom` at
`259e70e3`. Build `build_recom/` (cmake `-DCMAKE_BUILD_TYPE=Release -DRECOM_COUPLED=ON`, `env/albedo`, `make -j8`
under `taskset`: `configure.sh` runs `make -j$(nproc)`, 256 on the login node) -> `bin/fesom.x`, `lib64/libfesom.so`,
no warnings. Namelists `scripts/recom/fortran/namelists_albedo_fesom28/` (= `namelists_jaxphys` with the Albedo inputs
and nine `namelist.recom` keys REcoM 259e70e3 no longer reads; README there). pi `bin_2p3z2d`, 1 rank, 1 day:
job 48698080 aborted reading `&pavariables` (`HetRespFlux_plus`), job **48698081** completed (32 steps, 61 s, CG 35.5
mean iterations, no non-convergence), run `fesom_jax_runs/recom_ref/fesom28_vanilla_pi_1d`. All fields finite apart
from the below-bottom fill (45828 values in every 3-D field, T and S alike); sst -1.86..29.67, DIC 1848-2462, CO2f
-19..67. Production builds and the twins use `CMAKE_BUILD_TYPE=Debug`; the matched builds of step 5 follow them.

Restart check (on paper; the warm start of 5.5 is the real test): every restart variable 2.8 added (`d_eta`,
`hnode_new`, `sigma11/12/22`, `t_skin`, IDEMIX2, AB3) is defined with `def_*_var_optional`; a 2.6 file only has extra
ones (the tracer `_AB` fields), and `io_restart.F90` keeps `nc_restart_path_legacy` for the old
`fesom.<year>.oce.restart.nc` naming. 2.8 itself writes per-variable directories (`fesom.1958-01-02.oce.restart/`).

### 2. The fork's REcoM back in the tree

- Commit 0 (old base, side branch `recom_spectral_jaxref` off `_recom692_dumps` minus `63283bc9`): cherry-pick
  `4d96c3aa` and `21e0d02a`, so the REcoM source of Decisions exists as one commit on the old FESOM2. It is the
  "before" of check 5.2.
- Commit 1: `src/int_recom/` (including `recom/` mocsy) copied verbatim from commit 0; submodule `src/recom`
  removed from `.gitmodules` and the build; CMake builds `int_recom` as the fork did (`src/CMakeLists.txt` from the
  fork, merged into upstream's). Builds will fail at this commit; that is expected and keeps the diff readable.
- Commit 2: compile-driven adaptation of `int_recom` to the new host types: the 60 header includes replaced by
  explicit `mesh%...` / `partit%...` references (the mechanical form #1128 used for the host), `mod_parsup` and any
  renamed `MOD_ICE`/`MOD_MESH`/`MOD_TRACER` fields. No change to arithmetic or association order.

**Step 2 done 2026-11-02.**
- Commit 0: branch `recom_spectral_jaxref` (worktree `/albedo/work/user/sloza/SPECTRAL/recom_spectral_jaxref`) =
  `867cc6e3` + `1d758359` (cherry-pick of `4d96c3aa`) + **`ce142dcd`** (of `21e0d02a`), both clean. Against production
  `21e0d02a` it differs by the 11 REcoM commits and the dump hooks only (18 files, +906/-333; host files among them:
  `oce_ale_tracer`, `oce_ale_vel_rhs`, `oce_setup_step`, `io_meandata`, `io_netcdf_workaround_module`,
  `recom_ref_dump_hooks.F90`, which step 4 carries over). Built like the jaxphys twins: `build_rel/` Release,
  `RECOM_WAVEBANDS=ON` -> `install_rel/bin/fesom.x`.
- Commit 1 `250197af`: `src/int_recom/` from `ce142dcd`, submodule and `.gitmodules` removed; `src/CMakeLists.txt` globs
  `int_recom/*.F90 int_recom/recom/*.F90` into the fesom target (as the fork), new option `RECOM_WAVEBANDS` (ON) ->
  `__RECOM_WAVEBANDS`; upstream's `__recom USE_PRECISION=2 __usetp` kept (`__usetp` only declares MPI request arrays
  in `oce_modules`; int_recom has no `__usetp` code).
- Commit 2 **`eeb793e2`**: the four `associate_*.h` headers kept privately in `src/int_recom/` (from `e797b8ed^`, so
  they match the current `t_mesh`/`t_partit`), the 60 includes without `../`; `use MOD_PARSUP` (21) ->
  `use par_support_module, only: par_abort`, the five `par_ex` error exits -> `par_abort` (upstream #1073/`f3482bb0`).
  Nothing else in int_recom changed; every int_recom file compiles. Remaining errors are host call sites only:
  `ice_oce_coupling` (17, `use_virt_tracers`/`virtual_*` from the submodule's `recom_config`), `oce_ale_tracer` (13,
  `tracers_info`, `expflux/botflux`, `virtual_*`), `io_meandata` (17, `tracer_ids`); `fesom_module` and
  `gen_forcing_*` are not reached yet.

### 3. Host call sites back to the fork's interface

- Commit 3: replace upstream's array-interface calls with the fork's type-based ones at every site: `fesom_module`
  (`recom`, `recom_init`, `use recom_glovar`), `oce_ale_tracer` (`diff_ver_recom_expl`, `ver_sinking_recom`,
  `ver_sinking_recom_benthos` and their `use` lines), `gen_forcing_init`/`gen_forcing_couple` (`sbc_ini_recom`,
  `sbc_do_recom`: drop, since the fork's REcoM reads its own forcing), `ice_oce_coupling`, `io_meandata`,
  `gen_surface_forcing` (`use recom_declarations`).
- **Check that each call keeps upstream's position in the timestep** (where REcoM runs relative to the tracer
  advection, the ice coupling and the output). If upstream moved a call, follow upstream and write it down: it is a
  host-model change and belongs to this update.

### 4. The fork's host edits

- Commit 4: the 8 host files' changes from `cda35117..` commit 0 (OASIM/forcing reading in `gen_surface_forcing` and
  `gen_modules_read_NetCDF`, the spectral/per-group outputs in `io_meandata`, `fesom_module`, `oce_setup_step`,
  `gen_model_setup`, `oce_ale_tracer`), applied by hand onto the upstream versions. `config/bin_4p3z2d/namelist.recom`
  from the fork.

### 5. Verification

1. **Build**: DP, Intel, the production build's flags and cpp switches (`RECOM_CALC_ACDOM/APART`, `RECOM_BMASS`,
   `RECOM_CO2LIM_SPECTRAL` as used).
2. **Biology unchanged by the move.** A REcoM column with the same physics inputs must give the same output on
   commit 0 (old FESOM2) and on the new branch: compare the dumped SMS of one pi step bit for bit, feeding both the
   same dumped inputs, so the different physics does not matter.
3. **Biology = fesom-jax.** The existing column/one-step spectral gates (`test_recom_*_fork`,
   `test_recom_spectral_*`) against dumps of the new build, with the namelist switches matched
   (`RECOM_CO2LIM_SPECTRAL` = `spectral_co2lim`, `mort_phy = 0`): the current numbers (<= 2.4e-13 column, <= 2.0e-5
   one physics-matched step, VALIDATION.md) must hold.
4. **Physics now matches fesom-jax.** Re-run the physics-matched pi twin of the spectral gates (`test_recom_*_fork`,
   `docs/recom/VALIDATION.md`) with the new executable and fesom-jax **without** the `legacy_*` switches and the fork's
   level files. Target: the physics agreement of the 692f804 pair (4.1e-8) and the BGC agreement of the current
   spectral twins. GM/Redi on, if it now matches (it does not on the old base).
5. **CORE2.** Re-run the warm 1998 setup (`tmp_run_kdind15wjp`, the physics-matched one) on the new executable
   and compare with `fesom_jax_runs/core2_recom_spectral_warm1998_jaxphys_groups` using
   `scripts/compare_fortran_jax_groups.py`. The remaining difference should be fesom-jax's stated deviations only
   (ballast reset over the whole mesh instead of per rank, the first-call reset skipped at step 1) plus chaos.

### 6. Hand-over

- The user pushes `fesom2.8_recom_spectral` to `sveta-loza/fesom2`.
- Update `docs/recom/FORTRAN_JAX_DIFFERENCES.md` (host model rows), `VALIDATION.md` (twin numbers, SLURM job ids),
  `RUNNING_ALBEDO.md` (new build), and move this plan to `completed/`.

## Risks and open points

- **Restarts.** The warm 1998 runs start from a 1997 restart written by the 2.6 code. Upstream changed
  `io_restart` and `MOD_ICE`; if the new code cannot read it, a fresh spin-up (or a converter) is needed before step
  5.4. Check this early, in step 1.
- **Namelists.** Upstream 2.8 has new and renamed keys in `namelist.oce/ice/dyn/forcing/io/cvmix`; the fork's run
  directories cannot be reused unchanged. Start from upstream `config/` and carry the physics choices over
  explicitly, matching fesom-jax's configuration where the goal is a matched pair.
- **Upstream REcoM since `_recom692` is not taken.** The 11 commits fesom-jax carries are in; the later upstream
  REcoM work (the standalone-library refactor and whatever fixes came with it, up to `259e70e3`) is not, in either
  code. Listed as a known difference to upstream, optional later for both.
- **Production vs this branch.** The user's production results (`21e0d02a`, `tmp_run_kdind15*`) are not
  reproduced by this branch even before the host change, because of the 11 REcoM commits (`1cc2d0d3` changes
  values). Comparing them shows that effect plus the host change; the bisect separates the two.
- **Call order.** If upstream moved where REcoM runs in the step (step 3), the biology output changes slightly even
  with identical REcoM code. Step 5.2 does not see this; 5.3-5.4 do.
- **Interface later.** Moving the spectral code onto the standalone REcoM library (array interface) would let the
  fork follow upstream REcoM. That is a separate, larger job and not part of this plan.
