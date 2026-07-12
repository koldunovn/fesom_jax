# Porting lessons — FESOM2 → JAX

A **living log** of experiences, gotchas, and hard-won facts from porting FESOM2
to JAX. Mirrors the C/Kokkos "lesson log" discipline. **Rule: append to this file
whenever a session/task surfaces something non-obvious** — a config that differs
from the docs, a sign/index/association-order trap, an AD subtlety, a fidelity
surprise, or a "this cost me an hour" fact. One entry = one lesson. Keep entries
short and concrete; link the source file + line. Newest phase at the bottom.

Format per entry: **[area] one-line claim.** Then *why it matters* + *how to apply*.
Cite the C source (`file:line`) or dump probe that proves it.

---

## Cross-cutting (apply everywhere)

- **[fidelity] Bit-identity to C is impossible; target tolerance classes.** Scatters
  (`segment_sum`) and global reductions reassociate FP sums and JAX does not preserve
  C's edge/element order. Target: **~1e-15** for map/gather kernels, **~1e-12** for
  scatter/reduction kernels. Do *not* chase a scatter discrepancy below ~1e-12 — it is
  reassociation, not a bug (`port_kokkos/docs/SCATTER_STRATEGY.md`). This costs nothing
  for AD (a scatter's gradient is a gather).

- **[verify] Always truncate the JAX probe column to the record's `nlevels` before
  diffing.** The dump drops below-bottom padding; a full-length compare fails on the
  tail. Node → `nlevels_nod2D`; element → `nlevels`. `verify.compare_column` does this.

- **[masks] Layer vs interface validity is a two-class distinction, not one.** Layer
  fields (T,S,ρ,p,u,v,pgf) valid `k ∈ [ulevels-1, nlevels-1)` (exclusive bottom);
  interface fields (bvfreq,w,Kv,Av) valid `k ∈ [ulevels-1, nlevels-1]` (inclusive, one
  deeper). Use `node_/elem_ × layer/iface_mask` from `mesh.py`. Getting this wrong
  shows up as a wrong/zero value at exactly the bottom level in the probe diff
  (`fesom_eos.c:93-208`). Concretely in the substep-1 dump: density/pressure are 0 at
  index `nlevels-1` (masked layer tail) while bvfreq is *nonzero* there (interface
  bottom-pad) — same column, different last-level behaviour.

- **[constants] Use the truncated π = 3.14159265358979, not `jnp.pi`.** `RAD`, `OMEGA`,
  cyclic length all derive from it; full-precision π seeds ~1e-13 into every rotation
  and breaks gates. Already in `config.py`. Verified `config.py` ↔ `fesom_constants.h`
  match for PI, RAD, DENSITY_0=1030, G=9.81, R_EARTH=6367500, OMEGA.

- **[mesh] `nl = 48` globally for pi.** FRESH_START's "nl≈23" is the per-node count.
  Size every node/elem column to `nl=48`. Export is **already 0-based** (no 1→0
  conversion); `edge_tri`/`edge_up_dn_tri` use `−1` for boundary (masked by
  `ops.scatter_add`).

- **[probes] Node probes are 1-based gids; JAX index = gid − 1.** Pinned node gids
  `1001,1500,2000,2500,3000`; element probes = first incident cell `1757,2656,3688,
  4604,5575`. Element records carry the *element* gid, node records the node gid.

- **[golden-rule] Preserve the exact math + load-bearing association order, but express
  it as vectorized `ops.py` array ops.** Not a literal loop-by-loop translation; do not
  simplify the physics. When in doubt, dump the C value at a probe and match it.

---

## Phase 0 — Foundations

- **[env] Levante env is mamba `fesom-jax`, jax 0.10.1 x64, A100.** Use
  `/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python` for ALL python/pytest
  (NOT base conda). Login-node `cuInit 303` warning is the benign GPU-absent fallback;
  `JAX_PLATFORMS=cpu` silences it. Phase 2 is pure JAX → CPU is fine.

- **[oracle] The per-substep oracle is the *C-port* dump writer (Path A), not the
  Fortran shim.** `fesom2_port/src/fesom_dump.c` (branch `jax-mesh-export`) dumps node
  **and** element fields at the exact JAX config, so JAX↔C diffs are pure FP
  reassociation. The pre-existing Fortran dump uses a realistic stratified IC +
  KPP/opt_visc5 → **not** per-substep comparable; climate-level cross-check only.

## Phase 1 — Mesh & State

- **[pytree] `Mesh`/`State` are frozen dataclasses registered via
  `register_dataclass`.** For `Mesh`, the 31 arrays are leaves and the 7 scalar counts
  are static meta (they fix shapes + `segment_sum` segment counts → must be Python
  ints, not traced). For `State`, every field is a data leaf.

- **[AD] The two AD gates that matter were proven in Phase 1:** scatter transpose ==
  gather (analytic vjp), and TDMA grad == central FD (≤1e-6). Reuse `ops.scatter_add`
  (masked `segment_sum`, −1→0 in fwd **and** grad) and `ops.tdma` (two `lax.scan`
  sweeps) — do not re-derive.

## Phase 2 — Minimal forward step on pi

- **[IC] ⚠️ The pi reference dump is NOT a constant T=10/S=35 IC — it is constant +
  a Gaussian T-blob.** `fesom_main.c:744-753` adds `fesom_ic_tracer_T_blob` on top of
  the constant IC whenever no PHC path is given (the dump run gives none). The blob:
  centre (lon0,lat0)=(−45°,40°) geographic, σ_h=10°, σ_z=300 m, amp=+5 °C, added
  **additively** to T on every wet layer, with a **4σ horizontal cutoff** (`if r²_h >
  16: continue`) and a small-circle `cos(lat0)` correction (`fesom_ic.c:82-129`). S
  stays 35. **Consequence:** any kernel whose verification depends on T/S (EOS at
  substep 1, hence pressure → PGF → momentum → everything downstream) must reproduce
  the blob, not the bare constant. Probe 1001 sits inside the blob (stratified →
  bvfreq≠0); probe 3000 is outside (T=10, bvfreq=0). The constant-IC claim in
  `REFERENCE_RUNS.md`/the plan is the *base*; the blob is the actual field. *(Found
  while starting Task 2.1: density[0] differed across probes and bvfreq was nonzero —
  impossible under a truly constant IC.)*

- **[IC] T/S are effectively frozen over the 10 dumped steps.** Constant S and a
  smooth T-blob under analytical-wind-only forcing (no heat/water flux) drive only weak
  flow; density at substep 1 is identical to ~5 digits across steps 1–10. So substep-1
  EOS fields are step-independent in this config — a convenient but weak (no horizontal
  T/S evolution) gate. Horizontal-variation stress comes later (FCT / CORE2).

- **[eos] The substep-1 N² smoother (`fesom_smooth_nod3D`, single sweep) is an
  element→node area-weighted patch average; the dump's bvfreq is POST-smooth.** Per
  element el and level nz∈[ulevels[el]-1, nlevels[el]-1], scatter
  `area_el·(bv[v0]+bv[v1]+bv[v2])` to all 3 vertices; divide node sums by `3·Σarea`.
  The element's level range is always ⊆ its vertices' ranges (node nlevels = MAX over
  cells, ulevels = MIN), so the per-element clamp = `elem_iface_mask` and no extra
  node-side level clamp is needed (`fesom_eos.c:226-277`, `fesom_step.c:92`). Scatter
  class → ~1e-12. **Verified load-bearing**: raw N² *fails* the probe gate (rel ~1.1),
  smoothed *passes* (rel ~3e-16) — always test that a "decorative-looking" smoothing
  pass actually moves the field at the probe before trusting a green gate.

- **[eos] A pointwise map kernel can match C *bit-for-bit* (max|Δ|=0), not just
  ~1e-15.** The JM-EOS `density` column matched the dump exactly — Horner-form
  polynomial, identical constants, no scatter/reduction to reassociate. Useful
  expectation: for a pure pointwise map, a *nonzero* diff signals a real bug (wrong
  constant/association), not FP noise. `pressure` (a downward cumsum) lands at ~1e-11
  abs / 1e-16 rel — the sequential integration, near-exact. (Task 2.1, `eos.py`.)

- **[eos/AD] The unused `nz=0` N² level is a div-by-zero → NaN-gradient trap.** N²[nz]
  needs the `nz-1` layer; at `nz=0` there is none. A naive `0/0` there produces NaN
  that `jnp.where`-masking does NOT stop in the *backward* pass (NaN·0 = NaN). Fix:
  edge-replicate the level shift so ρ_up==ρ_dn at nz=0 (→ N²=0) and set the unused
  `zdiff[0]=1` to keep the divide finite. General rule: make masked-off lanes compute a
  *finite* value, don't rely on the forward mask to hide a NaN. (`eos.py:_shift_down`.)

- **[eos] N² surface/bottom interface padding vectorizes as a clip-gather.** C does
  `bvfreq[nzmin]=bvfreq[nzmin+1]; bvfreq[nzmax]=bvfreq[nzmax-1]`. Equivalent to
  `take_along_axis(bv, clip(arange(nl), nzmin+1, nzmax-1))` then mask to
  `node_iface_mask` — one op, per-node bounds, no scatter. (`eos.py:pressure_bv`.)

- **[pgf] Element dumps verify element kernels DIRECTLY — ignore the plan's older
  "indirect via ssh_rhs" hedges.** The C dump writer (Path A) records `pgf_x/y`,
  `uv_rhs`, `uv`, `Av` at element probes (first cell incident to each node probe:
  1757/2656/3688/4604/5575). PGF matched at all 5 to ~1e-20 abs (`gather` class).
  (Task 2.2, `pgf.py`.)

- **[verify] For fields whose tail values decay to numerical zero (~1e-20), the
  *relative* error is meaningless — the `atol` floor is what gates.** PGF deep-level
  diffs showed rel ~1e+287 (tiny-abs ÷ tinier-c) yet PASS because `|Δ|≤atol+rtol·|c|`
  with atol=1e-14 covers them. This is exactly why the gate is the isclose form, not a
  pure relative test. Don't panic at a huge `rel=` in the report if `max|Δ|` is ~atol.

- **[pp/masks] PP/convection write a THIRD level-range class: interior interfaces
  `[nzmin+1, nzmax)`.** Not the layer range `[nzmin,nzmax)` and not the full interface
  range `[nzmin,nzmax]` — `Kv`/`Av` are left **0** at the surface (`nzmin`) and bottom
  (`nzmax`) interfaces because the PP loops run `nz` from `nzmin+1` to `<nzmax`. The
  dump confirms: `Kv=[0, 1e-5, …, 1e-5, 0]`. Build this mask as
  `(k>=ulevels) & (k<nlevels-1)`; don't reuse `iface_mask`. (`fesom_pp.c:105`.)

- **[pp] The 3-loop order is load-bearing — compute the dimensionless `factor` ONCE.**
  C overwrites `Kv` with `factor = shear²/(shear²+5·max(N²,0)+1e-14)`, builds
  `Av = mix·mean(factor²)+A_ver` from THAT, *then* overwrites `Kv = mix·factor³+K_ver`.
  Av uses `factor²`, Kv uses `factor³` — if you compute Av from the final Kv you get
  the wrong viscosity. In JAX: compute `factor`, derive both from it. (`fesom_pp.c:62-145`.)

- **[verify/method] A step-1 dump gate is WEAK for any kernel that depends on
  velocity** — uv=0 at rest, so PP's shear path, momentum advection, etc. collapse to
  trivial background. Exercise the dormant path with a **synthetic-input unit test
  against an independent (loop-based, different-code-path) reference**, plus a
  later-step re-verification once the full `step()` exists (Task 2.11). A green step-1
  gate alone does NOT mean the kernel is right. (Tasks 2.3, 2.4 both hit this:
  momentum at rest collapses to `uv_rhs = −dt·pgf`.)

- **[momentum] The AB-slot read/overwrite order is load-bearing.** `compute_vel_rhs`
  (i) shifts the **OLD** `uv_rhsAB` into `uv_rhs` (`ab1·AB_old`), then (iii) **overwrites**
  `uv_rhsAB` with this step's Coriolis `(v·ff, −u·ff)`, then advection **adds** into that
  NEW slot, then assembly reads the NEW slot. In JAX: keep `AB_old` (input) and build a
  fresh `AB_new`; never alias. (`fesom_momentum.c:82-119`.)

- **[momentum] Momentum advection (momadv_opt=2) is TWO scatters, not one.** Vertical:
  element→node area-weighted interface-velocity scatter, ×`w_e`, then −d/dz over
  `3·hnode`. Horizontal: an **antisymmetric edge→node** scatter (`n1 += flux`,
  `n2 −= flux` — scatter `[+flux,−flux]` to `edges`). Then /`areasvol`, then
  vertex→element area·mean/3. The edge-replicated down-shift `0.5(u[j]+u[j−1])`
  *automatically* yields the C's surface term `u[0]` at `j=0` (since 0.5(u0+u0)=u0) —
  no special-case needed for non-cavity. (`fesom_momentum.c:156-271`.)

- **[verify/method] For an intricate multi-scatter kernel, transcribe the C loops
  verbatim into a numpy reference and diff against it with SYNTHETIC nonzero inputs.**
  Far stronger than the (rest-trivial) dump gate, and a different code path so shared
  bugs are unlikely. The momentum reference (~90 lines, loop-for-loop) caught nothing
  here only because the vectorization was right — but it's the gate that *would* catch a
  sign/index slip in the edge scatter. (Task 2.4, `test_momentum.py::_ref`.)

- **[AD] Use the double-`where` "safe sqrt" for any `sqrt(x)` that can hit x=0 —
  forward-identical, gradient finite.** `safe = where(x>0, x, 1); return where(x>0,
  sqrt(safe), 0)`. Plain `sqrt(0)` is fine forward but its grad is `1/(2·0)=∞`, and a
  downstream `where`/`max` that masks the value does NOT mask the NaN in the backward
  pass. The flow-aware biharmonic viscosity depends on `|∇u|=sqrt(|du|²)` (kink at
  rest) — without safe-sqrt the *whole* gradient is NaN at uv=0. Verified with a
  no-NaN-grad-at-rest test. Same pattern already used in `eos.py` for the unused N²
  level. (Task 2.5, `momentum.py:_safe_sqrt`.)

- **[visc] The biharmonic `visc_filt_bidiff` is two edge→element antisymmetric
  scatters with a shared per-edge `coef`.** Stage 1 builds the flow-aware Laplacian
  `U_c/V_c` at elements (`U_c[el1]-=u1·coef`, `+=` at el2); stage 2 scatters
  `-dt·coef·(U_c[el1]-U_c[el2])/area` back into `uv_rhs`. INTERIOR edges only (both
  `el1,el2≥0`) — boundary edges contribute nothing (the −1 sentinel + the level
  mask both zero them). Per-edge level range is the OVERLAP
  `[max(ulevels)-1, min(nlevels)-1)`, not either element's own range.
  (`fesom_momentum.c:654-762`.)

- **[forcing] ⚠️ The wind stress `impl_vert_visc` reads is DOUBLE-AVERAGED, not the
  raw cos pattern.** `set_analytical` writes the raw element stress AND an
  area-weighted node average; then `oce_fluxes_mom` (`fesom_ice_coupling.c:256`, run
  EVERY step before the ocean step via `fesom_main.c:983`, even with no ice — the ice
  blend is a no-op at `a_ice=0`) OVERWRITES the element stress with the **simple mean
  of the 3 vertices' node stresses**. So: raw elem → area-weighted node → simple-mean
  elem. Feeding the raw stress is a ~5e-4 surface-velocity error. A cross-module
  dependency hiding in the *ice* coupling file — easy to miss. (`forcing.py`, Task 2.6.)

- **[verify/diag] When ONE velocity component fails at ONE row and the other passes,
  the matrix/solve is correct and that component's FORCING is wrong.** The TDMA `v`
  matched the dump exactly while `u` failed only at the surface row — instantly
  localizing the bug to the u-only surface wind stress (not the solve, geometry, or
  drag). Per-component, per-row failure structure is a precise debugging signal.
  (Task 2.6.)

- **[tdma] Pad the per-column TDMA to full `nl` with `(b=1, a=c=0, d=0)` below the
  bottom → those rows solve to 0 and don't corrupt the real system.** The bottom valid
  row has `c=0` (no downward coupling) and the first pad row has `a=0` (no upward
  coupling), so `ops.tdma` over all `nl` rows gives the same answer as the C's
  `[nzmin,nzmax)` loop. Phase-2 simplifications that made this tractable: `w_i=0`
  (advective tridiagonal terms vanish) and no partial cells (`zbar_n=zbar`, `Z_n=Z`
  globally, computed once and broadcast). (`momentum.py:impl_vert_visc`, Task 2.6.)

- **[ssh/solver] ⚠️ The C CG stops at a LOOSE `soltol=1e-5`, so the dumped `d_eta` is
  the EARLY-STOPPED iterate — NOT the converged solution.** On pi `cond(S)≈800`, so PCG
  hits `‖r‖<soltol·‖b‖` in just **3 iterations** (residuals `[65, 1.0, 0.015]` vs
  `rtol=0.197`); the early iterate is ~2e-9 from the exact `S⁻¹b`. **Consequence: to
  match the dump you MUST replicate the C PCG (same static `S`, same MITgcm
  preconditioner, same `x0`, same stop) — converging *tight* gives a DIFFERENT `d_eta`
  (off ~2e-10 @ probe 1001, rel 2.5e-6 → fails the gate).** The replicated 3-iter PCG
  matches the dump to **~1e-18**. The huge residual margin (iter 2 is 5× above, iter 3
  is 13× below the threshold) makes the iteration count robust to `segment_sum`
  reassociation. *(The plan's "≤1e-12" gate is met by `d_eta`; the early-stop replication
  is what makes it possible.)* (`fesom_ssh.c:407-412,484`, Task 2.7.)

- **[ssh/AD] `custom_linear_solve` cleanly decouples a dump-matching forward from an
  accurate gradient via SEPARATE `solve`/`transpose_solve`.** Reverse-mode AD uses ONLY
  `transpose_solve` for the cotangent, so: forward `solve` = early-stopped PCG (matches
  the dump), `transpose_solve` = *tight* PCG → the gradient is the clean implicit-diff
  `S⁻¹·x̄` regardless of the loose forward stop. Verified: AD cotangent == an independent
  tight `S⁻¹w` (rel 2e-14) == central-FD, and is finite. The forward value and the
  gradient genuinely have different fidelity needs (dump-match vs accuracy); don't force
  one solver to serve both. (`ssh.solve_ssh`, Task 2.7.)

- **[ssh/precond] The MITgcm symmetric preconditioner is LOAD-BEARING — test that a
  Jacobi variant FAILS the dump.** Because the dump is the early-stopped iterate, the
  preconditioner (which shapes the Krylov path) directly changes `d_eta`. Zeroing the
  19336 off-diagonal `pr` entries (→ Jacobi) shifts `d_eta` by 2.9e-10 @ probe 1001 →
  fails the dump. Same discipline as the bvfreq-smoother: prove the "looks like a detail"
  pass actually moves the gated field. `pr[diag]=1/diag`, `pr[off]=−0.5·(S[r,c]/diag_r)/
  (diag_r+diag_c)` — off-diagonal, applied as a sparse matvec, not a diagonal scaling.
  (`fesom_ssh.c:239-253`, `ssh.ssh_precond`, Task 2.7.)

- **[ssh/rhs] `ssh_rhs` is a near-cancelling transport divergence → its abs floor is
  upstream `du` amplified by geometry (`dx·helem ~ 1e7`), NOT the ssh_rhs scatter.** The
  wind-forced convergence is a small residual of large opposing edge fluxes (~1e4), so at
  cancellation nodes (probe 1500: value 1.13) the abs diff vs the dump is ~5e-9 (rel ~4e-9)
  while at constructive nodes (probe 1001: value 2.8e4) it's rel ~1e-14. A
  numpy-*sequential* reference AND `segment_sum` both land ~5e-9 vs the dump — same floor,
  so it's the shared upstream `du` (~1e-12 rel) ×`dx·helem`, not the scatter order. Gate at
  **atol 1e-7**, not 1e-12; the relative error at cancellation nodes is meaningless.
  (`ssh.compute_ssh_rhs`, Task 2.7.)

- **[ssh/static-op] In linfs the stiffness operator is STATIC: the "−g·dt·α·hbar" factor
  uses the FIXED `zbar` depths, never the evolving `hbar`.** `depth = zbar[nlevels-1] −
  zbar[0] < 0` IS the `−hbar` (full static column depth); the positive `factor=g·dt·α·θ`
  carries the magnitude. So `update_stiff_mat_ale` is gated off (`fesom_ssh.c:9-12`), the
  operator is assembled ONCE (host scipy COO→CSR → a `segment_sum` matvec reused every
  step), and AD is clean (the operator carries no differentiable/evolving dependence — the
  whole `d(d_eta)/d(params)` path is through the rhs). Per-step rebuild is a Phase-5/zlevel
  concern. (`fesom_ssh.c:120-145`, `ssh.build_ssh_operator`, Task 2.7.)

- **[ssh/warmstart] The C warm-starts the CG from the previous step's `d_eta` (it's never
  zeroed between steps — only inited at `fesom_ic.c:57`).** Step-1 `x0=0` (a clean *linear*
  solve, ideal for `custom_linear_solve`). For step ≥2 the warm start makes the
  early-stopped iterate depend on `x0`, which would make the inner `solve` non-linear; keep
  it linear by folding the warm start into the rhs (`b_eff = b − A·stop_gradient(x0)`, solve
  `δ` from 0, return `x0+δ`). The *solution* is `x0`-independent — only the early-stop
  iterate isn't — so `stop_gradient(x0)` is correct. Exact warm-start dump-matching at step
  ≥2 (the C's stop threshold uses the original `‖b‖`) is finalized with the full `step()` in
  Task 2.11. (`ssh.solve_ssh`, Task 2.7.)

- **[hbar] ⚠️ `compute_hbar`'s `ssh_rhs_old` IS `compute_ssh_rhs` with `uv_rhs=0` and
  `alpha=1` — reuse it, don't re-port.** Substep 11's transport divergence
  (`fesom_momentum.c:796-830`) is the byte-identical antisymmetric edge→node scatter as
  substep 8 (`fesom_ssh.c:261`); the *only* differences are it uses the bare **new** velocity
  `u` (not `u+u_rhs`) and drops the `alpha` factor (`alpha=1`). So
  `ssh_rhs_old = compute_ssh_rhs(mesh, uv, zeros_like(uv), helem, alpha=1.0)` is exact. Edge
  range is `myDim_edge2D` (the C warns the `+eDim` double-count → CG NaN ~step 85-95 in MPI)
  but for single-rank pi `myDim_edge2D == edge2D == 8986`, so the all-edges JAX scatter is
  identical. (`ssh.compute_hbar`, Task 2.8.)

- **[hbar/fidelity] A downstream `÷ (large area)` RESTORES tight fidelity that the
  intermediate scatter lost — gate the OUTPUT, not the noisy intermediate.** `hbar =
  hbar_old + ssh_rhs_old·dt / areasvol[n,0]`. `ssh_rhs_old` is the *same* near-cancelling
  transport-divergence scatter as `ssh_rhs` (abs floor ~1e-7, amplified by `dx·helem~1e7`),
  yet `hbar` matches the dump to **~1e-17 absolute** — because `areasvol ~ 1e9–1e12 m²`
  divides that amplified error right back down (`1e-7·100/1e10 ~ 1e-15`). So the substeps
  10–12 dump gates are TIGHT (uv ~2e-17, hbar/eta_n ~1e-17), unlike the loose `ssh_rhs` gate
  (atol 1e-7). Moral: don't inherit an upstream field's loose tolerance — re-measure at the
  gated field; a `÷area`/average can recover map-class fidelity. (`ssh.compute_hbar`, Task 2.8.)

- **[update_vel] The SSH-gradient correction `(Fx,Fy)=∇N·(−gθdt·d_eta)` is BAROTROPIC
  (uniform over the column) and `uv` ACCUMULATES (`uv += du + F`).** `Fx,Fy` are a single
  per-element scalar added to *every* layer `nz∈[nzmin,nzmax)` (broadcast over `nz`), unlike
  the per-level increment `du` (`fesom_momentum.c:496-500`). At step 1 `uv=0` so this is the
  first wind-driven velocity (~1e-3 surface); at step ≥2 it increments the carried `uv`. uv
  matched the dump ~2e-17 (gather class) since both `du` (~1e-17) and the replicated
  early-stop `d_eta` (~1e-18) are near-exact. `d_eta` is *read* here, not consumed — it stays
  as the next step's CG warm-start `x0`. (`momentum.update_vel`, Task 2.8.)

- **[eta_n] With `SSH_ALPHA=1` the eta_n blend collapses to `eta_n = hbar` exactly** (the
  dump confirms `eta_n == hbar` at every probe). `eta_n = α·hbar + (1−α)·hbar_old`
  (`fesom_step.c:257-268`); the `(1−α)·hbar_old` term vanishes at `α=1` (same shape as the
  `ssh_rhs`'s `(1−α)·ssh_rhs_old` blend). Keep the blend form for generality, but in Phase 2
  `eta_n` is a renamed copy of the post-update `hbar`. Only non-cavity nodes
  (`ulevels_nod2D==1`, all of pi) are written; cavity nodes keep their prior `eta_n`.
  (`ssh.eta_n_update`, Task 2.8.)

- **[ale] ⚠️ `w` (substep 13) is the PER-LEVEL sibling of the ssh_rhs/hbar scatter —
  reuse the flux, keep it per-level, then reverse-cumsum + ÷area.** Same antisymmetric
  edge→node `(v·dx − u·dy)·helem` transport divergence as `compute_ssh_rhs`/`compute_hbar`
  (`alpha=1`, bare new `uv`, no AB-velocity), but NOT summed over levels — keep the
  `[edge,nl]` term, scatter `[+c,−c]→[n1,n2]` per level, then (3) a reverse bottom→top
  cumulative sum, then (4) ÷ area. (`fesom_ale.c:104-187`, `ale.compute_w`, Task 2.9.)

- **[ale] ⚠️ Stage-4 divides by `mesh.area` (upper-edge scalar CV area), NOT `areasvol`
  (which `compute_hbar` used) — they are DIFFERENT arrays.** Easy to grab the wrong one
  since both are `[nod2D,nl]` CV-area fields and the surrounding code (hbar) just used
  `areasvol`. The C is explicit: `w /= mesh->area[FESOM_NODE3D(n,nz,nl)]` with the
  `if (a>0)` guard → mirror as `safe_area = where(area>0, area, 1.0)` (AD-finite; the only
  nonzero `w` lanes are `[nzmin,nzmax)` where `area>0`, so it's exact). (`fesom_ale.c:178`,
  Task 2.9.)

- **[ale] The full reverse suffix-sum `lax.cumsum(div, axis=1, reverse=True)` == the C's
  bounded `for nz=nzmax-1..nzmin: w[nz]+=w[nz+1]` loop — for free.** Because the per-level
  scatter is masked to `elem_layer_mask` and every element's layer range ⊆ its vertices'
  node range (node nlevels=MAX, ulevels=MIN over cells), `div` is already 0 at and below
  each node's bottom interface `nzmax` — so the suffix-sum naturally preserves the no-flux
  BC `w[nzmax]=0` and equals the bounded loop. Mask the final `w` with `node_iface_mask` to
  zero a cavity node's suffix-sum spill above `nzmin` (a no-op for non-cavity pi, but
  correct in general). Verified `w[nzmax]==0` exactly at every node. (`ale.compute_w`, Task 2.9.)

- **[ale/fidelity] Like `hbar`, the ÷area (1e9–1e12 m²) crushes the near-cancelling
  divergence floor — `w` matches the dump ~4e-20 on CPU (a TIGHT, hbar-class gate), not the
  loose ssh_rhs-class ~1e-7.** Even though the per-level `div` carries the same amplified
  cancellation floor as `ssh_rhs` (`dx·helem~1e7`), the reverse cumsum's partial
  cancellation + the ÷area divide it back to ~1e-20. Step 1 is a REAL gate (post-`update_vel`
  `uv` is the first wind-driven ~1e-3 velocity → `w` ~1e-6, non-trivial). Synthetic O(0.1)
  uv vs a numpy loop ref agrees to ~1e-18 (rel ~3e-16); `w` is LINEAR in `uv` so AD==central
  FD exactly and is finite at uv=0. Gate at `W_ATOL=1e-12` (hbar precedent, GPU-safe).
  (`test_ale.py`, Task 2.9.)

- **[ale] `hnode_new = hnode` bit-for-bit in linfs (dh/dt=0, a memcpy) — confirms
  `State.rest().hnode` (the `zbar_3d_n` differences) EQUALS the C's static `hnode` exactly.**
  The substep-13 node dump of `hnode_new` matched `State.rest().hnode` to max|Δ|=0 at all 5
  probes (top layer 5 m, deepest ~250 m). The `helem` recompute + `hnode = hnode_new` commit
  is `fesom_ale_commit_thickness` = substep **16** (Task 2.10), NOT substep 13 — the plan's
  Task-2.9 "and helem" wording predates the substep map; 2.9 is strictly `w`+`hnode_new`.
  (`fesom_ale.c:10-16`, `ale.thickness_linfs`, Task 2.9.)

- **[ale/config] `use_wsplit=0` in Phase 2 ⇒ `w_e = w`, `w_i = 0` (no vertical-velocity
  split).** `fesom_ale_compute_wvel_split` (`fesom_ale.c:241`) reduces to a copy when
  `use_wsplit=0` (`fesom_constants.h:56`, off for the linfs reference runs — the split was
  what seeded a Fortran day-92 blow-up). So the `w` from substep 13 IS `w_e` (read by tracer
  advection, Task 2.10) and `w_i=0` confirms the Task-2.6 `impl_vert_visc` simplification
  (`w_i=0` ⇒ advective tridiagonal terms drop). `cfl_z`/`w_e`/`w_i`/`wvel_split` have no
  substep-13 dump → ported when consumed (Task 2.10/2.11), not here. (Task 2.9.)

## Phase 2 — tracers (Task 2.10, substeps 15–16)

- **[tracers/upwind] The horizontal upwind flux's 5 level-"zones" collapse to a masked
  per-element sum.** The C (`adv_tra_hor_upw1`, `fesom_tracer_adv.c:212`) splits each edge's
  column into 5 zones (el1-only-above / el2-only-above / both / el1-only-below /
  el2-only-below) purely to walk the union of the two cells' level ranges — vectorized this
  is just `vflux = mask₁·flux₁ + mask₂·flux₂` per level (each masked to `elem_layer_mask`).
  ⚠️ The per-element flux is the **NEGATION** of `compute_w`/`compute_ssh_rhs`'s term: el1
  uses `(u·dy₁ − v·dx₁)·h`, el2 `(v·dx₂ − u·dy₂)·h`. Upwind face value
  `-½(T₁(vflux+|vflux|) + T₂(vflux−|vflux|))` (the `|vflux|` is an AD kink, finite-grad).
  (`tracer_adv.adv_flux_hor`, Task 2.10.)

- **[tracers] Advection fluxes use `ttfAB` (AB2-extrapolated), but the ALE reconstruction
  updates `values` (T).** `init_tracers_AB_one` (`fesom_tracer_adv.c:174`) computes
  `ttfAB = -(0.5+ε)·valuesold + (1.5+ε)·values` (ε=0.1) and saves `valuesold := values`. At
  **step 1** `valuesold == values` (`ic` sets `T_old=T`) ⇒ `ttfAB == T`. Functional JAX:
  `advect_one` returns `(T_new, T_old_new=T)`; the caller sets the next step's `T_old`. The
  edge-replicated `T_above = T[nz-1]` makes the unified vertical formula reproduce the C's
  surface flux `-w·T·area` at `nzmin` (½·2w·T). (`tracer_adv`, Task 2.10.)

- **[tracers/constant] ⚠️ A constant tracer is preserved EXACTLY (bit-exact 0.0 on CPU) —
  this is the discrete-continuity consistency, and the reason `S=35` is the clean step-1
  gate.** The vertical divergence (via `w` = reverse-cumsum of the horizontal transport
  divergence ÷area, so `w·area` reconstructs that very divergence) and the direct horizontal
  edge scatter **cancel bit-exactly** because both reuse `ops.scatter_add` on the same edges
  with exactly-negated per-edge values. (GPU may leave ~1e-12 if the two `segment_sum`s
  reassociate differently — gate `S` at `kind="scatter"`.) (`test_tracers.py`, Task 2.10.)

- **[tracers/dump] ⚠️ The C dump runs FCT; this port runs UPWIND. `S=35` (constant) matches
  the dump bit-for-bit; `T` (the blob) differs by ~3e-7 = the limited antidiffusive flux —
  the tight `T` match is a Phase-4 (FCT) gate.** So gate `S` vs the dump (tight), verify
  upwind `T` against an independent numpy loop reference (bit-exact) + the constant-tracer
  property, and only *bound* `T` vs the dump (`< 1e-5`). This **corrects** the REFERENCE_RUNS
  "at step 1 the field is horizontally constant" claim — the **T-blob is not constant**; only
  `S` is. (`test_tracers.py`, Task 2.10.)

- **[tracers/diff] The vertical tracer diffusion is `impl_vert_visc`'s per-NODE 1-unknown
  sibling**, with two differences: an extra `area[iface]/areasvol[layer]` geometric ratio on
  the off-diagonals, and a **`hnode_new` mass diagonal** (`b = -a - c + hnode_new`, vs
  momentum's `+1`). Phase-2 reductions (all verified): `gm=NULL` ⇒ no Redi `K33`; `do_wimpl=0`;
  `bc_surface=0` (analytical forcing ⇒ zero heat/water/virtual-salt/relax-salt flux); `sw_3d=0`
  (`USE_SW_PENE` is gated on `use_jra`, off for analytical — `fesom_main.c:992`); full-cell linfs
  ⇒ `Z_n=Z`. Conserves `Σ areasvol·hnode·T` to ~1e-16; reuse `ops.tdma`.
  (`tracer_diff.impl_vert_diff_one`, `fesom_tracer_diff.c:85`, Task 2.10.)

- **[tracers/diff/AD] ⚠️ The `Z`-padding's exact 0 poisons `d/d(Kv)` (0·inf = NaN) — replace
  `dZ==0` with 1.** `Zp = concat([Z, Z[-1:]])` makes `dZ_dn[nl-2] = Z[nl-2]−Z[nl-2] = 0` at an
  always-masked lane; `c_full = …/dZ_dn` is `inf`/`NaN` there, which the forward `where`
  masks but whose **infinite local derivative × the where's 0 cotangent = NaN** in the
  backward pass (the masked value is finite, the *gradient* is not). Fix:
  `dZ = where(dZ==0, 1, dZ)` so the masked lanes are finite both ways. Same class as the eos
  unused-N²-level trap; `impl_vert_visc` has the same latent pattern but its `bot`-zeroing
  dodges it. Diffusion is linear in T (AD==FD) and `d/d(Kv)` matches FD where resolvable
  (FD underflows at the ~1e-5 gradient entries). (`tracer_diff`, Task 2.10.)

- **[ale] `commit_thickness` (substep 16) = `hnode:=hnode_new` + `helem = ⅓Σ_vertices hnode`.**
  Both static in linfs: the `hnode` node dump is bit-for-bit (like substep-13 `hnode_new`) and
  the recomputed `helem` equals `State.rest().helem` exactly. (`fesom_ale.c:18`,
  `ale.commit_thickness`, Task 2.10.)

## Phase 2 — assembled step() (Task 2.11, GATE 2)

- **[step/warmstart] ⚠️ The CG warm-start measures the residual against the ORIGINAL
  `‖ssh_rhs‖`, NOT the deflated `‖b_eff‖` — this is the step-≥2 fidelity the ssh lesson
  deferred to 2.11.** `solve_ssh` folds the warm start into `b_eff = ssh_rhs − A·x0` and
  solves δ from 0, but the C's early-stop threshold is `soltol·‖ssh_rhs‖`. Since the inner
  residual `b_eff − A·δ_k` equals the full residual `ssh_rhs − A·(x0+δ_k)`, passing
  `rtol_abs = soltol·‖ssh_rhs‖/√N` to the inner PCG replicates the C's warm-started
  early-stop exactly (a good warm start ⇒ `b_eff` already below threshold ⇒ 0 iters ⇒
  `d_eta=x0`); deriving rtol from `‖b_eff‖` over-converges. **Verified LOAD-BEARING:** step-2
  `d_eta` matches the dump 3–3000× better warm-started than from `x0=0`. (`ssh.solve_ssh`
  `rtol_abs`, Task 2.11.)

- **[step/multistep] ⚠️ A TIGHT multi-step dump match is impossible with upwind — `T`
  diverges ~3e-7 at step 1 (upwind vs the dump's FCT) and cascades via `density` into every
  T-dependent field at step ≥2** (density ~6e-8 → momentum/SSH ~1e-10, `ssh_rhs` ~1e-2 after
  the `dx·helem~1e7` amplification). So **step 1 is the tight integration gate** (one `step()`
  reproduces ALL per-kernel substep gates at the probes — confirms the order + threading),
  and step ≥2 is gated by INVARIANTS instead: `S` stays **exactly 35** (constant-tracer
  preservation — a sensitive AB2/threading check, a bug corrupts it), rest-state to machine
  precision, climate-close SSH/velocity, 100-step stability. The tight multi-step `T/S` match
  is a Phase-4 (FCT) gate. (`test_step_pi.py`, Task 2.11.)

- **[step/threading] The between-step bookkeeping (the whole point of 2.11):** `hbar_old`
  saved before `compute_hbar` overwrites `hbar`; `d_eta` carried as the next CG warm-start
  (**never zeroed between steps** — `fesom_main.c:570`'s `memset(d_eta)` is a one-time
  `do_sanity` CG self-test, NOT the time loop); `uv_rhsAB` (momentum) and `T_old`/`S_old`
  (tracers, from `advect_one`) are the AB2 histories; `eta_n`/`w_e` feed `compute_vel_rhs`
  **lagged** (previous step's). `is_first_step` only flips the AB2 `ff_step` (1.0 vs 1.6).
  (`step.step`, `fesom_step.c`, Task 2.11.)

- **[step/rest] Rest state (constant T/S, NO blob, zero wind) stays at rest to machine
  precision** (`max|uv|`~2e-16 after 5 steps; T/S exactly constant). Constant T/S ⇒
  horizontally constant density (depth-varying but identical per column) ⇒ PGF=0 ⇒ no flow;
  advection/diffusion of a constant field = 0. The fundamental no-spurious-flow gate — use a
  **zero** `stress_surf` (the analytical wind is nonzero). (`test_step_pi.py`, Task 2.11.)

- **[step/jit] XLA FMA-contracts the EOS density polynomial ⇒ the jitted step's `density`
  shifts ~1e-13 from the eager bit-exact value — past the `map` gate (1e-14).** So the TIGHT
  bit-exact step-1 gates run on **eager** `step()`; the jitted `step_jit` (the production /
  `lax.scan` entry, `static_argnames=(dt, is_first_step)` ⇒ 2 compiled variants) matches eager
  to ~1e-12 (FMA level), which is fine for the loose multi-step/stability gates. 100 steps:
  `max|uv|`~0.075, `|eta|`~0.35 m, no NaN, `S` exactly 35. (`step.step_jit`/`run`, Task 2.11.)

## Phase 3 — AD smoke test (Tasks 3.1/3.2, GATE 3)

- **[ad/eos] ⚠️ The `bvfreq` (N²) bottom-padding `1/zdiff` is a BACKWARD-ONLY NaN trap —
  the forward gate passed for two phases while the gradient was NaN.** `zdiff = Zd − Zp` is
  exactly 0 at **two** unused interfaces: the surface (`k=0`, edge-replicated) AND the
  **bottom padding** (`Zp = concat([Z, Z[-1:]])` duplicates `Z[-1]` in its tail ⇒
  `zdiff[bottom]=0`). `1/zdiff=inf` there ⇒ `bv[:,bottom]=inf` in the *forward* pass, but the
  output `take_along_axis(bv, clip(k,lo,hi))` **clips those lanes away**, so the forward
  bvfreq (and every Phase-0..2 gate) is correct. The backward pass, however, computes
  `0·inf = NaN` (the masked lane's 0 cotangent × the inf local derivative) and it flows to
  `d(loss)/d(T)` at exactly the `nl-2`/`nl-1` columns of *every* node (= 2·nod2D = 6280 lanes
  on pi). The old fix `zdiff.at[0].set(1.0)` only patched the surface. Fix:
  `zdiff = where(zdiff==0, 1, zdiff)` (covers both; forward unchanged — the lanes are clipped
  out). Same class as `tracer_diff`'s `where(dZ==0,1,dZ)` and the eos unused-`nz=0` trap; the
  rule "make masked-off lanes compute a FINITE value, don't trust the forward mask to hide a
  backward NaN" bit a THIRD time. (`eos.pressure_bv`, Task 3.2.)

- **[ad/method] ⚠️ `d/d(scalar param)` being finite does NOT prove `d/d(field)` is finite —
  the IC-field gradient is the strictly stronger masked-NaN probe.** `d(loss)/d(k_ver)` was
  finite (and FD-correct!) while `d(loss)/d(T₀)` was NaN, because the NaN lived in the
  `T₀→eos→bvfreq` backward at the masked lanes, and `k_ver` enters **additively** downstream
  (`Kv = mix·factor³ + k_ver` ⇒ `d/dk_ver` only needs `d(loss)/d(Kv)`, never `d(Kv)/d(bvfreq)`),
  so it never traverses the poisoned sub-path. Earlier per-kernel grad checks differentiated
  w.r.t. T at a *single wet node* and missed it. **Always include a `grad` w.r.t. a full IC
  field (incl. the below-bottom padding) — it is the test that catches these.** (Task 3.2.)

- **[scan/checkpoint] `integrate` (run step 1 eagerly with `is_first_step=True`, then
  `lax.scan` steps 2..N with `is_first_step=False` baked in) == the Phase-2 `run` loop
  BIT-IDENTICAL, and `jax.checkpoint` is forward-transparent (on==off exactly).** The
  `is_first_step`-outside-the-scan pattern keeps the scan body uniform with no traced bool;
  closing over the loop-invariant `mesh`/`op`/`stress_surf`/`params` keeps the carry minimal
  (just `State`). `scan(jax.checkpoint(body))` differentiates correctly for closed-over
  tracers (`scan` hoists them as consts and sums their per-step cotangents), so
  `d(loss)/d(params)` accumulates over the window. Forward `integrate==run` to ~4e-19 (uv),
  0.0 elsewhere; checkpoint on/off forward Δ = 0.0; same gradient with/without checkpoint.
  (`integrate.py`, Task 3.1.)

- **[ad/fd] The end-to-end FD floor is set by the loss's INTERMEDIATE-SUM magnitude (mean
  SST ~10 ⇒ ~`eps·10` round-off), NOT by AD accuracy — and the plateau is at LARGE `h`.**
  `d(mean SST)/d(k_ver)` is ~−2.6e-3, so the FD signal `g·2h·k₀` at `k₀=1e-5,h=1e-4` is
  ~5e-12 vs a ~3e-15 round-off floor (SNR ~1700 ⇒ rel ~1.5e-4, marginal). Because the loss is
  very *smooth/near-linear* in `k_ver` (tiny truncation), the sweep's best `h` is the LARGEST
  (1e-3), and rel error *grows* as `h→0` (round-off): `[h=1e-3→6e-7, 1e-4→1e-5, 1e-5→1e-4,
  1e-6→2e-3, 1e-7→5e-3]`. Two robust levers (both keep the gradient identical): evaluate at a
  larger background `k_ver` (1e-4 ⇒ plateau 6e-7; 1e-3 ⇒ 4e-7) to lift the signal off the
  floor, and/or a longer window. Subtracting a constant from the loss does NOT help (the
  round-off is baked into the mean before the subtract). So the gate asserts the **plateau**
  (min over the `h`-sweep) at a signal-lifted `k_ver=1e-4`, and only checks finite+sign+loose
  -FD at the physical `k_ver=1e-5`. (`test_gradient.py`, Task 3.2.)

- **[ad/ml-hook] The differentiable-parameter seam = a `Params` pytree threaded
  `step(...,params) → pp.mixing_pp(...,k_ver,a_ver)`, with `params=None ⇒ Params.defaults()`
  (the config constants) — numerically transparent (the 274-test suite stays bit-identical).**
  This is the first concrete ML-hook (Phase 7 swaps the PP mixing for an NN here; its weights
  join `Params`). `k_ver` routes through the CG `custom_linear_solve` *across* steps
  (k_ver→Kv→diffusion→T→[next step]density→…→ssh_rhs→CG); `a_ver` routes through it *within* a
  step (a_ver→Av→impl_vert_visc→du→ssh_rhs→CG) — both FD-confirmed, so the implicit-diff
  transpose solve is proven on the assembled model. (`params.py`, `pp.py`, Task 3.2.)

- **[scan/memory] Checkpointing is LOAD-BEARING for the backward pass — N=200 pi backward is
  4.23 GB checkpointed vs 48.7 GB (OOM on A100-40) without.** Reverse-mode through the N-step
  loop needs O(N · per-step intermediates) un-checkpointed; XLA's `hlo_rematerialization`
  couldn't get it below 28 GiB and tried to alloc 48.7 GiB → `RESOURCE_EXHAUSTED`. With
  per-step `jax.checkpoint` it is O(N · `State` carry) ≈ 4.23 GB (13% of the A100-40), compile
  +run 26 s. For *much* longer windows switch to nested/policy checkpointing (O(√N)); per-step
  remat suffices to N≥200. (`scripts/debug/phase3_grad_memory.py`, GPU job 25378918, Task 3.1.)

- **[xla] The host-assembled static scatters (mesh indices baked as constants) trigger XLA
  constant-folding warnings (`scatter-add … taking > 2s`) at compile — benign, ~5 s each.**
  The grad-of-scan compile constant-folds a few `f64[3140,48,2]` scatter-adds with constant
  index operands; it's a compile-time/runtime trade-off, not a correctness issue (the run is
  correct and fast). Ignore the `slow_operation_alarm` lines. (GPU job 25378918, Task 3.1.)

## Phase 4 — FCT (Task 4.1, GATE-4 forward)

- **[ic] ⚠️⚠️ THE bug that ate the session: the C's step-1 `valuesold` (AB2 `T_old`) is the
  **pre-blob base constant T=10**, NOT the blob field.** `fesom_ic_tracers_constant` sets only
  `values` (`fesom_ic.c:62`); `valuesold` is `calloc`'d to 0 (`fesom_tracers.c:17`). Then
  `fesom_main.c:721` runs a rest-state `advect_one(T)` *sanity check* whose `init_tracers_AB_one`
  saves `valuesold = values = 10` — and ONLY THEN (`:748`) is the blob added to `values`. So at
  step 1, `T_old=10` (base), `values=10+blob`, and the AB2 extrapolation is
  `ttfAB = -(0.5+ε)·10 + (1.5+ε)·(10+blob)`, **not** `ttfAB = values`. Our `ic.initial_state`
  set `T_old=T` (the blob) — wrong. **This was mis-attributed for two phases as the
  "upwind−FCT gap" (~3e-7).** It contaminated BOTH upwind and FCT at step 1; the FCT `T` dump
  match jumps from 3.4e-7 → **1.8e-15** once `T_old` is the base. `S` (constant) is insensitive
  (a constant tracer is preserved for any `S_old`), which is why `S=35` matched all along and
  hid the bug. Fix: `ic.py` sets `T_old`/`S_old` to the masked pre-blob base. **Lesson: at step
  1, `T_old` need not equal `T`; chase a "gap" to its first principles before labelling it a
  scheme difference.** (`ic.py`, Task 4.1.)

- **[verify/method] ⚠️ A "faithful port" that matches your own numpy reference can STILL be
  wrong — they can share an INPUT bug. Dump the C's intermediates.** JAX FCT == numpy-C-ref
  FCT to ~1e-10 (stage-by-stage), and BOTH disagreed with the dump by 3e-7. The shared error
  was not in the formula (verified the qr4c against the C *and* the Fortran `oce_adv_tra_ver.F90`
  line-for-line — identical) but in the `ttfAB` *input* (the `T_old` bug above). The decisive
  move: add temporary `fesom_dump_record_node` calls in the C for the FCT intermediates
  (`fct_LO`, `adv_flux_ver` after the limiter) right after the T-advection (before S overwrites
  the scratch), rebuild, run with `FESOM_NO_TRDIFF=1` (isolates advection from diffusion). The
  C's `adf_v[surface] = -5.6e4` while ours was 0 (at step 1 `ttfAB=values` ⇒ surface adf=0) —
  working backward `adf=-(ttfAB-values)·W·area` gave `T_old=10`, cracking it. (Task 4.1.)

- **[fct] The FCT structure: `T_new = LO + limited(HO − LO)`. LO fluxes use `values` (T), HO
  uses `ttfAB`, the element/up-dn gradient uses `values`.** Driver `fesom_tracer_advect_one_fct`
  (`:1199`): (2) LO upwind fluxes from **values** (NOT ttfAB — unlike the upwind-only driver);
  (3) `compute_fct_LO` = the upwind ALE solution; (4) HO with `init_zero=0` ⇒ `adf := HO − LO`
  (horizontal MFCT 3rd-order `num_ord=0`, vertical QR4C 4th-order `num_ord=1`); (5) Zalesak
  limit; (6) `flux2dtracer_fct` adds the LO transition `-T·hnode + LO·hnode_new` + the limited
  antidiff divergence; (8) reconstruct. The algebra collapses to `T_new = LO + antidiff_div/
  areasvol/hnode_new`. For pi: no cavities (`ulevels≡1`), single rank (`myDim_edge2D==edge2D`).
  (`tracer_adv.advect_one_fct`, Task 4.1.)

- **[fct] A constant tracer is preserved by FCT (the gradients vanish ⇒ HO==LO, antidiff=0).**
  `tracer_gradient_elements` of a constant is 0 (∑∂N/∂x = 0, partition-of-unity), so the MFCT
  reconstruction `Tmean1=Tmean2=const` ⇒ HO flux == LO flux ⇒ `adf=0`, and the limiter clips
  nothing ⇒ `T_new = LO = const`. This is why `S=35` is the clean bit-for-bit FCT gate, exactly
  as in upwind. (Task 4.1.)

- **[fct/AD] The Zalesak limiter is differentiated as a SUBGRADIENT (option a) — finite & NaN
  -safe because the C's `flux_eps=1e-16` floors every limiter ratio.** `min(1, fct_ttf_max/
  (fct_plus·dt/area/hnode + flux_eps))` is always finite (`fct_plus≥0` ⇒ denom `≥ flux_eps`);
  when the antidiff flux vanishes the ratio is large-but-finite and `min` picks the constant 1
  with a 0 cotangent on a *finite* value — no `0·inf`. The `±bignumber=±1e3` padding (a2) is
  finite, so the `max`/`min`/`segment_max`/`segment_min` reductions never see inf. So unlike the
  CG (which needed a separate tight `transpose_solve`), the limiter needs NO special AD
  machinery — the forward `flux_eps` is the whole fix. Plateau `d(SST)/d(k_ver)` = 5.7e-7
  (≈ the upwind 5.9e-7 — the limiter is inactive in the smooth blob, so the subgradient == FD
  there). Decision + rejected alternatives (b smooth-relax, c stop_gradient) documented in
  `docs/LIMITER_GRADIENTS.md`. (`tracer_adv.zalesak_limit`, Task 4.1.)

- **[fct/qr4c] The vertical 4th-order QR4C `Z`-stencil denominators vanish at the bottom-pad
  level (`Zp=concat([Z,Z[-1:]])` ⇒ `Z[nz-1]−Z[nz]=0`) — guard with `where(d==0,1,d)` (the
  recurring masked-divide rule, 4th time).** The masked-out interior formula is unused forward
  but `0·inf=NaN` in the backward pass without the guard. Same class as the eos `bvfreq` and
  `tracer_diff` `1/zdiff` traps. (`tracer_adv._z_stencil`/`adv_flux_ver_ho`, Task 4.1.)

- **[fct/method] Test the limiter where it's ACTIVE — the dump's smooth step-1 leaves it
  inactive.** The real blob is well-resolved ⇒ no overshoots ⇒ the limiter clips nothing ⇒ the
  dump verifies only the HO flux, not the min/max/sign-select limiter logic. Added a synthetic
  **sharp tracer + ×5000 velocity** test vs the numpy FCT reference that forces the limiter to
  bind — the strong check for the limiter branch. (`test_FCT_limiter_active_vs_numpy_reference`,
  Task 4.1.)

- **[fct/diff] Fixing `T_old` also closed the deferred Phase-2 tight multi-step `T/S` gate AND
  dump-verified the blob diffusion.** With FCT + the IC fix, `step()`'s substep-15 `T` matches
  the committed dump to 1.8e-15 (was the "Phase-4 deferred" gate), and the step-2 SSH fields
  (`d_eta`/`hbar`/`eta_n`) — which cascaded the old `T` error through density — now match to
  <1e-11 (was gated loose at 1e-7). The vertical tracer diffusion on a *non-constant* field
  (only ever property-tested in Phase 2) is now confirmed correct by the tight `T` dump match.
  (`test_step_pi.py`, Task 4.1.)

## Phase 4 — opt_visc7 verify + wsplit (Task 4.2)

- **[verify/method] ⚠️ "Ported AND tested" ≠ "every coefficient regime tested" — check the
  branch-selection statistics, not just that the test is green.** The opt_visc=7 flow-aware
  biharmonic was fully ported in Task 2.5, but its flow-aware branch was effectively unverified:
  a diagnostic (run `step()` 10 steps, count edges with `max(γ1·|du|, γ2·|du|²) > γ0`) showed
  **0 flow-aware-active edges at every dump step** — pi's edge-velocity differences grow only
  8e-5→8e-4, all ≪ the |du|>0.03 γ1-onset, so the dump can ONLY ever test the constant-γ0
  biharmonic. The existing `_synthetic` test (uv amp 0.1) *did* reach the γ1 branch (51% of
  edges) but **never** the quadratic γ2 (needs |du|>γ1/γ2=0.351; synthetic max 0.219). Moral:
  when a kernel has data-dependent branches, instrument which branch the test inputs actually
  exercise; a passing test over a too-mild input silently skips a code path. Fix: a strong-flow
  (~2 m/s) synthetic test that binds BOTH branches, with an explicit `assert g2_wins.sum() > 0`.
  (`test_momentum.test_visc_filter_flow_aware_branches_vs_reference`, Task 4.2.)

- **[config] `use_wsplit=0` in the pi/CORE2-d1800 reference config (`fesom_constants.h:56`), so
  `w_e=w, w_i=0` IS the dump-matching path — porting the split is CORE2-readiness, not pi
  correctness.** The vertical-velocity CFL splitter was disabled in the reference runs (it seeded
  a Fortran day-92 barotropic blow-up). Two consequences: (1) the step-1 substep-15 `T` 1.8e-15
  match already *proved* `w_e=w` (tracer advection reads `w_e`); (2) pi's max `cfl_z`~1e-4 ≪
  maxcfl=1.0, so the split would be the identity even if turned on at pi velocities. So
  `compute_wvel_split` is ported faithfully but its active branch is verified only via a
  synthetic super-critical CFL vs the numpy ref — wiring it into `step()` is numerically
  transparent (every gate + the gradient plateau 5.70e-7 unchanged). ⚠️ The implicit part `w_i`
  feeds `impl_vert_visc`'s advective tridiagonal terms, which the Phase-2 kernel drops under the
  `w_i=0` simplification — re-enabling those is a Phase-5 item, gated on `use_wsplit=1`.
  (`ale.compute_wvel_split`, `step.py`, Task 4.2.)

- **[ale/cfl] `cfl_z` is an interface field built from BOTH adjacent layers' `|w|·dt/h` — the
  "above" layer must be ZERO-padded at the surface (not edge-replicated).** The C accumulates
  per layer onto its top (`+=|w[nz]|·dt/h[nz]`) and bottom (`+=|w[nz+1]|·dt/h[nz]`) interfaces,
  so `cfl_z[i] = |w[i]|·dt·(1/h[i] + 1/h[i-1])` with the surface/bottom interfaces getting one
  term. Vectorized: `below = inv_h` (layer i), `above = shift_down_zero(inv_h)` (layer i-1, 0 at
  i=0 since no layer is above the surface). Using the momentum `_shift_down` (edge-replicate)
  here would wrongly double-count the surface layer. The `dt/h` divide is AD-guarded with the
  usual `where(h>0, h, 1)` masked-finite pattern. (`ale.compute_cfl_z`, `fesom_ale.c:204`, Task 4.2.)

- **[verify/method] A rest-trivial element gate becomes a real gate at step ≥2 once the
  trajectory is dump-tight.** The substep-6 `uv_rhs` (viscosity) dump gate was trivial at step 1
  (uv=0 ⇒ the biharmonic adds nothing, substep6==substep5). With FCT making the multi-step
  trajectory tight (Task 4.1), reconstructing substeps 1–6 from the post-step-1 state and
  comparing to the step-2 dump gives a real end-to-end viscosity gate — matched **~1e-17**
  (gather class; the small wind-driven velocities keep it in the constant-γ0 regime, ~1e-9
  viscous contribution). General: deferred "trivial-at-rest" element gates unlock at the first
  nonzero-flow step *after* the trajectory is verified tight against the dump.
  (`test_step_pi.test_step2_uv_rhs_visc_matches_dump`, Task 4.2.)

- **[step/stability] pi 1000 steps (dt=100, full physics) is stable in ~48 s, and the
  vertical CFL stays ≪ maxcfl over the whole window (max `cfl_z`=2.8e-3 ≪ 1.0) — so the
  use_wsplit=0 config is self-consistent long-window, not just at the dump's 10 steps.** The
  jitted `run` amortizes compile so 1000 steps cost ~2.5× the 100-step test, not 10× (~48 s vs
  ~20 s). Over 1000 steps: no NaN, max|uv|=0.17, max|eta|=0.63 m, **S exactly 35** (bit-exact —
  the strongest long-window AB2/threading regression guard), T∈[10.0,14.98]. The AD gate is NOT
  re-run at 1000 steps (the model is mildly chaotic via scatter reassociation — long windows sit
  on the FD chaos floor; `test_gradient.py` stays at N=20 by design). "Climate-close to C" at
  1000 steps stays **indirect** (no C 1000-step pi snapshot — the dump is 10 steps); the tight
  step-1..10 FCT dump match + S-exact + boundedness are the stand-in. (`test_step_pi.py`, Task 4.3.)

## Phase 5 — CORE2 (scoping + pre-port guards)

- **[scope] ⚠️ The C port is a DELIBERATELY SIMPLIFIED FESOM: linfs-only, full-cell, no
  cavities — match THAT, not real-FESOM/Fortran features the parent outline mentions.** The
  parent plan's Phase-5 outline listed "zlevel ALE / local-zstar / partial cells / re-enable
  w_i" — **none exist in the C port** (`fesom_ale.c` is linfs-only; the zlevel algorithm is
  only in the Fortran `oce_ale.F90`; `fesom_mesh.c:617-634` sets `zbar_3d_n[n,nz]=zbar[nz]`
  with no `Z_3d_n`; `use_wsplit=0` + FCT ⇒ `w_i≡0`). FRESH_START §14.7 even says
  `which_ALE='zlevel'` **"but we will use linfs."** Per the golden rule the C port governs:
  **Phase 5 = pi physics (PP/linfs/FCT/opt_visc7) on the CORE2 mesh + PHC IC + JRA55/SSS/runoff.**
  zstar/partial-cells are future and need C-side work first. *Lesson: when the plan outline
  and the C reference disagree, the C reference wins — verify against it before proposing
  scope, don't propagate the outline.* (Caught when a research read showed `fesom_ale.c`
  linfs-only; user flagged the zlevel error directly.)

- **[mesh/orientation] ⚠️ Triangle orientation is the pi↔CORE2 trap — CW is a CHECKED
  load-time invariant now, not an assumption.** The C `orient_cw` (`fesom_mesh.c:430-459`)
  computes `r = bx*cy − by*cx` (cyclic-wrapped) and swaps v2↔v3 whenever `r>0`, forcing
  **every** triangle CW, and runs at `fesom_mesh_read:1193` **before** any geometry is
  derived (`elem_area`@1219, `gradient_sca`@~1230). So both pi and CORE2 export in the same
  CW convention; `elem_area` is `abs` (orientation-free) and `edge_cross_dxdy` is
  centroid-based (orientation-free) — the only orientation-sensitive exported array is
  `gradient_sca` (post-swap ⇒ CW). CORE2's RAW mesh is ~all CCW (~244654/244659 swapped,
  FRESH_START §4); historically a missing swap ⇒ wrong SSH-stiffness sign ⇒ the
  Aleutian-Trench blow-up (§11/§14.8). Added `mesh.check_cw_orientation` + a `load_mesh`
  guard (raises on any `r≥0`) + tests; **pi verified 5839/5839 CW**. This makes a bad CORE2
  export fail loudly at load (Task 5.1) instead of diverging mid-run. (`mesh.py`,
  `test_mesh.py`, this session.)

- **[scope/discipline] ⚠️ Don't invent a "modeling choice" where the rule is "port the C
  exactly."** Phase-5 SSS/runoff was first framed as "match C-literal vs FRESH_START §9's
  shorthand" — a false choice (same error-class as the zlevel slip). The C `fesom_sss_runoff.c`
  mirrors the Fortran sbc and is validated (no SSS problems), so the discipline is a faithful
  1:1 port gated by the dump; §9's `water_flux += (S−Sclim)·v` / `−= runoff` is a *simplified
  description*, not an alternative. (In the no-ice Phase-5 path runoff enters only via the
  global-mean balance — the local term is in ice thermo, off here; the dump gate confirms
  JAX == the C-port-no-ice run.) *Lesson: FRESH_START is a description; the C port is the
  spec. No menu — port it and verify by dump.* (Task 5.5; user flagged.)

- **[mesh/CORE2] The CORE2 mesh port was genuinely ZERO JAX-code — the design held.**
  `load_mesh('data/mesh_core2')` worked unchanged: it reads `nl` from `meta.txt` and the four
  ragged masks already encode per-node variable depth (the only real pi→CORE2 mesh
  difference); full-cell ⇒ global `zbar`/`Z` stays valid so eos/ssh/pp/ale need nothing.
  `test_mesh_core2.py` (12) reuses the pi structural invariants verbatim + pins CORE2 counts
  (nod2D=126858, elem2D=244659, edge2D=371644, nl=48). The export at `npes==1` is cheap
  (job 25386129: 17 s, peak **5.6 GB** — the 32 G request was overkill, 8 G would do; NL is
  read from `aux3d.out`, not compile-time, so the same `build/fesom_port` exports pi and
  CORE2). (Task 5.1.)

- **[mesh/orientation] Empirical confirmation: CORE2 `orient_cw` swapped 244654/244659
  elements to CW** (job log) — exactly FRESH_START §4. So CORE2's raw mesh really is ~all
  CCW and the C normalization is load-bearing; the `check_cw_orientation` guard added to
  `load_mesh` re-verifies it survived export→load (CORE2: all 244659 CW). This is the
  concrete payoff of making CW a checked invariant rather than an assumption. (Task 5.1.)

- **[perf] ⚠️ An EAGER `step()` on CORE2 is ~32 s/step on CPU (~160× pi for ~40× the nodes
  — super-linear, the CG + eager/host-scatter overhead).** So CORE2 rest-state/smoke tests
  use a small step count, and real CORE2 correctness/stability work (Task 5.7) must use the
  **jitted** `run`/`integrate` (amortized compile) and/or GPU — not eager. `build_ssh_operator`
  itself is cheap on CORE2 (0.3 s; host scipy COO for ~1.5M entries). The CORE2 rest-state
  gate (`test_step_core2.py`) confirms `step()` produces no spurious flow on the big mesh
  (max|uv|=1.8e-14, T/S bit-exact) — but note PGF=0 at rest doesn't test the `gradient_sca`
  *sign* (constant field ⇒ Σ∂N=0 regardless); that's exercised by the non-rest dump gate in
  Task 5.7. (Task 5.1.)

- **[env] The fesom-jax env had NO Python NetCDF reader (netCDF4/xarray/h5py all missing);
  the PHC + forcing files are NetCDF-4/HDF5 (scipy.io can't read them).** Installed
  **netCDF4 via the env pip** (user-approved): `…/envs/fesom-jax/bin/python -m pip install
  netCDF4`. numpy (2.4.6) and jax (0.10.1, x64) **unchanged**. ⚠️ A benign
  `RuntimeWarning: numpy.ndarray size changed … Expected 16 … got 96` appears on `import
  netCDF4` (its wheel was built against an older numpy ABI) — harmless, the data path is
  correct (PHC matched the C to ~1e-14). Avoided **numba** (it pins numpy and could break
  jax) by doing the GS extrap in optimized pure Python. (Task 5.2.)

- **[phc/ic] The PHC IC is a faithful numpy port that matches the C to ~1e-14 (MAP class —
  no scatter, so near-bit-exact like the EOS).** `phc_ic.load_phc_ic` mirrors `fesom_phc.c`
  for npes=1/no-cavity: cyclic-pad lon, per-node bilinear bracket (`binarysearch_d` ported
  literally — bracket indices match the C **exactly**), bilinear-horizontal + linear-vertical
  interp onto `mesh.Z`, then `extrap_nod3D` + vertical fill + `ptheta`. The IC is
  **non-differentiable setup** (host numpy, not JAX), cached to `data/ic_core2/{T,S}_ic.npy`;
  `core2_initial_state` injects it via `dataclasses.replace`. (Task 5.2.)

- **[phc/extrap] ⚠️ The land-extrapolation is SEQUENTIAL Gauss-Seidel (order-dependent) and
  must be replicated as such — a Jacobi pass gives different values.** Each dummy ocean node
  is filled ONCE with the mean of neighbours valid *at fill time*; a node filled earlier in a
  sweep (lower index) is visible to later nodes that same sweep (`fesom_phc.c:318-342`). The
  faithful + fast port: per layer, collect dummy nodes in **ascending index order**, sweep
  updating the column **in place**, drop filled nodes, repeat until no progress (no numba
  needed — only the few-thousand coastal dummies are iterated). Post-load surface matched the
  C to ~1e-14 ⇒ the GS order was reproduced exactly. (Task 5.2.)

- **[phc/verify] The C `phc_dump_*` is SURFACE-ONLY — the vertical interp + deep `ptheta`
  are NOT directly gated by it.** `phc_dump_preextrap` (gid,T,S,bilin_i,bilin_j,lon,lat) and
  `phc_dump_postload` (gid,T,S) cover only level 0 (where `ptheta`'s pressure ≈|Z[0]|~2.5 m is
  near-zero). So `test_phc_ic` gates the surface tightly + checks the full field is physical;
  the deep column is verified indirectly by the Task-5.7 per-substep density/EOS gate. Add a
  full-column C dump for a few probes if 5.7 shows a depth mismatch. Also: `T_old`/`S_old`
  (step-1 AB2 history) is provisionally = the PHC field; the exact `valuesold` is finalized in
  5.7 against the dump (cf. the pi `T_old` base-vs-blob lesson). (Task 5.2.)

## Phase 5 — JRA55 forcing reader (Task 5.3)

- **[jra/fidelity] ⚠️⚠️ THE Task-5.3 trap: the C time-interp `field = rdate·coef_a + coef_b`
  CATASTROPHICALLY CANCELS, so a ~1e-13 reassociation in the bilinear gather blows up to ~1e-8
  in the interpolated field — the gather must be BIT-IDENTICAL to the C, not just "1e-13
  close."** `nc_time` is in **Julian days since year 0001** (~2.436e6 for 1958), and
  `coef_b = d1 − coef_a·nc_time[t0]`, so `field = rdate·coef_a + coef_b` subtracts two ~2.4e6
  numbers to get an O(1) result — a ~`coef_a·nc_time·eps` ≈ `164·2.4e6·2.2e-16` ≈ 1e-7 abs
  rounding floor. BOTH C and JAX incur it, but they land on *different* sides unless `d1`/`d2`
  (hence `coef_a`/`coef_b`) are bit-identical. My first gather folded `1/denom` into the weights
  (`Σ wₖsₖ`, `wₖ=dxₖdyₖ/denom`) — algebraically equal to the C's `(Σ sₖ·dxₖ·dyₖ)/denom` but
  ~1e-13 off by reassociation → the interp field came out **~6e-8** off the C (and the error
  *correlated with* `|coef_a|`, max error at the max-`coef_a` node — the smoking gun). Fix:
  compute each corner term as **`(s·dx)·dy` in the C's multiply order**, sum A→D left-to-right,
  **divide the sum by `denom` at the end** (store per-corner `dx,dy` + a per-node `denom`, not a
  folded weight). Result: the 6 scalar fields are **bit-exact (max|diff|=0 over all 126858
  nodes, both dates)**; only the wind carries ~3.5e-15 (the g2r `sin`/`cos`). *Lesson: when a
  downstream formula cancels large numbers, the usual "fold the constant in" optimization is
  WRONG — the division placement is load-bearing; replicate the C's exact op order.*
  (`jra55._build_stencil`/`_gather`, `fesom_jra55.c:480-516`, Task 5.3.)

- **[jra/method] The verification recipe that surfaced it: a step-1 *boundary* dump alone is a
  WEAK gate — add an *interior* dump that exercises genuine time interpolation.** At (day1,sec0)
  `rdate < nc_time[0]` ⇒ the `t_indx` boundary branch sets `coef_a=0`, so `field = d1` (just the
  bilinear gather, no cancellation) — it matched at ~1e-13 even with the folded-weight gather,
  hiding the bug. The (day100, 12:00) interior dump (genuine 2-slice interp + a `getcoeffld`
  cache refresh) is where the cancellation bites and the ~6e-8 error showed. The C dump job
  writes BOTH (`FESOM_JRA_DUMP_DIR` + `FESOM_JRA_DUMP_DAY`/`_SEC`). General: for any
  time/space-interpolated reader, gate at an *interior* point, not just the t=0 boundary where
  the scheme degenerates. (`jobs/jax_jra_dump_core2.sh`, `dump_jra_fields` in `fesom_main.c`,
  Task 5.3.)

- **[jra/traps] Three literal-parity traps the port had to honor (all dump-confirmed).**
  (1) **Field order is uas,vas,huss,rsds,rlds,tas,prra,prsn** — `tas` (air temp) is the **6th**
  field, NOT 3rd (`fesom_jra55.h:50`); a naïve alphabetical/physical ordering silently swaps
  T_air with humidity. (2) **Interp on GEOGRAPHIC coords, rotate the wind AFTER** — the bilinear
  bracket uses `geo_coord_nod2D/RAD` (deg, only a `<0`→`+360` wrap, no `>360` wrap unlike PHC),
  but the (uas,vas) result is then `fesom_vector_g2r`-rotated into the **model** frame (Euler
  50/15/−90); scalars are NOT rotated. (3) **Per-field mid-interval time shift** (`nm_nc_tmid=0`):
  instantaneous fields (uas/tas) are sampled on the 3-h marks, flux fields (prra) on the
  half-marks, and the shift `nc_time[i]=½(t[i+1]+t[i])` gives each field its **own** `nc_time`,
  so `getcoeffld` is per-field (not shared) even though the *spatial* stencil is shared (all 8
  files share one 640×320 grid → build the gather once). (`jra55.py`, `fesom_jra55.c`, Task 5.3.)

- **[jra/config] `flip_lat = 0` for JRA55-do v1.4.0** — `lat` is stored **ascending**
  (−89.57→89.57), so the C's north→south flip (`fesom_jra55.c:270`) is inert here. The reader
  implements it faithfully anyway (per-field, applied to both `nc_lat` and each data slice) so a
  future N→S-stored field still works; just don't expect the flip path to be exercised by the
  CORE2 gate. `Nlon=640+2=642` (cyclic halo), `Nlat=320`, `Ntime=2920` (3-hourly), cal=gregorian.
  (Task 5.3.)

- **[jra/scope] The reader is host-numpy, non-differentiable SETUP (like `phc_ic`) — the
  differentiable SST→flux / current→stress seam is the bulk (Task 5.4), not here.** Output is 8
  per-node physics-unit arrays (`u_wind`/`v_wind` rotated, `Tair` °C, `prec_*` m/s) that become
  per-step *device constants*. A simple per-field `getcoeffld` cache (refresh only when `rdate`
  leaves `[nc_time[t_indx], nc_time[t_indx_p1]]`, `fesom_jra55_step:651`) avoids re-reading the
  640×320 slices every step; the cache is a pure optimization (no effect on the result — the
  coefficients depend only on `rdate`'s bracket, not on call history, so a fresh reader == a
  sequentially-advanced one). netCDF4 reads use `set_auto_maskandscale(False)` to get the raw
  float32 the C's `nc_get_vara_float` sees (JRA has no scale/offset; bit-exact promotion to f64).
  (`jra55.JRA55Reader`, Task 5.3.)

## Phase 5 — L&Y09 open-water bulk formulae (Task 5.4)

- **[bulk/AD] ⚠️⚠️ "Drop the early break, run a fixed N, the result is IDENTICAL" was WRONG —
  it is a small but real, bounded divergence, and the sub-plan's "post-convergence iters are
  no-ops" claim is corrected.** The L&Y09 Monin-Obukhov coefficient loop (`ncar_ocean_fluxes_mode`,
  `fesom_bulk.c:89-172`) does **not** robustly converge at near-calm nodes: the C production
  breaks on `|Δcd|/(cd+1e-8)<1e-4`, but that "convergence" is a transient slowdown, so continuing
  to a fixed 5 iters lands elsewhere. Measured on CORE2 (year 1958): **`ch` differs by up to ~88%**
  fixed-5-vs-early-break at the calmest tropical nodes (`cd`/`ce` up to ~4.5%). The saving grace is
  it's **physically bounded**: `ch`/`ce` only enter the `ug`-scaled sensible/latent terms, so the
  heat_flux impact is **≤7.2 W/m² at ~4 nodes** (mean ~2e-4; <0.1 W/m² for 126848/126858 nodes),
  stress ≤~4e-3 N/m². *Decision: JAX runs fixed-5 (the AD-safe analog of the C's ≤5-iter cap — a
  data-dependent `while`-break is not reverse-mode differentiable), and is verified against a
  **fixed-5** C dump, not the early-break production. The residual vs production is this bounded,
  documented effect.* *Lesson: never assume "iterate-to-fixed-point" tolerates extra iterations —
  a capped non-convergent solver (M-O, sea-ice EVP, some EOS inversions) gives genuinely different
  answers per iteration count; measure the divergence, bound its PHYSICAL impact (not the raw
  coefficient %), and make the reference match your iteration scheme.* (`forcing.ncar_ocean_fluxes_mode`,
  `test_forcing.test_earlybreak_drop_is_physically_bounded`, Task 5.4.)

- **[bulk/method] The fix: a `fixed_iters` env-gated C dump so JAX-fixed-5 is compared to
  C-fixed-5 (apples-to-apples), separately bounding fixed-5-vs-production.** Added an
  `int fixed_iters` param to the C `ncar_ocean_fluxes_mode` (skips the break) + a `fesom_bulk_dump`
  (gated `FESOM_BULK_DUMP_DIR`) that runs fixed-5 and dumps `cd/ce/ch + heat_flux/water_flux/
  stress_node + elem stress`, **plus** the early-break `cd_eb/ce_eb/ch_eb` columns and the exact
  `T_oc` (so the JAX forward gate is fed the C's own SST and isolates the bulk from the ~1e-14
  PHC-IC residual). Result: **JAX-fixed-5 == C-fixed-5 to ~1e-17 (cd/ce/ch), ~6e-13 (heat_flux),
  ~5e-16 (stress)** over all 126858 nodes — essentially bit-exact (MAP-class, like the EOS). For
  Task 5.7 the matched per-substep reference must set `FESOM_BULK_FIXED_ITERS=1` or the calm-node
  coefficients won't match. (`fesom_bulk_dump`, `jax_bulk_dump_core2.sh`, Task 5.4.)

- **[bulk/AD] The AD-safe rewrite of `x2 = sqrt(|1−16ζ|); if(x2<1) x2=1` is
  `sqrt(max(|1−16ζ|,1))` — bit-identical to the C AND smooth through the ζ=1/16 singularity.**
  The naïve port hits `sqrt(0)` at ζ=1/16 (inf derivative ⇒ `0·inf` NaN backward even though the
  forward floors x2 to 1). Folding the floor INSIDE the sqrt argument (`max(arg,1)`) means: for
  arg≥1 it's `sqrt(arg)` (==C, and arg≥1 ⇒ ζ away from 1/16 ⇒ the abs is smooth there); for arg<1
  it's the constant `sqrt(1)=1` (==C's floor, gradient 0). One expression kills the kink, the abs
  kink, AND matches the C exactly — cleaner than a double-`where` safe-sqrt. The relative-wind `u`
  and stress `mag` still need the double-`where` safe-sqrt (their `sqrt(Δu²)` arg vanishes when
  wind==current, and that lane IS on the `current→stress` gradient path). The `copysign` step
  selectors (cd_n10 hi/lo-wind switch, the stab switch) are ported **literally** via `jnp.copysign`
  (gradient 0, exact at ±0) — not `where(>0)`, which mishandles `−0.0`. (`forcing._psi`/`_safe_speed`/
  `_cd_n10`, `fesom_bulk.c:99-160`, Task 5.4.)

- **[bulk/fidelity] The deliberate Fortran wind mismatch is load-bearing and was preserved.**
  The exchange coefficients (`ncar_ocean_fluxes_mode`) and the wind stress use the **relative**
  wind `|u_atm − u_ocn|` (floored at 0.3); but `obudget`'s `ug` (the sensible/latent multiplier)
  uses the **absolute** wind `|u_atm|` (`fesom_bulk.c:283`, mirroring `ice_thermo_oce.F90`). A
  synthetic-current dump mode (`current_mode=1`, an 8-entry exact-decimal table indexed by
  `(gid−1)%8`, reproduced bit-for-bit in JAX) exercises this: it moves the coefficients/stress via
  the relative wind while `ug` stays absolute, and validates the `current→stress` feedback that the
  zero-current IC state can't (uvnode=0 at setup). Also: `albw=0.1` (CORE2 `namelist.ice`, NOT the
  LY2004 0.066), bulk gravity `9.80` (NOT config.G=9.81), and `heat_flux = qns − qsr` is the
  bulk_compute output **before** shortwave-penetration removal (a Task-5.6 step; `USE_SW_PENE=1` in
  the C). (`forcing.bulk_surface_fluxes`/`obudget`, Task 5.4.)

- **[workflow] Cheap C dump jobs schedule far faster on `-p compute --time≤30:00` (debug QOS)
  than on `-p shared`.** The bulk dump sat minutes pending on `shared` (Priority); resubmitted to
  `compute` with a 30-min walltime it started in ~16 s (DKRZ Levante's short-walltime compute jobs
  land in the fast debug/devel QOS). Use `-p compute --nodes=1 --ntasks=1 --time=00:30:00 -A ab0995`
  (drop `--mem` — compute nodes are exclusive) for the 17–25 s mesh/IC/dump jobs. (User-flagged;
  `jax_bulk_dump_core2.sh`, Task 5.4.) [[fesom-jax-port]]

- **[workflow] Large generated artifacts go on `/work`, not `/home`** (user standing rule).
  `port_jax/data/` (499 M: mesh export, IC, C dumps) was moved to
  `/work/ab0995/a270088/port_jax/data` with a `data → /work/...` **symlink** at the repo root, so
  all relative-path code (`Path(__file__).parents[2]/"data"`) and the C job scripts (which write to
  `/home/.../data/...`) transparently land on `/work`. `.gitignore` needs **both** `/data` (the
  symlink) and `/data/` (a plain dir) — the trailing-slash form alone does not ignore a symlink.
  (User-flagged, Task 5.4.)

## Phase 5 — SSS restoring + CORE2 runoff (Task 5.5)

- **[sss/fill] ⚠️ The 30-cell missing-value fill is JACOBI (reads the ORIGINAL field), NOT
  sequential Gauss-Seidel like PHC's `extrap_nod3D` — so it VECTORIZES.** The C copies
  `ncdata → ncdata_temp` first, and every missing cell reads its neighbours from
  `ncdata_temp` (never modified during the fill loop), so fills do **not** cascade ⇒
  order-independent. A `scipy.ndimage.uniform_filter` box-mean (expand `k=1..30`, fill each
  cell at the SMALLEST k whose `(2k+1)²` window — clamped at the grid edge with
  `mode='constant',cval=0`, NOT cyclic — holds ≥1 valid cell) reproduces it. The `/count`
  crushes the box-sum reassociation: **105148/126858 SSS nodes bit-exact, ~35 coastal
  fill-bracket nodes ~1e-12** (the window straddles a land-extrapolated cell). Contrast PHC's
  extrap which *is* sequential GS and had to be replicated in index order (Task 5.2). *Lesson:
  read whether a fill reads from a frozen copy (Jacobi → vectorize) or in-place (GS → replicate
  order) before porting it.* (`sss_runoff._fill_missing_expand`, `fesom_sss_runoff.c:207-239`.)

- **[sss/interp] `interp_2d_field` is a THIRD distinct bilinear routine — lat CLAMPS, lon
  CYCLIC-WRAPS — not the JRA `extrp`-flag stencil nor PHC's ±halo padding.** Out-of-range
  latitude pins to the boundary grid value (clamp `y`, weight 1 on the edge node); out-of-range
  longitude wraps across the 0/360 seam with gap `lon[0]+(360−lon[last])` (here 1.0° between
  359.5 and 0.5). Ported as `clip(y)+searchsorted` (lat) + a 3-branch in-range/below/above
  select (lon); the corner blend keeps the C's `(s·rt_lon1+s·rt_lon2)·rt_lat` order. **Runoff
  bit-exact, SSS bit-exact at the 105k ocean-bracket nodes** ⇒ the bracket+blend is exact for
  any node not touching a filled cell. Each forcing reader has its own interpolation routine —
  do not assume one stencil fits all. (`sss_runoff._interp_2d_field`, `fesom_sss_runoff.c:34-113`.)

- **[sss/fidelity] The salt/water balance matched the C to ~1e-20 — the global-mean's
  ÷`ocean_area` crushes the reduction back to MAP-class (the hbar/w ÷area lesson again).**
  `virtual_salt = S_top·water_flux − ⟨·⟩`, `⟨x⟩ = Σ(x·areasvol_surf)/ocean_area`. Fed the C's
  own `S_top`/`water_flux`, the multiply is bit-exact; the only JAX↔C difference is the
  area-weighted global mean. The integral `Σ(x·area)` ~1e9 (x~1e-6, area~3e9, ×1.3e5 nodes) so
  the ~1e-7 sum reassociation, ÷`ocean_area`=3.6e14, lands at **~1e-21**. So a reduction
  divided by a huge constant area gates TIGHT (~1e-20), not at the loose 1e-12 reduction floor —
  measure at the output. The flux math is fed the dump's own inputs (apples-to-apples, like the
  bulk's `T_oc`), isolating it from the reader. (`sss_runoff.sss_runoff_fluxes`,
  `test_sss_runoff.py`, `fesom_sss_runoff.c:382-440`.)

- **[sss/config] ⚠️ `ref_sss_local=1` (rsss = LOCAL S_top, not 34.7) + NO legacy month +1 —
  both are C-comment-documented traps; port them, don't reinvent.** The CORE2 namelist sets
  `ref_sss_local=.true.`, so the virtual-salt reference salinity is the per-node surface
  salinity, NOT the constant `ref_sss=34.7` (using 34.7 over-strengthens the flux where SSS is
  low → Arctic freshwater bias — `fesom_sss_runoff.c:298-307`). And the monthly SSS read fires
  on the FIRST step of the new month (where `month_now` is already M+1), so there is **no `+1`**
  (the legacy Fortran fired on the LAST step of month M and added +1; keeping it would skip a
  month — `:351-359`). `surf_relax_S = 10/(60·3600·24) = 1.929e-6 s⁻¹`. (Task 5.5.)

- **[sss/method] The month-CROSSING dump (m4 = Apr/day100) is the real gate; m1 (Jan/day1)
  alone is the trivial first-month case.** Same shape as the JRA interior-vs-boundary lesson:
  m1 just reads SALT month 1 at the first step. The m4 dump steps jra to day100 (April),
  **recomputes the bulk** `water_flux`, and reads SALT **month 4** — exercising a different SSS
  slice + a different (April) bulk input, confirming the reader picks the right month and the
  flux math handles the seasonal target. The C `fesom_sss_runoff_dump` saves+restores
  `water_flux` (the one field the step both reads and writes in place) so two month dumps in one
  run stay independent. (`jax_sss_dump_core2.sh`, `fesom_main.c` SSS-dump block, Task 5.5.)

## Phase 5 — wire surface BCs + assemble CORE2 forcing (Task 5.6)

- **[ice/scope] ⚠️⚠️ The C "no ice" run is NOT ice-free — it keeps a STATIC `a_ice` mask
  that gates the surface fluxes.** `fesom_ice_initial_state` (`fesom_ice.c`, called
  `fesom_main.c:792`) sets `a_ice = 0.9` wherever **(non-cavity & the PHC IC SST < 0)** — and
  with `FESOM_NO_ICE_DYN/ADV/THERMO=1` the ice model never runs, so that mask is *frozen* for
  the whole run (37089/126858 nodes on CORE2). It has **two** couplings in the no-ice path:
  (1) **shortwave penetration is skipped where `a_ice>0`** (`fesom_bulk.c:381-382` —
  `cal_shortwave_rad`); (2) **the wind stress is blended** `stress = ice_drag·a_ice +
  atm·(1−a_ice)` with `ice_drag = ρ·Cd·|u_ice−u_w|·(u_ice−u_w)`, `u_ice=0` (static),
  `u_w = uvnode[:,0]`, `ρ·Cd = FESOM_DENSITY_0·cd_oce_ice = 1030·5.5e-3` (`fesom_ice_coupling.c:
  234-264`, `oce_fluxes_mom`). The **bulk itself does NOT gate on `a_ice`** (open-water fluxes
  everywhere — verified: at an ice node the C heat_flux == the JAX open-water bulk). `ocean2ice`
  runs even with ice off (so `u_w` updates each step → a step≥2 current→stress drag), but
  thermo/`oce_fluxes` are skipped (so heat/water/salt are the bulk+sss values). **User decision
  (2026-06-06): match the C — replicate the static mask** (not truly-ice-free). Symptom that
  found it: JAX (assuming `a_ice=0`) mismatched the C heat_flux by **122 W/m²** at an Antarctic
  node (the visible band `0.486·shortwave` it wrongly added under ice). *Lesson: "ice off" in a
  coupled model rarely means `a_ice≡0` — grep every `a_ice` reader before assuming ice-free.*
  (`surface_forcing.ice_ic_aice`/`compute_surface_fluxes`, Task 5.6.)

- **[tracers/T_old] ⚠️ The CORE2 step-1 `T_old`/`S_old` (AB2 `valuesold`) is the CONSTANT
  BASE 10/35, NOT the PHC field — the exact analog of the pi blob `T_old` trap.** The C order
  (`fesom_main.c`): set `values = const 10/35` (`:413`) → run the rest-sanity `advect_one`
  which saves `valuesold = values = 10/35` (`:724-756`, via `init_tracers_AB_one`) → **then**
  `fesom_phc_load_ic` overwrites `values = PHC` but leaves `valuesold` (`:778`). So at step 1
  `ttfAB = −(0.5+ε)·base + (1.5+ε)·PHC`, not `PHC`. `core2_initial_state` had `T_old=T=PHC` →
  corrupted the step-1 FCT advection: post-step `T` was off **2.4e-3** (a *large fraction of the
  one-step tendency*, surface-concentrated with an opposite-sign dipole at level 1). Fix:
  `T_old = masked base 10`, `S_old = masked base 35`. *Lesson (3rd time, pi+CORE2): at step 1
  `T_old ≠ T`; the IC's `valuesold` is whatever the last pre-IC-overwrite `advect_one` saved.*
  (`phc_ic.core2_initial_state`, Task 5.6.)

- **[bulk/fixed-iters] ⚠️ `FESOM_BULK_FIXED_ITERS=1` was honored ONLY by `fesom_bulk_dump`,
  NOT by the time-loop `fesom_bulk_compute` (hardcoded `fixed_iters=0` = early-break).** So the
  per-substep dump captured the *production early-break* bulk while JAX runs the AD-safe fixed-5
  loop → at the non-convergent calm/cold nodes (Task 5.4) the heat_flux diverged **1.6e-4 W/m²**
  (2.5e-7 rel on a 633 W/m² flux at an extreme-stability Arctic node). NOT an input difference:
  the PHC SST matched 3.5e-15 and a 1e-13 wind perturbation only moved heat_flux 6e-12 (×60).
  Fix: env-gate `fesom_bulk_compute` on `FESOM_BULK_FIXED_ITERS` too (`fesom_bulk.c:250-263,283`).
  After it, heat_flux → **1.1e-13**. *Lesson: when a fidelity flag exists, grep that EVERY code
  path which feeds the gate honors it — a dump-only flag silently leaves the real kernel on the
  production branch.* (`fesom_bulk.c`, Task 5.6.)

- **[forcing/method] The step-1 integrated `T`/`S` dump match is the COMPREHENSIVE CORE2 gate
  — bit-exact (~7e-15) once all three bugs above are fixed.** Post-step `T`/`S` is the end of
  the step and depends on every upstream kernel (EOS/PGF/PP/momentum/CG-SSH/ALE/FCT/vert-diff)
  AND the new surface BCs, so a tight match validates the whole assembled step in one number —
  no need for per-substep dynamics gates at 5.6 (those, with calibrated tolerances, are 5.7).
  Verified bit-exact: density 2.3e-13, bvfreq 1e-16, Kv 0.0, uv ~1e-10, d_eta ~2e-9 (CORE2 CG
  takes 43 iters vs pi's 3 → more reassociation), `w` 4e-12, **post-step T 7.1e-15 / S 2.1e-14**;
  multi-step T/S stay ~1e-9 over steps 2-3 (the loop-carried jra date + AB2 + step≥2 ice drag
  thread correctly). (`test_core2_step.py`, Task 5.6.)

- **[runoff/scope] ⚠️ Runoff is INERT in the Phase-5 no-ice config — by the C's DESIGN, not a
  bug. It works fully in the ice-on run.** The C routes runoff through **sea-ice thermo**, not
  the standalone sbc (`fesom_sss_runoff.c:376-380` "Phase C3b: removed runoff subtraction;
  runoff is now folded into `ice->flx_fw` inside `fesom_therm_ice`; subtracting again would
  double-count"). Ice-ON path: `runoff → fesom_ice_thermo.c:318 prec=rain+runo+snow(1−A) →
  :509 flx_fw → fesom_ice_coupling.c:139 water_flux=−flx_fw → :391 virtual_salt=rsss·water_flux
  → bc_S` (river-mouth freshening that advects — what makes runoff "work"). With
  `FESOM_NO_ICE_THERMO=1` (Phase 5) that block is gated off, so `water_flux` stays the **bulk
  evap−prec** (no runoff), `virtual_salt = rsss·(evap−prec)`, and the balance
  `water_flux += ⟨water_flux+runoff⟩` is inert in linfs (the only `water_flux` consumers are the
  **non-linfs** `ssh_rhs`/ALE paths, `fesom_ssh.c:322-324`). ⇒ runoff has zero effect on the
  Phase-5 trajectory (a known salty coastal/Arctic bias). **The port is faithful** (dump:
  `virtual_salt` ~1e-20). *Lesson: trace where a forcing's LOCAL term actually lives before
  declaring it "works/doesn't work" — runoff's local door is ice thermo, so "ice off" silently
  removes it.* (User-flagged; user decision = keep matching the no-ice C run, with the Phase-6
  activation plan locked.) (Task 5.6.)

- **[runoff/phase6] Runoff "comes online" for free once Phase 6 ports the ice freshwater
  budget — the reader + balance are done and the seam is pure.** No Phase-5 runoff code needs
  revisiting: `sss_runoff.runoff_node` (reader, bit-exact) + `sss_runoff_fluxes(water_flux, …,
  runoff_node, …)` (the balance, **pure in `water_flux`**) are already in place. Phase 6 only
  adds (a) `fesom_ice_thermodynamics` folding runoff into `flx_fw`, (b) `fesom_ice_oce_fluxes`
  setting `water_flux=−flx_fw`; then `surface_forcing.compute_surface_fluxes`'s ice-on branch
  feeds `−flx_fw` (incl. runoff) into the EXISTING `sss_runoff_fluxes` instead of the bulk's
  `evap−prec`. Verify ice-on with the dump recipe at river-mouth nodes; no double-count if the
  C3b design is followed (runoff in `flx_fw`, sbc local-term stays removed). Full spec:
  sub-plan "Runoff handoff to Phase 6". (Task 5.6.)

- **[ad/forcing] The surface forcing adds three differentiable seams into the tracer/momentum
  eqns, all AD-safe by construction.** `bc_T = −dt·heat_flux/vcpw` (SST→heat_flux via the bulk),
  `bc_S = dt·(virtual_salt+relax_salt)` (SST→water_flux→virtual_salt + S_top→relax_salt), and
  the ice-ocean drag `current→stress` (safe-sqrt for `|u_w|` at `u_w=0`). `sw_3d` is a per-step
  forcing **constant** (depends only on JRA shortwave + chl climatology + geometry, not state),
  so it carries no AD path — only its additive `swsurf_W` on `heat_flux` matters, and that's a
  constant offset (preserves `d(heat_flux)/d(SST)`). `cal_shortwave_rad`'s data-dependent break
  (aux<1e-5) vectorizes as a cumulative-OR mask (no monotonicity assumption). The chl reader is
  the SAME `read_other_NetCDF` routine as SSS (Sweeney = the C default; constant chl=0.1 is the
  `FESOM_CHL_SOURCE=None` seam). (`tracer_diff.py`/`forcing.cal_shortwave_rad`, Task 5.6.)

## Phase 5 — matched C dump run + CORE2 stability (Task 5.7)

- **[stability/scope] ⚠️⚠️ "No ice" is numerically stable for ~1 week, then the UNBOUNDED
  high-lat SUPERCOOLING destabilizes the dynamics — a PHYSICAL limitation (the C supercools +
  tracks JAX identically through the verified ~day 2.3 window), not a numerical bug.** The
  Phase-5 no-ice CORE2 run (PHC IC + JRA55 + SSS/runoff, dt=500)
  is numerically clean for **days 1–7** (no NaN; max|vel| ≤ ~1.9 m/s < 3; |SSH| ≤ ~2.8 m < 5;
  the Aleutian-Trench node 94122 stays warm ~3.2 °C and calm). But with no sea ice to cap
  high-latitude heat loss, SST **supercools monotonically without bound**: −1.9 (IC) → −5.8
  (day 1) → −16.5 (day 5) → −22.8 (day 8). Once SST drops below ~−20 °C the JM-EOS is being
  evaluated far outside its valid range, the spurious density field drives spurious convection,
  and at **model day ~8.1 (step 1399) max|vel| finally crosses 3 m/s**. This is the *anticipated*
  no-ice failure mode (sub-plan risk #1 / FRESH_START §15 "SST < −2 without ice"), NOT a port
  error. **The matched C arbiter supercools + tracks JAX identically through the verified window
  (~day 2.3 / step 396; see next lesson — the longer C run was cancelled, so the day-8 figures
  are JAX's, shared by the mechanism)** ⇒ the C does NOT numerically blow up at this config
  either, so the Task-5.7 "if the C itself blows up, ice must move to Phase 5" finding did
  **not** trigger — ice stays Phase 6, and a physically realistic SST simply needs
  the ice cap. *Lesson: distinguish numerical stability (bounded vel/SSH/CG, no NaN) from
  thermodynamic realism (SST in range); a no-ice ocean is the former for ~a week and never the
  latter at high latitudes.* (`scripts/archive/core2_stability_run.py`, Task 5.7.)

- **[stability/method] The matched C arbiter run is the decisive truth-teller: JAX tracks the C
  to 3 sig figs on the bulk min/max diagnostics, even though individual elements diverge
  chaotically.** Running the C at the IDENTICAL config (`FESOM_MIX_SCHEME=PP FESOM_NO_GMREDI=1
  FESOM_NO_ICE_*=1 FESOM_BULK_FIXED_ITERS=1`, dt=500) with the per-step monitor
  (`FESOM_PRINT_EVERY`) gives a step-by-step reference for SST-range / max|uv| / max|eta|.
  JAX vs C at step 216 (1.25 d): **SST_min −6.60 = −6.60, max|uv| 1.389 ≈ 1.39, max|eta| 2.715 ≈
  2.71** — agreement to ~3 sig figs, *despite* the per-element chaotic divergence (the step-1
  bit-exact match degrades to ~1e-6 by step 3). The reason: **min/max reductions are robust
  observables of the forced large-scale response** (the supercooling, the geostrophic
  adjustment), which both models share, while the FP/CG-iteration chaos lives in the small-scale
  detail that the reductions don't see. *Lesson: to cross-validate a chaotic multi-step
  trajectory you cannot bit-compare, gate on robust global reductions (range, max-speed) against
  the matched reference — they track far longer than any pointwise field.* (`jobs/jax_core2_stability.sh`,
  `core2_stability_run.py`, Task 5.7.)

- **[dynamics/calibration] Step-1 per-substep DYNAMICS are bit-exact-class (~1e-15); the spread
  blows to ~1e-6 by step 3 — and the 5.6 "uv~1e-10/d_eta~2e-9" was the steps-2/3 evolution, not
  step 1.** Because JAX and C start from the IDENTICAL PHC IC (incl. the `T_old`=base trap), every
  step-1 substep input matches and the only spread is FP reassociation: measured at the 7 probes,
  pgf/Av/hnode ~0..1e-17, uv_rhs/uv/d_eta/eta_n/hbar/w ~1e-16..8e-15 (all bit-exact-class). From
  step 2 the **discrete CG iteration count** (CORE2 takes 32–43 iters; a ±1-iteration difference
  near the residual tolerance makes the solutions jump apart) + the FCT limiter amplify the
  ~1e-15 to ~2.6e-8 (d_eta) / ~2.8e-7 (uv) at step 2, ~1e-7 / ~2e-6 at step 3 — bounded, not
  growing catastrophically. Two large-magnitude intermediates need scaled tolerances: `ssh_rhs`
  (~1e5, transport-divergence ×area/dt) and `pressure` (~5e5, hydrostatic integral) match to
  ~1e-11 *relative* (4.9e-7 / 2.2e-10 absolute). *Lesson: a per-substep gate is only "tight" at
  step 1 where inputs are shared; downstream of a chaotic solver, calibrate to the accumulated
  spread, and use relative tolerances for the big intermediate fields.*
  (`tests/test_core2_step.py::test_step1_dynamics_per_substep`, Task 5.7.)

- **[dump/layout] The C dump pairs each NODE probe with its incident ELEMENT gid — element
  fields land at element gids, node fields at node gids.** `fesom_dump.c` records, per probe,
  the node fields (T/S/density/Kv/pressure/d_eta/eta_n/hbar/ssh_rhs/w/hnode) at the node gid AND
  the element fields (pgf_x/y, Av, uv_rhs_u/v, uv_u/v) at `s_elem_gid` (an incident triangle's
  global id). On the CORE2 dump the 7 node probes `{1001,33778,43828,61202,66921,79663,94122}`
  came with 7 element probes `{307,747,25954,61526,99096,110065,154575}` (one >nod2D, the
  give-away that they are element ids). So the dynamics gate compares `uv[gid−1,:,comp]` at the
  ELEM probes, not the node probes — getting this wrong silently compares the wrong cells.
  (`tests/test_core2_step.py::_emaxabs`, Task 5.7.)

- **[perf] Jitted CORE2 step: GPU ~0.06 s, CPU ~3 s, eager ~32 s.** The jitted `step_jit` is
  ~10× faster than eager on CPU and **~500×** on an A100 — a 10-model-day run (1728 steps) is a
  ~2-minute GPU job (incl. ~48 s mesh+IC+forcing host build + a one-time compile of the two AB2
  step variants). Per-step host-sync of a handful of scalar diagnostics (NaN / SST-range /
  max|vel| / max|eta| / Aleutian node) is cheap enough to monitor every step and pinpoint the
  exact destabilization step. The single-rank C reference is much slower (~4.6 s/step) — it is
  an MPI code run on 1 rank — so for the C arbiter cap the run or give it a non-debug QOS.
  (`scripts/archive/core2_stability_gpu.sh`, Task 5.7.)

## Phase 5 — CORE2-slice gradient gate (Task 5.8, GATE 5)

- **[ad/scope] ⚠️⚠️ The multi-step FORCED CORE2 trajectory is genuinely NON-SMOOTH in the
  physics parameters — a clean FD↔AD plateau is only well-posed in SMOOTH regimes, NOT over
  the full forced model.** pi's clean `d(mean SST)/d(k_ver)` plateau (5.70e-7) works because
  the smooth Gaussian blob keeps the FCT Zalesak limiter AND the convective adjustment
  (`max(N²,0)` / instabmix) **dormant**. Under real CORE2 forcing (PHC fronts +
  supercooling-driven convection) those kinks are **active**, so the loss is genuinely
  non-smooth in `k_ver` at the FD scale: the N=20 `d(mean SST)/d(k_ver)` FD does **not**
  plateau (AD=+33.5; FD swings −15.1 → +6.4 → +222 → +37 across h=1e-2…1e-6; min rel 9.7e-2).
  The AD is a valid **(sub)gradient** (the smallest-h FD ≈ +37 is the closest); FD *across* a
  kink ≠ the slope. So the quantitative FD↔AD gate runs where the assembled model **is**
  smooth: **N=1** `d(mean SST)/d(k_ver)` (plateau **7.5e-10**) — at step 1 `uv=0` ⇒ the PP
  shear term vanishes ⇒ `Kv = k_ver` *additively* on stable columns (the convective mask is
  fixed by the IC density, so `k_ver` can't move it) — plus the isolated bulk seams + the
  linear-solve residual. The multi-step CG/FCT/EOS machinery is pi-proven (identical code) and
  AD-safe on CORE2 (the masked-NaN probe). *Lesson: re-validating a gradient by FD needs a
  smooth regime; a forced realistic trajectory exercises kinks (flux limiters, convective
  adjustment) that a synthetic smooth IC hides — match the FD regime to what is smooth, and
  lean on the (sub)gradient + masked-NaN finiteness for the non-smooth full model. This is the
  long-anticipated "model is mildly chaotic ⇒ test_gradient stays short/smooth" lesson, now
  shown to be STRONG (not mild) under real forcing.* (`scripts/archive/core2_grad_gate.py` [1]/[4],
  job 25394380, Task 5.8.)

- **[ad/observable] ⚠️ To FD-probe a sub-path's gradient the observable must actually DEPEND
  on the parameter — the barotropic SSH `d_eta` is PHYSICALLY INSENSITIVE to `a_ver`
  (`d(Σd_eta²)/d(a_ver) ≈ −7e-17`, correctly ~0, AD agrees — not a bug).** `d_eta` solves the
  *depth-integrated* transport divergence (set by the wind stress + bathymetry); `a_ver` only
  redistributes momentum **vertically**, conserving the column integral, so `d_eta ⊥ a_ver`.
  And `k_ver` doesn't reach the CG in one step (it enters the *final* vertical diffusion,
  downstream of the solve). So no obvious single-step param-gradient probes the CG transpose on
  CORE2. Verify the CG implicit-diff **directly by the residual**: for `f(b)=½‖S⁻¹b‖²`,
  `∇f = S⁻¹·d_eta`, so the AD cotangent must SOLVE `S·g_ad = d_eta` — `‖S·g_ad − d_eta‖/‖d_eta‖
  = 8.8e-14` confirms the tight transpose reached the true `S⁻¹` on the 40×-bigger CORE2 matrix.
  A residual check is **strictly stronger** than matching another run of the same `_pcg` (a
  non-converging / wrong-preconditioner solver passes *that*). *Lesson: choose a parameter the
  observable depends on; verify a linear solver's implicit-diff by the residual `S·g − cotangent`,
  not a parameter sweep.* (`test_gradient_core2.test_grad_cg_transpose_core2`, Task 5.8.)

- **[ad/bulk-seam] The NEW Phase-5 differentiable forcing seams are AD-correct to ~1e-11
  (FD↔AD), well-conditioned because the bulk is a per-node pure map.** `d(Σheat_flux)/d(SST)`
  plateau **5.3e-11** (physical sign **+**: warmer ocean ⇒ more heat loss ⇒ larger upward
  `heat_flux`), `d(Σstress)/d(u_current)` **3.6e-12**. Sum the directional derivative over a
  SMOOTH node subset (`|SST−Tair|>1`, `1<|Δu|<30`): the bulk has kinks at SST=Tair (the initial
  `stab` switch), **ζ_u≈0** (the in-loop neutral `stab` switch + the `ψ` branch flip), u10=33
  (drag switch) and Δu=0 (the safe-sqrt); nodes straddling a kink corrupt the large-h FD
  (rel ~4e-3 at h=1e-2), but the straddler count scales with h, so the **min-over-h plateau**
  lands kink-free at h≈1e-6. *Lesson: for a kinky per-node map, take the directional FD over a
  smooth node subset and sweep h — the plateau lives at the small-h end where the straddlers
  vanish.* (`test_forcing.test_ad_vs_fd_heat_flux_sst` / `_stress_current`, Task 5.8.)

- **[ad/memory] The checkpointed N=20 CORE2 backward peaks at 37.8 GB on an A100 — fits the
  80 GB card (59%); on a 40 GB card it's at the limit (drop to ~N=10 or use O(√N) nesting).**
  (pi N=200 was 4.23 GB; CORE2 ≈ 40× nodes × 1/10 the steps ⇒ ~9× ⇒ ~38 GB, consistent.) The
  masked-NaN probe `d(mean SST)/d(T₀)` at N=20 is finite everywhere, **exactly 0** on the
  below-bottom masked lanes and nonzero (5.5e-3) on wet — the strong AD-safety gate now on the
  *assembled CORE2 model* (the new bulk seams + the eos/tracer_diff/FCT masked-divide guards in
  one backward). ⚠️ A login/CPU node could **not** hold even the N=4 T₀ backward (the process
  was killed) → this is a GPU-only gate; the CPU suite runs the N=1 masked-NaN probe.
  (`scripts/archive/core2_grad_gate.py` [3], job 25394380, Task 5.8.)

# Phase 6 — Sea Ice (sub-plan `docs/plans/20260606-fesom-jax-phase6-seaice.md`)

## Task 6.1 — ice state + cold-start IC + IceConfig

- **[state] σ11/σ12/σ22 are PROGNOSTIC elastic-memory, NOT per-step scratch — they must be
  carried in `State` across ocean steps.** The EVP subcycle's stress update reads the *prior*
  σ (`si1 = det1·(σ11 + σ22 + dte·r1)`, `fesom_ice_evp.c:126-130`) and `EVPdynamics` only
  zeros `inv_areamass/inv_mass/rhs_a/rhs_m` at the top (`:265-270`) — **never σ**. So σ
  persists from the previous ocean step (the "elastic" memory of EVP). It joins
  `a_ice/m_ice/m_snow/u_ice/v_ice/t_skin` as carried ice state. *Lesson: before adding an ice
  field to `State`, check whether the C re-initializes it each step or carries it — σ and
  t_skin (the Newton warm-start) carry; eps/ice_strength/the velocity rhs are per-step
  scratch.* (`fesom_jax/state.py`, Task 6.1.)

- **[config] ⚠️ `Tevp_inv = 3.0/ice_dt` is the `ice_setup` value (`fesom_ice.c:233`), NOT the
  `evp_rheol_steps/ice_dt` written in the `fesom_ice_types.h:177` comment.** The comment is
  stale; the code overrides it in setup (`Tevp_inv = 3.0/ice_dt`, with `ice_dt=500` ⇒ 0.006).
  Also `dte = ice_dt/evp_rheol_steps` and `det1=det2=1/(1+0.5·Tevp_inv·dte)`. Encoded as
  `IceConfig` derived properties so the one definition is the C setup value. *Lesson: when a C
  struct comment and the setup code disagree, the SETUP code wins — grep the `*_setup`/`init`
  for the actual assignment, don't trust the type-declaration comment.* (`fesom_jax/ice.py`
  `IceConfig.Tevp_inv`, Task 6.1.)

- **[state/inert] Adding the 9 inert ice fields to `State` keeps the ocean (pi + Phase-5
  no-ice) path bit-identical — nothing reads them until Phase 6.** They default-zero in
  `State.zeros` and flow through `lax.scan`/`grad` as extra zero leaves (more cotangent leaves,
  all zero). The only test that broke was the explicit field-inventory assertion
  (`test_state.py::test_zeros_shapes_and_dtype`) — a deliberate drift guard, updated to list
  the new fields. *Lesson: a frozen-dataclass pytree tolerates additive inert fields with no
  numeric change; the field-inventory test is the intended tripwire — update it, don't loosen
  it.* (`fesom_jax/state.py`, Task 6.1.)

- **[verify] The cold-start ice IC is C-verified TRANSITIVELY — no new C dump needed.** The
  C `fesom_ice_initial_state` (`fesom_ice.c:246-277`) is a pure threshold rule of the IC
  surface T (`SST<0 ⇒ a_ice=0.9` + hemisphere-split m_ice/m_snow), and the JAX PHC SST already
  matches the C to ~1e-14 (Task 5.2). So the JAX IC matches the C's to the same tolerance; the
  only sensitivity is nodes with `|SST|` within FP noise of 0 — **counted, found 0**
  (`test_ice_ic.test_ic_threshold_not_fp_fragile`). Gated against an independent per-node loop
  reference instead. *Lesson: when a kernel is a pure function of an already-dump-verified
  field, verify it transitively (numpy ref + a threshold-fragility count) rather than spending
  a SLURM C dump.* (`fesom_jax/tests/test_ice_ic.py`, Task 6.1.)

- **[workflow] Probe re-pinning for the ice dumps is env-only (`FESOM_DUMP_PROBES`,
  `fesom_dump.c:19-21`), and the incremental ice configs are env knobs
  (`FESOM_NO_ICE_DYN/ADV/THERMO`) — no C edit to start dumping ice-ON.** Clone
  `jobs/jax_step_dump_core2.sh`, flip the NO_ICE flags, set `FESOM_DUMP_PROBES` to ice
  coverage. The per-substep `fesom_dump.c` does NOT capture ice thermo outputs
  (flx_fw/flx_h/a/m/msnow/t_skin) — those need a small additive all-node dump hook modeled on
  `fesom_bulk_dump` (Task 6.2). *Lesson: the Phase-5 dump harness already supports ice-ON via
  env; the only C addition Phase 6 needs is per-kernel output dumps, not new gates.* (Task 6.1
  prep for 6.2.)

## Task 6.2 — ice thermodynamics

- **[verify] The sea-ice thermodynamics is a per-node MAP (no scatter) and ports BIT-EXACT —
  all 7 outputs match the C dump to MAP-class over all 126858 CORE2 nodes.** h/hsn/A ~1e-16,
  t_skin 5.6e-14, fw 3.5e-19, **ehf rel 4e-10**, thdgr 4.5e-19. The only non-machine field is
  `ehf` (rel 4e-10) — because `ehf = ahf + cl·(dhgrowth + …)` with `cl = rhoice·3.34e5 ≈ 3e8`
  amplifies the ~1e-16 `h` reassociation into ~1e-10 W/m². The dedicated `fesom_ice_thermo_dump`
  (re-runs `therm_ice` on copies → per-node inputs+outputs) makes it a pure feed-the-C-inputs
  MAP gate; config-A (EVP+FCT off) isolates the thermo since the input a/m/snow == the cold-start
  IC. *Lesson: a per-node kernel with a big multiplicative constant (cl) downstream loses a few
  digits in the derived flux even when the prognostic is machine-exact — gate the flux on a
  RELATIVE tolerance scaled by that constant, not an absolute one.* (`test_ice_thermo.py`,
  job 25395803, Task 6.2.)

- **[ad] Runoff activation is provable as an EXACT analytic gradient: `d(fw)/d(runoff) == 1`
  on every node.** Runoff enters at one line (`prec = rain + runo + snow·(1-A)`,
  `fesom_ice_thermo.c:318`) and `fw = prec + evap + fwice + fwsnw`, so `∂fw/∂runo ≡ 1`
  identically — a cleaner gate than any FD. This is the Phase-6 mechanism that turns the
  (Phase-5-inert) runoff back on: once `water_flux = -flx_fw` feeds the existing
  `sss_runoff_fluxes` (Task 6.3), river freshwater reaches the salinity BC. *Lesson: when a
  forcing enters a kernel linearly, gate its activation by the exact analytic derivative
  (`grad == 1`), not a finite difference.* (`test_ice_thermo.test_runoff_activates`, Task 6.2.)

- **[ad] The thermo AD surfaces (the fixed-5 skin-temp Newton, the 4-way albedo `where`, the
  freezing/melt min/max, the masked divides) are all finite — and `d(ehf)/d(SST)` even gives a
  clean FD↔AD plateau (~1e-16) because the thermo is NEAR-LINEAR in SST.** SST enters only via
  `obudget` (smooth exp/pow) + `o2ihf` (linear in `T_oc - Tfrez`); the skin-temp Newton uses
  `S_oc` (not `T_oc`), so varying SST is smooth (no kink) on an interior-ice subset where the
  `max(qhst,0)`/`max(sh,0)`/`min(hsn,…)` melt clamps are inactive. Contrast the Task-5.8
  finding (the *forced multi-step* model is non-smooth): the ISOLATED thermo kernel IS smooth in
  SST, so a quantitative FD↔AD is well-posed here. The masked-NaN guards that mattered:
  `con/hice` (`where(hice>0,hice,1)` — the ice-free class has thact≈0), `/rsss`, the Newton
  `/A3`, `tfrez`'s `√(S³)`. *Lesson: "the model is non-smooth" is about the assembled forced
  trajectory; an isolated per-node kernel can still be smooth in a chosen input — pick that
  input + a kink-free node subset for the per-task FD↔AD.* (`test_ice_thermo.py`, Task 6.2.)

- **[fidelity] The 7-class growth-rate loop sequentially refines the SAME skin temperature `t`
  (each class's 5-iter Newton warm-starts from the previous class's result), and the
  `thick>hmin` gate must mask `t`/`rhice`/`subli` — NOT skip the loop.** In JAX (no Python `if`
  on traced `thick`) the loop runs for all nodes; the ice-free result is masked out
  (`where(thick>hmin, looped, original)`), and `con/hice` is guarded so the ice-free lane (where
  `thact≈0`) stays finite through the masked-out Newton. The snow-accumulation `_dhsngrowth`
  baseline is captured AFTER the snowfall add (`fesom_ice_thermo.c:321-322`), so `dhsngrowth`
  counts only melt, not the snowfall. *Lesson: a C `if(cond){ loop }` over a traced condition
  becomes "run the loop unconditionally + `where`-mask the outputs + guard every divide for the
  masked lane" — the masked lane still executes and still backprops.* (`fesom_jax/ice_thermo.py`
  `therm_ice_cell`, Task 6.2.)

## Task 6.3 — ice-ocean coupling (the runoff handoff)

- **[fidelity] The ice-on `oce_fluxes` reuses the Phase-5 `sss_runoff_fluxes` virtual_salt +
  relax_salt math VERBATIM but DROPS the standalone `water_flux += ⟨water_flux+runoff⟩` term.**
  In ice-on, `water_flux = -flx_fw` (flx_fw already contains runoff via the thermo `prec`), so
  the standalone freshwater-balance term would double-count. Implemented as a backward-compatible
  `balance_water_flux=True` flag on `sss_runoff_fluxes` (default = the Phase-5 no-ice path,
  unchanged; ice-on passes `False`). The Phase-5-verified salt-balance code is reused, not
  re-derived. *Lesson: when a kernel splits into a no-ice and an ice-on variant that share most
  math, add a default-preserving flag to the existing (verified) function rather than forking it
  — the default keeps the old gate green, the flag is the only new surface to test.*
  (`fesom_jax/sss_runoff.py` + `ice_coupling.py`, `fesom_ice_coupling.c:125-179`, Task 6.3.)

- **[ad] The runoff handoff is provable as an EXACT gradient through thermo∘coupling:
  `d(water_flux)/d(runoff) = -1` everywhere.** runoff →(thermo, `d(fw)/d(runo)=1`)→ flx_fw
  →(coupling, `water_flux=-flx_fw`, `balance_water_flux=False` so no mean coupling)→ water_flux,
  giving `∂water_flux/∂runo ≡ -1` (freshwater in). `virtual_salt = S_top·water_flux` with
  `S_top>0` then makes river mouths freshen. ⚠️ Do NOT test freshening via
  `d(Σvirtual_salt)/d(runoff)` — that entangles the area-weighted **global mean** (which flips
  sign on large-area nodes, so only ~85% of river mouths show `<0`); the mean-free `water_flux`
  gradient is the clean signal. *Lesson: gate a forcing handoff on the term WITHOUT the global-
  mean coupling (here `water_flux`, not `virtual_salt`) — a summed-gradient through a
  mean-subtracted field mixes in every node's area weight.* (`test_ice_coupling.py`, Task 6.3.)

- **[verify] `ocean2ice` is free: `srfoce_u/v == uvnode[:,0]`.** The C `ocean2ice`
  (`fesom_ice_coupling.c:84-110`) computes `u_w` as the area-weighted mean of the surrounding
  elements' surface UV — which is **exactly** the recipe that already produced `State.uvnode`
  (the C comment `:44-45` says so), and Phase-5 `surface_forcing` already taps `uvnode[:,0]` for
  the bulk current. So `ocean2ice` is five taps (`T/S[:,0]`, `hbar`, `uvnode[:,0]`), no new
  scatter. *Lesson: before porting a "compute X at nodes" coupling routine, check whether the
  ocean step already computed X under another name — FESOM reuses `uvnode` for the surface
  current.* (`fesom_jax/ice_coupling.py` `ocean2ice`, Task 6.3.)

## Task 6.4 — EVP dynamics (the 120-subcycle scan)

- **[ad] The EVP `Δ = √radicand` singularity is tamed by a double-`where` safe-sqrt THEN the
  C's `max(Δ, delta_min)` clamp — keep `delta_min=1e-11`, do NOT raise it.** At `u_ice=0`
  (every step's subcycle 0) ε=0 ⇒ radicand=0 ⇒ a bare `sqrt` gives a `1/√0=∞` backward even
  though the clamp picks `delta_min` forward (the classic `0·inf` via the non-selected branch).
  `_safe_sqrt(radicand)` returns 0 with a finite gradient, then `jnp.maximum(·, delta_min)`
  reproduces the C value exactly. An EVP-port reflex is to bump `delta_min` to ~1e-8 for
  "stability" — unnecessary here and it would break the bit-exact `Δ` match; the safe-sqrt is
  the right fix. `d(Σσ²)/d(u_ice)` finite at u_ice=0 confirms it. *Lesson: a clamped sqrt
  (`max(sqrt(x), c)`) still needs the double-`where` on `x` — the clamp protects the forward,
  not the backward.* (`fesom_jax/ice_evp.py` `stress_tensor`, Task 6.4.)

- **[ad/perf] The 120 EVP subcycles are a FIXED count → a `jax.checkpoint`ed `lax.scan` with
  carry = (u_ice, v_ice, σ11, σ12, σ22).** σ is the elastic memory (carried, not re-zeroed);
  the scan is checkpointed so the 120-deep backward rematerializes rather than storing every
  subcycle (the inner-loop analog of the outer time-loop checkpointing). `Tevp_inv=3/ice_dt`,
  `dte=ice_dt/120`, `det1=det2=1/(1+0.5·Tevp_inv·dte)`. *Lesson: a fixed inner solver loop is a
  plain `scan` — checkpoint it so it doesn't blow the backward memory when nested in the outer
  step scan (Task 6.7 will measure the combined cost).* (`fesom_jax/ice_evp.py` `evp_dynamics`,
  Task 6.4.)

- **[verify] Ice-free elements must be MASKED in BOTH `stress_tensor` (freeze σ) and
  `stress2rhs` (no contribution), exactly as the C `if (ice_strength<=0) continue` skips them —
  not just one.** `ice_strength=0` unless all 3 vertices have `m_ice>0 AND a_ice>0` (so
  cavity + ice-edge elements are 0). If `stress_tensor` updated their σ (decaying it via det1)
  while `stress2rhs` scattered it, the velocity rhs would pick up spurious stress at the ice
  edge. Freezing σ where `ice_strength≤0` + zeroing the scatter contribution there matches the
  C's double-skip. The step-0 σ/u_rhs gates are **bit-exact** (per-element/node maps, no
  reassociation); only the END after 120 subcycles drifts to ~1e-9 (the accumulated
  element→node scatter reassociation, max|u_ice|=0.21 ⇒ rel ~5e-9). *Lesson: when the C skips an
  element in two consecutive loops, the JAX port must mask it in both — a `where` in the stress
  update AND a `where` in the scatter.* (`fesom_jax/ice_evp.py`, `test_ice_evp.py`, Task 6.4.)

## Task 6.5 — ice FCT advection (the 2-D Zalesak module)

- **[reuse] The entire ice FCT ports CSR-FREE — every step is element gather/scatter, and it
  matches the C BIT-EXACTLY (~1e-15).** The C uses the SSH-stiffness CSR (`rowptr`/`colind`)
  for the mass-matrix product, the cluster bounds, and the flux sums — but all three have an
  element-local form: (a) `(mm·X)[row] = Σ_{elem∋row} area/12·(X[row] + ΣX_elem)` (the FE
  consistent-mass block `area/12·(I+11ᵀ)` scattered — its row sum is the node CV area, so
  `mm·1 = area` falls out, and `mass_matrix_fill` is unnecessary); (b) the Zalesak cluster
  min/max over a node's graph neighbours == `jax.ops.segment_min/max` over the elements
  touching the node (on a triangle mesh, edge-neighbours == element-co-vertices); (c) the +/−
  flux sums are element→node `scatter_add`. So no CSR is ported. The single-step FCT matched the
  C to ~1e-15 (not the ~1e-12 scatter floor — the cold-start IC has little cancellation).
  *Lesson: before porting a CSR-based FE kernel, check whether each sparse op (matvec, neighbour
  min/max) has an element-local form — for a P1 triangle mesh they usually do, and the
  element-scatter port is simpler AND avoids threading the CSR.* (`fesom_jax/ice_adv.py`,
  job 25396145, Task 6.5.)

- **[fidelity] Use the ice FCT's OWN limiter floor `1e-12` (`fesom_ice_fct.c:458`), NOT the
  ocean FCT's `1e-16` — match each kernel's own constant.** The deep-read brief recommended
  unifying on 1e-16 for "AD stability", but the golden rule wins: with 1e-16 the JAX limiter
  ratio would differ from the C wherever the flux sum sits in (1e-16, 1e-12), breaking the
  bit-exact dump match. 1e-12 is already a finite floor ⇒ NaN-safe; AD doesn't need it tighter.
  Also: **no positivity clip** (the C doesn't) — the small antidiffusive overshoot past a_ice=0.9
  (~0.0019) is FCT-physical and IDENTICAL in JAX and C; `cut_off` clamps a≤1 afterward. *Lesson:
  a deep-read's "improve it" suggestion (tighter eps, add a clip) is subordinate to bit-exact
  fidelity — port the C's constant, let the documented downstream guard (cut_off) do the
  clamping.* (`fesom_jax/ice_adv.py` `_fem_fct`, `test_ice_adv.py`, Task 6.5.)

## Task 6.6 — assemble the ice step

- **[verify] The per-kernel gates are BIT-EXACT; the ASSEMBLED multi-kernel step is
  climate-close (~1e-6) — the 120-subcycle EVP floor propagates.** Each ice kernel matches the
  C to ~1e-15 (thermo, FCT) or step-0 bit-exact (EVP), but the EVP's END velocity carries a
  ~1e-9 scatter-reassociation floor (120 subcycles, Task 6.4), and `u_ice` feeds ustar (thermo),
  the ice-ocean stress (momentum) and the FCT — so the assembled step-1 post-step T/S match the
  C dump at ~1e-6, NOT bit-exact. This is the right gate altitude: verify each kernel tight in
  isolation (its own dump), accept climate-close for the assembled trajectory (the Phase-5
  multi-step gates were ~1e-9 too). *Lesson: don't chase bit-exactness on an assembled step that
  contains a reassociating iterative solver — gate the kernels tight, gate the assembly
  climate-close.* (`fesom_jax/ice_step.py`, `test_ice_step.py`, Task 6.6.)

- **[design] `ice_cfg` is a STATIC jit arg (an `IceConfig` NamedTuple — hashable), `None` ⇒ the
  pi/Phase-5 path is bit-identical.** Threading the whole ice subsystem through `step`/`integrate`
  needed exactly one new arg: `ice_cfg=None`. When `None`, the ice branch is a dead Python `if`
  (no trace), so the 376 pi + 63 Phase-5 CORE2 gates are untouched; when an `IceConfig`, the ice
  step runs and its prognostic a_ice/u_ice replace the static mask in the two existing couplings.
  The `IceConfig` properties (`cc`/`cl`/`Tevp_inv`/…) bake in as trace-time constants. *Lesson:
  gate a big new subsystem behind one static config arg defaulting to None — the old path stays
  a compile-time dead branch (bit-identical), the new path is opt-in.* (`fesom_jax/step.py`
  `step_jit`, Task 6.6.)

- **[workflow] ⚠️ Do NOT run two heavy CPU-JAX processes on the login node at once — XLA's
  CPU threadpool init (`pthread_create`) hits the per-user thread/process limit and one crashes
  (a faulthandler dump at `PjRtCpuClient`, NOT a code bug).** A CPU stability-smoke launched
  while the full pytest suite was running crashed both. The fix: run ONE CPU-JAX job at a time
  (background the suite, don't launch a second), and put heavy runs on the GPU (separate node).
  Also: `pytest … | tail` reports `tail`'s exit code (0) even when pytest crashes — grep the
  output for "passed", don't trust the pipe's exit. *Lesson: serialize CPU-JAX on the login
  node; verify a backgrounded suite by its "N passed" line, not the pipe exit code.* (Task 6.6.)

## Task 6.7 — GATE 6: stability + gradient (PHASE 6 COMPLETE)

- **[physics] ✅ SEA ICE CAPS THE SUPERCOOLING — the defining Phase-6 result.** The Phase-5
  no-ice CORE2 run supercooled the high-lat SST without bound (−1.9 IC → −16.5 day 5 → −22.8
  day 8 → max|vel|>3 blow-up ~day 8). WITH prognostic sea ice the thermo `o2ihf` (ocean→ice
  heat flux) + the freezing point pin SST_min at **−1.91 °C** (the local freezing point) for the
  whole 10-day run, which stays numerically stable (max|vel|=2.72<3, |SSH|<2.1, no NaN); ice
  grows physically (m_ice→2.94 m, a=1.0, extent ~2.5e13 m², drift ~1 m/s). Both standing Phase-5
  findings (supercooling, inert runoff) are now RESOLVED. *Lesson: the no-ice supercooling was
  exactly the PHYSICAL gap the C-port-matched model predicted; porting the ice thermo (not any
  numerical band-aid) is what fixes it — vindicating the "match the C, the limitation is real"
  call.* (`scripts/archive/core2_ice_stability_run.py`, job 25396309, Task 6.7.)

- **[ad] The assembled-ice backward is AD-SAFE (masked-NaN clean) but the EVP IC-gradient is
  STIFF (~1e16, finite) via `1/delta_min` — this is fine for the ML use case.** `d(SST)/d(T0)`
  on the N=4 ice model is finite everywhere, exactly 0 on masked lanes, nonzero on wet (the
  backward flows through the thermo Newton + the 120-subcycle EVP scan + the FCT limiter + every
  masked guard). But the wet magnitude reaches ~1e16: `zeta = ice_strength/delta_clamped·Tevp_inv`
  with `delta_min=1e-11` gives `zeta ~ 1e15` at rigid ice, and `d(stress)/d(eps) ~ zeta`
  propagates that. This is the GENUINE plastic-rheology stiffness (the EVP is nearly
  non-differentiable at rigid ice), not a bug — it's finite (the gate's criterion). For Phase-7
  TRAINING the NN-parameter gradients flow through the `k_ver`/`a_ver` mixing seam — `d(SST)/
  d(k_ver)` is well-conditioned (FD↔AD plateau **4.5e-10**) — NOT through the EVP `1/delta`. So
  the stiff EVP IC-gradient is a documented characteristic, not a blocker; if a future objective
  needs ice-dynamics gradients, raise `delta_min` for the gradient or `stop_gradient` the EVP.
  *Lesson: a finite-but-huge gradient through a plastic/iterative solver is the solver's
  conditioning, not a NaN bug — gate on finiteness, and confirm the ACTUAL trainable path (the
  mixing seam) is well-conditioned separately.* (`scripts/archive/core2_ice_grad_gate.py` [1]/[3],
  job 25396293, Task 6.7.)

- **[memory] Two CORE2-ice GPU memory traps on the A100-40: (a) stacking the per-step forcing,
  (b) >1 N-step backward per process.** (a) `cf.stack(dates_for_steps(1728))` puts ALL 1728
  steps × ~10 fields × nod2D × f8 ≈ **17.5 GB** of forcing resident on the GPU at once → the
  model OOMs; fix = generate `cf.step_forcing(*dates[i])` per step in the loop (tiny). (b) the
  grad gate ran 3 separate N=4 backwards (`d/d(k_ver)`, `d/d(T0)`, `d/d(m_ice0)`) in one process;
  each compiles a fresh reverse graph (~26.5 GB peak) and they accumulate → the 3rd OOMs;
  fix = `jax.clear_caches()` between probes, or one probe per job, or the A100-80. *Lesson: on a
  40 GB card, don't hold the whole forcing trajectory resident (stream it per step) and run one
  heavy backward per process.* (`core2_ice_stability_run.py`, `core2_ice_grad_gate.py`, Task 6.7.)

## Phase 6B — GM/Redi (Task G.1 — sw_alpha_beta + the seam + the dump hook)

- **[eos] `sw_alpha_beta` (McDougall 1987) is a bit-exact pointwise map — max|Δ|=0 vs the C over
  all 3.7M CORE2 wet lanes, like `density`.** The two coefficients (`sw_beta` = the 10-term
  saline-contraction polynomial; `sw_alpha = a_over_b·beta`, `a_over_b` the 11-term ratio) are
  written **term-by-term** in the C (`fesom_eos.c:336-369`), NOT Horner — mirror that exact
  left-to-right grouping and it matches bit-for-bit on CPU-eager (no FMA divergence; same
  expectation as the JM `density`). Inputs `t1=T·1.00024`, `p1=|Z[nz]|` (pressure proxy), `s35=S−35`.
  Smooth (no sqrt/divide) ⇒ trivially AD-finite, no guards. (`eos.compute_sw_alpha_beta`, G.1.)

- **[eos/mesh] ⚠️ `mesh.Z` is `[nl-1]` (layer midpoints), NOT `[nl]` — pad to `[nl]` like
  `pressure_bv` does.** `mesh.zbar` is the `[nl]` interface depths; `mesh.Z` (layer-centre depths)
  has one fewer entry. The C indexes `Z[nz]` over the layer range `[nzmin, nzmax)` (max index
  nl-2), so it never overflows; in vectorized JAX you must broadcast `Z` against `[N, nl]`, so pad
  `Zp = concat([Z, Z[-1:]])` (the padded tail is below-bottom → masked out). Got a
  `(N,48)+(1,47)` broadcast error until padded. (`eos.compute_sw_alpha_beta`, G.1.)

- **[ad/ml-hook] The 2nd ML-hook seam (GM/Redi eddy diffusivities `k_gm`/`redi_kmax`) extends
  `Params` with `dataclasses.field(default_factory=…)` defaults — so the old `Params(k_ver=,
  a_ver=)` 2-arg construction stays valid AND the pytree round-trips.** Adding leaves to a
  registered-dataclass pytree changes its structure; giving the new leaves config-constant
  defaults (via `default_factory`, NOT a bare array default) keeps every existing constructor +
  the 17-test gradient/integrate seam **bit-identical** (when GM is off the leaves are unused ⇒
  `d/d(k_gm)=0`, finite). Mirror of how `k_ver`/`a_ver` seamed the 1st hook in Phase 3.
  (`params.py`, `config.K_GM_MAX/REDI_KMAX`, G.1.)

- **[verify] GM/Redi is STATELESS, so its dump hook just SNAPSHOTS the already-computed arrays
  all-node — no re-run-on-copies (unlike the ice-thermo dump).** Every GM field (`sigma_xy`,
  slopes, `fer_K`, `Ki`, `fer_gamma`, `fer_uv`, …) is recomputed each step from T/S/N², so after
  the GM coefficient block (`fesom_step.c:124-130`) the `gm->*`/`dyn->fer_uv`/`aux->sw_*` arrays
  are exactly the outputs — `fesom_gm_dump` `fwrite`s them as raw f64 blobs (C row-major) +
  `gm_meta.txt`. **Dump the INPUTS too** (`T,S,bvfreq,hnode,hnode_new`) so ONE GM-ON dataset
  (`data/gm_dump_core2/`, job 25397273) feeds the JAX kernels the C inputs and gates G.1-G.4
  output-for-output. GM is mixing-independent ⇒ dump with `FESOM_MIX_SCHEME=PP` + ice OFF +
  `FESOM_NO_GMREDI` dropped. Reader: `io_dump.load_gm_dump`. (`fesom_step.c fesom_gm_dump`, G.1.)

## Phase 6B — GM/Redi (Task G.2 — neutral slopes)

- **[gm] `compute_sigma_xy` is the `eos.smooth_nod3D` element→node area-weighted scatter, but
  ÷Σarea (not 3·Σarea) and carrying the per-element ∇T/∇S.** Per node: ⟨∇_c T⟩ =
  Σ_{el∋n}(area_el·∇_c T_el)/Σarea_el, then `sigma_xy = (-α⟨∇T⟩ + β⟨∇S⟩)·ρ0`. Vectorize: per-
  element gradient `∇T_el = Σ_v gradient_sca[:,v]·T[elem_nodes[:,v]]` ((E,nl)), stack the 4 grads +
  the area into one (E,nl,5) tensor, broadcast to the 3 vertices, ONE `ops.scatter_add` →
  (N,nl,5), split → tx/ty/sx/sy/vol; `inv_vol = where(vol>0,1/vol,0)`. Bit-exact vs the C dump
  (el-range ⊆ node-range ⇒ `elem_layer_mask` suffices, same as the smoother). (`gm.compute_sigma_xy`,
  `fesom_gm.c:124`, G.2.)

- **[gm/verify] ⚠️ `neutral_slope` (UNTAPERED) has enormous dynamic range — slopes reach ~1e5-1e6
  where N²→the eps² floor — so gate it RELATIVE, never absolute.** `ro_z_inv = 2g/ρ/max(N²,eps²)`
  with `eps²=2.5e-11` ⇒ `ro_z_inv` up to ~8e8, and `slope = sigma_xy·ro_z_inv` reaches ~3e5 at
  weakly-stratified deep lanes. An absolute 1e-13 gate is meaningless there (max|val|~1e5); the
  field is eager-bit-exact vs the C but a ~1e-15 *relative* shift = ~1e-10 absolute. Gate
  `|Δ| ≤ atol + rtol·|ref|` (rtol=1e-12). The physically-consumed field is `slope_tapered` (the
  taper kills these huge slopes). (`test_gm_slopes`, G.2.)

- **[gm/fma] ⚠️ `slope_tapered = neutral_slope·√c1` has a `huge×tiny ≈ 0` lane (huge untapered
  slope × taper→0) whose result ~1e-10 carries the huge factor's FMA noise (~4e-10 abs) — gate
  isclose with a NEAR-ZERO ABSOLUTE FLOOR (atol≈1e-9), not pure-relative (rel>1 there).** And the
  XLA FMA-contraction of `√(sx²+sy²)` is the density-lesson effect AGAIN: **eager is bit-exact
  (max|Δ|=0) vs the C, but a fused path (jit, or eager under some process states) shifts it ~2e-16
  relative (machine-ε)** — and WHICH of neutral_slope/slope_tapered shows it varies run-to-run with
  the fusion decision. The lane IS ~zero slope (negligible Redi flux). (`test_gm_slopes`, G.2.)

## Phase 6B — GM/Redi (Task G.3 — init_redi_gm + the 2nd ML-hook)

- **[gm] `init_redi_gm` has two level-bound regimes — F1 uses the CONSERVATIVE bounds
  (`ulevels_nod2D_max`/`nlevels_nod2D_min`), F2 the REGULAR (`ulevels_nod2D`/`nlevels_nod2D`).**
  F1: resolution `scaling = min(√(area_surf·2/refscalresol²), 1)`, `fer_K_top=max(scaling·k_gm,
  K_GM_min)`, `Ki_top=max(scaling·redi_kmax, K_GM_min)`, and the baroclinic wave speed
  `cm = max(Σ_cons hnode·½(√bv0+√bv1)/π/K_GM_cm, K_GM_cmin)` (a depth REDUCTION over the
  conservative range → `fer_C=cm²`, scatter/reduction class ~1e-15). F2: `zscaling =
  clip(smin+(1−smin)e^{−|z|/zref}, smin, 1)`; `fer_K = fer_K_top·zscaling` on the **iface** range
  (`node_iface_mask`); `Ki = Ki_top·½(zscaling[nz]+zscaling[nz+1])` on the **layer** range, then
  the taper. For no-cavity CORE2 the conservative/regular *upper* bound collapses (ulevels≡1 ⇒
  nzmin=0) but the *lower* differs (cm integrates only to the shallowest surrounding cell's
  bottom). Verified map-class vs the dump. (`gm.init_redi_gm`, `fesom_gm.c:345`, G.3.)

- **[gm/ad/ml-hook] The 2nd ML-hook gradient is LIVE: `d(Σfer_K)/d(k_gm)=2.03e6` (finite,
  positive) flows through `init_redi_gm`.** `k_gm`/`redi_kmax` thread from `Params` →
  `fer_K_top`/`Ki_top` = `max(scaling·k_gm, K_GM_min)`; `d/d(k_gm)=Σ scaling·zscaling` over the
  iface range (the `max` unclamped at the default 1000). The `Redi_Ktaper`
  `Ki·√c1 + Redi_Kmin·|√c1−1|` ⇒ where the taper kills c1 (unstable strat bv≤0, c1=0),
  `Ki=Redi_Kmin=100` — matches the C. Same seam pattern as `k_ver`/`a_ver` (Phase 3); Phase 7
  swaps the NN here. (`params.py`, `test_gm_coeffs`, G.3.)

## Phase 6B — GM/Redi (Task G.4 — streamfunction TDMA + bolus velocity)

- **[gm] `fer_solve_gamma`'s tridiagonal geometry is a STATIC precomputed constant in full-cell
  linfs (`zbar_n=zbar`, `Z_n=Z`) — verified `hnode_new == zbar thickness` bit-exact (max|Δ|=0).**
  So `a[nz]=fer_C·(1/(zbar[nz-1]−zbar[nz]))·(1/(Z[nz-1]−Z[nz]))`, `c` similarly with the lower
  iface spacing, `b=−a−c−max(N²,1e-8)` — only `fer_C`/`bvfreq`/`fer_K`/`sigma_xy` vary per node.
  Build the body coefficients on `[nzmin+1,nzmax)` (conservative inner bounds), set the
  Dirichlet/padding rows to `b=1, a=c=d=0` (→ Γ=0), and the full-column `ops.tdma` reproduces the
  C's bounded Thomas sweep exactly. The two components (x,y) SHARE the matrix → two `ops.tdma`
  calls with the same `(a,b,c)`. **`fer_gamma` matched ~8.9e-15** (sequential Thomas ≈ bit-exact).
  (`gm.fer_solve_gamma`, `fesom_gm.c:492`, G.4.)

- **[gm] `fer_gamma2vel` (bolus velocity) is a gather + interface-difference ÷helem — ~1e-16,
  essentially bit-exact.** `fer_uv(c,nz,el) = (1/3)·(Σ_v Γ[v,nz,c] − Σ_v Γ[v,nz+1,c])/helem`:
  gather Γ to the 3 vertices, sum, difference adjacent interfaces, ÷`helem` (safe-divide
  `where(h>0,h,1)`), mask to `elem_layer_mask & h>0`. `helem` (static linfs) = `⅓Σ_v hnode` =
  `gather_nodes_to_elem(hnode).mean(axis=1)`. (`gm.fer_gamma2vel`, `fesom_gm.c:1035`, G.4.)

## Phase 6B — GM/Redi (Task G.5 — the GM driver + the bolus vertical velocity)

- **[gm] ⚠️ `fer_w = ale.compute_w(fer_uv)` — a PURE REUSE of the dump-verified `compute_w`, no new
  kernel.** The C computes the bolus vertical velocity with the byte-identical edge→node
  transport-divergence scatter + reverse-cumsum + ÷area as `w`, just driven by `fer_uv` instead of
  `uv` (`fesom_ale.c:124-152,166-186` — `c2` mirrors `c1`). So `fer_w = compute_w(mesh, fer_uv,
  helem)`; verified by composition (compute_w bit-exact vs the `w` dump in test_ale + fer_uv
  ~1e-16 in G.4) + the no-flux BC (`fer_w[nzmax]=0` exact, the bolus is divergence-free/
  streamfunction-derived) + activity (max|fer_w|~3e-4). The C wraps tracer advection with
  `uv+=fer_uv; w_e+=fer_w` and subtracts after; in functional JAX just PASS `uv+fer_uv`,
  `w_e+fer_w` into the advection — the carried `uv`/`w_e` are untouched, so the subtract-back is
  automatic. (`gm`, `fesom_step.c:312-332`, G.5.)

- **[gm] `gm_diagnostics` composes G.1-G.4 (sw_alpha_beta→sigma_xy→neutral_slope→init_redi_gm→
  fer_solve_gamma→fer_gamma2vel) and reproduces `fer_uv` END-TO-END at 2.2e-16** (essentially
  bit-exact) fed the C's T/S/bvfreq/hnode_new — the strongest GM gate (the whole chain, all-node).
  It returns `(fer_uv, slope_tapered, Ki, fer_K, fer_C)` — `fer_uv` drives the bolus (G.5),
  `slope_tapered`/`Ki` the Redi terms (G.6). `d(Σfer_uv²)/d(T)` through the full chain (incl. the
  TDMA) finite + nonzero. (`gm.gm_diagnostics`, G.5.)

## Phase 6B — GM/Redi (Task G.6 — the Redi tracer terms)

- **[gm/redi] ⚠️ G7b's 5 partial-cell branches A/B/C/D/E collapse to 3 CASES by level-membership
  `(in1=nz∈el1, in2=nz∈el2)`: el1-only (A∪D), el2-only (B∪E), both (C).** A and D are the SAME
  formula (el1-only, above vs below the overlap); B and E likewise. So per (edge, level) select
  `(Tx, Ty, dz, CX, CY)` by the 3 cases (`both/only1/only2` from `elem_layer_mask[el1/el2]`):
  `c = (CX·Fx + CY·Fy)·dz`, `Fx=Kh(Tx+SxTz)`; el1-only `CX=dyL,CY=−dxL`; el2-only `CX=−dyR,CY=dxR`;
  both `CX=dyL−dyR,CY=dxR−dxL`. The node-endpoint `Kh/SxTz/SyTz` (the `COMPUTE_KH_TZ_S` macro) are
  the SAME in all branches. Antisymmetric edge→node scatter (`+c→e1,−c→e2`), then
  `÷(areasvol·hnode_new)`. **Matched the C at 1.07e-14 first try** — the same "5-zones→masked-sum"
  collapse as the ocean upwind advection. (`gm_redi.diff_part_hor_redi`, `fesom_gm.c:824`, G.6.)

- **[gm/redi] ⚠️ The Redi K33 is just "AUGMENT Kv" — `Ty(nz)==Ty1(nz-1)` is ONE per-interface value,
  and `impl_vert_diff` already builds `a[nz]∝Kv[nz]`, `c[nz]∝Kv[nz+1]`, so passing `Kv+K33_aug`
  reproduces the C's `a∝(Kv[nz]+Ty)`, `c∝(Kv[nz+1]+Ty1)` with NO change to the diffusion kernel.**
  `K33_aug[k] = (geo_up·zinv)·s[k-1]²Ki[k-1] + (geo_dn·zinv)·s[k]²Ki[k]` (`s`=`slope_tapered[...,2]`
  the |slope|, static linfs geometry). The (3,3) Redi-tensor term = the isoneutral vertical
  diffusivity. (`gm_redi.k33_augmentation`, `fesom_tracer_diff.c:167-246`, G.6.)

- **[gm/redi/verify] The G7a/G7b explicit terms read `valuesold` (the AB2 pre-step T) for their
  gradients but APPLY to the post-advection T — gate by capturing T before/after each Redi piece
  all-node.** The C composes the Fortran `del_ttf` accumulation + `ale_reconstruct`, so in JAX each
  is a `delta` added to T with the `/(areasvol·hnode_new)` factor. The `fesom_redi_blob` hook
  (`FESOM_REDI_DUMP_DIR`) dumps `T_old/T_pre/T_g7a/T_g7b/tr_xy/tr_z` all-node ⇒ gate `T_pre+G7a==
  T_g7a`, `T_g7a+G7b==T_g7b` exactly (the dt is `fesom_phase1_dt`=the runtime 500, NOT a separate
  constant). (`fesom_step.c fesom_redi_blob`, `test_gm_redi`, G.6.)

## Phase 6B — GM/Redi (Task G.7 — assemble + GATE 6B)

- **[gm/step] ⚠️ The assembled GM/Redi step is BIT-EXACT class (7e-15), NOT climate-close — because
  GM/Redi is fully deterministic (no EVP-like reassociation floor).** The full GM-ON CORE2 step-1
  post-step T/S match the C GM-ON substep dump (`gm_step_dump_core2`) at **T 7.1e-15 / S 2.1e-14** —
  the same bit-exact class as the per-kernel G.1-G.6 gates, NOT the ice assembly's ~1e-6 (whose
  120-subcycle EVP END velocity carries a ~1e-9 floor, Task 6.6). The whole chain — bolus advection
  + G7a + G7b + K33 — is pure map/scatter, so it composes without an iterative-solver floor. This
  also **CLOSES K33's tight gate**: K33 had no isolated dump (G.6 only sanity-checked it); the
  assembled 7e-15 match is its tight validation. Step 2 is scatter-class (1.2e-9, the CG iter-count
  amplifying step-1's spread, bounded). *Lesson: gate an assembled step at the altitude of its
  LEAST-deterministic kernel — bit-exact if all-deterministic (GM), climate-close if it contains a
  reassociating iterative solver (ice EVP).* (`test_gm_step.py`, G.7.)

- **[gm/step] ⚠️ The Redi G7a/G7b read the PRE-step tracer = `st.T`/`st.S` (the `T_old` RETURNED by
  `advect_one_fct`), NOT the AB2 history `st.T_old`.** The C's `valuesold` at Redi time is what
  `init_tracers_AB_one` saved DURING this step's advect call = the pre-step `values` = `st.T` (which
  `advect_one_fct` returns as `T_old_new`). `st.T_old` (the carry going IN) is the PREVIOUS step's
  pre-step T — different at step ≥2. At step 1 they coincide, so a single-step gate would NOT catch
  the bug; the step-2 evolution gate (1.2e-9, bounded) is what confirms it. The bolus wrap is
  functional: pass `uv+fer_uv`, `w_e+fer_w` into advection (the carried uv/w_e untouched ⇒ the C's
  post-diffusion subtract-back is automatic); `hnode_new` is hoisted once after EOS (static linfs)
  so the GM block (substep 2) and the Redi reconstruction (substep 15) share it; K33 augments Kv
  before `impl_vert_diff` (no diffusion-kernel change). `gm_cfg=None` ⇒ a dead Python branch ⇒
  bit-identical (the `ice_cfg` precedent; the full 453-test suite stays green). (`step.py`, G.7.)

- **[gm/ad/ml-hook] ✅ The 2nd ML-hook gradient `d(SST)/d(k_gm)` is WELL-CONDITIONED (clean FD↔AD
  plateau 3.5e-6), NOT stiff — the risk-list worry was unfounded.** The eddy-flux hook plateau is
  the textbook V-shape (truncation error at large h, round-off at small, min at h=1e-5), unlike the
  EVP's `1/delta_min` stiffness (~1e16, Task 6.7). Both ML hooks are now training-ready end-to-end on
  the assembled CORE2 model: `d/d(k_ver)` plateau 5.8e-10, `d/d(k_gm)` 3.5e-6, masked-NaN `d/d(T0)`
  finite everywhere / 0 on dry / nonzero wet (the backward flows through the GM slopes safe-sqrt,
  the streamfunction TDMA, the Redi scatters). Backward memory at N=4 CORE2 GM-ON = **37 GB / 64 GB**
  (the per-step GM TDMA + Redi scatters add to the backward; the A100-80 is comfortable, the -40
  would be tight at N=4). (`scripts/archive/core2_gm_grad_gate.py`, job 25402381, G.7.)

- **[gm/physics] ✅ GM does PHYSICAL WORK — it smooths fronts: 10-day front |∇T| 7.42e-6 (GM-ON) vs
  7.89e-6 (GM-OFF), a ~6% reduction that GROWS monotonically (d1 6.72/6.82 → d10 7.42/7.89).** The
  bolus advection + Redi neutral diffusion flatten isopycnals (the eddy parameterization is active,
  not inert). Both runs 10-day stable (max|vel| 2.84/2.72 < 3, SST capped −1.91 by the ice thermo,
  no NaN). The within-step DYNAMICS (SSH, |vel|) are GM-INDEPENDENT — identical between GM-ON/OFF —
  because the bolus only redistributes tracers and is subtracted back before momentum/SSH; GM acts
  purely on T/S. GPU steady-state is ~0.09 s/step (the 8-9 s I saw at s1/s2 was compile + the first
  device_get sync), so the full 1728-step (10-day) run finishes in ~4 min. *Lesson: verify a
  parameterization is doing work with a matched ON/OFF run on a physical diagnostic (front
  sharpness), not just "it ran stably".* (`scripts/archive/core2_gm_stability_run.py`, jobs 25402379/80, G.7.)

## Phase 6C — KPP (planning + research; the pivot to finish the full model)

- **[process/pivot] After GATE 6B the user redirected: finish the full FUNCTIONING model (KPP) BEFORE
  the Phase-7a parameter-tuning on-ramp.** Phase 7a was scoped (the `calibrate.py` seam + the
  perfect-model `k_gm` twin) and two facts were verified before the pivot — preserved in
  `docs/plans/20260607-fesom-jax-paramtune.md`: (1) **`optax 0.2.8` installed clean** (pip dry-run
  confirmed jax/jaxlib 0.10.1 untouched; only `optax`+`absl-py` added); (2) **the `k_gm` twin is
  well-posed** — `gm.py:198` `k_top=max(scaling·k_gm, k_gm_min=2.0)` has **only a lower floor, no
  upper clamp at `k_gm_max=1000`**, so injecting `k_gm=1500` is unclamped, `fer_K ∝ k_gm` linearly.
  *Lesson: when the user redirects mid-scope, capture the in-flight design + any verified de-risking
  into a deferred plan so zero work is lost; don't just drop it.*

- **[kpp/framing] ⚠️ KPP is the *real* FESOM2 CORE2 default mixing scheme (`mix_scheme='KPP'`,
  `mix_scheme_nmb=1`); the JAX port has been running the OPT-IN PP (`pp.py`, `nmb=2`) all along.** The
  C dispatch (`fesom_step.c:257` KPP / `:259` PP, selected by `FESOM_MIX_SCHEME`; default KPP since
  `8d0cdbc`) and the GM/Redi sub-plan (`20260607-fesom-jax-gmredi.md:49-51`) both confirm it — the GM
  dump ran on `FESOM_MIX_SCHEME=PP` to match the JAX port. So **every JAX gate to date is PP-vs-PP**;
  porting KPP brings the model to the production config. *Lesson: verify which branch the "default"
  config actually selects — the reduced-namelist port matched PP because that's what the dumps used,
  not because PP is the model's default.*

- **[kpp/validation] CONTROLLED REPLAY is the load-bearing validation technique for forcing-amplifying
  kernels.** The C KPP port found a **live-run** dump diffs at **~52 % of nodes** vs Fortran — NOT an
  algebra bug, but the **step-1 surface-forcing transient** (a known C↔Fortran flux mismatch)
  perturbing `bfsfc`/`ustar` at ~every node, which `blmix` amplifies (`f1 ∝ bfsfc/ustar⁴`). A
  whole-field live diff is uninterpretable. The fix: **inject the reference-dumped INPUTS into the
  kernel under test, diff only its OUTPUTS** → isolates algebra from forcing noise → bit-faithful
  (C hit max|Δ|=3.18e-13). The JAX port must adopt this per-kernel (K.2–K.7), gating against the
  **C** (the JAX SoT, already Fortran-validated K0–K11, climate RMS 0.005–0.013 °C). The end-to-end
  check is the **climate gate**: JAX-KPP ≈ C-KPP ≪ the genuine PP↔KPP scheme gap (~0.085 °C, ~18×).

- **[kpp/ad] KPP is the kink-heaviest scheme yet — it has STRUCTURAL discreteness PP/GM did not.** The
  AD bar stays "no NaN/Inf in backward, finite incl. masked lanes" + a clean gradient where one
  physically exists. Inventory + treatments (full list in `…-kpp.md` §4): (1) **`ustar =
  sqrt(sqrt(|τ|/ρ₀))`** double-sqrt — ∞ backward slope at zero wind, and `ustar` sits in many
  denominators (`u*⁴` in `f1`, `u*³` in `hmonob`) → `_safe_sqrt`, the #1 priority. (2) the **OBL depth
  `kbl`** chosen by a thresholded bulk-Ri search (+ the `wscale` `int()` bin index, `caseA` sign) is
  discrete → vectorize as a masked first-crossing, **`stop_gradient` the integer index** but keep the
  **`hbl` interpolation weight differentiable** ("which level" discrete, "where within" smooth — same
  treatment as the FCT/upwind kinks). (3) the **`EPSLN=1e-40` denominators** stop Inf but NOT gradient
  blow-up (`d(1/den)/d ~ 1/den²` is huge at 1e-40) → replace with **physical floors** on the
  physically-small ones (`/(Rib_k−Rib_km1+ε)`, `bfsfc/u*⁴`, `hekman/max(|f|,ε)`). *Lesson: `+1e-40`
  is a forward-Inf guard, not an AD guard — audit every such denominator for backward conditioning.*

- **[kpp/seam] The C KPP port is DONE + validated → port FROM it, mirror its K0–K11 decomposition, and
  the only missing JAX input is `dbsfc`.** Input audit at the seam (`step.py:130`): `bvfreq` ✓,
  `eos.compute_sw_alpha_beta` ✓ (GM added it), `heat_flux`/`water_flux`/`stress_node_surf`/`sw_3d` ✓
  (in the CORE2 forcing, thread them to the call), `uvnode` ✓ (reuse `pp.compute_vel_nodes`) — but
  **`dbsfc` (the EOS surface-buoyancy difference, `Ritop=zk·dbsfc`) is NOT computed under PP**
  (`eos.py:7` says so) → **add it in K.5** (mirror `fesom_eos.c`). KPP follows the `gm_cfg`/`ice_cfg`
  static-config gate exactly (`kpp_cfg=None ⇒ PP bit-identical`); it is STATELESS (no new `State`
  fields, like GM); double diffusion + nonlocal flux are GATE-ONLY for CORE2. (Phase 6C planning,
  `docs/plans/20260607-fesom-jax-kpp.md`.)

## Phase 6C — KPP (Task K.0 — scaffolding + reference dumps)

- **[kpp/scaffold] `kpp_cfg=None` threads bit-identically exactly like `gm_cfg` — verified `kpp_cfg=None
  ⇔ KppConfig()` give max|ΔT|=max|Δuv|=0 through both `step_jit` AND the checkpointed `integrate`
  (scan body).** `KppConfig(NamedTuple)` is plain Python scalars/bools ⇒ hashable ⇒ a valid
  `static_argname` (the GM/ice precedent); KPP carries NO differentiable leaves of its own (the
  seam tunables stay in `Params`; `Ricr`/`visc_sh_limit`/backgrounds become Phase-7a targets there).
  Threaded `step.py` (`:53` sig, `step_jit`+`run` static_argnames) + `integrate.py` (sig, both the pi
  eager+scan and CORE2 eager+scan bodies, `integrate_jit` static_argnames). The arg is THREADED but
  UNUSED through K.0 (the `if kpp_cfg: kpp.mixing_kpp` gate is K.8) — an unused static arg is free.
  (`fesom_jax/kpp.py`, `step.py`, `integrate.py`, K.0.)

- **[kpp/init] ⚠️ The derived scalars `Vtc/cg/deltaz/deltau` AND the full 892×480 wm/ws lookup table
  recompute BIT-EXACTLY (max|Δ|=0) vs the C dump — so K.1's table builder is already pre-validated by
  the K.0 reader check.** Compute the derived scalars at module load with `math.sqrt`/`math.pow` (the
  same libm routines the C `pow()` calls ⇒ bit-equal), VERBATIM from `fesom_kpp.c:130-138` (do NOT
  re-derive Vtc/cg from a paper — research disagreed; trust the C association). They are frozen at the
  CORE2 values in `KppConfig` (NOT auto-recomputed from the tuple's own fields — re-derive if Ricr/concv
  change under Phase-7a). The table-build `pow(conas·u³−concs·zehat, 1/3)` base can go negative → the
  K.1 builder must clamp ≥0 (the table is a constant → build once, freeze, no grad). (`kpp.py`
  `_VTC`/`_CG`/`_DELTAZ`/`_DELTAU`, `io_dump.load_kpp_init`, K.0/K.1.)

- **[kpp/dump] The C KPP dump is plain TEXT (not the GM `.f64` binary): one `kpp_dump_s<step>_<tag>_rank<R>.txt`
  per kernel field + `kpp_init_rank0.txt` + `kpp_wscale_rank0.txt`.** Run single-rank (`--ntasks=1`),
  so `*_rank0.txt` carries every node (gid 1..N) in `myList` order, and (single-rank) JAX node `i` ↔
  global gid `i+1` (the GM-gate node alignment, made explicit) — `load_kpp_dump` reorders by gid
  (`out[gid-1]=row`, robust to partition order; verified gids are the identity here). Fast parse via
  `np.fromstring(sep=' ')` (1.5 s for the 90 MB `dVsq`). The dump job isolates KPP (`FESOM_MIX_SCHEME=KPP`,
  GM OFF, ice OFF — the controlled-replay gate feeds C inputs so config doesn't affect validity; ice OFF
  drops EVP noise) — 57 s, ~890 MB text, all step-1. A 10-comp `iceforce` ice-debug tag shares the dir
  (reuses the harness; harmless — gates request named tags). (`jax_kpp_dump_core2.sh`,
  `io_dump.{read_kpp_table,load_kpp_dump,load_kpp_init,load_kpp_wscale_sweep}`, K.0.)

## Phase 6C — KPP (Tasks K.1 + K.2 — lookup tables + wscale)

- **[kpp] The wm/ws lookup tables (892×482) + the 4 derived scalars are bit-exact CONSTANTS
  (max|Δ|=1.7e-18 / 0 vs the C init dump) — build once (`lru_cache` on `cfg`, host numpy → jnp
  constant, no grad through the table).** The `pow(·,1/3|1/4|1/2)` base is positive in every KEPT
  branch lane (verified analytically: `conas·u³−concs·zehat ≥ 70·|zehat|` when `zeta≤zetas`;
  `conam·u³−concm·zehat>0` for `zehat<0`; `1−conc2·zeta`/`1−conc3·zeta>0` since `zeta<0`), so
  `np.power(np.maximum(base,0), …)` only suppresses NaN in DISCARDED `np.where` lanes ⇒ the clamp is
  exact. (`kpp.build_wscale_tables`, `fesom_kpp.c:140-165`, K.1.)

- **[kpp/⚠️BUG] `wscale`'s bilinear `zfrac`/`ufrac` use the UNCLAMPED numerator minus the CLAMPED
  integer index (`fesom_kpp.c:184-191`) — so ustar beyond the table's `UMAX=0.04` EXTRAPOLATES
  LINEARLY (`ufrac>1`), it is NOT a clamped table edge.** I first clamped `uq=min(udiff/deltau,nnj)`
  and used `uq` for BOTH the index and the frac → at `ustar=0.05` got `vonk·0.04=0.0159` vs the C's
  `vonk·0.05=0.02`. The fix: `ju=trunc(min(uq_raw,nnj))` (clamped index) but `ufrac=uq_raw−ju`
  (UNCLAMPED numerator). `zfrac` was already right (numerator `zq` never clamped; only `iz` is). The
  remaining table-region residual (4.3e-13) is this extrapolation amplifying the table's ~1e-15
  last-ULP by `ufrac≈121` (the C's libm class); the stable region (`zehat>0`) is EXACT (0.0). *Lesson:
  in a clamped table lookup, check whether the fractional weight uses the clamped or the raw
  coordinate — FESOM extrapolates, so the raw one. Strong wind drives ustar≈0.2 ≫ UMAX, so this
  extrapolation path is HOT, not a corner case.* (`kpp.wscale`, K.2.)

- **[kpp/ad] `wscale` AD is finite everywhere incl. the ustar=0 zero-wind column: the `(int)` bin
  index is `jnp.trunc` (zero grad a.e. ⇒ discrete selection carries no cotangent), the bilinear
  weights stay differentiable ⇒ the gradient is the table's piecewise-linear slope; the stable-branch
  denom gets a `jnp.where`-safe dummy (`=1`) in the masked unstable lanes so the unused branch can't
  emit a `0·inf` masked-NaN.** Gated via the C `kpp_wscale_rank0.txt` sweep (201×101 over
  zehat∈[−1e-6,1e-6], ustar∈[0,0.05]) — spans table/stable/clamp/zero-wind. (`kpp.wscale`,
  `test_kpp_wscale.py`, K.2.)

## Phase 6C — KPP (Tasks K.3 + K.4 — interior mixing + ddmix gate)

- **[kpp] `ri_iwmix` matches the C dump BIT-EXACTLY (max|Δ|=0 over 3.8M iface points) — but only
  because step-1 `uvnode=0` (cold start) ⇒ shear=0 ⇒ `frit∈{0,1}` (a pure `sign(N²)` map).** The
  shear=0 dump exercises the edge copies + masking + the static-instability (N²<0→frit=1) branch but
  NOT the intermediate cubic `frit=(1−min(Ri/Riinfty,1)²)³` ⇒ add a SYNTHETIC test (a linear
  u-profile sets a controlled shear, N² targets Ri across (0,Riinfty)) — matched 3e-18. The C's
  two-pass scratch (pass-1 Ri edge copies are FULLY overwritten by pass-2 viscA/diffK edge copies,
  since pass 2 reads only the interior Ri) collapses to a single edge-copy-of-the-result
  (`take_along_axis` with `clip(k, nzmin+1, nzmax-1)`). `diffKt is diffKs` (the `Kv0_const` branch ⇒
  one array for both). Output range = `node_iface_mask` = `[nzmin,nzmax]` (KPP FILLS surface+bottom
  via edge copies, unlike PP which leaves them 0). (`kpp.ri_iwmix`, `fesom_kpp.c:219`, K.3.)

- **[kpp/ad] `ri_iwmix`'s `Ri=max(N²,0)/(shear+epsln)` epsln=1e-40 is forward-inert AND backward-safe
  in practice.** Any realistic shear `(Δu/Δz)²≫1e-40` so epsln never bites the forward; where
  `shear→0` the outcome is `frit∈{0,1}` with a CLAMPED (`min`) or ZERO (`max(N²,0)=0`) ratio whose
  `d(ratio)/d(Ri)=0` ⇒ no `1/epsln²` blow-up at the relevant lanes (and dry lanes have N²=0→Ri=0). The
  dz reciprocal is clamped `where(dz==0,1,dz)` (the surface k=0 + the bottom-pad duplicate, both
  masked) to kill the `1/0` Inf the `pp`/`eos` shear pattern would otherwise leave in discarded lanes.
  d/d(uvnode), d/d(bvfreq) finite. (`kpp.ri_iwmix`, K.3.)

- **[kpp] K.4 `ddmix` (double diffusion) + the nonlocal flux are GATE-ONLY for CORE2** —
  `assert_no_double_diffusion(cfg)` is a no-op (`double_diffusion=False`) and raises `NotImplementedError`
  if enabled (the C `#error` analog, `fesom_kpp.c:828-831`); `ghats` is *computed* in blmix (K.6) but
  `use_kpp_nonlclflx=False` ⇒ never wired into the tracer flux. (`kpp.assert_no_double_diffusion`, K.4.)

## Phase 6C — KPP (Task K.5 — pre-step + dbsfc + bldepth, the highest-risk kernel)

- **[kpp/⭐] `bldepth` (the OBL bulk-Ri search, historically the buggiest KPP kernel) VECTORIZES
  cleanly + matched the C on the FIRST try: hbl 6.5e-12, kbl 0/126858 mismatches, bfsfc 8.6e-23,
  stable/caseA EXACT.** The key realization: `Rib_k[nz]` has NO inter-level dependence — each level is
  a pure function of its own forcing (bfsfc/zehat/ws/Vtsq/Ritop/dVsq at nz) — so the C's two
  sequential per-node loops become two **masked first-crossings** (`jnp.argmax` of a bool = first
  True): loop 1 `Rib_k>Ricr → kbl1` + interpolated hbl, loop 2 `|zbar|>hbl → kbl`. The only sequential
  quantity, `Rib_km1`, is the gather `Rib_k[kbl-1]` with a `Rib_k[nzmin]=0` SENTINEL so a first-level
  crossing recovers the C's `Rib_km1=0` init. (`kpp.bldepth`, `fesom_kpp.c:317`, K.5.)

- **[kpp/⚠️] The loop-1-end bfsfc (feeding the Ekman/Monin-Obukhov gate) = bfsfc at kbl1 in BOTH the
  crossed AND never-crossed cases — no case split needed.** In the never-crossed case the C's final
  sw-interp-to-hbl collapses to `sw_3d[nzmax]` (the interp fraction → 1 exactly when
  `hbl=|zbar[nzmax]|`), which equals the top-of-loop bfsfc at nzmax = bfsfc at kbl1 (kbl1=nzmax). So a
  single gather `Bo+coeff_sw·(sw_surf−sw_3d[kbl1])` reproduces it. The Ekman/MO clamp
  `max(min(hbl,hlimit),|zbar[1]|)` applies only where `bfsfc1>0 && nzmin==0` (stabilizing forcing). The
  final-loop bfsfc interp uses SIGNED zbar `(hbl+zbar_km1)/(zbar_km1−zbar_k)`. (`kpp.bldepth`, K.5.)

- **[kpp/ad] bldepth AD finite everywhere (d/d{dVsq,Bo,bvfreq,dbsfc,sw_3d}): stop_gradient the integer
  kbl1/kbl, keep the hbl interp weight `(Ricr−Rib_prev)/(Rib_at−Rib_prev+ε)` differentiable, `_heaviside`
  = `0.5+copysign(0.5,x)` for stable/caseA (zero-grad regime switch), `_safe_sqrt(|bvfreq|)` in Vtsq.**
  Gradients are large (d/dBo ~1e10 — the hbl interp + `hmonob/(bfsfc+ε)` amplify) but FINITE; the
  masked-NaN gate K.10 confirms end-to-end. (`kpp.bldepth`, K.5.)

- **[kpp/⚠️] `ustar = sqrt( sqrt(τx²+τy²) / ρ0 )` is TWO nested sqrts, BOTH hitting 0 at zero wind
  (inner `sqrt(τ²)`, outer `sqrt(·/ρ0)`) ⇒ BOTH need `_safe_sqrt`.** The plan's "`sqrt(sqrt(|τ|/ρ0))`"
  notation is loose — it is the standard `sqrt(|τ_vec|/ρ0)` with `|τ_vec|=sqrt(τx²+τy²)` (exponent 0.5
  on the magnitude, NOT 0.25 — a test-math trap I hit). d/d(stress) at τ=0 finite (the #1 AD priority —
  ustar is in many downstream u*³/u*⁴ denominators). `dVsq=0` at the cold-start step 1 (uv=0; dump
  ~2e-30). (`kpp.prestep`, `fesom_kpp.c:792-821`, K.5.)

- **[kpp/eos] `dbsfc = −g·(ρ_surf(z)−ρ_insitu(z))/ρ_insitu(z)` (the surface parcel brought adiabatically
  to z vs in-situ) is AD-clean (ρ_insitu≈1030 ⇒ no singular denom, unlike bvfreq's 1/Δz) and bit-exact
  (max|Δ|=0) vs the C dump.** Gated via the GM dump's step-1 T/S — the SAME PHC IC (the EOS runs on the
  pre-mixing state, so the mixing scheme is irrelevant; GM-dump=PP, KPP-dump=KPP, identical step-1 T/S).
  `dbsfc[surface]=0` automatic (surface parcel at its own depth). Added to `eos.py` (only KPP reads it —
  PP skipped it, hence absent until now). **Extended the C KPP dump (jax-mesh-export) with `sw_3d` +
  `sw_alpha`** (bldepth reads them live — not previously dumped) so the controlled-replay has ALL its
  inputs; surgical C edit + rebuild + rerun (1 min). (`eos.compute_dbsfc`, `fesom_eos.c:138`,
  `jax_kpp_dump_core2.sh`, K.5.)

## Phase 6C — KPP (Task K.6 — blmix, the C's hardest replay)

- **[kpp] `blmix` vectorizes as "per-node scalars → cubic-over-interfaces" and matched the C's hardest
  replay: blmc 1.9–3.0e-13 (the C hit 3.18e-13), dkm1 2.6e-14.** Per node (via gathers at the discrete
  matching level): `kn = min(kbl − int(caseA), nzmax-1)` (stop-grad), the one-sided slope
  `½(dvdz+|dvdz|) = max(dvdz,0)` (AD-safe kink), `gat1=visch/(hbl+ε)/(w+ε)`, `dat1=min(−slope/(w+ε)+f1·visch,
  0)` with `f1=stable·conc1·bfsfc/(u*⁴+ε)`. Then the cubic `blmc = hbl·w·sig·(1+sig·G)`, `G=a1+a2·gat1+a3·dat1`,
  over the BL interfaces `nz∈[nzmin+1, min(kbl-1,nzmax-1)]` (a masked range). Channel cross-wiring:
  blmcM←dcol ch0 (viscA)/wm, blmcT←dcol ch1 (diffKt)/ws, blmcS←dcol ch2 (diffKs)/ws. `dcol` = the
  ri_iwmix outputs directly (their nzmax edge-copy already gives `dcol[nzmax]=dcol[nzmax-1]`); `hnode`
  is passed in (a State field, static full-cell linfs ⇒ the GM dump's hnode for the replay). `dkm1` at
  kbl-1 uses σ from zbar (not Z). (`kpp.blmix`, `fesom_kpp.c:449`, K.6.)

- **[kpp/⚠️verify] `ghats` has the GM huge-dynamic-range signature — gate RELATIVE.** `ghats =
  (1−stable)·cg/(ws·hbl+ε)` reaches ~2e3 where the velocity scale `ws→0`, so its absolute residual
  (7.5e-11) carries that magnitude × FMA noise = relative ~3.7e-14 (bit-faithful). Gate
  `|Δ|≤atol+rtol·|ref|` (the `test_gm_slopes` neutral-slope pattern), not absolute. `ghats` is
  COMPUTED but CORE2 zeroes it outside the BL in the combine and never wires it into the tracer flux
  (`use_kpp_nonlclflx=False`). AD finite through all of blmix. (`test_kpp_blmix.py`, K.6.)

## Phase 6C — KPP (Task K.7 — enhance + smooth_blmc + combine + node→elem)

- **[kpp] The KPP driver tail is BIT-EXACT (viscA/viscAE 2.2e-16, diffKt/diffKs 6.7e-16, ghats 0.0) —
  better than the scatter-class ~1e-12.** `enhance` modifies blmc at the SINGLE interface kbl-1 per
  node (a masked `where(k==kbl-1, blend, blmc)` update; `delta=(hbl+zbar[kbl-1])/(zbar[kbl-1]−zbar[kbl])`,
  blend = `om·interior + delta·(om²·dkm1 + delta²·dkmp5)`, `dkmp5=caseA·interior+(1−caseA)·blmc`) and
  scales ghats[kbl-1] by (1−caseA). `smooth_blmc` = `eos.smooth_nod3D(channel, 3)` — the SAME
  N²-smoother, 3 sweeps. `combine` = `max(interior_ri, smoothed_blmc)` within the BL (nz<kbl), ghats=0
  below; `Av` = 3-vertex node→elem mean + bottom-fill + `minmix` floor; `Kv = diffKt` (T-channel, both
  T&S in CORE2). The bit-exactness comes from the combine's `max` picking the deterministic interior in
  most lanes + the smoother reassociating identically single-rank. AD finite. **K.1→K.7 = the complete
  KPP forward chain, all controlled-replay bit-faithful + AD-finite.** (`kpp.enhance`/`assemble_mixing`,
  `fesom_kpp.c:588-924`, `test_kpp_enhance.py`, K.7.)

## Phase 6C — KPP (Task K.8 — wire KPP into the assembled step)

- **[kpp/🎯headline] The wired KPP step is BIT-FAITHFUL to the C, not the "sanity match" the plan
  expected — because the ~52 % step-1 forcing transient is a *C↔Fortran* artifact, and the JAX forcing
  is a validated 1:1 port of the *C* forcing (Phase 5).** Running one assembled JAX KPP CORE2 step
  (PHC IC + JRA55 1958, KPP/GM-off/ice-off, dt=500) vs the C dumps: stress_node_surf 4.4e-16, Kv/Av @
  probes (post-mo_convect) 1.7e-21/0.0, all-nodes pre-mo_convect diffKt/viscA/viscAE 2.7–4.2e-12,
  hbl/ustar all-nodes 9.5e-9/6.0e-17. **Lesson:** before assuming a documented forcing-transient diff
  applies, check WHOSE forcing the reference used — a per-kernel replay isolates algebra from forcing,
  but here the *driver-level* gate is ALSO bit-faithful because both sides share the (ported) forcing.
  Don't pre-loosen a gate on a borrowed caveat. (`test_kpp_step.py`, K.8.)

- **[kpp/⚠️seam] The stress KPP reads for `ustar` is the ICE-BLENDED node stress, not the raw bulk
  stress — and `oce_fluxes_mom` runs even with ice dynamics OFF.** `forcing->stress_node_surf` is
  written by `fesom_bulk_compute` (raw) then OVERWRITTEN in place by `fesom_ice_oce_fluxes_mom`
  (blended: `sic·a_ice + atm·(1−a_ice)`, `fesom_ice_coupling.c:230-252`), which `fesom_main.c:1073-1075`
  calls unconditionally (only `FESOM_NO_WIND` skips it). The dump used the static-ice mask (a_ice=0.9
  where IC SST<0, u_ice=0), so the blend is active at cold nodes. The JAX `compute_surface_fluxes`
  already computed this blend (`sns_b`) for the element `stress_surf` but only exported the element
  mean — K.8 exports the node `sns_b` as `SurfaceFluxes.stress_node_surf` (and `ice_oce_fluxes_mom`
  now returns `(stress_surf, sns)`, threaded through `IceStepOut`). Verified vs the C `iceforce` dump
  (cols 8–9 = the final blended `stress_node_surf`) at 4.4e-16. (`surface_forcing.py`, `ice_step.py`,
  `ice_coupling.py`, K.8.)

- **[kpp] `mixing_kpp` mirrors `pp.mixing_pp`'s contract — `(Kv,Av,uvnode)` post-mo_convect — so KPP is
  a one-line drop-in at the step seam.** The C does compute_vel_nodes + (PP|KPP) + shared `mo_convect`
  in the step driver (`fesom_step.c:251-264`); the JAX `mixing_pp` already bundled compute_vel_nodes +
  mo_convect, so `mixing_kpp` does too (it imports `pp` for both — no cycle: pp has no kpp dep). The
  `DUMP_SUB_MIXING=4` probe is POST-mo_convect, so bundling mo_convect keeps the gate honest. KPP is a
  CORE2 forced-path feature → it **raises** on the pi path (no `step_forcing` ⇒ no heat/water/stress);
  the raise is at trace time (`kpp_cfg` is static) so it's clean. `kpp_cfg=None` ⇒ the PP branch is a
  dead `if` (no trace) ⇒ byte-identical. (`kpp.mixing_kpp`, `step.py` substep 4, K.8.)

- **[kpp/ad] The assembled-driver backward (`d/dT` through all of `mixing_kpp`) is finite + nonzero and
  runs in ~24 s on the LOGIN node** — the per-kernel safe-sqrt/stop-grad/physical-floor treatments
  compose cleanly through the full chain incl. compute_vel_nodes (element→node scatter), smooth_nod3D,
  and mo_convect. A single-kernel backward is light (unlike the full multi-step CORE2 trajectory
  backward, which RAM-thrashes the login node) — so a driver-level AD smoke fits in the suite; the
  full assembled-STEP masked-NaN grad gate is K.10 (SLURM). T enters via N²/dbsfc/α/β.
  (`test_kpp_step.py`, K.8.)

- **[kpp/⚠️jax-trap] `@lru_cache` returning *jnp* arrays leaks tracers across jit traces — cache the
  numpy build, cast to jnp FRESH per call.** `build_wscale_tables` was `@lru_cache` and returned
  `jnp.asarray(wmt), jnp.asarray(wst)`. The FIRST `step_jit` trace (is_first_step=True) called it,
  creating trace-bound `DynamicJaxprTracer`s that the cache stored; the SECOND trace (is_first_step=
  False, or the eager step-1 vs the `lax.scan` body in `integrate`) reused those cached arrays →
  `UnexpectedTracerError: a reference to an intermediate value ... escaped the scope`. The fix: split
  into `_build_wscale_tables_np` (`@lru_cache`, returns **numpy** — trace-independent) + a thin
  uncached `build_wscale_tables` wrapper doing `jnp.asarray` (each trace bakes its OWN constant; the
  expensive host build stays cached, only the cheap host→device cast repeats). **Why it hid through
  K.1–K.8:** the kernel tests + the K.8 step test all ran a SINGLE eager trace (no cross-trace reuse);
  the bug is jit/scan-trace specific and only the K.9 multi-step `step_jit` run (is_first_step True→
  False) first surfaced it. **Lesson:** a one-step forward gate is necessary but not sufficient — add
  a 2-step *jitted* smoke (both `is_first_step` variants) to exercise the second trace, and never
  cache device arrays keyed on a static config. (`kpp.build_wscale_tables`,
  `test_kpp_step.py::test_kpp_two_jitted_steps_no_leak`, K.8/K.9.)

## Phase 6C — KPP (Tasks K.9 + K.10 — climate/stability + the gradient gate → GATE 6C)

- **[kpp] The end-to-end "JAX-KPP ≈ C-KPP" claim lives at the STEP level (K.8, 1e-12), NOT a multi-day
  field diff — the multi-day gate is stability + the distinct-from-PP scheme signal.** KPP+GM+ice ran
  10 days stable (worst |vel|=1.885 m/s, SST capped −1.91 °C, ice bounded) AND a matched PP+GM+ice
  baseline gave a surface SST/SSS RMS of 0.129 °C / 0.063 psu vs the KPP run — the genuine scheme
  difference (~the C-class C-PP-vs-KPP 0.085 °C). So the discriminating chain is **JAX-KPP ≈ C-KPP
  (1e-12, the bit-faithful step) ≪ JAX-PP↔KPP gap (0.13 °C)** — ~11 orders of separation. A multi-day
  *field* comparison to the C diverges by FP chaos regardless of correctness (the Task-5.8 finding), so
  don't gate on it; gate on the step-level fidelity + the physical scheme signal. (`core2_kpp_stability_run.py`,
  K.9.)

- **[kpp/ad] The masked-NaN `d(mean SST)/d(T0)` through the ASSEMBLED multi-step KPP model is clean
  (non-finite=0, masked max|g|=0.0, wet 7e-5) — the kink-heaviest scheme survives the assembled
  backward.** The per-kernel AD treatments (safe-sqrt ustar/Vtsq, stop-grad kbl + differentiable hbl
  interp, f1/gat1/dat1 physical floors, the smooth_blmc/node→elem linear ops) compose through the
  4-step checkpointed scan backward at CORE2 scale (28 GB peak on the A100, 44 %). Do NOT require a
  smooth plateau through the discrete kbl — the bar is finite-everywhere + a well-conditioned gradient
  where one physically exists. (`core2_kpp_grad_gate.py` [3], K.10.)

- **[kpp/ad] A static-NamedTuple config field can still be a gradient target — `cfg._replace(field=
  traced)` traces it through any kernel that reads `cfg.field` directly (no hashing).** The KPP-tunable
  gradient `d(mean Kv)/d(K_bg)` = +0.9952 (additive interior diffusivity ⇒ FD plateau 1.1e-11) was
  taken by replacing the one `KppConfig` field with a traced scalar and running the mixing chain with
  the wscale tables PREBUILT from the static cfg (the only consumer that hashes cfg is the lru_cache'd
  table build — keep it on the static cfg). This is the Phase-7a pattern preview: KPP's `Ricr`/
  `visc_sh_limit`/backgrounds become tuning targets by lifting them from the static `KppConfig` into the
  traced `Params`. (`core2_kpp_grad_gate.py` [Kbg], K.10.)

## Sea-ice climate bias (Phase 6 follow-up — first real multi-year climate comparison)

- **[🎯big] Step-level bit-faithfulness does NOT guarantee a matching multi-year CLIMATE — run the
  end-to-end comparison.** Every gate through GATE 6C was step-level/per-kernel (controlled replay,
  step-1 bit-faithful). The FIRST annual-mean comparison vs the C-port-KPP + Fortran-KPP refs (via
  `m32_climate_compare.py`) found JAX-vs-C SST **0.49 °C RMS / −0.15 °C bias** vs the **0.005–0.014 °C**
  inter-reference budget (C-vs-Fortran, Kokkos-CUDA-vs-C) — a ~35–100× excess. **Lesson:** add a climate
  comparison to the acceptance gate; "≈ C at step 1" ≠ "≈ C climate". The C/Fortran/CUDA references
  agree to ~0.01 °C, so that IS the achievable bar — a from-scratch vectorized port is NOT exempt.
  (`core2_kpp_climate_run.py`, `kpp_bias_map.py`.)

- **[ice/diagnostic] The bias localized cleanly: high-lat marginal sea-ice only; open ocean
  (−45..+45°) matches C to the bit-faithful 0.006–0.024 °C ⇒ KPP/dynamics/open-ocean-forcing are
  SOUND.** Surface-trapped (gone by 200 m), in the seasonal-ice seas (Okhotsk/Bering). The fingerprint:
  `m_ice` **flips sign by hemisphere** (Arctic too thin −0.15 m, Antarctic too thick +0.27 m) while
  `a_ice` is high at both poles. Opposite-sign-N/S ⇒ first suspect was a hemisphere-dependent term
  (Coriolis `∝sin lat` / metric `∝tan lat`), but the **entire EVP dynamics + `metric_factor` (max|Δ|=0
  vs `tan(rot_lat)/R`) + Coriolis are bit-faithful to C** — RULED OUT. So it's THERMO/FORCING acting on
  two regimes (perennial vs seasonal ice). **Lesson:** opposite-N/S is a strong localizer; spatial +
  lat-band maps (`kpp_bias_map.py`) beat global RMS for diagnosis. (Investigation plan
  `docs/plans/20260607-fesom-jax-seaice-climate-bias.md`.)

- **[⚠️blind-spot] Cold-start step-1 gates can't see velocity/shear-dependent bugs.** At step 1
  `uv=u_ice=0` ⇒ shear/`dVsq`/strain = 0, so KPP shear-Ri, the EVP metric terms (`mfac·v̄` etc.), and
  any drift-dependent path are multiplied by zero and pass trivially. The EVP `metric_factor` value was
  only exercised *within* step 1's 120 subcycles (u builds up) — but a thermo/forcing offset that needs
  the spun-up circulation stays invisible for months. **Add a later-step (nonzero-velocity) dump-gate
  and a climate comparison.** (Phase 6/6C gates, the sea-ice bias.)

## Sea-ice climate bias — ROOT CAUSE (2026-06-07): `ice_dt` desynced from the ocean `dt`

- **[🎯ROOT-CAUSE] The high-lat sea-ice climate bias was NOT a kernel port bug — it was a CONFIG
  desync: the climate run stepped the ocean at `dt=1800` but built `IceConfig()` with the default
  `ice_dt=500`, so the ENTIRE ice subsystem integrated 3.6× too slowly.** `IceConfig.ice_dt`
  defaults to 500 (a build-time placeholder the docstring says to override per-run); the C
  `fesom_ice_setup` instead DERIVES it (`ice_dt = ice_ave_steps*dt`, `fesom_ice.c:231`). With the
  desync, every ice rate is wrong at once: thermo growth/melt (`rhow*ice_dt`, `o2ihf*ice_dt/cl`),
  FCT transport (`vol*ice_dt*…` in `_tg_rhs`), and EVP timing (`dte=ice_dt/120`, `Tevp_inv=3/ice_dt`).
  The ice "clock" ran at 500/1800 = 0.28× real time ⇒ sluggish, under-evolved ice. **Fix:**
  `cfg = cfg._replace(ice_dt=cfg.ice_ave_steps*dt)` at the top of `ice_surface_step` — derive it from
  the ocean `dt` so the desync is structurally impossible (`ice_step.py`). **VERIFIED (re-run, 1958):
  SST RMS 0.490→0.0107 °C (46×), m_ice RMS 0.196→0.0030, polar bands 0.71–0.77→0.009–0.017 — now
  INSIDE the 0.005–0.014 °C inter-reference budget at all latitudes. The whole high-lat bias was this
  one config line.**

- **[⚠️why-masked] The bug hid through ALL of Phase 6/6B/6C because every gate ran at `dt=500`,
  where `ice_dt=500` is COINCIDENTALLY correct.** The step dump gates (`test_ice_step`/`test_kpp_step`,
  `DT=500`), the kernel dumps (the C dump runs used dt=500), and EVERY stability/grad script
  (`core2_{ice,gm,kpp}_stability_run`, `core2_ice_grad_gate` — all `DT=500.0`) sat exactly on the one
  timestep where the default is right. The `dt=1800` climate run (the Fortran-KPP timestep) was the
  FIRST thing to run the ice off its default ⇒ the first to expose it. **Lesson: a config field with a
  default that is only valid for ONE value of another field is a latent footgun — gate at ≥2 distinct
  values of the coupling field (here dt), or derive the dependent field so it can't desync.** The
  `_replace` fix is a no-op at dt=500 (so all 55 ice/step tests stay green) and only bites at dt≠500.

- **[ice/fingerprint→cause] The opposite-sign-N/S `m_ice` fingerprint is explained by sluggish ice
  relaxing toward an IC seeded with the OPPOSITE asymmetry from equilibrium.** The cold-start IC
  (`fesom_ice.c:246-280`, `ice.ice_initial_state`) seeds SH `m_ice=2.0` THICKER than NH `m_ice=1.0`;
  the true (C) 2-yr climate is the reverse — NH 1.06 > SH 0.89. A 3.6×-too-slow ice is pulled toward
  the IC ⇒ too THIN in the NH (1.0→0.91 vs C 1.06) and too THICK in the SH (2.0→1.15 vs C 0.89) =
  the observed opposite sign; the retained concentration ⇒ `a_ice` high at BOTH poles ⇒ cold
  surface-trapped SST in the marginal-ice seas. **Lesson: "opposite-sign by hemisphere" need not be a
  hemisphere-dependent TERM — it can be a uniform rate error acting on an asymmetric IC.** This is
  why the metric/Coriolis hunt (`∝sin lat`/`∝tan lat`) was the wrong tree: those were bit-faithful;
  the asymmetry lived in the IC, not the physics. (Investigation plan §0/§1.)

- **[audit/method] Ruling out EVERY kernel is what forced the search up to the config/threading
  level — the negative result was the signal.** A full static re-audit (this session) confirmed
  `ice_thermo`↔`fesom_ice_thermo.c`, the FCT `ice_adv`↔`fesom_ice_fct.c` (TG-RHS, the `_mm_times`
  CSR-mass reconstruction, low/high-order solves, the full Zalesak limiter), the EVP `evp_setup`
  (`ice_strength` incl. the load-bearing `0.5`, `inv_mass`, `inv_areamass`), `ice_coupling`
  (`ocean2ice`/`ice_oce_fluxes`/`_mom`), and `atm_ice_stress`/bulk are ALL faithful, AND every
  `IceConfig` constant matches `fesom_ice.c:53-111` (incl. `h0=h0_s=0.5` ⇒ `lid_clo` is NOT
  hemisphere-split). When the kernels are all faithful and a systematic bias remains, the bug is in
  the WIRING (what dt/state each kernel is fed), not the algebra. (Sea-ice bias investigation.)

- **[ice/threading] Minor (NOT the bias): JAX's `srfoce_u/v` is one extra step lagged vs the C.**
  `ocean2ice` reads the carried `state.uvnode[:,0]`, which substep-4 computed from the PREVIOUS
  step's input `uv`; the C `fesom_ocean2ice` recomputes the area-weighted node velocity fresh from
  the current `dyn->uv`. So the ice sees `uv_out(N-2)` where the C sees `uv_out(N-1)` — a ~30-min lag
  at dt=1800. `compute_vel_nodes` IS the C's exact area-weighted incident-surface-element recipe, so
  the only diff is the lag, not the recipe. Left as-is (sub-dominant); revisit if a residual remains
  after the `ice_dt` fix. (`ice_coupling.ocean2ice`, `step.py` uvnode threading.)

## Phase 8 — sharding (Task S.1 — the `dist_<NP>` partition reader)

- **[🎯ownership-asymmetry] Only NODES are uniquely partitioned; ELEMENTS and EDGES are redundantly
  owned at partition boundaries.** Verified on CORE2 `dist_2`: `Σ_d myDim_nod2D == nod2D` exactly
  (126858, interior lists disjoint), but `Σ_d myDim_elem2D = 245221 > elem2D = 244659` (overlap 562)
  and `Σ_d myDim_edge2D = 372225 > 371644` (overlap 581). A boundary element whose 3 vertices span
  two ranks sits in the **interior** (`myDim`, not the halo) of *both* ranks — that IS the "redundant
  compute over the halo" model (both ranks compute it; the broadcast then only needs to refresh the
  outer halo). **Load-bearing for S.5:** a reduction over elements/edges must assign each shared
  entity a UNIQUE owner (e.g. lowest-rank, or a precomputed owner flag) — naively summing over
  `myDim` double-counts the boundary. Node reductions (CG dots, `integrate_nod_2D`) are safe. The
  global counts are therefore `max(gid)+1` (the id space is dense `[0,count)`), NOT `Σ myDim`.

- **[index-conventions] `dist_<NP>` is 1-based Fortran; shift to 0-based — but NOT every field.**
  Confirmed by reading how `fesom_halo.c` consumes each: `rPE`/`sPE` are MPI **rank** ids fed
  straight to `MPI_Isend`/`MPI_Irecv` (`:166,182`) ⇒ already 0-based, **no shift**; `rlist`/`slist`
  are LOCAL field indices used as `[...]-1` (`:178,196`) ⇒ shift; `rptr`/`sptr` are 1-based cumulative
  offsets (`rptr[0]==1`) ⇒ shift so `rptr[0]==0` and `rlist[rptr[k]:rptr[k+1]]` slices cleanly;
  `myList_*` are 1-based gids ⇒ shift. After the shift: `rlist` ∈ `[myDim, myDim+eDim)` (the halo
  lanes a rank receives into), `slist` ∈ `[0, myDim)` (the interior lanes it sends). Getting the
  rPE/sPE shift wrong (subtracting 1 from a rank id) is a silent off-by-one that only surfaces as a
  wrong neighbour at exchange time — read the *consumer* (`fesom_halo.c`), not just the file, to know
  which fields are indices vs ids.

- **[reader-design] Mirror `fscanf(" %d")` with a tokenise-once `_IntStream`, not `np.loadtxt`.**
  `rpart.out`/`my_list`/`com_info` are free-format (whitespace == newline, ragged rows), and the C
  reads a known *count* of ints per block then stops — `np.loadtxt` (regular columns) can't model
  that, but `f.read().split()` + a cursor + `np.array(tok[i:i+n], int32)` does, and the slice-parse
  keeps even the 63k-entry `myList` blocks to one numpy call. `rpart.out` actually holds
  `npes + npes counts + nod2D owner-ids`; the C reads only the first `1+npes` (builds the vestigial
  `part[]` prefix, never indexed downstream) and ignores the per-node owner vector — mirror that.

- **[pytree-ragged] A registered-pytree dataclass can hold ragged per-rank data as `tuple`-of-arrays.**
  `Partition` is read for all `npes` ranks in one process (vs the C's per-rank `mype`), so per-rank
  `myList`/`com` are ragged. Storing them as `npes`-long tuples of numpy arrays (counts as rectangular
  `[npes]` arrays) flattens cleanly: `tree_flatten` recurses into the tuples and the nested (also
  registered) `ComStruct`, yielding all arrays as leaves, with the global scalar counts as static
  meta. It is **host metadata** (consumed by S.2's numpy build), never device-put — registering it is
  house-style + lets `tree_map` work, not a correctness need. The serial `npes==1` `synth_serial`
  (identity `myList=arange`, empty coms) makes the sharded path reduce to the dense model.

## Phase 8 — sharding (Task S.2 — sharded-mesh build + export)

- **[exchange-as-gather] For the gate, the broadcast exchange is an `all_gather` + per-lane gather,
  encoded as `(src_dev, src_lane)` `[P,Lmax]` per kind — no `slist`/`rlist` segment bookkeeping.**
  A halo lane reads its owner's *interior* value; an **interior lane reads itself (identity)** — that
  is exactly the C broadcast (owner→halo overwrite, interior untouched). Built from a `_owner_map`
  (global id → lowest-id interior owner + that owner's interior lane), NOT from the `com_struct`:
  halo lanes are *by construction* `[myDim:myDim+eDim)` (FESOM's `myList` order), so `rlist` only
  *reorders* them by neighbour — irrelevant to a gather that refreshes every halo lane from its owner.
  Verified gid-consistent on dist_4 (every halo lane's owner gid == the lane's gid; halo never
  self-owned). `ragged_all_to_all` (the scalable form that *does* need the segments) is a perf
  follow-up; the `com_struct` stays in `Partition` for it.

- **[interior-identity] Never refresh interior lanes — only halo.** Elements/edges are redundantly
  owned (S.1), so a boundary element is interior on ≥2 devices; each computes it independently
  (~1e-15 apart from FP reassociation). Refreshing interior from a canonical owner would impose one
  device's value on the others — a deviation from the C (which leaves them independent). The
  identity-on-interior rule keeps the N-vs-1 gate clean: each device's owned copy = its own compute =
  the 1-device value to ~1e-12. Halo source choice among redundant interior owners is immaterial
  (same value to ~1e-15).

- **[omit-CSR] `nod_in_elem2D` (the node→elem CSR) is used ONLY by the host PHC IC builder
  (`phc_ic.py`), never a step kernel** (grep-confirmed) — so it is **omitted** from the per-device
  bundle. S.2b builds the IC on the host then partitions the result, so the ragged-CSR pad (the one
  genuinely awkward field to shard) is never needed. Audit "who reads this field" before paying to
  shard it.

- **[pad-for-AD] Pad value by dtype: float→`1.0`, int→`0`, connectivity→`-1`, bool→`False`.**
  Floats pad to **nonzero** (1.0) not 0 so a masked pad lane that feeds a denominator (`1/area`,
  `1/elem_area`) stays finite — the masked-NaN AD rule, now on the device-pad axis. The gathered masks
  are `False` on pad lanes, so a padded "entity" is fully masked regardless; the pad value only
  matters for finiteness of unmasked intermediates. Connectivity pads `-1` (the existing boundary
  sentinel); owned elements carry **no** `-1` (their 3 vertices are all local — 0 sentinels in
  `elem_nodes[:myDim]` on dist_4), so no owned output depends on a sentinel gather (S.2's safety
  proof). Only halo/eXDim lanes carry `-1`, and those are masked / refreshed.

- **[noop-invariant] The `npes==1` sharded mesh is array-equal to the dense `Mesh`** (all non-static,
  non-CSR fields, squeezing the `P=1` axis) — the additive-sharding guarantee. `g2l` is the identity
  `arange` at `npes==1`, so connectivity remap is a no-op; replicated `zbar`/`Z` are kept global.
  This test is the cheap proof that the single-device path is structurally untouched.

## Phase 8 — sharding (Task S.2b — partition State / forcing / IC)

- **[same-Lmax] State and mesh MUST pad to the same `Lmax`** — factor a single `local_sizes(partition)`
  used by both `build_sharded_mesh` and `partition_state`. Then the state's pad lanes `[n_local:Lmax]`
  coincide exactly with the mesh's invalid (mask-`False`) lanes ⇒ the padded state is provably inert
  (masked out of every owned computation). If the two derived `Lmax` independently they could drift and
  a "padded" state lane could land on a *valid* mesh lane — a silent corruption. One source of truth.

- **[detect-leading-dim] Shard a pytree field by DETECTING its entity axis (size == nod2D / elem2D),
  not a hardcoded node/elem list.** `State`'s 40 fields are node- or elem-leading; `StepForcing` is
  `[nod2D]` for one step but `[n_steps, nod2D]` when scanned. A size-match finds the node axis in both
  (the `n_steps`-vs-`nod2D` sizes never collide), so one `_shard_along_axis(arr, ml, Lmax, axis)`
  handles single + stacked + node + elem uniformly and survives a field-list change.

- **[host-IC] Build the IC globally on the host, then `partition_state` — do NOT port a distributed IC.**
  `State.rest` / PHC IC / ice cold-start all already produce a *global* `State`; gathering it to
  per-device padded form is a pure reshape. This **sidesteps the C's PHC `extrap_nod3D` per-sweep halo
  exchange** (a startup cost the C pays because it builds the IC already-distributed) — the same
  host-build trick used for the SSH operator. The serial-`npes==1` `partition_state` is array-equal to
  the dense `State` (the no-op invariant), so the single-device IC path is untouched.

- **[forcing-pytrees] `ForcingStatic`/`StepForcing` are `NamedTuple`s ⇒ already JAX pytrees** (no
  registration needed). Partition them field-by-field: node fields gather to `[P, Lmax_nod]`, the scalar
  `ocean_area` stays replicated (it becomes a `psum` over owned nodes in S.5), and a scanned stack
  `[n_steps, nod2D]` → `[P, n_steps, Lmax_nod]` (node axis sharded, `n_steps` preserved for the scan).

## Phase 8 — sharding (Task S.3 — broadcast halo-exchange primitive + identity gate)

- **[shard_map-convention] Fold the device axis INTO the leading dim — `[P*Lmax, …]` sharded
  `PartitionSpec('p')`, NOT `[P, Lmax, …]`.** `shard_map` keeps a sharded axis at its *local* size: a
  `[P, Lmax]` global with `P('p')` gives each device `[1, Lmax]` (a stray size-1 axis the body must
  squeeze everywhere). Reshaping to `[P*Lmax, …]` and sharding `P('p')` gives each device `[Lmax, …]`
  directly — so the step body (S.7) operates on the natural `[Lmax, …]` shape **unchanged**. The
  `(P, Lmax)`-stacked S.2 arrays just `.reshape(P*Lmax, …)` at device-placement.

- **[exchange=all_gather+gather] The broadcast exchange is `all_gather` then a per-lane gather** —
  `g = all_gather(field, 'p', axis=0); out = g[src_dev, src_lane]`. The fancy index on `g`'s leading
  two axes (`[P, Lmax]`) handles `[Lmax]`, `[Lmax,nl]`, `[Lmax,nl,2]` in one line (trailing axes ride
  along). Interior lanes are identity (`src_dev=self`, `src_lane=self`), halo lanes read their owner's
  interior — exactly the C `fesom_halo_exchange` (owner→halo overwrite, interior untouched). `src_lane`
  is always `≥0`, so the exchange gather NEVER hits a sentinel (no masked-NaN risk inside the
  collective). `all_gather` is the simplest verifiable collective and correct for 2–4 devices;
  `ragged_all_to_all` (the scalable form, needing the `com_struct` slist/rlist) is a perf follow-up.

- **[identity-gate] Ported `fesom_halo_identity_test`**: set owned lanes to their gid, halo to the
  sentinel, exchange, assert every halo lane now carries its owner's gid (+ corruption recovery — clobber
  a halo lane, re-exchange, restored). Passes for all 3 kinds (nod/elem/elem-full) × {2,4} devices and a
  multi-level field. The exchange is **linear** in `field` (gather is linear); its vjp is the reverse
  exchange (`all_gather` transpose = reduce-scatter `psum`, gather transpose = scatter-add → halo
  cotangents flow additively back to owners) — JAX handles it automatically; FD-grad-checked on interior
  AND halo lanes.

- **[fake-device-gate] Collective tests need ≥2 CPU fake-devices, set at process start
  (`XLA_FLAGS=--xla_force_host_platform_device_count=N`) BEFORE jax init** — so they `pytest.skip` in
  the default 1-device suite and run as a separate `run_suite.sbatch` SHARDING group (4 devices). The
  host-side foundation (`partit`/`shard_mesh`/`partition_state`) needs no fake-devices (pure numpy +
  pytrees) and stays in the ocean group. A 4-device process can test `dist_2` too (subset the mesh to
  `jax.devices()[:2]`).

## Phase 8 — sharding (Task S.4 — exchange schedule + scatter gate)

- **[🎯loop-bound-verified] The `PORTING_LESSONS §4` halo-bound rule HOLDS for the JAX sharding — a
  LOCAL scatter gives each OWNED entity its complete sum, no special loop bound.** Verified on dist_4:
  every owned node has ALL its incident edges AND incident elements in its local list, and every owned
  element has all its `edge_tri`-contributing edges local (0 violations). So a kernel run over a device's
  local (owned+halo) entities with the existing `segment_sum` produces, on **owned** entities, the SAME
  sum as the global single-device kernel (modulo FP reassociation ~1e-13). The post-kernel broadcast is
  needed ONLY to refresh the (incomplete) HALO copies for the next kernel — it does not fix owned values.
  Confirmed by the scatter gate: owned edge→node and edge→element scatters match the global to 1e-11
  *before* any broadcast; the broadcast then makes the halo match too. This is why the C "redundant
  compute over `myDim+eDim(+eXDim)` + broadcast" model is correct, and why no JAX kernel scatter needs a
  change for sharding — only the local connectivity (S.2) + the post-kernel exchange (S.7).

- **[schedule-as-data] The per-substep exchange schedule is a DATA module (`halo_points.py`), not inline
  code** — ported from the C `MPI_PORT_REPORT.md` "Halo exchanges per timestep" table (ocean, ~30
  exchanges) + the `fesom_exchange_nod2D` call sites in `fesom_ice_{evp,fct,coupling,thermo}.c`. Each
  `Exch` records (substep, field, kind, **post/intra**, C-ref). S.7 iterates the `post` ones as simple
  inserts; the `intra` ones need a kernel split (recorded in `FUSED_KERNELS_NEEDING_SPLIT`).

- **[intra-kernel-splits] FIVE fused JAX kernels exchange MID-kernel ⇒ must be split in S.7:**
  `momentum.visc_filt_bidiff` (the bilaplacian exchanges `u_b/v_b` then `u_c/v_c` mid-kernel),
  `tracer_adv.advect_one_fct` + the `ice_adv` FCT (Zalesak exchanges `fct_LO` then `plus/minus` around
  the limiter), `ssh._pcg` (CG exchanges `pp`/`rr` per iteration — S.6), and the **EVP subcycle**
  (`u_ice/v_ice` exchanged INSIDE the 120-step `lax.scan` — a collective *inside* `scan` under
  `shard_map`, which must lower/transpose). A fused 2nd stage that reads a halo-stale 1st stage gives an
  owned-node boundary error — the C's hardest surface; splitting exposes the seam.

- **[elem-full=superset] The `all_gather` exchange refreshes the FULL local elem extent
  (`eDim+eXDim`), so one `'elem'` map serves both the C's `elem2D` and `elem2D_full` exchanges.** A
  superset refresh is always correct for the N-vs-1 gate (no kernel relies on a *stale* halo); only the
  per-substep C-N dump diff (S.9c) would need the exact `eDim`-only intermediate, restricting the refresh
  to `[myDim:myDim+eDim]`.

## Phase 8 — sharding (Task S.5 — distributed reductions)

- **[all-node-reductions] Every per-step reduction is NODE-based, so the S.1 element/edge
  redundant-ownership caveat does NOT bite the reductions.** `_area_mean` (virtual-salt / relax-salt /
  water-flux balances) and the CG dots are all sums over nodes; `ocean_area` is `Σ areasvol_surf` over
  nodes. Nodes are uniquely owned, so `owned_mask` (`i<myDim`) IS the unique-owner mask ⇒ owned-node sum
  + `psum` is exact. (`ice_coupling.ice_oce_fluxes` routes through `sss_runoff._area_mean` — there is one
  reduction primitive, not two.) If a future element/edge reduction appears, it would need the
  min-owner mask, not `owned_mask`.

- **[reduction-gating] `global_sum(vals, owned_mask, axis_name=None)` — `axis_name=None` ⇒ plain masked
  sum (single-device), a real `axis_name` ⇒ owned-sum + `jax.lax.psum`.** Routing `_area_mean` through it
  with `owned_mask=None` default keeps the `npes==1` graph the **exact** `jnp.sum(x·area)/ocean_area`
  (byte-identical — the 9 sss tests + the dead-branch discipline confirm); S.7 threads the real
  `owned_mask`/`axis_name` through the step's call chain. `psum` is only valid inside `shard_map` (it
  needs the mapped axis), so the `None` path is mandatory off the sharded path, not just an optimization.

- **[psum-out-spec] A `psum`'d scalar is replicated across devices ⇒ `shard_map` `out_specs=PartitionSpec()`
  (empty)** returns it directly (all devices hold the same total). 2/4-device owned-sum + `psum` matches
  the single-device global sum to ~1e-12 (reduction reassociation), and a deliberately-corrupted halo
  value leaves the result unchanged (masked) — the owned-mask correctness check.

## Phase 8 — sharding (Task S.6 — distributed CG solve)

- **[🎯operator-loop-bound-by-VALUE] The SSH stiffness stencil EXCEEDS the node halo — but every excess
  owned-row entry is EXACTLY zero, so a local matvec is still exact on owned rows.** Unlike the S.4
  *scatter* loop-bound (which held topologically), the global `S`/`M⁻¹` operator has owned-row columns
  *outside* the local node list (11664 entries on dist_2, 20466 on dist_4). Keeping only
  (row-local ∧ col-local) entries would silently DROP them — but **all of them have exactly-zero stiffness
  AND preconditioner value** (the operator deliberately keeps the full topological pattern incl. numeric
  zeros, `fesom_ssh.c`; the far "wing" columns reached through eXDim-halo elements are all zeros, and the
  MITgcm precond is `∝ S[i,j]` so it is zero wherever `S` is). So `(S_local·x)[i] == (S·x)[i]` EXACTLY for
  owned `i` (the dropped terms are `0·x = 0`). `partition_ssh_operator` **asserts no NONZERO owned-row entry
  is dropped** — a mesh/config that ever violated it fails loudly instead of corrupting the owned matvec.
  Lesson: the operator analog of the loop-bound must be checked on VALUES, not just topology — "is the
  stencil inside the halo?" can be *no* and the scheme still correct because the overshoot is numeric zero.

- **[fold-exchange-into-matvec] Fold the halo exchange INTO `ssh_matvec`/`ssh_precond` ⇒ the `_pcg` body is
  structurally UNCHANGED (only the dots → `global_dot`, `n` → global count).** The C's per-iteration
  schedule ("exchange `pp` before each SpMV, `rr` after the residual update") maps EXACTLY: the matvec
  broadcast-exchanges its input (`pp`) before the local SpMV; the precond exchanges its input (`rr`) right
  before its SpMV — which is exactly *after* `r = r − α·Ap`. So the CG loop needs no per-step exchange
  plumbing: the matvec/precond closures carry the `SSHHalo`, and `custom_linear_solve`'s `matvec` refreshes
  the halo automatically every SpMV. The carry vectors' halo lanes are **scratch** (refreshed inside
  matvec/precond, never trusted); only OWNED lanes are the real state, masked into every dot. The dense
  path (`halo=None`) guards the exchange behind `if halo is not None` and the dots default to `jnp.sum`
  (`reduce=None`) ⇒ the **exact `v1.0` graph** (43 single-device ssh tests stay green, dump ~1e-18 + AD).

- **[🔴iteration-count-robust] CORE2 CG = 127 iters (cold) / 130 (warm) — NOT pi's ≈3 — yet the count is
  robustly device-deterministic, and `d_eta` matches to MACHINE PRECISION.** The `ssh.py` docstring's
  "≈3 iters, cond≈800" is **pi**; the real CORE2 operator (dt=1800, nod2D=126858) is far stiffer and the
  loose `soltol=1e-5` stop lands deep in the trajectory. Captured the residual-vs-threshold margin on the
  REAL KPP+GM+ice rhs (`scripts/debug/capture_core2_ssh_rhs.py`): consecutive residuals near the stop cross the
  threshold by only a factor **~1.09** (tightest margin: the last *above* iterate sits 0.93 % above `rtol`)
  — but that is **~10 orders of magnitude** above the ~1e-15 `psum` reassociation, so the count CANNOT
  drift. Verified N==1 iteration count (127/130 on 2 and 4 devices) AND owned `d_eta` agreeing to
  **~3e-16 abs (1e-15 rel)** — far tighter than the 1e-12 budget, because (a) each owned row's local
  `segment_sum` is over the same nonzero terms in the same order (bit-identical per row) and (b) the
  contracting CG damps the ~1e-15 dot-reassociation. The residual RMS divides by the GLOBAL node count
  (`halo.n_global`), not `b.shape[0]` (= local `Lmax`) — getting that wrong would shift `rtol` per device.

- **[collective-in-while_loop-lowers] `all_gather` + `psum` inside a `lax.while_loop` inside
  `custom_linear_solve` inside `shard_map` LOWERS and runs (review #4 resolved).** The data-dependent CG
  trip count is safe because the `psum`'d residual is identical on every device (no deadlock — all devices
  exit the loop on the same iteration). `custom_linear_solve(symmetric=True)` reuses the (exchange+SpMV)
  matvec as its own transpose; that matvec represents the symmetric global `S` on owned lanes, so the
  implicit-diff cotangent (`S⁻¹·x̄` via the tight `transpose_solve`, also sharded) is structurally intact —
  the gradient is gated in S.8, *not* AD-through-the-`while_loop`. Confirmed on 4 CPU fake-devices (~54 s,
  9 tests) and an early real-4×A100 run (the formal multi-GPU gate is S.9).

- **[capture-realistic-rhs] Gate the distributed CG on a CAPTURED real-config rhs, not a synthetic one.**
  The iteration-count margin is a property of the operator (mesh+dt) AND the rhs spectrum (config), so the
  fixture is `ssh_rhs` read straight off `state.ssh_rhs` after a real assembled `step()` (KPP+GM+ice,
  dt=1800) — 2 steps give the cold-start (`x0=0`) and warm-start (`x0=d_eta_step1`) cases. Saved on `/work`
  (gitignored, ~1 MB each) like the dumps. The serial `npes==1` `partition_ssh_operator` is byte-equal to
  the dense operator (rows/cols/vals), and the serial sharded solve reproduces the dense `d_eta` — the
  no-op invariant proving the sharded code path collapses to the single-device model.

## Phase 8 — sharding (Task S.7 part 1 — device-mesh placement + local reconstruction)

- **[reconstruct-local-mesh] Run the UNMODIFIED `step` under `shard_map` by reconstructing a per-device
  LOCAL `Mesh` with `Lmax` STATIC sizes.** The kernels use `mesh.nod2D`/`elem2D`/`edge2D` only as
  `segment_sum` `num_segments` / array-shape bounds (audited — `myDim_edge2D` is `build_ssh_operator`-only,
  not a step kernel), so setting the reconstructed mesh's static sizes to the LOCAL `Lmax` makes every
  scatter allocate `[Lmax]` and the kernels run on each device's shard with **zero code change**. The
  omitted node→elem CSR (`nod_in_elem2D`, S.2 — IC-only) is a step-unused dummy. Pass the `Mesh` to
  `shard_map` as a "folded" container — `Lmax` static meta + `[P*Lmax_kind, …]` leaves + a `Mesh`-shaped
  `PartitionSpec` tree (`'p'` for entity fields, `()` for replicated `zbar`/`Z` + the CSR dummy) — and
  inside the body it IS a valid local `Mesh`. `npes==1` reconstruction is array-equal to the dense `Mesh`
  for every step-read field ⇒ the whole step under `shard_map` is **byte-identical** to dense (`max|Δ|=0`).

- **[🎯check_vma-false] `shard_map(..., check_vma=False)` is REQUIRED to run the unmodified kernels.**
  JAX 0.10's `shard_map` tracks "varying manual axes": a value derived from a sharded input is typed
  `{V:p}`. The kernels' tridiagonal-solve (Thomas) and FCT `lax.scan`s init their carry with a CONSTANT
  `jnp.zeros` (NOT varying) while the body produces a varying carry → the strict check rejects the
  `float64[n]` vs `float64[n]{V:p}` carry-type mismatch ("manual axis types do not match"). `check_vma=False`
  treats every value conservatively as per-device-varying (always correct here — no cross-device replication
  to exploit inside a `shard_map` body), so the scans lower unchanged. ⚠️ Contrast S.6: the CG `while_loop`
  lowered with `check_vma=True` (default) because ALL its carries derive from the sharded `b` (uniformly
  varying). The constant-carry scan is the case that needs the relaxation — reach for it whenever an
  unmodified body has a `lax.scan`/`while_loop` seeded by a literal.

- **[interior-match-diagnostic] Without exchanges, the deep-interior owned nodes already match single
  device — a cheap proof the local kernels are correct on real shards.** On CORE2 `dist_2`, 58 % of owned
  nodes match the dense full-step `T` to 1e-10 with NO halo exchanges (their multi-substep stencil never
  reaches a halo lane); the boundary 42 % is the halo footprint the exchanges (rest of S.7) refresh. So a
  multi-device step that LOWERS + matches on the interior validates the placement + the per-shard kernel
  correctness independently of the exchange wiring — debug the plumbing before the boundary.

## Phase 8 — sharding (Task S.7 part 2 — interleave the halo exchanges + split the fused kernels)

- **[🎯JAX-redundant-compute-needs-FEWER-exchanges-than-C] In the JAX sharding model the kernels run over
  the FULL local extent `[0,Lmax)`, so per-NODE intermediates are auto-complete on the halo — only SCATTER
  results need an exchange.** This is the key divergence from the C MPI port. The C computes per-node fields
  over OWNED entities only (`myDim`) and must EXCHANGE every intermediate a downstream kernel reads at the
  halo (e.g. it exchanges the raw `bvfreq` BEFORE `smooth_nod3D`). In JAX, a per-node field like raw `bvfreq`
  = f(T,S at the node) is computed for owned AND halo lanes (T/S halos are fresh), so it is already complete
  on the halo — **no pre-smooth exchange needed**. The exchanges that ARE needed are exactly where a kernel
  produces an incomplete-on-halo value (a SCATTER over edges/elements, whose halo entity's contributing
  edges aren't all local) and a LATER kernel reads it at the halo (a gather-to-element or a per-node
  cluster). So the C `§4` loop-bound rule ("who reads this into the halo?") still applies, but the
  "producing loop covers `myDim+eDim`" half is automatic — only the scatter-result exchanges remain.

- **[🎯use-the-Kokkos-SYNC_MAP] The reference ports' per-substep exchange map is the authoritative checklist
  — read it BEFORE wiring, not after debugging.** `port_kokkos/docs/SYNC_MAP.md` lists every substep's
  internal-exchange (`D21`) bracket. It caught two scatter-result exchanges the `MPI_PORT_REPORT` table folds
  into a kernel and I had missed: (1) `momentum_adv_scalar`'s node advection `un_u/un_v` (a scatter), gathered
  back to elements at the cell vertices — needs a `nod` exchange before the gather; (2) the FCT element tracer
  gradient `tr_xy` (wrong on eXDim halo elements), read by `fill_up_dn_grad` — needs an `elem` exchange. Both
  are the "scatter/incomplete value read at the halo" pattern. The per-field N-vs-1 diagnostic (diff every
  State field owned-and-halo) localizes a missing one in one run; the reference map prevents needing the run.

- **[🎯FCT-upwind-flip-is-climate-close-NOT-a-bug] The Zalesak FCT amplifies the ~1e-12 input reassociation to
  ~1e-3 on the tracer via UPWIND FLIPS — the documented "climate-close, not bit-identical" non-determinism,
  not a missing exchange.** After all exchanges were wired, the N-vs-1 step matched to <1e-9 on EVERY field
  except the FCT tracers (T,S) and the heavily-**cancelling** SSH divergences (`ssh_rhs`/`ssh_rhs_old`).
  Three independent proofs it is NOT a halo gap: (a) the per-field diagnostic showed ALL FCT *inputs*
  (`uv,w_e,helem,hnode,T_old`) match to 1e-9 on owned AND halo — a missing exchange would diverge an input's
  halo; (b) `S` (constant in the test ⇒ zero advection) matches to <1e-9 while `T` (with a gradient) does
  not — the error is advection-magnitude-dependent; (c) the owned and halo errors are EQUAL (a boundary
  exchange bug makes the halo worse). Mechanism: the upwind flux `±0.5(vflux±|vflux|)·Tmean` flips which face
  value it takes when `vflux` (the edge volume flux) crosses zero, and a 1e-12 reassociation near a zero-flux
  edge flips it ⇒ an O(1) flux swing. The C and Kokkos ports both accept this (`SCATTER_STRATEGY.md` D22:
  "Serial bit-identical, OpenMP/CUDA climate-close"). So the per-substep gate is **field-appropriate**:
  momentum/SSH/ALE/EOS to the clean reassociation floor (<1e-7, the proof the wiring is right), FCT tracers
  + cancellation fields to the flip/cancellation budget (scales DOWN with the velocity/gradient, so it is far
  smaller on a physical field than on a sharp test bump). This IS Phase 8's bar (Decision 4: per-substep
  correctness, not bit-identity — "the C port sees the same chaotic Allreduce-order divergence").

- **[exch-closure-gating] One `_exch(field, kind)` closure threads every exchange; `halo_ctx=None` ⇒ the
  identity ⇒ byte-identical `v1.0`.** `step` builds `_exch = halo_ctx.exchange` (sharded) or `lambda f,k: f`
  (dense), inserts `field = _exch(field, kind)` after each producing kernel (the `OCEAN_SCHEDULE` posts), and
  passes `exch=_exch` into the fused kernels that split (`visc_filt_bidiff` exch `Uc/Vc`; `momentum_adv_scalar`
  exch `un_u/un_v`; `advect_one_fct` exch `fct_LO`+`tr_xy`, `zalesak_limit` exch `fct_plus/minus`). The
  `None`→identity makes every insertion a structural no-op ⇒ the 483-test single-device suite stays GREEN
  (dump gates byte-identical). Over-exchanging is harmless (refreshing an unread halo is a no-op), so insert
  the WHOLE schedule and let the per-field N-vs-1 diagnostic flag any genuinely-missing one.

- **[⚠️compute-node-not-login] Run every multi-minute `shard_map` compile via `sbatch` on a COMPUTE node,
  NOT the login node.** The full assembled step under `shard_map` is a ~2 min compile + GBs of RAM; iterating
  it on the shared login node (`levante0`, ~40 users, RAM-limited, one-CPU-JAX-process) is antisocial and can
  be killed. Only the lightweight host-side numpy checks (stencil/connectivity audits) belong on login. The
  slower `sbatch` debug cycle (queue + run) is the correct cost; batch several diagnostics into one job.

## Phase 8 — sharding (Task S.7 part 3 — GM/Redi forced-path exchanges)

- **[🎯GM-needs-5-exchanges-fer_gamma-is-the-trap] The GM/Redi chain needs only FIVE halo exchanges, and
  `fer_gamma` (the streamfunction) is the easy-to-miss one — the Kokkos `SYNC_MAP` row 1b caught what the
  plan's "likely only fer_uv/slope_tapered/Ki" underestimated.** The C/Kokkos exchanges ~10 GM intermediates
  (it computes per-node fields over `myDim` only); the JAX redundant-compute model needs only the fields a
  downstream kernel reads at the HALO of an entity whose value is INCOMPLETE there: **`fer_gamma`** (nod,
  INTRA — before `fer_gamma2vel`), **`fer_uv`** (elem), **`slope_tapered`**/**`Ki`** (nod), all in
  `gm.gm_diagnostics`, + **`fer_w`** (nod) in `step.py`'s bolus wrap. The trap is `fer_gamma`: `fer_gamma2vel`
  GATHERS it at the element's 3 vertices, and a boundary OWNED element has HALO-node vertices (S.1 redundant
  element ownership), whose `fer_gamma` is incomplete (its per-node TDMA RHS reads the element→node SCATTER
  `sigma_xy`, incomplete on the halo). So the owned element's `fer_uv` is wrong unless `fer_gamma`'s halo is
  refreshed BEFORE the gather. `sigma_xy`/`neutral_slope`/`fer_K`/`fer_C` need NO exchange (per-node maps,
  read per-NODE downstream — owned-complete). **READ THE REFERENCE MAP FIRST** (user hard-rule #2): the plan's
  per-field guess missed `fer_gamma`; the `SYNC_MAP` row had it as the one explicit "re-push" (L30).

- **[🎯Redi-tr_xy/tr_z-auto-complete-in-JAX] The Redi diffusion (`gm_redi`) needs NO internal exchange,
  unlike the C.** The C exchanges `tr_xy` (elem) inside `diff_ver` and `tr_z` (nod) inside `diff_hor`
  (`SYNC_MAP` §6) because it builds them over `myDim` only. In JAX both are recomputed per call from
  halo-complete `T_old` over the FULL local extent (`tr_xy` = per-element ∇T_old, `tr_z` = per-node ∂z T_old),
  so they are auto-complete — the edge loop's owned-node output is correct GIVEN `slope_tapered`/`Ki` are
  exchanged (read at halo edge endpoints). The "JAX needs fewer exchanges than the C" rule again: a
  recomputed-from-complete-inputs intermediate never needs its own exchange; only the persistent
  scatter-results read at the halo do.

- **[🎯per-kernel-gm-gate-is-BIT-EXACT] The per-kernel GM-exchange gate (`run_gm_diag_sharded`, the S.4
  scatter-gate analogue) matches single-device to EXACTLY 0.0 on owned — definitively proving the exchanges
  before the noisy FCT.** Running `gm_diagnostics` alone under `shard_map` (npes=2) and diffing `fer_uv`/
  `slope_tapered`/`Ki` on owned gave `max|Δ|=0.000e+00` (bit-exact, not just ~1e-9): each owned node/elem's
  GM output is the same scatter terms in the same order as the dense, and the exchanges only touch the halo.
  This ISOLATES the GM exchange correctness from the FCT tracer floor — so when the full GM step then matched
  every clean field to MACHINE PRECISION (uv 2e-16, w 7e-17, d_eta 3e-16, Kv/density tiny) and only T/S were
  elevated (T≈8.6e-3, S≈3.9e-3), it was PROVABLY the upwind-flip floor (GM-diag bit-exact ⇒ the bolus + Redi
  inputs are correct), not a missing exchange. Build the per-kernel gate when a composite (GM/KPP/ice) feeds
  the FCT — it discriminates "missing exchange" (would be O(1) on owned boundary) from "flip floor" cleanly.

- **[GM-FCT-floor-larger-on-PHC-IC] The GM+PHC-IC FCT flip floor (T≈8.6e-3) is LARGER than the part-2
  sharp-bump (~1e-3) — realistic fronts + the bolus-augmented advecting velocity make more upwind flips.**
  The bolus `uv_adv = uv + fer_uv` carries `fer_uv`'s ~1e-12 scatter reassociation into the FCT, and the real
  PHC IC has sharper tracer gradients (thermocline, western-boundary currents) than the test bump — so the
  flip floor scales UP with gradient × velocity. T > S (8.6e-3 vs 3.9e-3) because T's gradients are sharper.
  Set the FCT-tracer budget per-config (sharp-bump 5e-3, GM/PHC-IC 2e-2); it is climate-close, not a bug
  (Decision 4). The non-FCT/clean fields stay at the machine-precision floor regardless — gate THEM tightly.

- **[GM-needs-stratified-state] Gate GM on the REAL PHC IC (stratified), NOT a depth-uniform perturbed-rest
  state — the latter degenerates (N²≈0 ⇒ the ODM95 slope taper collapses, the slopes blow up).** GM is purely
  diagnostic (no surface forcing, no reductions), so it gates WITHOUT the forced path — but it needs genuine
  vertical stratification or `compute_neutral_slope`'s `denom=max(bv0+bv1, eps²)` floors to `eps²` and the
  `sigma_xy·(2g/ρ₀/eps²)` slopes explode. The cached `core2_initial_state` (PHC IC) is the right state; it
  isolates the GM exchanges from the forcing/reduction wiring (which KPP/ice need).

## Phase 8 — sharding (Task S.7 part 3 — reductions routing + the forced-path forcing fold)

- **[reduction-threading-is-pure-plumbing] Routing the `_area_mean` balances through `owned_mask`/`axis_name`
  is a 5-file thread with a `None` default => byte-identical `v1.0`.** The S.5 `global_sum` helper already
  existed; S.7-part-3 threads `owned_mask=None, axis_name=None` (keyword-only) down the call chain
  `step.py -> compute_surface_fluxes / ice_surface_step -> sss_runoff_fluxes / ice_oce_fluxes -> _area_mean`.
  `owned_mask=None` keeps the `if owned_mask is None: return jnp.sum(x*area)/ocean_area` branch (the EXACT v1.0
  graph - the 9 single-device `sss` tests stay green). `step.py` derives `(_red_mask, _red_axis)` from
  `halo_ctx` (`owned_mask["nod"]`, `"p"`) or `(None, None)` when dense. The sharded owned-sum + `psum` matches
  single-device on owned to ~1e-12. The `_area_mean` subtracts a GLOBAL scalar mean, so each owned node's
  balanced flux is `local_value - global_mean` = correct (local value owned-complete, mean `psum`'d) - no
  per-node halo issue.

- **[fold-the-forcing-as-a-sharded-input-NOT-closed-over] On the sharded FORCED path the `StepForcing`/
  `ForcingStatic` MUST be folded to `[P*Lmax]` `shard_map` inputs - closing over the `[P, Lmax]` partitioned
  forcing REPLICATES it (every device sees all ranks' forcing).** `run_step_sharded` previously passed
  `step_forcing`/`forcing_static` as Python closures into the body; that is correct only at `npes==1`. For
  `npes>=2`, `_fold_forcing` folds each NamedTuple field `[P, Lmax_nod, ...] -> [P*Lmax_nod, ...]` (spec
  `'p'`), EXCEPT the 0-d scalar `ocean_area` which stays replicated (`PartitionSpec()`, it becomes a `psum`),
  and passes them as varargs through `shard_map`'s `in_specs` (a same-typed NamedTuple-of-`PartitionSpec` -
  JAX accepts nested-pytree specs). `forc=()` when `step_forcing is None` => the no-forcing pi/GM path traces
  EXACTLY as before (the GM gate stayed byte-identical). The partition helpers `partition_step_forcing`/
  `partition_forcing_static` (S.2b) produce the `[P, Lmax]` form; `_fold_forcing` is the device-placement step.

## Phase 8 — sharding (Task S.7 part 3 — KPP forced-path exchanges)

- **[🎯KPP-smoother-must-exchange-PER-SWEEP] The KPP 3-sweep `blmc` smoother needs a halo refresh BEFORE
  EVERY sweep, not just once — each sweep is an element->node SCATTER, so its halo is incomplete for the
  next.** `eos.smooth_nod3D(arr, n_smooth, exch)` exchanges `arr` at the start of each of its `n` sweeps: the
  first refresh fixes the INCOMPLETE input (`blmc` is uvnode-derived: `ri_iwmix(uvnode)` where `uvnode` is the
  element->node scatter `compute_vel_nodes`), the later refreshes fix the inter-sweep scatter incompleteness
  (the sweep reads `arr` at the element's HALO vertices). Mirrors the C's "the smoother does its own internal
  exchanges" (`SYNC_MAP` M2.3). The single-sweep `bvfreq` smoother (substep 1) passes `exch=None` — its input
  is a halo-complete per-node T/S map, so one sweep is correct unrefreshed. **Proof:** `Kv` (the smoother
  output) matched single-device on owned to **2.4e-14** (machine precision), npes=2.

- **[KPP-2nd-exchange-viscA-before-the-Av-gather] After `smooth_blmc`+combine, refresh the node `viscA` BEFORE
  the node->elem `Av` average.** `_node_to_elem_visc` GATHERS `viscA` at the element's 3 vertices (HALO nodes
  for a boundary OWNED element), but `viscA = max(viscA, smoothed blmcM)` is still incomplete on the halo
  (the smoother's final halo is incomplete). The second `SYNC_MAP` KPP exchange point. `Kv` (=combined
  `diffKt`) is refreshed by `step.py`'s post-mixing `Kv` exchange (read per-NODE downstream), so it needs no
  in-kernel exchange. **Proof:** `Av` matched on owned to **9.1e-15**, npes=2.

- **[🎯KPP-needs-FEWER-than-the-C-uvnode/sw_alpha-auto-complete] KPP's per-node-COLUMN kernels are
  auto-complete, so only the 2 horizontal ops (smoother + `Av` gather) need exchanges — NOT the C's `uvnode`/
  `sw_alpha`/`sw_beta`/`dbsfc` pre-exchanges.** `ri_iwmix` (shear Ri), `prestep` (ustar/Bo), `bldepth` (the OBL
  search) and `blmix` are all per-node-COLUMN (they read `uvnode`/forcing at the node's own column, vertically)
  -> their OWNED outputs are complete from the OWNED (scatter-complete) `uvnode`, with no horizontal
  neighbour read. `sw_alpha`/`sw_beta`/`dbsfc` are per-node maps of halo-complete T/S -> auto-complete. The C
  exchanges all of them because it computes per-node over `myDim` only; JAX computes over the full extent. The
  forced-path inputs (`heat_flux`/`water_flux`/`stress_node_surf`) are per-node maps of the (folded,
  halo-complete) forcing + the `_area_mean`-balanced global scalar -> complete on owned.

- **[KPP-forced-compile-is-heavy] The forced KPP step under `shard_map` is a ~17 min CPU compile — the most
  collectives of any step (9 `blmc`-smoother `all_gather`s + ~18 ocean exchanges + the CG + the `psum`
  reductions).** Budget the `sbatch` time accordingly (`--time=00:30:00`) and split the npes==1 byte-id +
  npes==2 owned gates so a failure is localized. The npes==1 byte-identity is the proof the whole forced
  machinery (forcing fold + reductions + KPP exchanges + the smoother `exch`) collapses to `v1.0`; the npes==2
  `Kv`/`Av` machine-precision match is the proof the exchanges are correct on real shards.

## Phase 8 — sharding (Task S.7 part 3 — ice forced-path exchanges + the multi-step scan)

- **[🎯collective-in-CHECKPOINTED-scan-lowers] An `all_gather` (the `u_ice/v_ice` halo exchange) inside
  `jax.checkpoint` inside `lax.scan` inside `shard_map` (`check_vma=False`) LOWERS and runs — the hardest
  collective placement in the port, validated by the ice npes==1 byte-id.** The EVP momentum subcycle is a
  120-step `lax.scan` with a `jax.checkpoint`'d body (Phase-6 backward-memory cap); the sharded port adds a
  per-subcycle `u_ice/v_ice` `exch` INSIDE that body (each subcycle's `velocity_update` is a per-node update
  of the element→node SCATTER `u_rhs`/`v_rhs`, incomplete on the halo, and the next subcycle's `stress_tensor`
  reads `u_ice` at the element's HALO vertices). This extends the S.6 result (collective in a `while_loop`
  inside `custom_linear_solve`) to a CHECKPOINTED scan — the forward pass lowers cleanly (the checkpoint only
  affects the backward recompute, S.8). The ice FCT's `_solve_high_order` per-iteration `dvalues` refresh +
  the `a_l/m_l/ms_l` low-order + the `icepplus/icepminus` limiter splits are the same per-sweep idiom as the
  KPP smoother. **Result:** every ICE prognostic field (`a_ice`/`m_ice`/`m_snow`/`u_ice`/`v_ice`/`sigma`)
  matched single-device on owned to **0.0 bit-exact** (npes==2) — the EVP in-scan + FCT split exchanges are
  correct.

- **[🎯exchange-before-the-CONSUMER-not-at-step-end] A field read by a node→elem GATHER must be halo-refreshed
  BEFORE that gather, not at the end of the step.** The bug the npes==2 ice gate caught: `uv`≈7e-4 on owned
  (a CLEAN field) while EVERY ice field was bit-exact. `ice_oce_fluxes_mom`'s `stress_surf` is a node→elem
  gather of the blended node stress `sns`, which reads the FCT-derived `a_ice` at the element's HALO vertices
  (a boundary OWNED element) — but the `a_ice` exchange was placed at step-END (for the next step's EVP), so
  the gather read INCOMPLETE-halo `a_ice` ⇒ wrong OWNED `stress_surf` ⇒ wrong ocean `uv`. Fix: exchange
  `a_ice` RIGHT AFTER thermo, before `ice_oce_fluxes_mom`. The lesson generalizes the S.4 "who reads this at
  the halo?" rule across the ice→ocean SEAM: an ice OUTPUT consumed by an ocean kernel's gather needs its halo
  fresh at the consumer, and a single end-of-step refresh is too late if an earlier consumer gathers it.

- **[ice-bit-exact-ocean-amplifies-localizes-the-bug] When the per-field diagnostic shows the ICE fields
  bit-exact (0.0) and only the OCEAN fields elevated + ordered by coupling depth, the gap is in an ice→ocean
  OUTPUT (a surface BC), not an ice-internal exchange.** The breakdown — ice fields not even printed (= 0.0),
  ocean `uv` 7e-4 → `d_eta` 1e-5 → `w` 6e-8 (descending by how deep in the coupling chain) — immediately
  pointed at `stress_surf` (the ice momentum BC, fed to the ocean `impl_vert_visc`), not the EVP/FCT. Read the
  per-field ordering as a dependency graph: the SHALLOWEST elevated field (closest to the gap) is the suspect.

- **[🎯global-boundary_node-not-local] The EVP coastal BC needs the GLOBAL `boundary_node` partitioned in —
  the local-mesh recompute mis-flags partition-boundary nodes as coastal.** `boundary_node_mask` counts
  boundary edges (`edge_tri[:,1]==-1`); on a device's LOCAL mesh a partition-boundary edge has its off-rank
  element unmappable (`-1`), so an interior node gets mis-detected as coastal and its `u_ice` forced to 0 —
  diverging from single-device. Compute the mask on the dense mesh, `_shard_along_axis` it, and thread it
  through `run_step_sharded(boundary_node_p=…)` → `step(boundary_node=…)` → `ice_surface_step` → `evp_dynamics`
  (the C uses `partit->myList_edge2D`, `SYNC_MAP` M4.3b). The dense step derives it from the full mesh itself,
  so only the SHARDED side passes it.

- **[🎯free-running-multistep-decorrelates-use-TEACHER-FORCING] A free-running N-step N-vs-1 compare is NOT a
  tight gate — the step-1 FCT flip floor amplifies chaotically through the coupled system within a few steps;
  gate per-step with TEACHER-FORCING instead.** A 2-step OCEAN compare showed the fields ordered by coupling
  depth: `uvnode` 2.6e-17 → `bvfreq` 2.9e-10 → `density` 9.4e-7 → `uv` 1.1e-5 → `Kv`/`Av` 0.1 → `ssh_rhs` 67 —
  the ~5e-6 step-1 tracer flip floor propagating density→PGF→momentum→(PP-mixing + `mo_convect` binary
  flips)→SSH, exactly Decision 4's chaotic divergence, visible at just 2 steps. So: (a) gate the multi-step
  SCAN MECHANISM by "lowers + runs + FINITE + physically bounded" (the `run_steps_sharded` collective-in-scan
  works); (b) gate PER-STEP CORRECTNESS by teacher-forcing — each sharded step reads the SINGLE-DEVICE's
  previous state (partitioned), so the only N-vs-1 difference is the within-step reassociation (clean except
  FCT). A threading bug shows as a CLEAN field diverging under teacher-forcing; chaos cannot. `T_old`/`S_old`
  (the AB2 histories of FCT tracers) are FCT-class — add them to the climate-close set.

## Phase 8 — sharding (Task S.8 — the AD gradient gate)

- **[🎯the-sharded-REVERSE-pass-exposes-masked-NaN-traps-the-dense-XLA-folds] The forward of the
  sharded model is N-vs-1 correct (S.7), but `jax.grad` of it NaN'd — because the sharded BACKWARD
  does NOT fold the `0·inf` / `0·(±inf)` that single-device XLA silently folds.** A masked lane that
  carries an `inf` forward intermediate (so the forward `where`-mask hides it — the output is finite)
  poisons the backward: the cotangent into the masked branch is `0`, but `d/d(input)` of the inf-producing
  op is `±inf`, and `0·(±inf)=NaN`. `shard_map(check_vma=False)` + the manual-mode graph keeps that NaN
  where the single-device graph constant-folds the structural zero. This is the **masked-NaN rule on the
  device-pad axis** the plan flagged — a NEW masked axis the Phase-3/5/6 discipline must cover. **7 guards
  across 5 kernels**, ALL forward-byte-identical (the inf lanes were always masked — the 2-yr v1.0 run + 123
  single-device tests prove no live output changed):
  - **`pp.py`** (PP `pp_mixing`): `dz_inv = 1/dz`, `dz==0` at the `Zp=concat([Z,Z[-1:]])` duplicated-tail
    interface ⇒ `shear = 0·inf=NaN` backward. Guard the divisor.
  - **`momentum.py`** (`impl_vert_visc`): `Av/dZ_up` with `dZ_up==0` (same `Zp` tail) and `Av==0` (masked)
    ⇒ `0/0=NaN`. Guard `dZ_up`/`dZ_dn`.
  - **`tracer_adv.py`** (ocean FCT `zalesak_limit`) + **`ice_adv.py`** (ice FCT): `segment_max`/`segment_min`
    return their identity **`±inf`** on **empty pad-node segments** ⇒ `fct_ttf_max/min = ±inf` ⇒
    `0·(−fct_ttf/flux²)=NaN` backward. Clamp to finite on non-wet lanes.
  - **`kpp.py`** (`bldepth` + `blmix`): three `(hbl+zk)/(zk−zk1)` and `…/dth_kn` interpolations whose
    layer-spacing divisor is `0` on pad / degenerate-`kbl` nodes ⇒ `inf` ⇒ `0·inf=NaN`. Guard the divisors.
  `tracer_diff.py`/`kpp.py`(dz)/`eos.py`(zdiff)/`ice_thermo.py` ALREADY had these guards (their authors hit
  the same trap in single-device AD — the docstrings cite it); pp/momentum/the-FCTs/kpp-OBL were the gaps
  the device-pad backward newly exposed. **Lesson: every `1/<geometry-that-can-be-0>` and every
  `segment_min/max`/`±inf`-sentinel reduction is a masked-NaN trap unless the divisor is guarded BEFORE the
  divide / the `±inf` is clamped — a forward `where`-mask is NOT enough (it stops the forward, not the
  `0·inf` backward).**

- **[🎯debug-method: jax_debug_nans + a cheap focused probe, iterate; a FORCED probe pre-clears the heavy
  gate] A scalar `d/d(a_ver)` grad under `jax_debug_nans` (npes=2, ~1 min) pinpoints each trap by source
  line; fix, re-run, repeat.** `debug_nans` halts at the FIRST NaN in execution order (incl. harmless masked
  ones), so it walks the traps one per run (pp → momentum → ocean-FCT here). A separate `d/d(T0)` probe
  reaches EVERY kernel (EOS→PGF→KPP→momentum→FCT→ice), catching what the `a_ver` probe (which starts at the
  mixing) misses; the FORCED `d/d(T0)` probe (assembled KPP+GM+ice, ~20 min) found the KPP-OBL traps and then
  confirmed the **ice-EVP 120-subcycle `jax.checkpoint`'d scan backward runs FINITE** — all far cheaper than
  discovering NaNs inside the full forced gate. Proactive grep (`Zp=concat`, `segment_max/min`, unguarded
  `1.0/`) batches siblings (found `kpp:blmix`/`ice_adv` before their probe iteration).

- **[CG-transpose-backward-runs-sharded-CLEAN] The CG `custom_linear_solve` `transpose_solve` backward is
  AD-correct under `shard_map` — isolated probe: `grad_b 0.5‖solve_ssh(b,halo)‖²` is finite (max 7e-11), the
  matvec-only control finite too.** So the implicit-diff transpose (the S.6 forward's reverse-mode) carries
  through sharded; the `a_ver` NaN was NOT the CG (it was upstream `impl_vert_visc` + downstream FCT). And the
  **closure-grad of a REPLICATED param** (the `params` pytree closed over `run_step_sharded`'s `shard_map`)
  correctly `psum`s its cotangent (toy probe rel 0.0 with the real `jax.sharding.Mesh` API; the gate: `d/d(k_ver)`
  matches single-device to **3.75e-8**) — Decision 6's "`psum` transpose = `psum`" holds. ⚠️ `jax.make_mesh`
  (the newer explicit-sharding API) breaks closure-grad of a replicated scalar ("device assignment … not
  equal to mesh size"); the older `jax.sharding.Mesh` (what `halo.device_mesh` uses) works.

- **[🎯gradient-gate-is-FIELD-APPROPRIATE — the gradient analog of the forward Decision-4 gate] A sharded
  param/field gradient matches single-device to the floor of the PATH it traverses, not a uniform tol.**
  `k_ver` → CLEAN tracer vertical DIFFUSION ⇒ machine floor (rel 3.75e-8); `a_ver` → the FCT tracer
  ADVECTION (via `uv`) ⇒ the upwind-flip floor ON THE GRADIENT (rel 4e-4, and its gradient is tiny ~3e-8 so
  the absolute reassociation dominates the rel — within the dense path's own FD accuracy, `test_grad_flows_through_cg`);
  `T0`/`k_gm` likewise FCT-influenced. The **T0-field grad reconstruction** `Bᵀ(g_p)` (scatter-add the sharded
  cotangent over each global node's owner-interior + halo copies = the `all_gather` transpose) matches dense
  to **max |Δ|=7.4e-8, median ~1e-22** (the reverse-exchange AD is exact; the bulk is machine-precision, a few
  near-flip nodes ride the FCT floor). ⚠️ Gate the T0 reconstruction on the **ABS** diff, not rel — the rel
  blows up (1e4) at nodes where the dense grad ≈ 0 (a meaningless near-zero divide, not an error). The
  masked-NaN-across-devices check: `d/d(T0)` is FINITE everywhere (halo/pad/below-bottom), exactly 0 on
  dry/pad lanes, nonzero on owned-wet.

- **[🎯grad-of-a-jax.checkpoint'd-scan-under-shard_map-needs-jax.jit-AROUND-the-shard_map] The
  multi-step gradient (`run_steps_sharded`, a `jax.checkpoint`'d `lax.scan` under `shard_map`) raised
  `NotImplementedError: Eager evaluation of closed_call inside a shard_map isn't yet supported` — fixed
  by wrapping the shard_map-decorated body in `jax.jit`.** The 1-step `run_step_sharded` grad (param +
  T0) lowers WITHOUT a jit (no scan ⇒ no checkpoint ⇒ no `closed_call`); but `jax.checkpoint` emits a
  `closed_call` primitive, and JAX 0.10's reverse pass cannot eagerly evaluate a `closed_call` *inside* a
  `shard_map` unless that shard_map is under a `jax.jit` trace (the error message prescribes exactly this).
  The FORWARD lowered fine without the jit (the S.7p3 multistep gate), so this is a BACKWARD-only
  requirement. The fix is forward-transparent (jit is semantically identity ⇒ the npes==1 byte-identity +
  the forward gate are unaffected). **Lesson: when a `shard_map` body contains `jax.checkpoint` (or any
  `closed_call`-emitting primitive — custom_vjp, custom_call), `jax.jit` the shard_map before taking its
  gradient.** The 2-step `d/d(k_ver)` is then finite (+3.2e-6); a free-running multi-step compare still
  decorrelates chaotically (Decision 4), so this gates the scan-backward MECHANISM, not a tight dense match.

- **[🎯S.9 — the model runs CORRECTLY on real A100s; byte-identity is a CPU property, and the EVP stress is
  a VP-kink diagnostic not a prognostic] The first real-GPU run (`scripts/debug/phase8_s9_gpu.sbatch`, 4×A100)
  validated the sharded model: every PROGNOSTIC field matched single-device — ocean dynamics at the clean
  floor (uv 1.1e-9, d_eta 2.6e-11, w 2.3e-13, Kv/Av 4e-14…2e-14), FCT tracers T/S climate-close (9.7e-3/
  6.0e-3), prognostic ice u_ice/v_ice/m_ice/a_ice/m_snow 1e-7…6e-9 — and the OCEAN gradient (`jax.grad`-thru-
  `shard_map` over NCCL) matched at d/d(k_ver) rel 3.75e-8.** Two GPU truths the CPU gates didn't expose:
  **(1) byte-identity is a CPU property.** GPU XLA fuses/reorders the same arithmetic differently (a larger
  reassociation floor), so the 1-device serial-COLLAPSE worst across all State fields was **7.66e-9** (CPU is
  ~0). The CPU-calibrated `< 1e-9` byte-id asserts were physically too tight for GPU — NOT a bug. Fix:
  `_PLATFORM = jax.devices()[0].platform; _BYTE_ID_ATOL = 1e-9 if cpu else 1e-7` (the CPU branch is unchanged,
  so the single-device CI stays exactly as tight). The clean N-vs-1 owned-matches use the same platform-aware
  floor. **(2) the EVP internal stress σ11/22/12 is a NON-PROGNOSTIC VP-kink diagnostic, not gated.** σ = ζ·ε
  with ζ = ice_strength/Δ and Δ = max(√radicand, Δ_min): near-rigid ice rides the viscous-plastic yield kink
  where Δ≈Δ_min, so a ~1e-15 reassociation wiggle in the strain is multiplied by a HUGE viscosity → an O(0.5)
  branch flip in the RAW stress on a handful of near-kink elements. **The decisive tell that the physics is
  fine: the u_ice/v_ice that σ drives matches single-device to 1e-7** — the net stress DIVERGENCE (the force
  on each node) is correct, only the per-element stress branch flips at the non-smooth kink. **Lesson: gate
  the PROGNOSTIC state, not the kink diagnostic.** σ excluded from the N-vs-1 gate via `_DIAG_FIELDS` (still
  PRINTED so the floor stays visible in the log). This is the same "non-smooth diagnostic at a kink, smooth
  prognostic downstream" pattern as the FCT upwind flip (Decision 4) — the EVP yield curve is just a sharper
  kink. **(3) the FORCED gradient (the full EVP-scan backward) OOM'd on GPU** (RESOURCE_EXHAUSTED, 249 KiB
  after 2.5 h) — memory-bound, NOT a correctness failure; the OCEAN-grad pass already validates AD-thru-
  `shard_map` on the hardware. The EVP-scan backward materializes its ~120-subcycle intermediates; making the
  forced grad fit GPU memory (more aggressive `jax.checkpoint` on the EVP scan, or fewer subcycles for the
  gate) is deferred to its own task — it does not block the S.9 correctness verdict.

- **[Phase 8b B.0a — derive the ragged point-to-point halo maps from the OWNER MAP, not the C `ComStruct`]**
  The scaling fix replaces the O(P·N_local) `all_gather` halo with halo-only `lax.ragged_all_to_all`
  (confirmed in JAX 0.10.1 **with a registered transpose + jvp** — `_ragged_all_to_all_{transpose,jvp}` —
  so the gradient survives). The per-device send/recv index maps could come from the C `ComStruct`
  (`rPE`/`rlist`, `sPE`/`slist`, already parsed in `partit.py`), BUT the `Partition` has **no
  `com_edge2D`** — edges have no C communicator — whereas the existing `all_gather` `_exchange_map` derives
  ownership uniformly for nod/elem/edge from `_owner_map` (the lowest-id interior owner of each global id).
  So build the ragged maps (`shard_mesh.RaggedExchange`) from the **same `_owner_map`**: it is (a) uniform
  across all three kinds, and (b) **provably consistent with the `all_gather` oracle** (same ownership, only
  the transport differs). **Canonical ordering for `ragged_all_to_all`:** order each per-`(receiver e,
  source d)` block by **increasing halo-lane index on the receiver**, and build BOTH sides from the same
  `recv_pairs[e][d]` list → `send_sizes[d,e] == recv_sizes[e,d]` and the transported chunks align
  element-wise without any extra sorting. The forward is then a gather (`operand=field[send_idx]`) →
  `ragged_all_to_all` → scatter into halo lanes (interior+pad untouched); a host-numpy applier reproduces
  the `all_gather` exchange on every valid lane (the B.0a gate). ⚠️ An owned lane sent to several neighbours
  is gathered multiple times into `operand` — correct, because the transpose scatter-ADDs the cotangents
  back (the same additive-reverse-exchange property the `all_gather` AD relies on). ⚠️ The host builder uses
  per-halo-lane Python loops (fine to dars; vectorize for NG5's 7.4 M nodes).

- **[Phase 8b B.0c — JAX 0.10.1 `lax.ragged_all_to_all` FORWARD is correct but its reverse-mode AUTODIFF
  TRANSPOSE is WRONG (cotangent scales with device count) — wrap in `custom_vjp`]** The B.0 GPU gate
  (`scripts/debug/phase8b_b0_gpu.sbatch`, 4×A100, job 25438454) found: the halo-only `ragged_all_to_all` exchange
  matches the `all_gather` exchange **byte-identically on the forward** (all 3 kinds, npes 2 & 4 — the maps,
  the `output_offsets = recv_offsets.T` argument semantics, and the NCCL movement are all correct), **but its
  gradient is wrong by an order-unity amount that scales ~linearly with `npes`**: nod/elem/edge grad max|Δ|
  ≈ 4.3 at npes=2 and ≈ 8.0 at npes=4 (a clean ~2× doubling). Since `halo_exchange_ragged` is a composition
  of only linear ops (gather → `ragged_all_to_all` → gather-back → masked `where`) and the forward is exact,
  the culprit is **JAX's registered `_ragged_all_to_all_transpose`** — the ~linear-in-P error is the
  signature of the cotangent being SUMMED over the device axis instead of routed point-to-point. **This is
  why the AD gate exists** — it caught a silent gradient corruption before it could poison training.
  **Suggested fix (B.0d, deferred — forward scaling doesn't need it):** give `halo_exchange_ragged` a
  `jax.custom_vjp` so we control the transpose. Two backward options: (A) **reuse the proven `all_gather`
  exchange's VJP** in the backward (correct on every meaningful lane — pad-lane cotangents are inert; simple,
  but the backward still moves O(P·N_local)), or (B) a **hand-written reverse `ragged_all_to_all`** with
  swapped routing (`input_offsets=recv_offsets`, `send_sizes=recv_sizes`, `output_offsets=send_offsets.T`,
  `recv_sizes=send_sizes`) for a fully-scaling backward. Until B.0d, `use_ragged=True` is **FORWARD-ONLY
  safe** (the default `use_ragged=False` all_gather path keeps gradients correct). A forward-only scaling run
  never triggers the backward, so the bug does NOT block the scaling work — only training-at-scale with the
  ragged halo.

- **[benchmarking JAX/XLA per-step time: EXCLUDE compile or you measure the wrong thing — it can be a
  10× error]** The Phase-8b scaling bench first reported the full CORE2 model as "10× slower than Kokkos"
  (1.19 s/step vs 0.117). It was a TIMING BUG: the timed window included the **XLA compile**. Two causes:
  (1) warming up at `n=2` steps but timing `n=N` — `lax.scan` bakes the trip count into the executable, so
  different `n` ⇒ different compile ⇒ the timed call recompiled; (2) `run_steps_sharded` builds a fresh
  `shard_map` + `jax.jit` closure each call ⇒ a fresh cache key ⇒ recompile EVERY call (a one-shot call's
  wall-time is dominated by compile). The full model's graph (the 120-subcycle EVP `lax.scan` + KPP + GM) is
  far bigger than ocean-only ⇒ a far longer compile ⇒ the spurious 10×. **CORRECTED: JAX full CORE2 = 92.6
  ms/step (allgather) / 86.7 (ragged) vs Kokkos 117 — COMPARABLE / slightly faster.** Fix pattern: have the
  runner return the jitted executable (`run_steps_sharded(return_executable=True)`) so you compile ONCE then
  time a reused 2nd call; and use the **subtraction method** `per_step = (t_N − t_W)/(N − W)` over two warm
  runs (N and W steps) — the JAX analog of "omit the first W steps" (Kokkos' warmup exclusion), which cancels
  compile + per-call dispatch overhead + the AB2 first-step transient. ⚠️ ALSO benchmark the REAL workload:
  use the real PHC IC + JRA55 forcing + prognostic ice (`phc_ic.load_phc_ic` + `surface_forcing.build_surface_forcing`
  are BOTH mesh-agnostic — they interpolate the global `/pool` datasets onto any mesh), NOT synthetic constant
  forcing — the SSH-CG iteration count (the dominant comm) is state/forcing-dependent, so a degenerate state
  understates it. (Per-step cost IS forcing-value-independent, so holding the real forcing constant across the
  timing window is fine.) Decomp (CORE2 full): ocean+forcing 63%, ice/EVP 19%, KPP+GM 16% — like Kokkos's
  profile; ragged starts WINNING on the full model (~500 exchanges/step from EVP 120×2 + CG) where its
  per-exchange volume savings outweigh its per-call overhead, even single-node.

- **[multi-process / large-mesh sharding: BUILD global arrays on HOST (numpy), never `jnp` — or GPU 0 OOMs
  in SETUP before the model even runs]** dars/NG5 full-model OOM'd the GPU during setup (identical
  `1.34 GiB jit__where` for BOTH all_gather and ragged — 1.34 GiB = one full global 3-D field), single- AND
  multi-node. Root cause: the data pipeline materialized the FULL GLOBAL arrays as `jax.numpy` on the
  default device (GPU 0) BEFORE sharding — `State.rest`/`State.zeros` (`jnp.full`/`where`), the bench
  `phc_state`/`perturbed_state`, and `integrate_sharded._fold`/`folded_state`/`folded_mesh`/`folded_operator`/
  `_halo_arrays` (`jnp.asarray(...)`). `shard_mesh.partition_state` already returns host numpy, but `_fold`
  re-uploads it to GPU 0. So GPU 0 had to hold the entire global model (dars ≈ 20+ GiB) before `device_put`
  could shard it. **Fix: keep the whole pipeline HOST numpy (`np`) until a single final `jax.device_put` to
  a `NamedSharding` — which places ONLY the addressable shards per process, so GPU 0 never holds the full
  global.** The host keeps the full global numpy (fits a 512-GB node: dars ~22 GB, NG5 ~80 GB; use
  `--mem=0`); each GPU holds 1/P. This is the contained alternative to true per-subdomain loading (only
  needed if a mesh's global exceeds one node's host RAM). Lesson: in JAX, `jnp.*` runs on the default
  device — a "build then shard" pipeline silently routes the *whole global* through GPU 0; for sharded/
  multi-process data, build with `np` and shard at `device_put`. (Single-node CORE2/farc hid this — their
  globals fit one GPU; dars/NG5 are the first that don't.)

- **[host-build rewrite IMPLEMENTED + VALIDATED — the `_fold` must stay POLYMORPHIC, and the setup OOM ≠
  the model OOM]** Executed the host-build fix above (`State.zeros/rest` gained `xp=jnp|np`; `integrate_sharded._fold`
  + `folded_mesh`/`_fold_forcing`/`_halo_arrays` host-numpy; `run_steps_sharded` ALWAYS `device_put`s the
  folded inputs to a `NamedSharding`, dropping the `process_count>1` guard; bench `phc_state`/`perturbed_state`
  numpy). **The one non-obvious trap: `_fold` is differentiated THROUGH** — the S.8 IC-field gradient gate does
  `jax.grad(loss)(state_p.T)`, which folds a *tracer*. A blanket `np.asarray` in `_fold` raises
  "can't convert tracer to numpy". So `_fold` must be **polymorphic**: `a = arr if isinstance(arr, np.ndarray)
  else jnp.asarray(arr)` — concrete host arrays stay numpy (off GPU 0), tracers stay `jnp` (autodiff flows).
  The differentiated inputs in `run_steps_sharded` are CLOSED OVER the body (params/`k_ver`), not in the
  device_put'd `args`, so `device_put` of the (constant) folded inputs is grad-safe. **Validated:** the full
  CORE2 gate suite stayed GREEN on CPU fake-devices — State/partition byte-identical (68 tests), ocean sharded
  forward (npes 1/2/4), ocean grad (param/IC/FD/multistep — the multistep is the `run_steps_sharded` device_put
  path), forced assembled forward + backward (KPP+GM+ice, `_fold_forcing` + boundary_node device_put). It only
  changes WHERE arrays live, so every value is identical — exactly as the placement-only change should be.
  ⚠️ **Surprise: "host-build fixes single-node dars-4" (the prep claim) was WRONG.** The host-build DID remove
  the SETUP OOM (dars-4 now reaches XLA step compile instead of dying in the data build), but dars-4 FULL then
  OOMs on the **MODEL working set** — `hlo_rematerialization` floor 48.6 GiB/GPU at dist_4 (790k nod/GPU; the
  compiled full-step's live intermediates: EVP 120-subcycle scan + KPP + GM + CG + FCT), which exceeds even an
  80 GB A100. That is the SAME limit Kokkos hits (`SCALING_M524`: "dars/NG5 don't fit 4×A100, both start at
  2N"). **Lesson: separate the SETUP OOM (data build on GPU 0 — fixed by host-build) from the MODEL OOM (per-
  device step working set — fixed only by MORE devices / smaller shards).** dars needs 2 nodes (dist_8).

- **[the first HARD ragged win: at dars/8GPU multi-node, `ragged_all_to_all` FITS where `all_gather` OOMs]**
  dars (3.16M × nl57) FULL model (real JRA1958 + PHC IC + prognostic ice, dt=180) on **2 nodes / 8×A100**
  (dist_8, `jax.distributed`, 1 proc/node): **RAGGED runs — 0.934 s/step, peak_gpu 35.90 GiB; ALLGATHER OOMs**
  (needs a 43.34 GiB collective buffer; `hlo_rematerialization` floor 52.5 GiB vs ragged's lower working set).
  This is the FIRST place the ragged halo is not just faster but **necessary** — all_gather's O(P·N_local)
  gather volume literally doesn't fit at 8 GPU, exactly the "you copy too much data" failure mode the rewrite
  targets. (Single-node CORE2/farc could NOT show this — NVLink made all_gather's volume ~free, so ragged only
  showed a per-call-overhead penalty there; the win is fundamentally a multi-node / bandwidth-bound regime, as
  RevLog #4 predicted.) **The host↔GPU transfer is one-time** (`device_put` of the IC; `peak_gpu_after_setup =
  8.10 GiB` ≪ the old 40+ GiB build-on-GPU-0); all N steps run in ONE `jax.jit(shard_map(lax.scan(...)))` with
  the state carry resident on GPU — per-step traffic is GPU↔GPU only (the halo + `psum`). **vs Kokkos:** JAX
  dars-2N 0.934 s/step vs Kokkos M524 CUDA dars-2N **0.814** → ~15% slower (hand-tuned CUDA/MPI overlap; CORE2
  JAX was actually *faster* — the multi-node gap is XLA collective overlap). ⚠️ Confirm the Kokkos M524 dars
  level count (old `SCALING_DARS.md` says 47; the JAX mesh ran nl=57 — a ~20% vertical-work caveat on the exact
  ratio). NEXT: dars-16/32 (the JAX scaling curve) + NG5 (the headline 7.4M multi-node goal; IC cached).

- **[`jax.jit(shard_map)` + `device_put` of a *folded global* array stages it on ONE device — the NG5 wall;
  fix = `make_array_from_callback` (per-shard host slicing)]** After the host-build fix, dars scaled fine
  (dist_8/16) but **NG5 dist_16 + dist_32 OOM'd in a `jit__identity_fn` allocating ~the *full folded global*
  `[P·Lmax, nl]` (125.81 GiB at dist_32) on ONE GPU** — even though the model's own working set FIT (66.63 GiB
  at dist_32, scaled down from dist_16's 140). The culprit was the INPUT placement: `_to_global_sharded` used
  `jax.device_put(folded_global_numpy, NamedSharding)`, which routes a global-sized staging copy on a single
  device — ~56 GB for dars (fit an 80 GB A100, so it silently worked) but ~125.81 GiB for NG5 ⇒ OOM. It does
  NOT shrink with node count (`P·Lmax` ≈ global + P·halo grows with P), so more nodes can't fix it.
  **Fix: `jax.make_array_from_callback(shape, sharding, lambda idx: host_numpy[idx])`** — JAX calls the callback
  per ADDRESSABLE shard and pulls only that slice from the host numpy, so the global never lands on a device
  (CPU-verified bit-identical to `device_put`; single- and multi-process). After this, **NG5 dist_32 (8 nodes,
  32 A100) FULL model RUNS: 0.840 s/step** — vs Kokkos M524 CUDA NG5-8N **0.810** → **~3.7% slower** (the gap
  *closes* with scale: dars-2N 15% → dars-4N 8% → NG5-8N 4%; ragged halo + process-local I/O make the JAX port
  competitive with hand-tuned CUDA at 7.4 M nodes multi-node). Lesson: with manual `shard_map`, the `jit` I/O
  boundary is the *global* logical array — `device_put` of it can stage a global copy on one device; for
  big-mesh multi-process, **place via `make_array_from_callback`/`make_array_from_process_local_data`, never a
  global `device_put`.** (`integrate_sharded._to_global_sharded`.)

- **[sharded, gather-free model OUTPUT to Zarr — each GPU writes its own shard in parallel, no rank-0 gather]**
  Writing NG5 output the C/Kokkos way (gather the global field to rank 0 → one NetCDF) re-hits the
  single-device-materialization wall (the Kokkos `SCALING_NG5` "step-0 ~66 GB global gather" OOM). Instead
  (`fesom_jax/zarr_output.py`): write the **folded** `[P·Lmax_kind, …]` State to Zarr **chunked at `Lmax_kind`**
  along axis 0, so each device's shard is exactly one chunk ⇒ different processes write DISJOINT chunk files ⇒
  fully parallel, no locking, no gather (rank 0 creates the `.zarray` metadata + per-kind `gid`/`owned` index
  maps → `multihost_utils.sync_global_devices` barrier → every process writes its `arr.addressable_shards`).
  `reconstruct_global` scatters the OWNED lanes (`owned_<kind>` is True only on each entity's unique owner, so
  `owned.sum() == nod2D` exactly — no double-write) by `gid` back to a dense `[nod2D, …]` host array on read.
  **Verified: NG5 wrote 19 GB across 8 nodes, T had exactly 32 chunk files (one per GPU), `owned_nod == nod2D`
  (7,402,886).** The output analogue of the input `make_array_from_callback` fix — nothing global on one device.
  (zarr v2; `bench_forward_scaling.py --out-zarr`.) ⚠️ Reconstructing an NG5 *global* field (~8 GB) OOMs the
  *login* node's per-process cap — reconstruct on a compute node, or read chunk-wise.

## Phase 9a — zstar vertical coordinate (Task JZ.0 — scaffolding)

- **[ale/seam] `typing.NamedTuple` FORBIDS a `__new__` override (`AttributeError: Cannot overwrite
  NamedTuple attribute __new__`) — so config validation goes at the step SEAM, not construction.** The
  plan offered "a `__new__` override or factory"; the language picks for you. But the seam guard is
  actually *more faithful*: the C validates at runtime in `fesom_ale_mode_init` (`exit(1)` when
  `FESOM_ALE` is neither `linfs` nor `zstar`, `fesom_ale.c:25-30`), not at config-parse. So
  `AleConfig.validate()` is called from `step()` whenever `ale_cfg is not None` (a trace-time Python
  guard on the static arg — zero runtime cost). Verified it fires through `step_jit` too. (`ale.AleConfig`.)

- **[ale/seam] linfs is the ABSENCE of the cfg (`ale_cfg=None`), NOT `AleConfig(zstar=False)`.** The
  presence of an `AleConfig` ⇒ zstar; `zstar=False` is the *unsupported* request that raises. The two
  derived properties mirror the C's mode globals exactly: `use_virt_salt = not zstar`
  (`fesom_use_virt_salt = !fesom_ale_zstar`, `fesom_ale.c:31`) and `is_nonlinfs = 1.0 if zstar else 0.0`
  (`fesom_ale.c:32`). Threaded as a `static_argname` through `step`/`integrate`/the two sharded drivers
  (the `gm_cfg`/`kpp_cfg`/`ice_cfg` precedent); `None` vs `AleConfig()` is bit-identical (max|Δ|=0.0) at
  JZ.0 since no kernel branches on it yet — the standing `ale_cfg=None` byte-identity invariant. (Task JZ.0.)

- **[io_dump/oracle] The ALE dump is the SAME gid-keyed text format as KPP but MULTI-RANK, each rank
  dumping ONLY its OWNED rows (`myDim_nod2D`/`myDim_elem2D`, keyed by `myList_*`) — merge by gid.** So
  `read_kpp_table` generalizes verbatim to `read_gid_table` (parser unchanged; KPP aliases kept), and
  `load_ale_dump` globs all 16 ranks and scatters `out[gid-1]=row`. **Nodes are a clean disjoint
  partition** (z2_cdump: 126858 rows = 126858 unique gids, union = 1..nod2D, no dupes). **Elements
  OVERLAP on a thin boundary ring** (247199 rows → 244659 unique) **but the duplicated rows are
  BIT-IDENTICAL** (verified: distinct full-row tuples == distinct gids), so merge-by-gid is
  conflict-free; the loader asserts both invariants (`strict=True`: node full-coverage + element
  overlap bit-identity). (`fesom_ale_dump.c:38-55`; `io_dump.load_ale_dump`, Task JZ.0.)

- **[oracle] z2_cdump is COMPLETE — the JZ.0 audit worry is resolved, NO regeneration needed.** 12 tags
  × 3 steps × 16 ranks = 576 files, the full feature set. Tags + ncomp + component order (the C getter
  packing, `fesom_ale_dump.c:82-153`): `forcing`(4: water_flux, virtual_salt, relax_salt,
  real_salt_flux), `sshsolve`(2: ssh_rhs, d_eta), `hbar`(4: hbar, hbar_old, ssh_rhs_old, eta_n),
  `dhe`(1), `pgf_x`/`pgf_y`(47 layer), `Wvel`(48 iface), `hnode_new`/`hnode`/`Z_3d_n`/`helem`(47 layer),
  `zbar_3d_n`(48 iface). NG5 dist_16 mesh: nod2D=126858, elem2D=244659, nl=48. (Task JZ.0.)

- **[io_dump/verify] ⚠️ The ALE dump is DENSE to `ncomp` (full column 0..ncomp), NOT truncated to each
  node's `nlevels` like the binary Fortran dump.** `get_col` reads `arr[i*nl + c]` for c in `[0,ncomp)`
  (`fesom_ale_dump.c:69-73`). So when gating JZ.1+ thickness/geometry columns, mask per-node validity
  with the **mesh masks** (`node_layer_mask`/`node_iface_mask`) — `verify.compare_column`'s
  "truncate to record.nlevels" does NOT apply here (the C array is 0/nominal beyond bottom, included in
  the dump). (Forward note for JZ.1, found while building the reader, Task JZ.0.)

## Phase 9a — zstar vertical coordinate (Task JZ.1 — thickness machinery)

- **[ale/zstar/AD] ⚠️ `mesh.zbar_3d_n` is 0-PADDED below bottom (`dz=0` on EVERY node) — the dense-JAX
  inf factory the plan warned about.** Verified all 3140 pi nodes have ≥1 zero-`dz` lane in
  `mesh.zbar_3d_n` (a shallow node's column is `[0,-5,-10,-20,-30,0,0,…]`). So `live_geometry` must NOT
  pass the stored geometry through below bottom — it fills the non-stretch / below-bottom lanes with the
  strictly-decreasing **nominal `mesh.zbar`** (`dz>0` everywhere, verified min spacing 5 m). In the wet
  range `mesh.zbar_3d_n == zbar` (non-cavity full-cell), so the output still matches the static geometry
  where consumers read it; the below-bottom lanes (masked out in the forward) just stay finite for the
  backward. `d(zbar_3d_n)/d(hbar)` finite on every lane. (`ale.live_geometry`, Task JZ.1.)

- **[ale/zstar] `live_geometry` telescopes EXACTLY on pi (bitwise == static), ~ulp only on real
  bathymetry.** The reconstruction `zbar_3d_n[nz] = anchor + Σ_{j≥nz} hnode_stretch[j]` (a reverse
  cumsum over the stretch range, anchored at the nominal interface `min_f−2`) reproduces the C bottom→top
  recurrence `zbar[nz]=zbar[nz+1]+hnode[nz]` (`fesom_ale.c:238`) with the SAME association. Under nominal
  `hnode` the round pi depths (5,10,20,…) sum exactly ⇒ `max|Δ|=0` vs static `zbar_3d_n` AND `mesh.Z`;
  gate at ≤1e-9 for general meshes (cumsum reassociation, lesson #13 class). `Z_3d_n[nz] = zbar[nz+1] +
  hnode[nz]/2`. The `ale_cfg=None` path never calls this (it keeps the static arrays — the bitwise gate).
  (`ale.live_geometry`, Task JZ.1.)

- **[ale/zstar] Cold start (hbar=0) ⇒ the zstar init is BIT-FOR-BIT the linfs rest init (the free Z1
  degeneracy gate).** `init_thickness_zstar` writes the whole column as `(zbar[nz]−zbar[nz+1])·(1 +
  (hbar/dd)·stretch_mask)` — at `hbar=0` the factor is 1 ⇒ nominal; `eta_n=ssh_rhs_old=0`. Verified
  `hnode`/`helem` Δ=0 vs `State.rest()`. ⚠️ The JAX mesh has **no `bottom_node_thickness`/
  `bottom_elem_thickness`** field (linfs never needed it), but in full-cell those equal the nominal
  `zbar` difference, so the whole-column form needs no separate bottom field — the non-stretch +
  bottom layers are nominal by construction. (`ale.init_thickness_zstar`, `fesom_ale.c:45-100`, Task JZ.1.)

- **[ale/zstar] init `eta_n` uses the REVERSED AB weights `α·hbar_old + (1−α)·hbar` — the MIRROR of the
  per-step blend `α·hbar + (1−α)·hbar_old` (lesson #7).** At the reference `α=1`: init `eta_n=hbar_old`
  (=0 cold), per-step `eta_n=hbar`. Getting the two weights swapped is the classic landmine; the C is
  explicit (`fesom_ale.c:62-63` vs the per-step `ssh.eta_n_update`). (`ale.init_thickness_zstar`, Task JZ.1.)

- **[ale/zstar] The zstar commit REUSES the linfs `commit_thickness` (hnode:=hnode_new + full
  vertex-mean helem) — the C's `update_thickness_zstar` bottom-helem (which keeps `bottom_elem_thickness`,
  not the mean) AGREES with the full mean in FULL CELL.** The bottom layer is nominal/unstretched (only
  `nz ≤ min_f−3` stretch), and `mean(3 nominal node thicknesses) = bottom_elem_thickness` in full-cell, so
  the two differ only at ≤1 ulp (lesson #13) — and at partial cells (out of scope). The geometry-rebuild
  part of `update_thickness_zstar` is our `live_geometry` (D1); only the `hnode`/`helem` commit reuses the
  existing kernel. (`ale.commit_thickness`, `fesom_ale.c:245-264`, Task JZ.1.)

## Phase 9a — zstar vertical coordinate (Task JZ.2 — forcing flip)

- **[ice/zstar] The C already computes `evap` (bundled) AND `subli` SEPARATELY — the JAX split is
  free.** `fesom_ice_thermo.c:407-408` outputs `evap = _evap + _subli` and `subli = _subli`; the JAX
  `therm_ice_cell` had the same `evap` (line 182, open-water·(1−A)) and `subli` (line 183, sublimation·A)
  internally and just bundled them at the return. So `evaporation = evap`, `ice_sublimation = subli`,
  `bundled evap = evaporation + ice_sublimation` (verified exact, Δ=0). No new physics — only surfacing
  the two halves through `ThermoOut`/`ThermoState`/`IceStepOut`. (`ice_thermo.therm_ice_cell`, Task JZ.2.)

- **[ice/zstar] The real-salt producer is a one-line gate on `fwice` + the flooding correction
  (`use_virt_salt` static branch).** Virtual-salt (linfs): `fwice = −dhgrowth·ρice/ρwat·(rsss−Sice)/rsss`,
  `rsf=0`, flooding corrects `fw`. Real-salt (zstar): `fwice = −dhgrowth·ρice/ρwat` (UNSCALED — real
  volume change), `rsf = fwice·Sice`, flooding corrects `rsf` (`rsf −= iflice·ρice/ρwat·Sice`). Verified
  `rsf` matches the closed form exactly (Δ=0); `thdgr`/bundled-`evap` are path-independent (the growth
  physics doesn't change, only the salt bookkeeping). (`fesom_ice_thermo.c:359-404`, Task JZ.2.)

- **[ice/zstar] ⚠️ The water-flux global balancing `flux` is a SEPARATE re-derivation, NOT `−flx_fw`
  with the global mean removed.** `flux = evaporation − ice_sublimation + prec_rain + prec_snow·(1−a_ice_old)
  + runoff − thdgr·ρice/ρwat − thdgrsn·ρsno/ρwat` (`fesom_ice_coupling.c:203-209`) — different sign
  convention AND `evaporation−ice_sublimation` / `snow·(1−a_ice_old)` vs `flx_fw`'s `evap+subli` /
  `snow·(1−A)`. So the balancing needs its OWN inputs (evaporation, ice_sublimation, thdgr, **thdgrsn**,
  prec_rain, prec_snow, **a_ice_old = prev-step a_ice**, runoff), not derivable from `flx_fw`. `net =
  ⟨flux⟩` via `sss_runoff._area_mean` → `reductions.global_sum` (sharded-correct); `water_flux += net`
  (uniform). Gated `!use_virt_salt`. (`ice_coupling.fresh_water_balance_zstar`, Task JZ.2.)

- **[bc/zstar] ⚠️ The `bc_T` `sval·water_flux` term needs the POST-ADVECTION surface T, so it lands in
  the STEP (substep 15), NOT the forcing step.** The C `sval = trarr[surface]` is read AT the diffusion
  (`fesom_tracer_diff.c:292`), i.e. the post-advection+Redi `T_adv[:,0]` — the plan's hoped "start-of-step
  T from the forcing step" does NOT hold. So `bc_T = −dt·heat_flux/vcpw` is built in the forcing step, and
  `bc_T −= dt·T_adv[:,0]·water_flux` is added in `step.py` right before `impl_vert_diff` (gated `ale_cfg`).
  `bc_S` IS fully forcing-step: `dt·(virtual_salt[≡0] + relax_salt + real_salt_flux)` — **+dt, NO sval·wf
  term** (the S sign-trap, C lesson #3, `fesom_tracer_diff.c:65`). (`step.step`, `ice_step`, Task JZ.2.)

- **[ice/zstar] `use_virt_salt=True` is the universal default ⇒ every JZ.2 file stays byte-identical on
  the linfs path** (verified: `test_ice_thermo`/`test_ice_coupling`/`test_ice_step`/`test_core2_step` all
  green unchanged; the new `ThermoOut`/`IceOceFluxes`/`IceStepOut` fields are added, not reordered, so
  named access is intact and positional test calls still bind). The zstar inputs (`real_salt_flux`, the
  balancing args) are keyword-only with `None`/0 defaults — a dead branch under linfs. (Task JZ.2.)

- **[verify/zstar] ⚠️ The `forcing` ale_dump gate vs `z2_cdump` is a CONFIG-MATCHING problem, not a
  code gate — the JAX `build_surface_forcing` harness (tuned to the dt=500 dumps) does NOT reproduce the
  z2_cdump's exact step-1 inputs.** First run: `virtual_salt` matches the C **exactly** (Δ=0, the zstar
  flip), but `water_flux`/`relax_salt`/`real_salt_flux` differ by ~1e-5. The DIAGNOSTIC: `relax_salt` is
  **path-independent** (identical math in linfs/zstar) yet among the worst-matched ⇒ the gap is the
  JAX↔z2_cdump SSS/runoff/forcing-date INPUTS, NOT the zstar code. So the robust JZ.2 code gate is the
  **linfs↔zstar FLIP** (`test_forcing_flip_linfs_vs_zstar`, config-independent): same ice step, two
  `use_virt_salt` values ⇒ relax_salt **bit-identical**, virtual_salt→0, rsf 0→live. Combined with the
  existing linfs forcing dump gates (validate linfs forcing vs the C at dt=500), this transitively
  validates the zstar forcing. The direct z2_cdump match is a follow-on: localize the input mismatch at
  the worst nodes (start with the SSS-climatology interpolation timing + the step-1 forcing date), match
  the zstar reference namelist (`port2/.../zstar_reference_namelists/`). (Task JZ.2.)

- **[verify/zstar] ⚠️ The z2_cdump forcing config gap is 488 BRACKISH MARGINAL-SEA IC nodes (Baltic +
  Kara/Siberian estuaries), NOT a broad mismatch — the bulk 99.6 % matches to ~7e-9 (reduction class).**
  Decompose `relax_salt = surf_relax_S·(Ssurf − S_top) − ⟨·⟩` on the LOGIN node (host numpy only — no
  JRA/device/ice needed): both JAX and C relax are zero-mean, so `(relax_jax − relax_c)/surf ≈
  S_top_c − S_top_jax (+ const)` isolates the IC surface salinity. Result: `|Δrelax|` p50=p99=**7.2e-9**,
  p99.9=**5.0e-5** — a 488-node tail (`>1e-7`). The outliers carry PINNED fill constants (S_top=33.1175
  ×136, 33.7520 ×63 — impossible from bilinear interp ⇒ a Gauss-Seidel land-fill seed propagated across a
  connected brackish basin), geolocated in the Baltic (148) + Kara-Sea estuaries (lon 54-60/lat 69-70). So
  the JAX `phc_ic` GS-extrapolation fills these enclosed seas with an open-ocean constant where the C
  z2_cdump carries the true brackish PHC salinity (~7-8 PSU). The `virtual_salt` "Δ=0 match" is the trivial
  zstar flip (both ≡0), NOT evidence S_top matches — the real S_top check is this relax decomposition.
  **CONSEQUENCE:** z2_cdump is a TIGHT gate (reduction class) on the **config-clean subset** — gate JZ.3+
  dumps on a robust statistic (mask the 488 brackish outliers + a halo), NOT `max|Δ|`. Matching the IC
  extrapolation at marginal seas is a deep GS-fill-ordering follow-on, out of Phase-9a scope (phc_ic was
  bit-verified vs `phc_dump` at the open-ocean probes; the discrepancy is the enclosed-sea fill, which the
  open-ocean probes never exercised). (login-node `relax_salt` decomposition, Task #11 config-match probe.)

## Phase 9a — zstar vertical coordinate (Task JZ.3 — SSH plumbing)

- **[ssh/zstar/D2] The stiffness-as-function-of-state is a recomputed matvec increment, NOT a carried
  CSR — and its foundation is CONFIG-INDEPENDENTLY dump-verifiable.** The C `update_stiff_mat_ale`
  (`fesom_ssh.c:238-296`) adds, per step, the base edge-assembly with the column depth replaced by `−dhe`
  (`dhe=mean₃(hbar−hbar_old)`, `fesom_step.c:216-227`). It telescopes: `Σ dhe ≡ mean₃(hbar−hbar_init)`,
  so at cold start (`hbar_init=0`) the live matrix is `A_base + ΔA(−mean₃(st.hbar))`, recomputed inside
  the `custom_linear_solve` matvec from the carried `hbar` — no cumulative CSR, no new State field
  (D2). `ΔA` is the **same antisymmetric edge→node scatter** as `compute_ssh_rhs` with the "velocity"
  replaced by the **element gradient of the iterate** `x` (`∂ₓx=Σ_k ∂N_k/∂x·x[node_k]`), so it is linear
  & symmetric (a weighted Laplacian `div(−mean₃(hbar)·grad)`) ⇒ `custom_linear_solve(symmetric=True)`
  still holds; the closure makes implicit-diff propagate into `A` via `hbar`, not just the rhs. **The C
  dump proves the telescoping for free** (login-node, NO JAX forcing ⇒ dodges the config gap): the
  `dhe` tag ≡ `mean₃(hbar−hbar_old)` recomputed from the `hbar` tag to **Δ=0** at all 3 steps, and
  `Σ_{s=1..3} dhe_s ≡ mean₃(hbar_3)` to **1.7e-16** (hbar grows 0.35→0.68 m, real signal). Preconditioner
  stays the frozen base (lesson #11). (`ssh.stiff_increment_matvec`, Task JZ.3.)

- **[ssh/zstar] The wf tail lands in BOTH `compute_ssh_rhs` (substep 8) and `compute_hbar` (substep 11),
  with different signs of the same `−wf·areasvol[:,0]`.** `compute_ssh_rhs`: `+= −α·wf·areasvol[n,0]`
  (`fesom_ssh.c:413-421`). `compute_hbar`: `ssh_rhs_old −= wf·areasvol[n,0]` **before** the hbar update,
  so the wf-modified `ssh_rhs_old` is BOTH what hbar consumes AND what next step reads via `(1−α)·ssh_rhs_old`
  (`fesom_momentum.c:839-846`). The inner transport divergence is the bare `uv` (no wf) — add the tail at
  the call site, not inside the inner `compute_ssh_rhs`. Both arms are non-cavity-only (cavity unported).
  (`ssh.compute_ssh_rhs`/`compute_hbar`, Task JZ.3.)

- **[verify/ssh] ⚠️ Testing the wf tail by differencing two `compute_ssh_rhs`/`compute_hbar` calls
  CATASTROPHICALLY CANCELS — the ssh_rhs base is a near-cancelling ~1e6 scatter (ULP ~1e-9), so
  `(zstar − linfs) − want` is ~1.5e-8, not 0.** The base `ssh_rhs_old` carries the `dx·helem~1e7`-amplified
  cancellation floor (ssh/rhs lesson); adding the ~1e3 wf tail to a ~1e6 base then subtracting the base
  recovers the tail with only ~8 digits. Fix: probe the tail at **`uv=0`** (the transport divergence is
  exactly 0 ⇒ the tail IS the whole field ⇒ bit-exact `<1e-18`), and check additivity/gating separately
  via the `water_flux=None` byte-identity on a real `uv≠0` base. Same "don't difference two large
  near-equal fields" discipline as the ssh_rhs gate. (`test_ale_zstar.py`, Task JZ.3.)

## Phase 9a — zstar vertical coordinate (Task JZ.4 — vert_vel distribute)

- **[ale/zstar] The zstar Wvel correction is the VERTICALLY-INTEGRATED `(zbar_3d_n[nz]−dd1)·dd/dt`, NOT
  the per-layer `h·dd/dt` — and the stretch range is exactly the `_stretch_mask`.** Per non-cavity node
  (`fesom_ale.c:169-201`): `dd=(hbar−hbar_old)/(zbar_3d_n[0]−dd1)` with `dd1=zbar_3d_n[min_f−2]` (the
  first non-stretch interface), then over `nz < min_f−2` (the SAME `_stretch_mask` as JZ.1):
  `w[nz] −= (zbar_3d_n[nz]−dd1)·dd/dt` and `hnode_new[nz] = hnode[nz] + (zbar_3d_n[nz]−zbar_3d_n[nz+1])·dd`.
  The `(zbar_3d_n[nz]−dd1)` factor IS the depth-from-anchor integral (bottom→top Σdh/dt), so it grows
  toward the surface — a per-layer `h·dd/dt` would be wrong. The surface freshwater BC `w[0] −= wf` is
  SEPARATE (applied unconditionally on non-cavity, `fesom_ale.c:189`, on top of any stretch correction
  at nz=0). Uses the **pre-commit** live geometry `live_geometry(st.hnode)` (the carried thickness; the
  commit is substep 16) + the **post-`compute_hbar`** `hbar`/`hbar_old`. Cold start `hbar=hbar_old ⇒ dd=0`
  ⇒ w unchanged, `hnode_new=hnode`. (`ale.vert_vel_zstar_distribute`, Task JZ.4.)

- **[step/zstar] The dual-geometry (lesson #6) needs NO un-hoist — just OVERRIDE `hnode_new` after
  vert_vel.** The hoisted `hnode_new = thickness_linfs(st.hnode) = st.hnode` (substep 2) is exactly the
  OLD committed thickness the GM coefficient block must read (`fesom_step.c`: the C's `hnode_new` still
  holds the previous commit there). Reassigning `hnode_new` at substep 13 (after `vert_vel_zstar_distribute`)
  leaves GM's already-executed read untouched (Python value semantics) and feeds the NEW thickness to the
  substep-15 tracers/Redi/impl-diff (which take BOTH `st.hnode` for QR4C and `hnode_new` for the
  flux-limited/diffusion pieces) + the substep-16 commit. So the plan's "un-hoist" is a no-op: the single
  override gives the C's exact dual-geometry. linfs ⇒ the block is skipped ⇒ byte-identical.
  (`step.step`, Task JZ.4.)

## Phase 9a — zstar vertical coordinate (Task JZ.5 — Shchepetkin PGF)

- **[pgf/zstar] The C's running vertical integral `int_dp += aux` IS a cumsum — vectorize as
  `pgf[k] = cumsum(aux)[k] − ½·aux[k]`.** The C writes `pgf[surf]=½aux; int_dp=aux`, then
  `pgf[k]=int_dp+½aux; int_dp+=aux` (`fesom_eos.c:419-496`). Telescoping: `pgf[k] = Σ_{j<k}aux_j +
  ½aux_k = cumsum_incl(aux)[k] − ½aux_k`. Since `aux` is masked to 0 above the surface and below the
  bottom, the full-column cumsum equals the C's `[ule..nle]` integral exactly. So the whole
  surface→bottom sequential dependency collapses to one `jnp.cumsum` — no scan. (`pgf.pressure_force_shchepetkin`,
  Task JZ.5.)

- **[pgf/zstar] The 3 vertical stencils (forward/centered/backward) are STATIC case masks from the
  integer level arrays; compute all three everywhere, then `where`-select.** Surface→forward `(k,k+1,k+2)`
  where `k==ulevels[e]−1 AND k==ulevels_nod2D[node]−1`; bottom→backward `(k−2,k−1,k)` where
  `k==nlevels[e]−2 AND k==nlevels_nod2D[node]−2`; centered `(k−1,k,k+1)` else. Bottom wins the
  single-mid-layer (`nlevels==2`) overlap (the C's block order; that case is C UB in the backward
  stencil anyway ⇒ gate tests on `nlevels≥3`). Edge-padded shifts + the masked-NaN rule (**safe
  denominators on every `drho_dz` divide** — `dx10`, `dx20·dx21·dx10`) keep the overridden/below-bottom
  lanes AD-finite; the live `Z_3d_n` is strictly decreasing so REAL denominators are nonzero — the guard
  is only for the padded lanes. (`pgf._drho_dz`/`pressure_force_shchepetkin`, Task JZ.5.)

- **[pgf/zstar] Two depth fields, don't conflate: the ELEMENT `Z_n` (from `helem`, static bottom
  anchor) sets the evaluation depth; the NODE `Z_3d_n` (live) sets the stencil points.** `drho_dz` is
  the quadratic through the node's `(Z_3d_n, ρ)` triplet, EVALUATED at the element mid-depth `Z_n[k]`
  (`(Z_n−Z0)+(Z_n−Zm)`). `Z_n` is built bottom→top from `helem` anchored at the **static** `zbar[nlevels−1]`
  (the C reads static depth there — mirror, don't "fix" to live). `Z_n = reverse_cumsum(helem) + zbar_bot`,
  `Z_n[k]=zbar_n[k+1]+½helem[k]`. (`pgf.pressure_force_shchepetkin`, Task JZ.5.)

## Phase 9a — zstar vertical coordinate (Task JZ.6 — geometry re-point sweep)

- **[step/zstar] The geometry re-point is a uniform `Z3d=None` threading — byte-neutral by
  construction because the STATIC branch is kept bit-for-bit and only zstar passes a live array.**
  Each consumer gains an optional `Z3d=None` (and/or `zbar3=None`): `None` ⇒ the existing
  `mesh.Z[None,:]`/`mesh.zbar` broadcast (UNCHANGED — so linfs is byte-identical, no `live==static`
  reassociation risk); a 2-D `[nod2D,nl]` live array ⇒ the per-node moving-coordinate depth. Done so
  far: `eos.pressure_bv`/`compute_sw_alpha_beta` (density compressibility depth + N² spacing),
  `pp.pp_mixing` (shear `dz`), `tracer_diff.impl_vert_diff` (layer-center spacings). The key discipline:
  do NOT "always build from hnode" — that would reassociate the cumsum and break linfs at ~1e-10; gate
  on `Z3d is None`. (`step.step` + consumers, Task JZ.6.)

- **[step/zstar] ⚠️ The dual-geometry needs TWO live arrays, not one: `Z3d_live = live_geometry(st.hnode)`
  (pre-commit/carried) AND `Z3d_new = live_geometry(hnode_new)` (the about-to-commit).** Consumer→source
  map (the C dual-geometry, lesson #6): `st.hnode` (`Z3d_live`) for EOS/PP/PGF/KPP/forcing + the tracer
  QR4C high-order reconstruction + horizontal Redi; `hnode_new` (`Z3d_new`) for the impl vert diff + the
  flux-limited FCT pieces + GM fer (vertical). `Z3d_live` is hoisted once before substep 1 (reused by
  EOS/PGF/PP); `Z3d_new` is computed at substep 15 (after vert_vel produces `hnode_new`). At cold start
  both equal the static geometry, so every re-point is a step-1 no-op — the live branch is first
  exercised at step ≥2 and only the JZ.7 multi-step dump gate validates it (the linfs suite only gates
  the `Z3d=None` byte-neutrality). (`step.step`, Task JZ.6.)

- **[step/zstar] JZ.6 COMPLETE — the per-consumer which-side map, confirmed against the C source
  (`fesom_gm.c`/`fesom_momentum.c`/`fesom_tracer_diff.c`), not guessed.** A 4-agent C extraction
  pinned each re-pointed consumer to its exact thickness time-level + array kind:
  * `eos.compute_dbsfc` — a consumer the grep-by-array audit caught that the comment-scan missed
    (the "dbsfc denom"; the adiabatic-compression depth `z=Z[nz]` → `Z3d_live`). **Moral: audit by
    ARRAY (`mesh.Z`/`mesh.zbar`/`zbar_3d_n`), not by reading docstrings** (C lesson #5).
  * `momentum.impl_vert_visc` — **per-ELEMENT** geometry, not per-node: the C (`fesom_momentum.c:321-333`)
    rebuilds `zbar_n`/`Z_n` per element from `mesh->helem` (the carried/OLD element thickness), anchored
    at the static bottom `zbar[nlevels-1]`. Needed a NEW helper `ale.live_geometry_elem(helem)` (the
    element analogue of `live_geometry`, same stack as the shchepetkin PGF) + the kernel's 1-D `zinv`/
    `dZ_up`/`dZ_dn` become 2-D `(elem2D,nl)` (drop the `[None,:]`; gathers → `take_along_axis`).
  * `gm.fer_solve_gamma`+zscaling — per-NODE from `hnode_new`@prev-step ≡ `st.hnode` (`fesom_gm.c:517-527`),
    so `Z3d_live`/`zbar3_live`. `gm_redi.diff_ver_part_redi_expl` (G7a) — geometry from `hnode` (OLD,
    `Z3d_live`) but the `÷(areasvol·hnode_new)` divisor is the NEW side (already the arg). `gm_redi`
    horizontal Redi (G7b) — already uses `st.hnode`/`st.helem`/`hnode_new` directly (NO static-geometry
    ref ⇒ no re-point). `gm_redi.k33_augmentation` — the ONLY GM consumer on the NEW side: `hnode_new`
    (`Z3d_new`/`zbar3_new`, `fesom_tracer_diff.c:134-158` — matches `impl_vert_diff`).
  * `kpp` (5 sites: `ri_iwmix`/`bldepth`/`blmix`×2/`enhance`), `tracer_adv` QR4C, `forcing` sw-pen — all
    start-of-step ⇒ `Z3d_live`/`zbar3_live`. The QR4C 2-D `_z_stencil` uses `_shift_down`×2 for `Z[nz-2]`
    (matches the static edge-pad on the `is_int` range k≥nzmin+2; the surface/cent lanes are don't-care).
  * The per-node TDMA geometry (`fer_solve_gamma`, G7a, k33) goes 1-D `jnp.zeros(nl).at[1:nl-1].set(...)`
    → 2-D `jnp.zeros((N,nl)).at[:,1:nl-1].set(...)`; live geometry is strictly decreasing (nominal fill
    below stretch) ⇒ no zero divides, no extra guard needed. (`eos`/`momentum`/`gm`/`gm_redi`/`kpp`/
    `tracer_adv`/`forcing`, Task JZ.6.)

## Phase 9a — zstar vertical coordinate (Task JZ.7 — assembled step-1 gate)

- **[integration] The full 4-config CORE2 model — KPP + GM/Redi + EVP ice + zstar — assembles and runs
  FINITE the first time all four knobs run together; no config-interaction bug.** Each phase validated
  its knob in isolation (KPP-only, GM-only, ice-only dumps); the z2_cdump reference is all-on. The
  assembled `step(state, mesh, op, None, kpp_cfg=KppConfig(), gm_cfg=GMConfig(), ice_cfg=IceConfig(),
  ale_cfg=AleConfig(), step_forcing=…)` runs clean — the config seams compose. (`test_jz7_assembled_zstar_step1`,
  Task JZ.7.)

- **[verify/zstar] The step-1 dump tags split into THREE tolerance classes — gate each on its own,
  don't apply one bar.** On the config-clean subset (the 488 brackish nodes excluded; task #11): (1)
  **bit-faithful** — `pgf` p50=**2e-18** (the C's exact level), `Wvel` p50=2.8e-11 (the ÷area crushes the
  scatter); gate `p50<1e-12` + a robust tail. (2) **CG early-stop tolerance** — `d_eta`/`hbar`/`eta_n`
  p50~3e-7 (≈ soltol=1e-5 relative for cm-scale SSH): the C's dump is the EARLY-STOPPED iterate
  (Phase-2 ssh/solver lesson), and on CORE2's ~127-iter solve the bulk matches to that tolerance, not
  to 1e-18; gate `p50<1e-6` + `p99<1e-3`. (3) **near-cancelling** — `ssh_rhs` p50~5e-2 (the transport
  divergence's `dx·helem~1e7`-amplified floor, Phase-2 ssh/rhs lesson): diagnostic only, no tight
  absolute gate. (Task JZ.7.)

- **[verify/zstar] ⚠️ The config gap spreads BEYOND its 488 nodes through the elliptic SSH solve — the
  `d_eta`/`hbar` config-clean gate is `p99`, not `p99.9`.** The 488 brackish nodes corrupt the local
  density→pgf→`ssh_rhs`, and the global elliptic CG solve spreads that RHS perturbation into a decaying
  HALO — so ~1 % of the *config-clean* nodes (the halo around the brackish basins) carry a `d_eta`/`hbar`
  tail up to ~0.1 m, while the bulk is at the solve tolerance. The robust statistic for an elliptic-solve
  field is therefore `p99` (worst 1 % = the halo), one decade looser than the local fields' `p99.9`.
  (Local fields — `pgf`/`Wvel` — only see the gap at the brackish elements themselves, so `p99.9` holds.)
  (Task JZ.7.)

- **[verify/zstar] ⚠️ A pgf `p99.9` tail on the "config-clean" subset was a DEEP IC mismatch, NOT a
  pgf bug — the surface-`relax` proxy is blind to deep T/S.** The `pgf` was bit-faithful on the bulk
  (p50=2e-18, p99=1e-16) but had a ~0.1 % tail (max ~3e-5) at IC-matched (surface-`relax`=floor) shelf-break
  elements. The full debug ladder (`scripts/debug/jz7_pgf_debug.py`): (1) the cumsum→recurrence geometry change
  left it **byte-identical** ⇒ not geometry; (2) **k≤14 are bit-faithful (1e-18), divergence starts at
  exactly k=15** (the deep-PHC grid-coarsening, `helem` 30→40 m) ⇒ not the kernel (same kernel all levels);
  (3) `dz_dx=0` (column-uniform geometry at step 1) ⇒ `∂ₓρ` is purely the IC density's horizontal gradient,
  a NEAR-CANCELLING sum (~1e-6 from ~1e-2 terms) that amplifies tiny inputs; (4) the `density−DENSITY_0`
  double-subtraction is **harmless** — `Σ ∂N_i/∂x = 0` to 1e-20 (shape-function gradients sum to 0 by
  construction), so any constant offset in ρ cancels in BOTH `∂ₓρ` (`Σgs=0`) and `∂_zρ` (differences).
  ⟹ by elimination the IC T/S matches the C in the upper ocean but DIVERGES at depth (k≥15) on shelf-break
  columns — a Phase-5 `phc_ic` deep-interpolation input difference (extends `[[zstar-forcing-dump-config-gap]]`
  to deep T/S; the surface `phc_dump` verification never exercised it). **Moral:** when a bit-faithful
  kernel shows a localized tail, check the INPUT depth profile (first-diverging-level) before suspecting the
  kernel; and a surface-only IC-match proxy cannot certify deep fields. (Task JZ.7.)

## Phase 9a — zstar (Task JZ.7 follow-on: the deep-IC mismatch ROOT-CAUSED & FIXED — full bit-identity)

- **[phc_ic] ⚠️ The C IC is PARTITION-DEPENDENT: `extrap_nod3D`'s Gauss-Seidel land fill runs per-rank
  in LOCAL node order with halo values frozen between `exchange_nod` calls — so a 1-rank and a 16-rank
  C run produce DIFFERENT ICs (up to 25.8 PSU at fill nodes: Baltic/Kara).** Matching a C oracle
  bit-for-bit therefore requires replicating its PARTITION, not just its algorithm. The serial-order
  port was "correct" yet couldn't match the 16-rank z2_cdump at any GS-filled node. Fix:
  `phc_ic._extrap_nod3D_mpi(rank_nodes=…)` simulates the concurrent ranks (per outer iteration: all
  ranks sweep their own nodes against the same post-exchange snapshot, owned results merge after) —
  rank node lists in C local order come from the per-rank dump gid columns. Also mirrors the C's
  **surface-only outer-loop continuation** (`fesom_phc.c:299-309`): deep layers whose cross-rank
  propagation is unfinished when the surface converges are left to the vertical fill. Result: surface
  bit-identical (0 diffs, all 126858 nodes, incl. the 488 brackish), and the JZ.7 pgf tail COLLAPSED —
  full-mesh `pgf_x` max|Δ| 3e-5 → **2.7e-20** (p99.9=2.5e-21). BOTH facets of
  `[[zstar-forcing-dump-config-gap]]` (488 brackish surface nodes + deep k≥15 shelf-break T/S) were this
  one mechanism. **Moral:** an order-dependent fill makes the OUTPUT depend on the domain decomposition;
  "which partition produced the oracle?" is part of the config. (`phc_ic.py`,
  `scripts/tools/rebuild_ic_dist16.py`, `test_phc_ic.py::test_dist16_*`.)

- **[phc_ic] The other half of bit-identity was ~1-ulp FP association in the bilinear interp: C groups
  `((v·wx)·wy)` (`fesom_phc.c:215-218`); the numpy port grouped `v·(wx·wy)` — ~27k surface nodes off by
  ~1e-15.** That ulp noise was the pgf's 2e-18 "bit-faithful" p50 floor. After regrouping, the pre-extrap
  stage is byte-identical to the C dump (0 diffs). **Moral:** "map-class ~1e-14" agreement on a literal
  port usually means an association-order mismatch, not an algorithm difference — chase it; the exactness
  is then a much stronger regression gate (`np.array_equal`, not ATOL). (`phc_ic._load_one_variable`.)

- **[verify] Dump-vintage discipline: the /work C dumps span THREE code generations (April = pre-lon-wrap-fix,
  1477-node 0.58 K band at lon 0/360; May/June = fixed). Before chasing a mismatch against an old dump,
  verify the dump's vintage against the current C source — `md5sum` the source trees and prefer the
  newest same-partition dump (mevp/cdump_16r Jun-11 == z2_cdump partition, gid-verified).** (Task JZ.7.)

- **[phc_ic] ⚠️ Follow-on: the partition-dependence cuts BOTH ways — the LEGACY CORE2 oracles
  (core2_cdump probes, kpp/gm/ice-coupling dumps) were 1-RANK C runs, so switching the default cache to
  the dist_16 build broke 12 of their bit-exact gates (water_flux 8.7e-14 vs tol 1e-14, step-1 T 2e-7 vs
  1e-11). One IC cannot serve oracles from different partitions: keep TWO caches — `data/ic_core2` =
  SERIAL order (legacy 1-rank oracles; `cache_phc_ic.py`), `data/ic_core2_dist16` = dist_16 order
  (z2_cdump zstar gates; `rebuild_ic_dist16.py`) — and point each test group at its oracle's partition.
  With the association fix in both, the two caches differ at exactly the C's own 1r-vs-16r footprint
  (1086 surface nodes / ~13.6k (node,k) entries). **Moral:** before changing a shared fixture to match
  one oracle, enumerate ALL oracles that consume it and their provenance. (Suite job 25538063 →
  caught by the full-suite rerun discipline.)

## Phase 9a — zstar (Task JZ.7 — multi-step gate: the SSH solve is byte-faithful; the chained divergence is upstream)

- **[verify/zstar] ⚠️ CORRECTED (the controlled-replay): the chained 3-step gate's `d_eta`/`hbar`
  diverge to ~mm at step ≥2, but the SSH SOLVE IS BYTE-IDENTICAL on CPU — the divergence is the
  UPSTREAM velocity/ssh_rhs reassociation amplified, NOT a solve/early-stop property.** A controlled
  replay (`test_jz7_ssh_solve_controlled_replay`, `scripts/debug/jz7_ssh_replay_check.py`) feeds the C's
  OWN dumped `ssh_rhs`+warm-start+`hbar` (instead of the JAX chained state) into `solve_ssh`:
  step 1 (`x0=0`,`hbar=0`) → max|Δ|=**7.2e-16**, step 2 (`x0=C_d_eta1`,`hbar=C_hbar1`, the **D2
  increment LIVE**) → **9.7e-16** — both at the map/gather floor. So with IDENTICAL inputs the
  iterated near-null-space CG + the zstar D2 closure reproduce the C bit-for-bit (the early-stop is
  a red herring: identical inputs ⇒ identical early-stopped iterate). The chained ~mm is because the
  JAX computes its OWN `ssh_rhs` from the JAX velocity (~1e-12-different from C's per the scatter
  floor), which the near-cancelling `ssh_rhs` (Phase-2 ssh/rhs `dx·helem~1e7` floor) + the
  near-null-space `S⁻¹` blow up to ~mm in `d_eta` — the FP-trajectory butterfly EVERY ocean model
  has (the C vs Fortran too ⇒ the year-scale climate gate, not multi-step bit-identity). BOUNDED
  (s2≈s3≈7e-3). **Moral: the assembled step-1 5.5e-7 / chained-mm in an SSH-derived field is the
  INPUT difference, NOT the kernel — controlled-replay (feed the dump's own inputs) to separate the
  two before suspecting the solve.** (`test_jz7_ssh_solve_controlled_replay`, Task JZ.7.)
- **[verify/zstar] The DISCRIMINATING gate that the JZ.6 live geometry is correct (not the divergence
  source) is `pgf`, NOT `d_eta`/`hbar`.** pgf (shchepetkin) reads BOTH the density (T/S — the tracer
  chain) AND the live `Z_3d_n` (built from hbar). It stays **p50≈4e-9 / p99≈3.7e-7 / max≈4.5e-6** at
  steps 2-3 (vs 4e-16 at step 1) — a wrong geometry reconstruction would corrupt it at the *geometry*
  scale, and a diverged tracer chain at the *density* scale; 4e-9 is consistent ONLY with a correct
  re-point fed inputs carrying the ~1e-6-relative (mm-over-km-depth) hbar noise. So gate pgf tight
  (the JZ.6 precision gate) + finiteness hard; treat `d_eta`/`hbar`/geometry as the bounded
  "no-blow-up" class (their chained `ssh_rhs` differs; the solve+D2 byte-fidelity is gated separately
  by `test_jz7_ssh_solve_controlled_replay` + JZ.3, not by the chained multi-step).
  **Moral:** when validating a chained multi-step run against a partial-state dump (no T/S/uv to
  reset), pick a gate field ROBUST to the SSH machinery's near-cancellation amplification (a
  tracer/density-driven one like pgf), not the SSH-derived fields — and controlled-replay the
  SSH-derived ones separately. (`test_jz7_assembled_zstar_steps123`, Task JZ.7.)

## Phase 9a — zstar (Task JZ.7 — sharded N-vs-1: the whole zstar path shards clean)

- **[sharding/zstar] The full zstar path shards N-vs-1 to the clean reassociation floor — but ONLY
  a WARM hbar seed exercises it; a cold-start sharded step is a live-geometry no-op.** At hbar=0 the
  JZ.6 re-points + the vert_vel distribute + the D2 stiffness increment are all `live==static`, so a
  cold sharded step tests only the forcing-flip reductions + the (trivial) hnode_new exchange. Seed a
  ~0.5 m smooth hbar bump + rebuild hnode/helem via `ale.init_thickness_zstar` (`_warm_zstar_state`)
  so the column is genuinely stretched (hbar/dd~1e-4 ≫ the 1e-9 byte-id floor ⇒ a live-path sharding
  gap is detectable). Result (npes=2 owned-match, the generic `dataclasses.fields(State)` loop):
  **hnode_new 2.8e-14** (the JZ.4 `exchange_nod(hnode_new)` OCEAN_SCHEDULE row is correct), **hnode
  2.8e-14 / helem 5.7e-14 / hbar 1.9e-13** (the live geometry is a per-node/elem map ⇒ halo-complete
  from the redundant compute, no new exchange needed), **d_eta 3.3e-16** (the D2 stiffness-as-state
  increment — the `custom_linear_solve` matvec closing over `st.hbar` — shards correctly inside the
  distributed CG: its ΔA edge→node scatter rides the SAME halo as the base stiffness). T/S at the
  upwind-flip FCT floor (9.7e-3), ssh_rhs at the cancellation floor — the documented N-vs-1
  non-determinism, NOT a missing exchange. npes=1 collapses to dense byte-identically. **Moral:** to
  N-vs-1-test a config-gated feature whose effect is a step-1 no-op (cold start), seed the state PAST
  the degeneracy first. (`test_zstar_{serial,assembled}_sharded_*`, `scripts/debug/jz7_shard_zstar.sbatch`,
  Task JZ.7.)

## Phase 9a — zstar (Task JZ.8 — climate: the IC-partition provenance bites a SECOND oracle)

- **[verify/zstar] ⚠️ A year-scale climate MISMATCH (JAX-zstar↔C-zstar SSS 0.12 vs the C↔Fortran
  ref 0.0015) was NOT a code bug — it was the partition-dependent IC ([[zstar-forcing-dump-config-gap]])
  biting a DIFFERENT oracle than the one the IC was built for.** I built `data/ic_core2_dist16` to
  match the **16-rank** `z2_cdump` (the step gate). But the climate oracle `c_zstar_2yr` was run on
  **864 ranks** (the C plan's Z9, job 25495449) — and the C `extrap_nod3D` GS land fill is
  order-dependent per-rank, so the dist_16 and dist_864 ICs differ by **up to 25.7 PSU at the Baltic
  fill nodes** (`scripts/tools/rebuild_ic_dist864.py` confirmed: 12258 surface nodes differ, 512 in the
  Baltic). The smoking-gun signature that it's the IC and not the physics: the divergence is (a)
  **largest at month 1, decaying** (a spin-up transient from a different start), (b) **localized to
  the GS-fill nodes** (global p50 SSS = 1.7e-3 = the ref level), (c) **global salt budget conserved**
  (no leak). Fix = build the IC for the ORACLE's partition (dist_864) and re-run — which COLLAPSED
  the SSS error 41x (0.12 -> 2.98e-3 psu) to the C<->Fortran port-fidelity (JAX<->C-zstar SST
  3.46e-3 / SSS 2.98e-3 vs C0=3.74e-3/1.52e-3; B/A=5.7x/7.4x, inside the C's 3-9x coordinate contrast
  => PASS), PROVING the whole "failure" was the IC. **Moral (a repeat, now load-bearing): "which
  partition produced THIS oracle?" must be answered PER ORACLE — the dump oracle (z2_cdump=16r) and
  the climate oracle (c_zstar_2yr=864r) were different decompositions, and an order-dependent IC fill
  makes each demand its own matched IC.** (`scripts/rebuild_ic_dist864.{py,sbatch}`,
  `scripts/archive/core2_zstar_climate_compare.py`, Task JZ.8.)

## Phase 9b — classical-TKE vertical mixing (Task JT.0 — scaffolding, NO behavior change)

- **[seam/tke] TKE is the FIRST prognostic mixing scheme — the new-State-field checklist, not the
  config-gate, is the load-bearing part of the scaffolding.** PP/KPP/GM are all stateless (Kv/Av
  recomputed each step from T/S/N²); TKE carries `tke [nod2D, nl]` across steps, so it touches every
  State-enumeration site at once. Adding the field is one line in `state.py`; keeping the
  `tke_cfg=None` path byte-identical is the discipline: (a) the `zeros`/`rest`/PHC-IC default of 0 is
  inherited for free (rest builds on zeros, ICs `replace` only T/S) — cold start needs NO extra
  wiring; (b) the carry is written via a **conditional `dataclasses.replace(new, tke=tke_new)` keyed
  on `tke_cfg is not None`** (the ice precedent) so the None path never touches `state.tke`; (c)
  `partition_state` + the sharded field loops + the pytree registration are all GENERIC over
  `dataclasses.fields(State)` ⇒ zero changes; (d) the deliberate `test_state.py:_expected_shapes`
  tripwire is the guard that the inventory didn't silently drift. **Moral: for a new prognostic field,
  the only hand-written sites are the declaration, the zeros entry, the conditional carry-replace, and
  the test tripwire — everything else (IC, partition, pytree, zarr) is generic or default-inherited;
  audit the checklist rather than trusting that "it's just one field".** (`state.py`, `step.py`,
  `test_state.py`, Task JT.0.)
- **[seam/tke] The mixing dispatch goes 3-way, and the two un-ported guards belong at DIFFERENT
  layers.** `tke_cfg`'s structural-validity guard (IDEMIX/Langmuir/Dirichlet/mxl_choice≠2) is a
  `TkeConfig.validate()` (the `AleConfig` parity — a `NamedTuple` can't override `__new__`, so the
  C `fesom_tke_alloc:247-253` abort is a method called at the seam); the OPERATIONAL guards
  (both-`kpp_cfg`-and-`tke_cfg`-set ⇒ "one scheme per process"; TKE on the pi path ⇒ no
  `stress_node_surf`) live in `step.py` at the dispatch. Order matters: check `tke_cfg` first, and
  inside it check `kpp_cfg`-also-set BEFORE the forcing guard, so "both set" wins over "no forcing"
  (a both-set call on the pi path reports the real misconfiguration). The trainable constants
  (`tke_c_k/c_eps/cd/alpha`) go in `Params` (the GM `k_gm` `default_factory` precedent — a deliberate
  divergence from KPP's static-only constants, justified by TKE being the designated ML seam), which
  means THREE edits move together: the dataclass field, `Params.defaults()`, AND the
  `register_dataclass` `data_fields` — miss the last and the leaf silently drops out of `jax.grad`.
  (`tke.py`, `params.py`, `step.py`, Task JT.0.)
- **[oracle/tke] The cdump IS the replay gate; the `replay/` dir is a C-internal artifact, not a JAX
  input.** The 16-rank `cdump/dump/` carries all 20 tags PER STEP — the 5 column inputs (normstress,
  vshear2, bvfreq2, dztrr, tkeold) AND the 3 outputs + 10 diags + 2 wired (kv/av) — so the JAX column
  gate (JT.1) reads the inputs and compares the outputs with NO live forcing in the loop. The
  separate `replay/` dir is the C model run *with Fortran-injected inputs* (its outputs are .nc, used
  to validate C-vs-Fortran) — irrelevant to the JAX port. The reader is the multi-rank merge-by-gid
  (`load_tke_dump`, the `load_ale_dump` clone): 16 disjoint owned-node partitions reassemble to the
  dense `[126858, nl]` global with no NaN (strict), `av` the one element tag (`[244659, nl]`,
  boundary-ring overlap bit-identical). **Moral: confirm WHICH oracle dir is the gate input before
  building the reader — "replay" named a C validation run, not the replay-injection set the JAX needs;
  the cdump's self-contained input+output bundle is what makes controlled-replay a pure-algebra gate.**
  (`io_dump.py` `TKE_TAGS`/`load_tke_dump`, `test_tke_replay.py`, Task JT.0.)

## Phase 9b — classical-TKE vertical mixing (Task JT.1 — column core, replay-gated bit-exact)

- **[port/tke] A CONSTANT `1e-8` (= `mxl_min`) replay diff on ~0.3% of nodes is a STRUCTURAL
  off-by-one in the scan range, NOT a "threshold flip" — trace it, don't wave it through.** The
  Blanke–Delecluse mxl wall is THREE passes with DIFFERENT ranges: a forward min-scan `k=1..nlev-1`
  (`fesom_cvmix_tke.c:678`), a special pre-step AT `nlev-1` (`mxl[nlev-1]=min(mxl[nlev-1],
  mxl_min+dzw[nlev-1])`, `:681`), then a backward min-scan `k=nlev-2..1` (`:682`). I reused the
  forward's `is_interior` mask (`1..nlev-1`) for the BACKWARD pass, so it also updated `k=nlev-1` —
  and `min(mxl_min+dzw[nlev-1], mxl[nlev]+dzw[nlev-1]) = min(dzw+1e-8, dzw) = dzw` silently dropped
  the just-set special value by exactly `mxl_min`. The trap: the diff was EXACTLY `mxl_min` and
  *looked* like the documented min-scan flip class, so I nearly accepted it as noise. But a flip is
  RANDOM (1-ulp either way, ~machine-eps); a diff that equals a NAMED CONSTANT on a value-dependent
  subset is deterministic ⇒ an index/range bug. Excluding `nlev-1` from the backward mask
  (`is_interior_bwd = is_interior & ~(k==nlev-1)`) made the column core BIT-EXACT (≤3e-17, from
  ~1e-8). It ALSO collapsed the "`tke` 1e-12" residual — that was downstream (wrong mxl → wrong
  KappaM → wrong `ke` → wrong tridiagonal). **Moral: in a multi-pass min/max-scan with per-node
  bounds, the loop RANGES differ between passes; port each range literally and gate per-tag — a
  constant-offset replay diff (≠ random 1-ulp) is an off-by-one, a class apart from FP flips.**
  (`cvmix_tke.py` `mixing_length`, `test_column_core_replay`, Task JT.1.)
- **[oracle/tke] ⚠️ The cdump replay oracle was STALE — built with the `(float)6.6` literal bug the
  port was specifically warned against — so my CORRECT port (double `6.6`) "failed" the prandtl tags
  until I REGENERATED it.** The plan's #1 landmine: "all `-r8` literals are DOUBLE; port `6.6` as
  float64." I did. But the cdump (SHA `8260deae`, dumped from UNTRACKED TKE files BEFORE the fix was
  committed as `45afc01`) compiled `TKE_C66` as `(double)(float)6.6 = 6.5999999046`, so its
  `pr`/`tkekv`/`tbpr` carried the bug on the ~0.6% UNCLAMPED prandtl nodes (the clamped 1/10 nodes
  absorb it ⇒ p99=0, p99.9=1e-7 — a bimodal "exact-or-1e-7" distribution, NOT FP noise). Smoking gun:
  the diff was a CONSTANT *relative* factor `0.9999999856 = (float)6.6/(double)6.6`, and recomputing
  prandtl with `(float)6.6` reproduced the cdump to **0.0** while double-6.6 differed by `1.45e-7`.
  `git show 8260deae:src/fesom_cvmix_tke.c` confirmed the file didn't exist at that SHA (untracked).
  Fix = regenerate with the fixed binary (`build/fesom_port` rebuilt 20:23, but the cdump was 00:36
  — never refreshed): 29 s, my port then bit-exact across all 13 tags. **Moral: when a
  controlled-replay diff is a clean constant *relative* factor on a value-dependent SUBSET (the
  unclamped band) with a bimodal exact-or-X distribution, suspect a LITERAL-PRECISION bug in the
  ORACLE'S build, not the port — check the oracle's SHA vs the validated source and the binary's
  mtime vs the dump's. A replay oracle is only as good as the build that made it; a stale one fails a
  correct port.** (cdump regenerated → canonical; stale preserved as `cdump/dump_stale_6.6f`,
  `jobs/job_tke_cdump_v2`, Task JT.1.)

## Phase 9b — classical-TKE vertical mixing (Task JT.2 — driver tke.py, bit-exact wiring)

- **[port/tke] The driver is a clean composition — its only genuinely-new logic (the node→elem `Av`
  mean + the geometry assembly) is bit-exact by REUSING already-gated kernels; and the cdump's `kv`
  tag is REDUNDANT with `tkekv` (owned-row exchange invariance), so it validates the full-slab `Kv`
  adoption for free.** `mixing_tke` assembles 5 column inputs and wires `Kv`/`Av`: `vshear2` is the
  `kpp.ri_iwmix` interface shear (`_shift_down(Z)-Z`, already gated) masked to the TKE interior;
  `dz_trr` is interior `|ΔZ|` + `hnode/2` end caps (bit-exact 0.0 vs the dumped `dztrr` — the same
  geometry the JT.1 dzw-reconstruction confirmed); `bvfreq2 = where(is_interior, bvfreq, 0)` (the
  surface-leak the plan flagged, avoided); the `Av` is the node→elem 3-vertex mean over INTERIOR
  element interfaces (`pp.pp_mixing`'s `gather_nodes_to_elem`, minus KPP's `minmix` floor). Replay of
  the WIRING (inject the dumped `tkeav`/`tkekv` at the driver boundary) hit **kv 0.0 / av ≤1.4e-17**
  first try. The non-obvious part: the C dumps `tkekv` (the zeroed node `KappaH`, owned rows) AND `kv`
  (`aux->Kv` after `exchange_nod`, owned rows) — but the exchange only fills HALO rows, so the OWNED
  rows are identical ⇒ `kv ≡ tkekv` on every dumped node. So the full-slab `Kv = KappaH` adoption
  needs no separate check; the `av` (node→elem mean) is the only wiring step the `kv`/`av` tags
  genuinely exercise. **Moral: when a driver is a thin shell over gated column kernels + a stdlib
  scatter, the replay gate is a formality (bit-exact first try) — spend the rigor on the ONE new op
  (here the boundary-vertex `Av` mean, whose exch the sharded N-vs-1 gate at JT.5 will stress), and
  recognize when an oracle tag is redundant with another.** (`tke.py` `mixing_tke`/`_wire_kv_av`,
  `test_driver_wiring_replay`, Task JT.2.)

## Phase 9b — classical-TKE vertical mixing (Task JT.3 — step wiring; the cdump-forcing gap)

- **[oracle/tke] ⚠️ A replay oracle's LIVE tags are only comparable to a JAX live step if the FORCING
  config matches bit-for-bit — and the TKE cdump (a C-vs-Fortran validation dump) is NOT
  forcing-matched to the JAX, so the assembled live gate xfails on the forcing, NOT on TKE.** The
  plan assumed the K.8 precedent ("the JAX forcing is a validated 1:1 port ⇒ no JAX↔C transient"), but
  that held for KPP because the KPP step dump was generated SPECIFICALLY to match the JAX (same dt=500,
  same date convention). The TKE cdump was generated by the C's own TKE validation at **dt=1800**, and
  the JAX `build_surface_forcing` step-1 wind stress there differs by **~7e-4 (≈60% of the 1.2e-3 scale)**
  — and CRUCIALLY it is **IC-independent** (identical to the digit under `ic_core2` vs the 16-rank
  `ic_core2_dist16`), which RULES OUT the land-fill-IC hypothesis (`[[zstar-forcing-dump-config-gap]]`)
  and points squarely at a step-1 forcing **time/convention** mismatch (the wind at a different model
  time), not FP and not the TKE assembly. The diagnostic that nailed it: switching the IC changed the
  failure by 0.0 (same `0.0007184606646158276`). So the step WIRING is validated by the 4 passing
  integration tests (state.tke evolves; TKE≠KPP scheme-engaged; both-cfgs raise; jit-twice), the TKE
  PORT by the JT.1/JT.2 bit-exact replay gates — only the live forcing→assembly→cdump composition is
  blocked, behind an `xfail`. **Moral: before building a "live vs C-dump" gate, confirm the dump's
  FORCING was generated to match yours (dt, date convention, ice) — a validation dump made for a
  DIFFERENT consumer (here C-vs-Fortran) compares its physics, not your forcing; isolate forcing
  mismatches with an IC-swap (IC-independent ⇒ it's the forcing config, not the IC).** (`test_tke_step.py`
  `_FORCING_GAP` xfail, Task JT.3.)

## Phase 9b — classical-TKE vertical mixing (Task JT.4 — gradient gates, the ML seam)

- **[AD/tke] The ML-seam parameter gradients are EXCELLENT (FD↔AD plateau ~1e-8); the only snag was
  GPU MEMORY, not differentiability — a multi-gate grad script must `jax.clear_caches()` between
  full-`integrate` backward passes.** TKE is the project's primary hybrid-ML hook, and the headline
  result confirms the seam: `d(mean ML Kv)/d(tke_c_k)` = +2.97e-2 with a plateau of **8.2e-8**, and
  `d(mean surf tke)/d(tke_cd)` plateau **7.8e-9** (both ≪1e-4) — both well-conditioned because the
  tunables enter linearly where `tke>0` (`KappaM = c_k·mxl·√tke`, `forc = cd·|stress|^{3/2}`). The
  masked-NaN `d(SST)/d(T0)` is clean (0 on masked lanes) and the new `tke` scan-carry path is finite
  (`d_tri = tke_old + dt·forc` ⇒ the IC propagates linearly even at the cold-start `tke=0`, where the
  safe-sqrt zeros the COEFFICIENT path but not the RHS). The trap: unlike the KPP gate (ONE full
  `integrate` grad + a cheap replicated-chain param grad), TKE's tunables are `Params` leaves, so
  ALL four gates trace the full `integrate` — and TKE's backward is heavier (the extra `tke` carry +
  the two mxl `lax.scan`s + the thomas scan), so the 4 compiled grad executables blew the 40 GB A100
  on the 2nd gate. `jax.clear_caches()` after each gate frees the executables; `n=2`–`4` is plenty
  (step 1 spins `tke` up so `c_k` engages `KappaM` at step 2). **Moral: a script that grads several
  losses through a big checkpointed model accumulates GPU executables — `clear_caches()` between them
  (or one-gate-per-job); and when a tunable is a traced `Params` leaf rather than a static-cfg field,
  every gate pays the full-model backward, not a cheap single-step chain.** (`scripts/core2_tke_grad_gate.{py,sbatch}`,
  Task JT.4.)

## Phase 9b — classical-TKE vertical mixing (Task JT.5 — stability + sharded; the Av exch confirmed)

- **[sharding/tke] The plan-review MAJOR (the internal node-`tke_Av` exchange) is empirically
  CONFIRMED REQUIRED by the npes=2 owned-match — and the `tke` field's NO-exchange design is
  confirmed by the SAME test (the generic field loop).** Two opposite facts, one gate: (a) `tke_Av`
  (= node `KappaM`) MUST be halo-exchanged before the node→elem 3-vertex `Av` mean (a boundary OWNED
  element has HALO vertices) — `test_tke_sharded_step_owned_matches[2]` passes BECAUSE `_wire_kv_av`
  does `exch(KappaM, "nod")` first; omit it and owned-boundary `Av` corrupts by O(mixing) ≫ the 1e-9
  CPU floor; (b) the `tke` FIELD itself is NEVER exchanged (each column self-contained on owned data)
  — the generic `dataclasses.fields(State)` loop in `_owned_match` shows owned `tke` == dense, the
  proof the no-exchange design is right. The serial (npes=1) byte-id confirms the exchanges collapse
  to identity at 1 device. **Moral: a single N-vs-1 owned-match gate validates BOTH a required
  internal exchange (its absence corrupts owned-boundary outputs) AND a deliberately-absent one (the
  field still matches) — the generic State-field loop is what makes the second free, so design new
  prognostic fields to ride it.** Separately, the 480-step linfs+TKE run is stable (max|vel| 1.53 m/s)
  and the TKE↔KPP SST RMS 0.43 °C cleanly resolves the scheme (TKE ≠ KPP ≫ FP noise) — the
  scheme-engaged climate check. (`test_step_sharded.py` TKE tests, `scripts/core2_tke_stability.{py,sbatch}`,
  Task JT.5.)

- **[verify/tke] ⚠️ DON'T pre-judge a climate from a step-1 forcing snapshot — RUN it. The 7e-4
  step-1 stress diff vs the cdump WASHES OUT: the 1-yr climate matches the C oracle at SST/SSS RMS
  4.68e-3/2.74e-3 ≈ the C↔Fortran floor (0.0049/0.0028).** I characterized a step-1 forcing
  difference (the JAX `build_surface_forcing` stress ~5× smaller than the cdump at ~10% low-wind
  open-water nodes) and OVER-REACHED to "a bulk difference that blocks the climate" — then deferred
  the climate behind that assumption. A reviewer's objection ("the KPP/zstar climates ran fine, why
  a forcing gap now?") forced the right test: run the JAX-TKE 1-yr climate (`ic_core2_dist864` to
  match the 864r oracle) and compare. Result: stable 365 days, **A(JAX↔C-TKE) ≈ C0(C↔Fortran)** for
  both SST and SSS ⇒ the JAX TKE climate is as faithful as the C is to Fortran, the step-1 diff is a
  pure transient (consistent with the C's own live s1–s3 diffs vs Fortran being threshold-flips that
  replay sidesteps). The live-step-1 forward gate stays xfail BECAUSE it is transient-sensitive — but
  that does NOT block the climate, which is the gate that matters. **Moral: a step-1 forward
  comparison and a year-scale climate measure DIFFERENT things; a step-1 mismatch (even a structured
  one) can be a forcing-init transient that integrates out — never conclude "climate-blocked" from a
  single-step snapshot; the climate run is cheap relative to being wrong, and IS the arbiter.**
  (`scripts/core2_tke_climate{,_compare}.py`, Task JT.5.)
- **[verify/tke] ⚠️ CORRECTION to the lesson above: "transient" was WRONG too (a reviewer pushed a
  second time, rightly). The forcing-gap is a PERSISTENT, UNDIAGNOSED low-wind diff — not washed-out.**
  Same bulk inputs → different stress is a per-node FORMULA/INPUT difference, not a step-1 artifact. I
  disproved every mechanism I confidently floated: NOT gustiness (the C stress is plain
  `cd*rho*|U_rel|*U_rel`, no such term); NOT the `BULK_U10MIN=0.3` min-wind floor (the JAX ports it
  faithfully, `forcing.py:151,195`); and `fesom_bulk.c` is UNCHANGED across the KPP & TKE C builds
  (`git log`) — yet the JAX matches the KPP dumps <1e-12 and differs from the TKE cdump by ~7e-4 at
  ~10% open-water LOW-WIND nodes. So it is NOT the bulk formula; it must be the INPUTS the C fed its
  bulk when generating the cdump (the JRA55 wind field, or the node's surface state) — and it remains
  **OPEN**. The climate still passes (low wind ⇒ tiny absolute stress ⇒ ~floor-level impact), so TKE
  is climate-faithful and the gate is legitimately green — but the diff is real and persistent. To
  close it: a C-side instrumented re-run dumping the bulk inputs (`u_wind/v_wind/T_oc/u_w`) + the
  computed `cd` at the max-Δ node vs the JAX. **Moral: a year-scale climate match proves SMALL IMPACT,
  not NO DIFFERENCE; an unverified mechanism-story ("gustiness", "transient") dressing an undiagnosed
  diff is WORSE than logging it honestly OPEN. State what you've ruled out + what's still unknown.**
  (`test_tke_step.py::_FORCING_GAP`, Task JT.5.)
- **[verify/tke] ✅ RESOLVED (the reviewer was right all along): the JAX forcing is FINE — the TKE
  CDUMP is the OUTLIER. The lesson is CROSS-CHECK A SUSPECT ORACLE AGAINST A TRUSTED ONE before
  blaming your code.** The reviewer's objection ("KPP/zstar forcing is fine, the code is
  scheme-independent, why would TKE differ?") was the key — and a 3-way comparison settled it
  immediately: at the disputed low-wind nodes, `|JAX−KPP-oracle| = 6.8e-5` (BIT-IDENTICAL at the worst
  node 56098: both 1.566710e-04) while `|JAX−TKE-cdump| = |KPP−TKE-cdump| = 7.185e-4`. So
  **`JAX == KPP-oracle ≠ TKE-cdump`** — the JAX matches the trusted KPP forcing oracle everywhere, and
  the TKE cdump (a separate old C job) disagrees with BOTH. The JAX forcing is the same validated
  `build_surface_forcing` code KPP/zstar use; the cdump just has different forcing inputs (likely a
  different JRA55 snapshot). My whole chase (gustiness → min-wind → transient → "undiagnosed") was
  WASTED EFFORT born of one error: I anchored on the cdump as ground truth and never cross-checked it
  against the KPP oracle (which I already had, `data/kpp_dump_core2`). **Moral: when ONE oracle
  disagrees with your code, before theorizing about your code, ask "do my OTHER trusted oracles agree
  with this one?" — a 30-second 3-way check (your_code vs oracle_A vs oracle_B) beats hours of
  mechanism-hunting, and `code==oracle_A≠oracle_B` immediately fingers oracle_B, not your code.**
  (`test_tke_step.py::_FORCING_GAP`, `data/kpp_dump_core2`, Task JT.5.)

## Phase 9c — mEVP sea-ice rheology

- **[mevp/JM.0] Validating a `typing.NamedTuple` constructor: patch `__new__` POST-creation
  (the class body forbids it).** The plan wanted `IceConfig(whichEVP=2)` to raise (C abort
  parity for unported aEVP), but `typing.NamedTuple` raises `AttributeError: Cannot overwrite
  NamedTuple attribute __new__` if you put `__new__` in the class body, and it has no `__init__`
  hook. The clean fix: after the class is created, save `_orig = IceConfig.__new__`, define a
  wrapper that calls `_orig`, validates, and returns, then assign `IceConfig.__new__ = wrapper`.
  Python looks up `__new__` on the type at call time, so direct construction validates;
  `_replace` rebuilds via `tuple.__new__` (`_make`) and cleanly SKIPS the wrapper (fine — the
  dispatch re-guards); the NamedTuple stays a JAX pytree. Verified empirically before wiring
  (`fesom_jax/ice.py`, the `_validating_ice_config_new` patch).
- **[mevp/JM.0] The mEVP dump header carries NO `ncomp` — infer it from the row width.** Unlike
  the KPP/ALE/TKE gid tables (`# step=.. tag=.. N=.. ncomp=..`), the C `maevp_dump`
  (`fesom_ice_maevp.c:57`) writes `# step=.. point=.. array=.. rank=.. N=..` because the column
  count varies per point (Q=4, U0/UF=2, F=4, P_node=4, P_elem=1, it*_node=2, it*_elem=3). So the
  shared `read_gid_table` parser (which int-casts `ncomp`) can't read it; `read_evp_table` reads
  the header without `ncomp` and computes `ncomp = total_numbers/N − 1` from the data. Same
  multi-rank merge-by-gid as ALE/TKE otherwise (`io_dump.py`). Confirmed the C bc_index build
  (`fesom_ice.c:249-258`) is exactly `1.0 − boundary_node_mask` (1.0 everywhere, 0.0 at
  boundary-edge endpoints) — the plan's bc_index claim verified against source.
- **[mevp/JM.1] To prove a "pure refactor" is BITWISE-identical, capture a baseline BEFORE
  touching the code, then assert max|Δ|==0 — passing the C-dump gate is necessary but not
  sufficient.** JM.1 extracts `strain_rates` + `stress_div_scatter` from `ice_evp.py` so mEVP can
  share them. The existing EVP gate (vs the C dump, tol 1e-10/1e-12) would catch a *typo* but NOT
  a within-tolerance reassociation (e.g. reordering a sum), which is exactly what a "graph-identity"
  claim must exclude. So I ran the pre-refactor `ice_evp` once, saved σ/rhs/velocity/`evp_dynamics`
  outputs to a committed `.npz`, refactored, then asserted the new outputs equal the baseline to
  the bit (all 12 fields max|Δ|=0.000e+00). Cheap, and it makes "I only moved code" a *measured*
  fact rather than a hope. (`fesom_jax/tests/data/evp_baseline_jm1.npz`, `test_evp_graph_identity`.)
- **[mevp/JM.1] The bc_index partition-seam trap needs a NON-VACUOUS test: assert seam nodes
  EXIST, then assert none are flagged.** `bc_index = 1 − boundary_node_mask` derived from the GLOBAL
  mask is correct by construction, so naively "assert no interior node is coastal" passes
  vacuously. The meaningful check loads the real dist_16 partition, takes the union of every rank's
  HALO nodes (the eDim tail of `myList_nod2D` — nodes owned elsewhere, i.e. partition-boundary),
  keeps the ones that are interior in the global mesh (>1000 of them — proves seams exist), and
  asserts bc_index==1 there. That contrasts the global-mask build against the local-recompute trap
  the C documented (`fesom_ice.c:243-247`). ⚠️ `read_partition(mesh_dir, npes)` wants
  `mesh_dir/dist_<npes>/` — pass the pool parent `/pool/.../core2`, not the mesh data dir.
  (`test_bc_index_no_seam_flagged`.)
- **[mevp/JM.2] The mEVP kernel ported BIT-FAITHFUL on the first run — the 14-trap checklist did
  its job.** Loading the cdump entry inputs (Q/U0/F merged across 16 ranks), running `mevp_setup`
  + 120 eager `mevp_iterate` calls, and diffing vs the C: P-precompute (inv_thickness/mass/tilt/
  pressure_fac) ALL max|Δ|=0.000e+00 (bit-identical maps); it1 velocity =0 (cold start is wind-
  driven — u_w=elev=0 at rest ⇒ it1 is a pure per-node `det·obd·rdt·inv_thickness·stress` map,
  NO scatter); it2=1e-19, it60=1.8e-16, it120=1.5e-12 (the accumulated element→node scatter-
  reassociation floor, 16-rank-C vs single-device-JAX). The traps that would have silently broken
  it (rdt=dte not ice_dt; the 0.5 in pressure_fac; max-clamp δmin; EVP's else-zero; has_ice on the
  tilt) were all pre-identified — porting straight from the C with each `[Tn]` cited inline meant
  zero debugging. **Moral: when the C plan hands you a fidelity-trap list with line citations, port
  trap-by-trap with the citation in the code, not "port then debug against the dump".**
- **[mevp/JM.2] The entry-anchor is LOAD-BEARING and the cold-start it2 dump proves it — make the
  anchor an explicit argument so you can test the wrong version.** mEVP's backward-Euler rhs anchors
  on the FROZEN ENTRY velocity (`rhsu = u_ice + … + β·u_aux`), NOT the current iterate like std-EVP.
  I gave `mevp_iterate(…, u_anchor, v_anchor, …)` an explicit anchor arg; production passes the
  frozen entry, and the test runs 2 iterations BOTH ways: entry-anchored matches the C it2 at
  1.08e-19, iterate-anchored (the std-EVP-template bug) diverges to 1.4e-5. At it1 the two coincide
  (iterate==entry==0 at cold start), so ONLY it2 catches it — exactly why the C plan flagged the
  "cold-start it2 dump" as the discriminator. **Moral: when two implementations differ only after
  ≥2 iterations, a 1-step gate is blind; thread the disputed choice as a parameter so the test can
  drive both and show the dump fingerprints the right one.** (`test_mevp_entry_anchor`.)
- **[mevp/JM.2] lax.scan (jitted) ≠ the eager Python loop bit-for-bit (~1e-11), and that's fine —
  σ amplifies it to 1e-4, velocity stays 1e-11.** The per-iterate gates drive `mevp_iterate` in an
  eager loop (to snapshot it1/2/60/120 — scan hides intermediates); the production `mevp_dynamics`
  (checkpointed `lax.scan`, jitted) differs from that eager loop by ~2e-11 on velocity (XLA fuses
  the scan body and reassociates the scatter reductions differently than op-by-op eager). σ, being
  VP-kink noise-amplified near rigid pack, blows that to 1.4e-4 — while velocity stays bound. BOTH
  paths match the cdump within the scatter floor (eager-vs-it120 1.5e-12; scan-vs-UF 6e-11). So:
  gate the kernel MATH eagerly (tight, intermediate access), gate the WIRING via scan-vs-final
  (looser, fusion floor), and keep σ out of the binding set (the std-EVP `_DIAG_FIELDS` precedent).
- **[mevp/JM.3] The s2 controlled replay is a free σ-persistence (T11) test — feed the C's
  step-2 entry + the s1-FINAL σ and it must reproduce s2.** The s1 replay starts from a 0 σ carry
  (cold start), so it can't catch a σ that's wrongly zeroed on entry. The s2 replay can: I feed
  the C's step-2 Q/U0/F entry AND s1's it120 σ (the carried State), and the kernel reproduces s2
  it*/UF to 7e-14 — TIGHTER than s1 (the velocity is more settled). If σ were zeroed on entry,
  s2 would diverge immediately. Bonus: s2's entry has a fully-evolved ocean (|u_w|=0.39,
  |elev|=0.35, entry |u_ice|=0.23), so it exercises the ocean-drag + ssh-tilt + Coriolis paths
  that s1's at-rest cold start leaves at zero. **Moral: a 2nd-step replay seeded with the 1st
  step's carried state validates state PERSISTENCE for free — design replays to inherit, not
  re-zero.** (`test_mevp_s2_replay`.)
- **[mevp/JM.3] Diff-of-diffs liveness across DIFFERENT temporal samples caps the correlation at
  ~0.4 — and that's still a valid liveness proof.** The GATE wants (JAX-mEVP − JAX-EVP) to
  pattern-match (C-mEVP − C-EVP). My JAX legs are a day-10 SNAPSHOT; the only C oracle is a
  January MONTHLY MEAN. The correlation came out +0.36/+0.37/+0.45 (sst/a_ice/m_ice) — clearly
  positive, domain-robust (ice-zone and |Δ|-top-decile give the same), but not the C's own
  +0.96 (which is C-vs-C at IDENTICAL sampling). The moderate value is the snapshot-vs-mean
  smoothing + ~5-day centering offset, NOT a fidelity gap — the kernel is bit-faithful to 1e-12
  at every dumped iterate. **Moral: when the only oracle is time-averaged and your run is a
  snapshot, expect a pattern correlation in the 0.3-0.5 band even for a perfect port; judge
  liveness by sign + robustness + the independent bit-level dump gates, not by hitting 0.9. A
  longer (monthly) run would tighten it but isn't needed once the kernel is dump-exact.**
  (`core2_mevp_climate_compare.py`.)
- **[mevp/JM.4] mEVP-ON leaves the trainable-seam gradient UNCHANGED — the rheology and the
  mixing seam are cleanly decoupled.** The whole point of the mEVP port is that it doesn't break
  the hybrid-ML seams. The assembled GPU gate (`MEVP_GRAD_GATE_OK`) confirms it: d(SST)/d(k_ver)
  plateau = 1.31e-10 (well-conditioned, FD↔AD agree) — IDENTICAL class to the EVP-ON value,
  because k_ver is an OCEAN-column mixing parameter and mEVP is an ICE-momentum solver; they only
  couple weakly through the surface fluxes. The assembled masked-NaN probe d(SST)/d(T0) is clean
  (non-finite=0, masked-lane=0 exactly) through the 120-iteration mEVP scan + thermo Newton + FCT
  limiter. The ice→ocean path d(SST)/d(m_ice0) is finite at 6.3e9 — the documented stiff-but-finite
  rheology class (∂p/∂Δ near rigid pack is huge but the additive δmin keeps it finite). **Moral:
  when adding an option, the gradient gate isn't just "does it differentiate" — it's "does turning
  it on perturb the EXISTING trainable seams"; the plateau-unchanged check is the one that matters
  for a hybrid-ML codebase.** (`core2_mevp_grad_gate.py`, 25GB/40% on the A100-80 at N=4.)

- **[mevp/JM.5] mEVP's sharded velocity came out BIT-IDENTICAL N-vs-1 (owned diff = exactly 0),
  tighter than std-EVP — the cold-start it1 wind-map has no scatter, and per-owned-node scatter
  preserves order.** The sharded N-vs-1 (CPU 4 fake-devices, npes=2) printed only FCT tracers
  (T 5.8e-3, S 4.8e-3) and clean fields (eta_n 1.2e-13, uv 1.5e-16) — u_ice/v_ice/a_ice/m_ice/σ
  did NOT appear, meaning their owned max|Δ| is 0.0 (`_owned_match` prints `if diff>0`). So I
  moved u_ice/v_ice OUT of the FCT bucket into the STRICT clean-floor gate (the plan's intent),
  keeping only the FCT-advected ice scalars (a/m/snow/t_skin) at the climate-close floor and σ
  excluded (`_DIAG_FIELDS`, VP-kink). Why bit-identical: at step 1 from the cold IC the velocity
  is a pure per-node wind-stress map at it1 (no scatter), and for the later iterates an owned
  node's incident-element scatter is the same set in the same relative order on each partition
  (the halo exch makes the vertex velocities identical). **Moral: don't assume a scatter-based
  field needs the loose reassociation floor under sharding — MEASURE it; mEVP's owned velocity is
  bit-exact, so gate it strictly and let the result document the fidelity.** ⚠️ Practical: the
  full-model `shard_map` compiles are too heavy for the login node (~17 min for 2 tests) — run the
  sharded group on a COMPUTE node (`-p compute --cpus-per-task=16 --mem=200G`, the suite-sbatch
  pattern), not the login node. (`test_mevp_sharded_step_owned_matches`, `mevp_jm5_shard.sbatch`.)

## Phase 9c follow-up — year-scale climate validation (JAX / C / Fortran)

- **[mevp/climate] mEVP got the year-scale C comparison it lacked, and it lands at the C↔Fortran
  floor.** mEVP's GATE 9c was 10-day stability + diff-of-diffs (lighter than zstar/TKE's full
  year-1 climate). The user flagged the inconsistency; the oracles existed, so I ran a year-1
  JAX-mEVP climate (KPP+GM+mEVP+linfs, dist864 IC matching the c_mevp_2yr partition) and the 4-way
  RMS: A=RMS(JAX-mEVP,C-mEVP)=5.00e-3°C/2.43e-3psu **== C₀=RMS(C-mEVP,Fortran-mEVP)=5.29e-3/2.49e-3**
  (A/C₀=0.95/0.98), with the mEVP↔EVP contrast B=1.30e-2 (B/A=2.6). The JAX port adds no more error
  than the C-vs-Fortran discrepancy. **Moral: when one option gets a lighter validation gate than
  its siblings, fill the gap if the oracles exist — consistency across the option set is worth a
  GPU-hour.** (`core2_kpp_climate_run.py --mevp on`, `core2_mevp_climate_compare.py --annual`.)
- **[mevp/climate] The C port CAN run option COMBINATIONS — and so can Fortran; "no combined oracle"
  ≠ "can't compare to C".** I'd said the all-3 (zstar+TKE+mEVP) couldn't be compared to C "because
  the C never ran them combined" — WRONG. The C options are independent env vars (FESOM_ALE /
  FESOM_MIX_SCHEME=TKE / FESOM_WHICH_EVP=1) with no mutual guard, and a combined zstar+TKE C job
  already existed. Adding one env line gave a C-all-3 oracle; flipping `whichEVP=0→1` in the Fortran
  `work_zstar_tke` namelist gave the Fortran ground truth. The 3-way: A=RMS(JAX,C)=4.5e-3,
  A′=RMS(JAX,Fortran)=4.1e-3, C₀=RMS(C,Fortran)=4.7e-3 (SST) — all at the climate-close floor, JAX≈C≈Fortran.
  All three engines ran the triple-option combo STABLE for a year. **Moral: "we haven't" is not
  "we can't" — before declaring a comparison impossible, check whether the reference *can be made to
  produce it*. The user caught both my overstatements.** (`port2 jobs/job_all3_1yr`, `work_all3`.)
- **[mevp/climate] Averaging a STILL-RUNNING model's monthly output gives a seasonal artifact — guard
  the record count.** The first all-3 3-way pass showed A′=C₀=0.26°C (50× the floor) and I almost
  reported a "physical-validation failure" — but the Fortran job was still running and its
  `sst.fesom.1958.nc` had only 10/12 months. A 10-month mean (missing Nov–Dec winter) differs
  seasonally from a 12-month mean by ~0.26°C. Added `_n_records≥12` so the compare skips a partial
  file (2-way until the run completes) instead of silently averaging it. **Moral: a metric over
  time-averaged output is only valid once the averaging window is COMPLETE; assert the record count
  before trusting an "annual mean" from a run you didn't confirm finished.** (`core2_mevp_climate_compare.py`.)

## Paper experiments (PART A infra) — `docs/plans/20260614-fesom-jax-paper-experiments.md`

- **[calibrate/A3] The optimizer seam is generic over the tunable pytree — keep it that way; don't
  bake in physics couplings.** `calibrate.optimize` jits `value_and_grad` once and host-loops; the
  *same* call trains a scalar `{'k_gm': θ}`, a `[nod2D]` field, or the `tke_nn` MLP because array
  leaves already differentiate. `build_params(dict)->Params` is the dict→pytree seam — deliberately
  generic: it does NOT auto-sync `Redi_Kmax=K_GM_max` (the GM caller passes both keys; the namelist
  writer enforces the sync on the Fortran side). A hidden coupling in the seam would surprise every
  *other* tunable. `stop_fn` is checked BEFORE the optax update so the returned params are the exact
  iterate that crossed the threshold. (`fesom_jax/calibrate.py`, `test_calibrate.py`; CALIBRATE_SEAM_OK.)

- **[obs_compare/A2] A differentiable model→obs operator must recompute the VERTICAL interp
  from live geometry; only the HORIZONTAL node→cell map is static.** `build_h_map` host-precomputes
  the node→obs-cell index (nodes are horizontally static); `to_obs` does a masked area-weighted
  node→cell `segment_mean` of BOTH the field and the live `Z_3d_n`, then a per-cell linear interp
  on the live depth axis — so `d(misfit)/d(hnode)` flows through the zstar coordinate (FD-probe
  nonzero, AD==FD confirmed). Pre-baking vertical weights would silently sever that gradient. Key
  AD-safety details: (1) the interp bracket index is a non-diff step but the WEIGHTS differentiate
  w.r.t. the gathered depths (guarded denom, clipped weight) — no `jnp.interp` needed; (2) empty
  cells use the `ops` 0/0 sentinel-mask; (3) below-bottom levels are filled with a strictly-
  decreasing nominal axis so the interp stays monotone; (4) MLD is the linear-interpolated σ-excess
  crossing (NOT an argmax), and the EOS gets safe `S>0` at dry lanes (its `sqrt(S)` has ∞ grad at 0).
  Obs depths above the top layer constant-extrapolate (the standard model↔surface-obs convention) —
  analytic exactness holds only strictly within `[cell_bottom, cell_top]`. (`fesom_jax/obs_compare.py`,
  `test_obs_compare.py`; OBS_OPERATOR_OK.)

- **[obs_ice/A2b] The sea-ice obs operator is forward-only — no live-geometry path — but the
  pole hole and hemisphere mask are load-bearing.** Sea-ice concentration is a 2-D surface field,
  so `obs_ice` is a plain area-weighted node→polar-stereo-cell mean (no vertical interp, no AD
  contract — nothing calibrates through it). The polar-stereographic projection is the spherical
  NSIDC convention (true scale at the standard parallel; verified pole→origin, lat_ts→R·cos(lat_ts)).
  Two masks matter: wrong-hemisphere nodes → `node_cell=-1` (ops sentinel), and the passive-microwave
  **pole hole** (cells with radius < ρ(87.2°)) is dropped in BOTH model and obs for a consistent
  comparison. Test trap: a ±3000 km north grid only reaches ~64°N, so test nodes must sit at high
  latitude (ρ(lat) < grid extent) or they project outside and silently drop. (`fesom_jax/obs_ice.py`,
  `test_obs_ice.py`; OBS_ICE_OK.)

- **[tke_nn/A5] A structure-preserving NN closure = `m = exp(s·tanh(raw))` on the constants, with
  a ZERO last layer for an exact-identity fallback.** The bounded multiplier `m ∈ (1/m_max, m_max)`
  is ALWAYS positive (exp) ⇒ positive-definite diffusivities for any weights/inputs (the structural-
  stability guarantee, no clamping). Zero last layer ⇒ `raw≡0` ⇒ `tanh(0)=0` ⇒ `m≡1` EXACTLY, so
  `c_k·m == c_k` bit-for-bit ⇒ default TKE recovered to the ULP — this is BOTH the `params=None`
  regression invariant AND the deployment net. Wiring trap: `m=1` only bit-identical if the multiply
  is `scalar·ones[N,1]` then `c_k·mxl` (broadcast commutes, products identical) AND the FEATURES are
  finite everywhere — a NaN feature at a dry column would survive the zero last layer as `0·NaN=NaN`,
  so `column_features` arcsinh-normalizes with count-guarded interior means (finite for every finite
  input). `tke_nn` is a `Params` leaf (`None`⇒no leaves, structure unchanged ⇒ the `len(leaves)==8`
  test still holds), trained by the same `grad(loss)(params)`. (`fesom_jax/tke_nn.py`, `tke.py` consume
  site, `test_tke_nn.py`; TKE_NN_OK.)

- **[eki/A4] EKI is the right tool where the adjoint can't reach — a forward-only, vmap-parallel
  ensemble update, no adjoint anywhere.** `eki_update` is pure ensemble linear algebra
  (`θ_j += C_θg(C_gg+Γ)⁻¹(y*+η_j−g_j)`); `eki_step`/`eki_run` vmap the single-member forward over
  the ensemble. The ensemble MEAN converges to the minimizer (verified: recovers known scalars +
  multi-param vectors from noisy linear AND mildly-nonlinear forwards to <2%). Γ accepts scalar σ²
  / diagonal / full matrix. This is the slow-target (GM→stratification) calibrator the chaos+memory
  ceiling rules out for the adjoint, and the §1 cross-check partner on `k_gm`. The real-model budget
  (warm-started 16–32-member few-year ensemble) is the plan's; the unit test is analytic.
  (`fesom_jax/eki.py`, `test_eki.py`; EKI_OK.)

- **[write_namelist/A6] Scalar→Fortran transfer = a FORMAT-PRESERVING namelist patcher, exact-key
  match, with the Redi auto-sync honored on EVERY input path.** The "killer app" (tuned scalar →
  operational Fortran, zero code) needs a writer that replaces only the named value and leaves every
  comment/column/blank byte-identical (regex captures indent/key/`=`/value/comment; re-pad comment
  with 2 spaces). Two traps: (1) match the EXACT key token before `=` — `K_GM_max` must not touch
  `K_GM_min`/`K_GM_bvref`; (2) the K_GM_max↔Redi_Kmax auto-sync (`Redi_Kmax<=0`⇒Fortran syncs) must
  fire on BOTH the Params-leaf path AND the raw-namelist-key/CLI path, else `--set K_GM_max=1500`
  silently leaves Redi_Kmax at 0 — moved the sync into a `_sync_redi_keys` helper applied to the final
  namelist-keyed dict. Real keys verified against `fesom2/work_all3` (oce: K_GM_max/Redi_Kmax; cvmix:
  tke_c_k/c_eps/cd/alpha). Test lives in `scripts/tests/` (pure-Python, runs outside the JAX suite).
  (`scripts/write_namelist.py`, `scripts/tests/test_write_namelist.py`; FORTRAN_TRANSFER_OK.)

- **[adjoint-window/A7] MEASURED: the CORE2 TKE→MLD adjoint window is N_max≈20 (~0.42 d / ~10 h) on an
  80GB A100 — NOT the ~1 day I first extrapolated. Measure the ceiling; don't extrapolate it.** A
  single-GPU N-sweep of `d(mean MLD)/d(tke_c_k)` (zstar+TKE) gives a CLEAN gradient (FD↔AD plateau
  2e-5 @ N=4, 6e-7 @ N=20 — the signal is real, FD==AD) with peak backward memory ≈ **33 + ~0.95·N GB**:
  N=4→37GB, N=20→52GB. N=50's backward needs a single **70 GiB** tensor that can't coexist with the
  working set even on a full 80.8GB card (OOM), and N≥100 exceeds the card outright ⇒ **N_max=20**. (I'd
  extrapolated N=50 would fit at a higher mem fraction; the rerun with `MEM_FRACTION=0.95` proved it does
  NOT — the 70 GiB single allocation + overhead/fragmentation > 80 GB. A linear memory fit doesn't
  predict the largest single-tensor allocation.) **Decision:** fast MLD/SST are adjoint-reachable at
  ~10-h BATCHED windows (feeds D2a/E2); slow GM→T/S (multi-year) ⇒ EKI. Reconciles the inherited
  37.8 GB-at-N=20 GM figure: TKE+zstar+MLD is heavier (52 GB at N=20 — the tke scan-carry +
  live-geometry + density + MLD diagnostic). TWO infra traps cost a GPU run each: (1) Levante's `gpu`
  partition mixes **a100_40 and a100_80** nodes — `--gres=gpu:1` can land on a 40GB card →
  OOM-on-CUBIN-load; request `--gres=gpu:a100_80:1`. (2) With `XLA_PYTHON_CLIENT_PREALLOCATE=false`
  (avoids the CUBIN-load OOM), XLA's default 0.80 fraction caps usable memory at 64GB on an 80GB card;
  set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` for the full card. (`core2_adjoint_window_sweep.py`+`.sbatch`,
  `fig_window_snr.py`; **WINDOW_DERISK_OK**, N_max=20.)

- **[sensitivity/C1] A scalar→`[nod2D]` field-leaf sensitivity map costs ZERO kernel change, and ONE
  backward delivers BOTH the map AND the scalar grad-gate proof.** Promoting a `Params` scalar to a
  field is just `build_params({'k_gm': field})` — the seam already differentiates array leaves (the
  "scalar calibration is the zeroth-order NN" thesis). The only trap is the **broadcast shape**, which
  differs per consumption site: `k_gm` promotes to `[nod2D]` (it enters as `scaling*k_gm` with
  `scaling=area[:,0]` → `[nod2D]`, then `[:,None]` over levels), but `tke_c_k` MUST be `[nod2D,1]` (it
  enters as `c_k*mxl*√tke` with `mxl` being `[nod2D,nl]` — a bare `[nod2D]` mis-aligns with the level
  axis). Verify the broadcast on the pi mesh BEFORE the GPU run (a 2-line `jax.grad` probe). **The
  uniform-broadcast identity** `dJ/dθ_scalar == Σ_x ∂J/∂θ_field(x)` (a uniform field's cotangent sums)
  is exact (pi mesh: rel 2.8e-16) — so the field map's SUM *is* the scalar gradient the existing grad
  gates report, FD-verified with one forward sweep ⇒ no separate scalar backward needed. The pi-mesh GM
  `d(meanT)/d(k_gm)` is tiny (~7e-10) yet FD↔AD-clean to 1e-9 (smooth GM response) — small magnitude ≠
  noisy gradient. Label the map HONESTLY as the **fast/instantaneous** (single ~10-h window) sensitivity,
  NOT the equilibrium; the **adjoint↔EKI cross-check** (`cov(θ,J)/var(θ)` == adjoint scalar grad for the
  locally-linear `k_gm`) validates both tools and motivates EKI for the slow equilibrium the adjoint
  can't reach. (`scripts/paper/core2_paper_sensitivity.py`+`.sbatch`, `scripts/paper/fig_sensitivity.py`,
  `fesom_jax/tests/test_sensitivity.py`; **SENSITIVITY_SEAM_OK** (CPU) + **SENSITIVITY_MAP_OK** (GPU).)

- **[sensitivity/C1 GPU memory] The FIELD-leaf backward is heavier than the SCALAR backward → its N_max
  is SMALLER than A7's (TKE+MLD: N_max=12, not 20). Two distinct OOM traps, both real.** (1) **CUDA-graph
  command buffers**: a driver that compiles several big executables (loss + grad) accumulates CUDA graphs;
  the first run OOMed not on the working set but on `instantiate command buffer ... (38 alive graphs)`. Fix
  = `XLA_FLAGS=--xla_gpu_enable_command_buffer=` (the error recommends it) AND reuse ONE jitted forward for
  every scalar eval (FD sweep + EKI ensemble) instead of separate executables. (2) **A 63.27 GiB "Failed to
  allocate" message is a NON-FATAL XLA probe, NOT the verdict** — XLA tries a big fused buffer, fails, and
  falls back to a streamed path; mld_ck N=20 genuinely OOMed (working set too big) but N=12 SUCCEEDED at
  peak **44.5 GB** *with the same 63 GiB failure printed*. So don't trust the scary alloc line; trust the
  exit code / the reported `peak`. ⇒ a **descending-N safety net** in the .sbatch (20→12→6, the script
  returns exit 2 on OOM so the loop tries smaller) finds the field N_max in one job — measure it, like A7.
  Result: mld_ck N=12 clean (d(meanMLD)/d(c_k)=+2.91, plateau **8.3e-7**, FD spot-check rel **1.8e-6**), and
  the map peaks in the **Weddell/Labrador/Southern-Ocean deep-convection regions** (physically correct —
  where TKE mixing controls MLD). Single-GPU, `--gres=gpu:a100_80:1`, PREALLOCATE=false, MEM_FRACTION=0.95.
  (`scripts/core2_paper_sensitivity.{py,sbatch}`; **SENSITIVITY_MAP_OK**.)

- **[sensitivity/C1 FD spot-check] A single-h single-node FD verifies a map node UNRELIABLY — use an
  h-sweep over the top-K |g| nodes.** Validating one map cell by `(J(θ+h·eᵢ)−J(θ−h·eᵢ))/2h` at the |g|-max
  node with ONE step (h=1e-3·θ) gave rel **0.79** for the tiny-signal GM target (ts_kgm map max|g|≈6.5e-10),
  even though the AGGREGATE scalar adjoint==FD was perfect (plateau 6.5e-7). Cause was NOT a true kink and
  NOT round-off (signal ≫ floor): the secant over Δk_gm=1.0 crossed local curvature; the SAME node is clean
  to **1.2e-4 at h_rel=1e-4**. Fix = sweep h ∈{1e-2,1e-3,1e-4} at the top-3 |g| nodes and take the best
  plateau (one strong-signal SMOOTH node passing proves the map values are FD-correct; robust to a single
  extremal node). After the fix BOTH targets pass (mld_ck 3.4e-7 @ Weddell, ts_kgm 3.0e-5 @ Fram Strait).
  Lesson: a per-cell FD check needs the same h-plateau rigor as the scalar gate — never trust one h. Also
  reconfirmed: the GM short-window adjoint signal is **tiny** (6.5e-10 vs MLD/c_k's 2e-2) ⇒ the slow GM→T/S
  equilibrium is beyond the adjoint window — exactly why §2 GM calib uses EKI (the adjoint↔EKI rel-6.6%
  cross-check ties them). (`scripts/paper/core2_paper_sensitivity.py`; **SENSITIVITY_MAP_OK**.)

- **[AD / sea ice] The all-on (zstar+TKE+mEVP+GM) FORWARD is fine but the naive ADJOINT explodes
  through the mEVP sea-ice rheology — fix with a frozen-ice adjoint (full forward, `stop_gradient`
  the ice in the backward).** D1's perfect-model `k_gm` twin on the FULL model had a textbook-clean
  misfit bowl (argmin exactly at the truth) yet the backward gave `|g|≈9e51`, *mis-directed* (k_gm ran
  800→516→159). The ocean-only adjoints are clean at N=12 (C1: zstar+TKE, zstar+GM) and the mEVP
  grad-gate already showed huge ice gradients (`d/d(m_ice)=6e9`) at N=4 ⇒ the culprit is the mEVP
  120-iter plastic-yield solver's adjoint — the classic VP/EVP sea-ice adjoint instability MITgcm/ECCO
  hit. Fix (`IceConfig.adjoint_mode="frozen"`, `step.py`): run the full mEVP in the forward but
  `jax.tree.map(jax.lax.stop_gradient, ice_out)` so the backward never enters the rheology adjoint.
  `stop_gradient` is the identity in the forward ⇒ **forward bit-identical** (verified: frozen
  J0=2.676528e-06 == exact). Result: `|g|` 9e51→**2.31**, and the all-on twin recovered **k_gm=1498.88
  (rel 0.075%)**, misfit 2.68e-6→1.78e-10. Lesson: in AD you can run the full physics forward and
  substitute a stable adjoint for one block — `stop_gradient` (skip) or `custom_vjp` (the planned
  free-drift adjoint = drop ∇·σ, keep wind/ocean/Coriolis/tilt). An approximate adjoint ⇒ FD≠AD by
  construction; validate by descent + EKI cross-check, not a gradient check. (`fesom_jax/ice.py`,
  `fesom_jax/step.py`, `scripts/paper/core2_paper_calib_twin.py`; **TWIN_RECOVER_OK**.)

- **[optimizer] A tiny-signal misfit needs the loss normalized by its initial value, or Adam's `eps`
  swamps the gradient.** The GM short-window misfit is ~1e-6 (the GM→T signal is ~1e-9; C1), so the raw
  `d(loss)/dθ` is ~1e-11 — far below Adam's default `eps=1e-8`, making the `m/(√v+eps)` step ≈ `lr·g/eps`
  ≈ tiny ⇒ the parameter barely moves (a CPU pi-twin caught this: recovery stalled at the start). Fix:
  optimize `loss/J0` (J0 = the misfit at the start; an O(1) loss ⇒ O(1) gradient) on a scale-free leaf
  `u=θ/θ_ref`. A pi convergence probe then sets `lr`/`iters` (lr=0.05/80→2.2%, *missed* the 2% bar;
  lr=0.1/100→<0.1%) — probe the cheap mesh first, the normalized-bowl dynamics are mesh-independent.
  (`scripts/paper/core2_paper_calib_twin.py`, `fesom_jax/tests/test_calib_twin.py`; **TWIN_RECIPE_OK**.)

- **[calibration / §2] MLD is a strong FAST adjoint target — the TKE→MLD twin recovers in the FULL
  all-on model via the frozen-ice adjoint, far faster than the GM→T twin.** D2a injects `tke_c_k=0.15`,
  freezes the model's density-threshold MLD field (`obs_compare.mld_density_threshold`) as the synthetic
  obs, and recovers from 0.08 through the all-on + frozen-ice adjoint (`calibrate.optimize`): **c_k=0.14974
  (rel 0.176%) in 9 iterations**, MLD misfit 3.74→3.69e-6 m² (6 orders). vs the GM twin's 43 iters — the
  MLD↔c_k signal is ~10 orders larger (C1: ∂MLD/∂c_k=+2.91 vs ∂T/∂k_gm~1e-7), so MLD calibration is cheap
  and well-conditioned (loss-normalization barely needed, J0~O(m²)). Confirms (a) the C1 sensitivity
  signal is optimizer-grade, and (b) the frozen-ice adjoint generalizes to a 2nd parameter/target. MLD is
  the right OMIP metric for TKE calibration (fast ⇒ adjoint, not EKI). No CPU twin: TKE raises on the pi
  `integrate` path (`mixing_tke` is the faithful site) — the recipe is guarded by the D1 `test_calib_twin`.
  (`scripts/paper/core2_paper_calib_tke.py`; **TKE_TWIN_OK**.)

- **[obs] A density-threshold MLD must be computed PER MONTH then averaged — never from the
  annual-mean WOA profile (Jensen's inequality).** dBM is IFREMER-blocked, so we derive the MLD
  climatology from WOA. Computing it from the annual-mean T/S profile collapses to ~20 m everywhere
  (Labrador crosses 0.03 kg/m³ by 20 m) because the deep winter convection averages out — MLD(mean
  profile) ≠ mean(MLD per profile). Fix: compute MLD for each of the 12 monthly WOA files (the model's
  own `obs_compare.mld_density_threshold`, for a consistent comparison) then average → a real seasonal
  cycle, max 319 m. (Still under-represents sporadic deep convection vs dBM's individual profiles —
  ice-masked out anyway.) Also: WOA SST is NaN on land ⇒ the obs misfit needs `where(isfinite & ok, x,
  0)` or `0·NaN=NaN` poisons the weighted sum even at weight 0 (the masked-NaN rule, obs side).
  (`scripts/tools/make_woa_targets.py`.)

- **[AD / JAX] A `RESOURCE_EXHAUSTED: Failed to load in-memory CUBIN` OOM at iteration 2 of an optimizer
  loop is a WEAK-TYPE RECOMPILE, not a memory limit.** D2a-obs (TKE→WOA MLD+SST) died at it=2 with this
  error for 8 attempts, **independent of config (all3/tkegm), N (12→4), and MEM_FRACTION (0.95/0.85)**,
  with tens of GB free — while the twin's loop ran 43 iters fine. Root cause (CPU-confirmed with a trace
  counter): the init `u0 = {"ck": jnp.asarray(1.0)}` is **weakly typed**; the first Adam update produces
  a **strongly** typed float64, so the loop calls the jitted step with a *different signature* at it=2 →
  XLA **recompiles** → a 2nd executable's CUBIN loads atop it=1's still-resident one → OOM. The twin
  never hit it because its init was explicit `jnp.asarray(x, jnp.float64)` (strong). Fix: strong-type the
  optimizer init (`jnp.asarray(1.0, jnp.float64)`). Lesson: "Failed to load CUBIN" mid-loop ⇒ suspect a
  per-iter **recompile** (weak types, Python-int/shape drift), not the allocator — verify with a trace
  counter. Then the obs calib runs all-on (job 25601631, 32 it, 52.6 GB; MLD misfit −8.3%, SST −4.8%).
  ⚠️ The 2-param `{c_k,c_eps}` fit overfits — `c_eps`→0 (unphysical) — the structural-bias compensation
  the recovered-value plausibility report catches; use `--params ck` or bound `c_eps`.
  (`scripts/paper/core2_paper_calib_tke_obs.py`.)

- **[Calibration / EKI] The slow-target GM→T/S calibration uses gradient-free EKI on the FULL all-on
  model with LIVE mEVP ice — forward-only ⇒ immune to the A8 sea-ice-rheology adjoint instability** (no
  frozen-ice approximation the adjoint twins D1/D2a needed; the whole point of the adjoint↔EKI split). D2b's
  perfect-model EKI twin (`scripts/paper/core2_paper_calib_gm_eki.py --mode twin`) recovered a planted `k_gm=1500`
  to **0.034 %** (1500.51; ensemble 1500.5±0.5) in **5 EKI iters** at a **5-day** window (10 members; the 2-day/
  6-member de-risk gave 0.065 %), misfit 1.4e-5→4.2e-11, peak **24 GB** (forward-only is light — no tape).
  Observable = basin-mean upper-ocean/thermocline T/S
  profiles (5 lat bands × 8 WOA levels × {T,S} = an 80-vec) via the tested `obs_compare.{to_obs,
  basin_mean_profiles}`; the SAME fixed reduction (fixed basin weights + a fixed common-validity mask) hits
  model AND obs ⇒ a clean *linear* observable where members differ only through the physics. ⚠️ The GM→T/S
  signal over a short window is TINY (~3e-4 °C / 1.9e-4 psu RMS across `k_gm`∈[600,1800] at 2 days) but CLEAN
  (≫ float64 noise) — EKI's covariance update is scale-free so it recovers it, **but only if Γ ≪ the
  signal²**: use auto-Γ = `(ε·ensemble-signal)²` (ε=0.05) for the twin. A *physical* absolute σ_T=0.5 °C
  would swamp the 3e-4 °C signal (Γ ≫ C_gg ⇒ no update) — the EKI analogue of D1's "normalize the loss by
  J0". This is also exactly why the short-window *obs* calibration (which must use physical σ) is weakly
  constrained — **empirically confirmed**: the obs run (`--mode obs`, WOA basin T/S over a 10-day window) recovered
  `k_gm=1356` but with an ensemble spread of **±347 (26 %, no collapse)** and a misfit that barely moved (−0.2 %,
  bowl flat ≈2.35–2.40 across [600,1800], argmin at the grid edge). The WOA misfit is IC/spin-up-dominated over
  short windows ⇒ the **twin proves the OPTIMIZER works, the obs proves the WINDOW is the limiting factor** (so the
  equilibrium GM→stratification calibration needs the multi-year production ensemble — short-window adjoint OR
  short-window EKI both under-constrain it). The recovered-value-with-uncertainty report (±347) is the rigor catch,
  the D2a `c_eps`→0 analogue: report VALUES + plausibility, never just "misfit reduced".
  (`scripts/paper/core2_paper_calib_gm_eki.py`, `fesom_jax/obs_compare.basin_mean_profiles`, `fesom_jax/eki.py`.)

- **[AD / JAX / perf] A jit that closes over the pre-stacked CORE2 forcing makes XLA constant-fold the
  `to_obs` node→cell scatter over the mesh constants — a slow ONE-TIME compile** (D2b: a single
  `scatter-add f64[64800,48]` folded for ~20 s, plus `reduce-window`/`gather` alarms over mesh constants). It
  completes (XLA's guard caps the folding) and is amortized over the many forwards of an EKI run (one compile
  per mode), so it is tolerable — but for the multi-year production EKI, pass the forcing as a **traced
  argument** (not a closed-over constant) to skip it. Two hard limits the forward-only EKI also exposed:
  **(1)** `StepForcing` is ~10 MB/step ⇒ `cf.stack(N)` caps a single pre-stacked window at ~weeks (N=480 ≈
  4.8 GB), so a multi-year window needs **chunked re-stacking** (carry state across chunks — the
  `core2_kpp_climate_run.py` per-step pattern, but in fast scanned chunks). **(2)** The all-on forward runs
  **~4.3 steps/s on an A100-80** (forward-only, mEVP live, peak 18 GB) ⇒ a few-year × 16-member × 8-iter
  production EKI is hundreds of GPU-h — a production run, NOT overnight; the overnight EKI is a scoped
  demonstration + the GPU-h projection (`--mode budget`, the EKI analogue of A7's adjoint-window de-risking).
  (`scripts/paper/core2_paper_calib_gm_eki.py`.)

- **[calibration / rigor] Held-out CV answers TWO different questions for a global scalar — pick the split to
  match the question.** Validating the TKE `c_k`→WOA-MLD calibration on independent cells: a **random** 50/50
  cell split (train+held-out share the bias structure) gives held-out reduction **≈ train** (~+2 % each) ⇒ the
  clean "**not overfitting**" test (it fits real signal, not per-cell noise). A **blocked** 60° longitude split
  (spatially independent) instead exposes spatial **transferability**: the recovered `c_k` stays robust
  (~0.22–0.24 either fold) but the held-out misfit moves **asymmetrically** — a global `c_k` *helps* the
  held-out deep-convection sectors (**+2.75 %**) yet slightly *over-mixes* the held-out low-bias sectors
  (**−0.15 %**), because the `c_k`-sensitive bias is spatially concentrated (matches the §1 C1 map). A single
  global scalar can't fix a spatially-structured bias ⇒ report BOTH (random gates "not overfitting"; blocked
  motivates spatially-varying params). The recovered VALUE being robust across ALL splits (here spread 7.3 %,
  random cross-fold scatter 1.2 %) is the strongest rigor metric — report values + plausibility, never just
  "misfit reduced". Two practical notes: **(1)** score the held-out cells via `has_aux` (their weights never
  enter the loss) so no gradient leaks from them; `--holdout none` then stays bit-identical to the full-domain
  path. **(2)** A short-window calibration only moves what the window constrains: MLD ΔRMSE is real but **SST
  ΔRMSE (~0.003 °C) sits UNDER the C↔Fortran 0.0049 °C floor** — honestly, MLD is the constrained channel, not
  SST (and the EN4 interannual-spread bar needs the multi-year series, not the staged seasonal climatology).
  (`scripts/paper/core2_paper_calib_tke_obs.py` `--holdout`/`build_holdout`, `scripts/paper/fig_calibration.py`.)

- **[§3 hybrid-ML / AD memory] The CORE2 NN-of-TKE backward is RUNTIME-peak-bound, not compile-temp-bound
  — `memory_analysis()` underpredicts by ~2.4×, and a SECOND large jitted program is what actually OOMs.**
  Training a `tke_nn` end-to-end through the full all-on global model (zstar+TKE+mEVP+GM, frozen-ice adjoint)
  OOMed the 80 GB A100 at EVERY window N≤12 — not from un-checkpointed unrolling (`integrate` already wraps
  each step in `jax.checkpoint`), but from the **single-step VJP working set** (one step recomputed during the
  scan reverse tapes ALL its full-3D-mesh intermediates — FCT edge fields at 143 MB, Redi/GM/momentum — at
  once). Decompose the OOM-vs-N requests (43.66 GiB@N12 → 38.57@N6): **~0.85 GiB/step (scan carries) + ~34 GiB
  fixed (the per-step tape, independent of window length)** — so descending N never helps. Two fixes that
  compose: **(1) nested in-step remat** — a `remat_blocks` flag through `integrate`→`step` wrapping each heavy
  block (GM, TKE/KPP/PP, momentum×3, FCT×2, Redi, vert-diff, EOS, pgf) in its OWN `jax.checkpoint` so the
  reverse sweep recomputes ONE block at a time (`jax.checkpoint` is the identity in the forward ⇒ bit-identical
  values, gradient-identical; `remat_blocks=False` default ⇒ byte-identical no-trace-change path — verified
  pi forward Δ=0, grad rel=0). **(2) driver: compute baseline/final misfits by reusing the already-compiled
  `model_ts` executable, NOT a separate `jax.jit(raw_loss_TS)`** — with `XLA_PYTHON_CLIENT_PREALLOCATE=false`
  (mandatory — `=true` OOMs the truth forward's CUBIN load, the D2a it=2 gotcha) BFC never returns a freed
  region, so a second ~11 GiB executable arena fragments the pool and the grad can't grow its ~38 GiB region.
  RESULT: runs at **N=10** (peak 39.3/81 GB) where it OOMed before — grad-check FD↔AD `rel=1.6%` (the
  NN-weight gradient through the GLOBAL model is correct) and the T/S evolution recovers (misfit 0.15×). Two
  measured truths: **(a)** the **recovery LOOP** (`value_and_grad`) has huge headroom (39/81 GB at N=10) — only
  the OPTIONAL standalone grad-check OOMs above N=10 (a fragmenting 2nd program right after the truth forward),
  so `--no-grad-check` (gradient already proven) admits a much longer window. **(b)** a 5-h window recovers the
  T/S evolution + the gradient but NOT the induced-mixing FIELD (corr_active 0.118 ≪ 0.9): a rich per-column NN
  multiplier is **under-constrained** by short-window evolution (classic non-uniqueness; cf. D2c spatial
  transfer) — field recovery needs a longer window (or the cheap-long-window pi recipe `test_nn_twin.py`).
  Build a compile-only `memory_analysis()` probe to size forward-vs-backward and validate a remat fix BEFORE a
  full GPU run — but trust RUNTIME OOM (the BFC `bfc_allocator.cc` "ran out trying to allocate" line + the
  region-growth attempts) for the true ceiling, not the compile-temp. (`fesom_jax/step.py` `remat_blocks`/`_ckpt`,
  `fesom_jax/integrate.py`, `scripts/paper/core2_paper_nn_twin.py`, `scripts/archive/core2_nn_twin_memprobe.py`.)

- **[§3 hybrid-ML / identifiability] The per-column NN mixing FIELD is NOT recoverable from short-window
  T/S evolution — equifinality — and batching DIVERSE short windows does NOT fix it; only a CONTINUOUS
  long window can (O(√N) checkpointing makes one fit a single GPU).** The CORE2 single-window adjoint is
  memory-capped at N≈10 (~5 h). At that window the NN-weight gradient is exact (FD↔AD 1.6 %) and the T/S
  evolution recovers (misfit ratio 0.05–0.15), but the induced c_k-multiplier FIELD does NOT: corr_active
  +0.118 (early-stop) → **−0.006** (converged, N=10) → **−0.246** (K=8 consecutive 1.67-d chunks) →
  **−0.484** (K=8 SEASONAL chunks, Apr–Dec). Recovery gets *more anti-correlated with more/diverse
  windows* — classic mixing-closure equifinality: over a 5-h window the multiplier PATTERN barely imprints
  on T/S (only the net mixing does), so the optimizer fits the evolution via a compensating, uncorrelated
  field. **Batching adds STATE diversity but each chunk's gradient is severed (independent) ⇒ each is still
  a 5-h problem ⇒ no imprint.** What imprints the field is a CONTINUOUS gradient over a long window (days).
  Three practical results: **(1) batched windows** (`core2_paper_nn_twin_batched.py`) — K chained/seasonal
  N-step chunks, FULL-BATCH grad accumulation (G=Σ_k g_k, one chunk backward live at a time via
  device_put/free) — memory is K-INDEPENDENT (peak ~43 GB at any K), the right tool for E2 obs-training
  (reduce obs misfit, which does NOT need field identifiability) but NOT for perfect-field recovery.
  Seasonal-diverse chunk states come from `core2_kpp_climate_run.py --snapshot-every` (monthly State
  pickles; `core2_nn_snapshots_1yr.sbatch`). **(2) O(√N) two-level checkpointing**
  (`integrate.py` `remat_segments=-1`) — restructures the scan into S≈√N outer segments × M inner steps,
  both checkpointed ⇒ the backward stores ~**2√N** carries not N (the N×|State| stack is what OOMs long
  adjoints). Forward bit-identical, gradient identical (pi Δ=0, rel=0) ⇒ a *continuous* N≫10 window fits
  one A100. **(3) multi-GPU adjoint workaround:** the ragged_all_to_all transpose is the buggy collective
  ([[jax-ragged-a2a-grad-bug]]); the **all_gather halo (`use_ragged=False`) is autodiff-correct** (tested
  sharded-grad==dense, `test_gradient_sharded`) and `run_steps_sharded` already routes the full all-on NN
  config — so per-device tape ~1/P for even longer windows, composes with O(√N). New time bottleneck once
  memory is solved: a continuous N-step backward costs ~N×recompute/iter, so very long windows (N≫100) are
  GPU-h-bound, not memory-bound. (`fesom_jax/integrate.py` `_run_steps`/`remat_segments`.)

- **[§3 hybrid-ML / multi-GPU adjoint] The sharded all_gather adjoint is CORRECT (sharded NN-grad == dense,
  twin reproduces the dense recovery), but it took two fixes — and the multi-GPU run FALSIFIES the "only a
  long continuous window imprints the field" hypothesis: the stiff all3 adjoint amplifies EXPONENTIALLY with
  window length (the chaotic-adjoint horizon), so the usable window is SHORT and field recovery is BEST at the
  SHORTEST window, not the longest.** Unblocking the multi-GPU NN twin ([[e1-nn-twin-memory-fix]]) over the
  all_gather halo (`use_ragged=False`, the autodiff-correct collective; ragged_all_to_all's transpose is the
  buggy one, [[jax-ragged-a2a-grad-bug]]) needed: **(1) the TracerArrayConversionError** — wrapping
  `run_steps_sharded` in `jax.grad` hit `_to_global_sharded`'s eager `np.asarray` on tracers; fix
  (`integrate_sharded.py` `return_grad_fn`): make `params` a REPLICATED `shard_map` INPUT and device-place the
  constant folded mesh/state/forcing/op ONCE outside the returned closure, so the grad trace never reaches the
  placement. **(2) The sharded-grad NaN** (forward finite, `|g|=nan`, sharding+multi-step ONLY) — a
  **masked-NaN-in-reverse-mode** bug. Sharding pads local arrays to `Lmax` (`_default_pad`: float→1.0,
  bool→False, int→0), creating all-dry PADDING columns the dense mesh never has. `cvmix_tke`'s surface-flux
  denom `dzt_surf = where(is_surf, dzt, 1)` divided by `dzt[0]=0` there: the driver only sets
  `dz_trr[0]=hnode/2` where ITS `is_surf=(k==nzmin)` fires, but a padding node has `nlevels_nod2D=0` ⇒
  `nzmin=-1` ⇒ `dz_trr[0]` stays 0, while the core's hardcoded `is_surf=(k==0)` still computes the flux. The
  forward is MASKED finite (`is_wet_iface` all-False on the padding column ⇒ the `d_tri` identity override
  drops it), but REVERSE hits `cotangent(0)·∂surf_flux/∂cd(inf)=NaN` that leaks via `cd=tke_cd·m_NN` into the
  NN weights. Fix: the `dzt>0` guard `where(is_surf & (dzt>0), dzt, 1)` (mirrors Part 4's `dzt_s`) —
  **bit-identical on real meshes, 38/38 TKE tests pass.** GENERAL RULE: a forward that is finite-because-masked
  can still NaN in reverse — guard the non-smooth op (`/ sqrt pow log`) at its SOURCE (double-where), not at
  the masked output; and SHARDING introduces degenerate (all-dry padding) lanes a dense-clean adjoint never
  exercised, so re-audit every masked op for the sharded path. (XLA's CPU `shard_map` backend SEGFAULTS at the
  CORE2 mesh size ⇒ verify sharded autodiff on GPU, not CPU.) **(3) The validated capability + the falsified
  hypothesis.** Sharded multi-step NN-grad == dense (`|g|`=9.62 vs 9.63 at N=4) and the sharded twin reproduces
  the dense recovery (evolution misfit ratio ~0.14, corr_all ~0.80, corr_active ~0.25 at N=4) for the full
  TKE+NN+GM+ice(frozen)+zstar config at P=4 N=48, 24.7 GB/dev — so the all_gather reduce-scatter transpose is
  autodiff-correct end-to-end. BUT the it=0 gradient norm explodes with window: `|g|` = 9.6(N4) → 12(N6) →
  **2.5e4(N8)** → 4.6e15(N12) → 2.4e124(N48), a sharp onset between N=6 and N=8; recovery (evolution gate)
  holds only for **N≤6**, N≥8 DIVERGES (loss 2→8). This is the recurrent state-transition Jacobian compounding
  (stiff/chaotic adjoint): finite post-NaN-fix, but the gradient DIRECTION becomes the fastest-growing unstable
  mode, so `optax.clip_by_global_norm` bounds the magnitude (no float64 g²-overflow) yet does NOT fix the
  divergence (raw AND clipped Adam both diverge at N≥8). AND field corr is BEST at the SHORTEST window (N=4:
  corr_all 0.80–0.83) and DEGRADES with N (N=6 0.51, N=8 negative) — so the multi-GPU run FALSIFIES the
  [[d2-calibration-complete]] "only a continuous long window imprints the field" expectation: longer continuous
  windows do not help and (N≥8) break the adjoint. Equifinality is fundamental for this stiff closure; the
  usable-adjoint window is shorter than any window that would push corr past ~0.83. **Multi-node N≫48 is moot
  for field recovery** (it explodes harder, not better) — the binding constraint is the chaotic-adjoint horizon,
  not memory; gradient-based stiff-closure recovery needs shadowing/regularization, not bigger windows.
  **FIELD recovery at the usable short window (the locked §3 twin result):** an amp-sweep at N=4 found a
  Goldilocks truth perturbation `--truth-amp 2.0` (vs 4.0): `corr_active` 0.32→0.56, `corr_all` 0.83→**0.88**,
  evolution misfit 0.052. Weaker (amp 1.5) gives too little signal (evolution fails); DRIVING THE LOSS LOWER
  BACKFIRES (the stiff optimization destabilizes past loss~0.15) and `corr_active` is INSENSITIVE to further
  loss reduction ⇒ the strong-anomaly quartile is equifinality/saturation-limited (those columns re-equilibrate
  within the short window ⇒ least identifiable). So the twin gate was REFRAMED: PRIMARY = `corr_all` (bulk
  multiplier field, tol 0.8) + `corr_pw` (perturbation-weighted) reported; `corr_active` kept as a DIAGNOSTIC,
  not gated. Canonical §3 sharded twin = amp=2.0 N=4 ⇒ evolution recovered + bulk field recovered (corr_all
  ~0.88) = `NN_TWIN_SHARDED_OK`. (`fesom_jax/cvmix_tke.py`, `fesom_jax/integrate_sharded.py`,
  `scripts/paper/core2_paper_nn_twin_sharded.py`, `scripts/debug/verify_sharded_tke_grad.py`,
  `scripts/debug/repro_sharded_grad_nan.py`.)
- **[§3 hybrid-ML / obs training / offline-online gap] An NN mixing closure trains end-to-end through the
  global adjoint to reduce REAL held-out obs misfit — but the short-window optimum does NOT deploy, and the
  fix for the *instability* does not fix the *misalignment*.** E2 (`scripts/paper/core2_paper_nn_obs.py`, all3
  frozen-ice adjoint) trains `tke_nn` over the 12 monthly snapshots (batched SEASONAL short windows, N=12,
  gradient accumulated) to reduce MLD/SST-vs-WOA through the D2a obs operator. It WORKS as an obs reducer:
  held-out MLD **−2.1%** = train −2.1% (the D2c clean no-overfit bar), bounded multiplier ⇒ PD diffusivities,
  a spatially-structured correction a global scalar can't make. BUT a long forward-only deployment (90 d,
  default NN→0 vs trained) exposes the **offline-trained/online-deployed gap** in two layers: **(1) naive
  training is UNSTABLE** — minimizing the 5-h MLD misfit drives a **bang-bang over-mixing** multiplier (mean
  2.8×, saturating BOTH [1/3,3] caps) that lowers the short misfit but **blows up the 90-d MLD ×2–4**; the
  data loss itself OVERSHOOTS (non-monotonic — best at it≈15, worse by it≈43). **(2) even when stabilized,
  the benefit does NOT persist** — three fixes (KEEP-BEST iterate not last; a **trust-region reg** =
  area-weighted penalty on `(log m)²` toward default mixing, which is 0 at NN→0 so the bit-identical
  invariant holds; a tighter `m_max`=2) make the NN deploy STABLY (drift ≈ default, T/S 1.01×, finite +
  physical throughout) but its long-forward MLD is still WORSE than default at EVERY horizon (+11%
  seasonal-mean). Root cause: the short-window adjoint optimizes the **FAST** (6-h) MLD response, which is
  **misaligned** with the **SLOW** deployed equilibrium — the recovered net multiplier even flips *below* 1
  (less mixing helps 6 h) opposite D2a's held-out-validated "more mixing deepens MLD" equilibrium direction.
  A **uniform-multiplier deployment diagnostic** (`--const-mult`) PROVES the misalignment is a wrong-sign, not
  an unimprovable metric: a global **2× IMPROVES** the 90-d MLD (+0.5%, stable) and a **0.7× WORSENS** it
  (−0.4%) — monotone in D2a's direction — yet the short-window-trained NN sits at net 0.97× (wrong side).
  So the **deployed MLD is a SLOW target** (a longer differentiable rollout or EKI, or a physically-informed
  prior toward the D2a optimum), the SAME adjoint↔EKI
  boundary as GM→T/S ([[d2-calibration-complete]] D2b) — now for the NN closure. **Methodological takeaway:
  the long-forward drift+persisted-benefit gate is ESSENTIAL** — a held-out *short-window* obs reduction is
  necessary but NOT sufficient for a deployable closure; reporting it alone (as offline ML-closure papers
  do) hides the online failure. The structure-preserving bounded multiplier + trust-region reg are what buy
  *stability*; *persisted benefit* needs the slow-target tool. (`scripts/core2_paper_nn_obs.{py,sbatch}`,
  `scripts/paper/core2_paper_nn_obs_diag.sbatch`, `scripts/paper/fig_hybridml.py`.)

## Long-window experiments (PART A) — `docs/plans/20260618-fesom-jax-longwindow-experiments.md`

- **Task A1 — Foundation multi-year DRIFT VERDICT (`FOUNDATION_MULTIYEAR_OK`).** The 3-yr all-on foundation
  climate (zstar+TKE+mEVP+GM, job 25740424) is a USABLE base — and the verdict is decided by NUMERIC bars on
  the drift **slope**, never a field's level. Upper-ocean (0–700 m) volume-weighted basin-mean T drift
  **halves year-over-year** (−0.049 → −0.024 °C yr⁻¹; tail ≪ the 0.1 bar), S is flat (−0.0006 PSU yr⁻¹), and
  Arctic m_ice annual-mean drift decelerates (+0.32 → +0.15 m yr⁻¹, multi-year ice building toward
  equilibrium). Physical across ALL 36 monthly snapshots: finite, min wet-layer hnode 4.55 m, max|vel| 2.43,
  |SSH| 1.98. **Two gotchas that would false-fail a naive check:** (1) the **hnode>0 test MUST mask to wet
  layers** (`node_layer_mask`) — the raw `min(hnode)` is 0 on dry/below-bottom lanes; (2) the local-SST
  physicality floor must be the **salinity-dependent freezing point** (`T_f ≈ −0.054·S` ⇒ ~−2.1 °C at
  S≈39), NOT −1.9 — an instantaneous ice-edge SST of −2.11 °C is physical, and a −1.9 floor would wrongly
  reject it. The drift-metric reducer (area/volume-weighted, NaN-as-mask, guarded-denominator → finite) is
  unit-tested (`LW_DRIFT_SEAM_OK`). Drift source = the monthly-mean netCDF (snapshot sampling noise would
  swamp the signal); snapshots give only the hnode/|vel|/finite physicality series. (`scripts/
  core2_lw_foundation_check.{py,sbatch}`, `scripts/tests/test_lw_foundation_check.py`.)

- **Task A3 — restartable + parameter-injectable forward seam (`LW_FORWARD_SEAM_OK`).** `core2_kpp_climate_run.py`
  gains `--load-state` (restart from a saved State pickle) and `--ck` / `--nn-pkl` (inject `Params` via
  `calibrate.build_params({'tke_c_k': …})` / `{'tke_nn': …}`) — the one fix B1 (restart from the spin-up),
  D2 (restart + c_k±δ FD) and F (restart + the trained NN) all needed. **The LOAD-BEARING restart detail:**
  a State saved after step k already carries the AB2 history (`T_old`/`S_old`/`uv_rhsAB`/…), so the
  continuation must run **`is_first_step=False` throughout** (`is_first_step=(i==0) and not loaded`) — re-
  cold-starting the AB2 first step would corrupt it. With that, a restart is **BYTE-IDENTICAL** to never
  having stopped (pi-mesh save→pickle-roundtrip→continue test, k=1 and k=3). **Second subtlety:** even on
  restart `sst0` is STILL computed from the IC (`core2_initial_state`), because it builds the forcing's
  static `a_ice` mask — deriving it from the restart state would change the mask and break continuity with
  the trajectory being continued. No-flags ⇒ `params=None` passed (not `Params.defaults()`) ⇒ the byte-
  identical default path is untouched. Plus a `/work`-discipline guard: `--save-state`/`--snapshot-dir` must
  resolve under `/work` or the run aborts. (`scripts/tests/test_lw_forward_seam.py`.)

- **Task B0 — the ensemble-averaged adjoint seam (`LONGWINDOW_SEAM_OK`).** `fesom_jax/longwindow.py` is the
  GENERIC mechanism (no physics) for the Lea/Allen/Haine 2000 ensemble adjoint: `spread_indices`/
  `seed_starts` (host-side seed selection along a reference trajectory — even-spaced or rng-jittered,
  works for in-memory stacks AND on-disk snapshot files), `average_grads` (Welford **streaming** mean +
  across-burst **standard error**, leafwise over a pytree, masking non-finite burst lanes per the project
  masked-NaN contract — a diverged burst's NaN doesn't poison the mean), `convergence` (the running-mean-
  vs-#bursts diagnostic + the burst count at which it settles). **Validated on Lorenz-63** (the canonical
  chaotic benchmark): FD truth d⟨z⟩/dρ = **1.009**; a SINGLE long-window adjoint **DIVERGES** (+1.1 at
  T=0.5 → −368 at T=10 → +9.5e5 at T=20 — exponential in the window); the ENSEMBLE AVERAGE of K=4000 short
  bursts (T=0.5, N=50 dt=0.01) recovers it within **16.2%** (T=0.75 → 12.8%), the running mean stabilizing
  well before K. **The window-length sweet spot is the empirical preview of the C1 crux:** too short
  (T=0.3 → 44% err, the climate response undeveloped) OR too long (T≥1.5 → >27% err with huge across-burst
  stderr, bursts starting to blow up) both degrade — only a MODERATE window both develops the response AND
  stays below the chaotic blow-up. Efficiency gotcha: collect the reference trajectory with ONE `lax.scan`
  emitting the full state per step + host-subsample, NOT a Python loop of many short scans (10000 scan
  dispatches → minutes; one scan → ~1 s). Total test 2.2 s on CPU. (`fesom_jax/tests/test_longwindow.py`.)

- **Task C1 — frozen-ice clean-gradient horizon + the sign-flip (`SIGNFLIP_HORIZON_OK` → Part D).** THE make-
  or-break long-window measurement. Frozen-ice adjoint (`IceConfig(adjoint_mode="frozen")`), full all-on
  config, **window-mean MLD** observable = the per-step area-weighted mean MLD accumulated into a **(State,
  MLD-acc) tuple carry** run through `integrate._run_steps` (the tested √N two-level checkpointing — reused,
  NOT re-implemented; scripts-only, zero library surface). Seeding at a DEVELOPED snapshot ⇒ `is_first_step=
  False` throughout ⇒ a uniform scan, **no eager first step** (the simplification that makes the accumulator
  clean). Swept N at two foundation yr-3 seasonal states (NH deep-winter `037440` + summer `044640`).
  **RESULT (both seeds, near-identical):** `d(window-mean MLD)/d(c_k)` is **+slow/right** (D2a's "more mixing
  DEEPENS MLD") from the SHORTEST window (N=12 = 6 h: winter +0.77, summer +0.90) — **there is NO fast-wrong
  regime to flip from** — and stays clean (|grad| O(1), best-h FD sign-agree, 37–43 GB) through **N=48 (1 d)**,
  then **BLOWS UP EXPONENTIALLY**: N=96 (2 d) |grad| 2e4 / 1e6, N=192 (4 d) **5e15 / 2e10**. So **N\* ≤ 12 ≪
  N_blow = 96** ⇒ a clean, well-conditioned, slow-signed window (≤ 1 d) exists below the blow-up ⇒ **simple
  ensemble-averaging (Part D) is VIABLE; the adjoint-escalation Part E (shadowing/NILSAS) is NOT needed.**
  D1 burst window = **N=24 (0.5 d)** (cleanest; summer N=48 plateau 0.61 = blow-up onset). **Two methodology
  notes:** (1) the MLD density-threshold crossing is piecewise-constant in the level it picks ⇒ the SECANT
  FD is **kink-noisy across h** (winter N=12 h=1e-4 gave FD −7.56 vs AD +0.77 — a level-straddle artifact, NOT
  an AD error) ⇒ gate blow-up on **|grad| magnitude + best-h FD SIGN agreement, NOT plateau<1%**; (2) the
  frozen-ice clean horizon (N≤48, 1 d) is much LONGER than the stiff EXACT-ice NN-twin's (N≤6,
  [[multigpu-sharded-adjoint-horizon]]) — **freezing the mEVP rheology adjoint is what buys the usable window**.
  The §3 "wrong sign" ([[e2-nn-obs-offline-online]]) was the deployed-NN-over-90-d problem, NOT this window-
  mean adjoint sensitivity (correctly signed even at 6 h). (`scripts/core2_lw_signflip.{py,sbatch}`,
  `scripts/paper/fig_signflip.py`.)

- **Task A2 — 5-yr spin-up: a "blow-up" that was a PHYSICAL current vs a too-tight cap (`SPINUP_5YR_OK`).**
  The 5-yr all-on spin-up tripped the run's `max|vel|≥3.0` stability abort at year ~4.6. **Lesson: localize
  before assuming a model instability.** Loading the last pre-trip snapshot and finding the max-velocity
  elements showed it was **LOCALIZED** (~13 elements, 0.017% of the domain) at **lon~47°E lat~4°N — the
  Somali Current** — at the **surface**, **ice-free**, in **July**: exactly where/when the real Somali
  Current peaks (~3–3.5 m/s under the SW monsoon, one of Earth's fastest currents). So the 3.0 cap was an
  arbitrary safety threshold too tight for a *physical* boundary current (and a 10-yr reference hits it every
  summer). **Fix:** raised `VEL_ABSMAX` 3.0→5.0; a restart-probe from the pre-trip snapshot showed the jet
  **peaks ~3 and SUBSIDES** (worst |vel|=2.96 over the next 175 d, did NOT climb to 5.0) ⇒ physical, base
  recoverable. The completed base equilibrated: upper-ocean 0-700 m T drift yr3→4 0.018 °C yr⁻¹ (falling).
  Also added `--start-month/--start-day` to the driver for mid-year `--load-state` restarts (the probe needed
  the July forcing; ±1-day forcing offset is negligible for a developed base). (`scripts/paper/core2_lw_spinup_finish.sbatch`.)

- **Task D1 — probing-design findings (multi-target + window choice + robust ensemble).** The 10-yr reference
  is target-agnostic: `core2_lw_avgadj.py --target {mld_ck, t100_kgm}` gives d(any climate diagnostic)/d(any
  param) from ONE reference (2 swaps: observable + parameter; all-on config drives both). **Window:** N=48
  (1 d) beats N=24 (0.5 d) — it captures MORE of the slow response (MLD +2.5 vs +1.5 m; GM d(100m-T)/d(k_gm)
  **+7.9e-7 vs +2.4e-7 — ~3×**, the GM signal is window-dependent ⇒ still developing ⇒ the slow eddy
  response, the adjoint↔EKI boundary visible as window-growth ⇒ FD likely larger). **But N=48 is the C1
  clean-horizon EDGE:** individual unstable (summer) seeds amplify before the N_blow=96 average, contaminating
  the plain MEAN (smoke: mld_ck mean +52 vs median +3, one burst +303). **Fix = a MAD robust filter** (drop
  bursts with |g−median| > 5×MAD — the seeds past THEIR clean horizon; a no-op at N=24). ⚠️ a `median(|g|)×k`
  threshold is FRAGILE at small K (multiple outliers inflate the median → one slips through, even flipping the
  kept-mean sign) — use **MAD from the median**, the standard robust scale. (`scripts/core2_lw_avgadj.{py,sbatch}`.)

- **Task B1 — the 10-yr reference + REF_10YR_OK verdict.** The 10-yr all-on forward (restart from
  `spinup5_state.pkl`, job 25748365, 8.6 h) ran clean: **worst |vel|=2.735 over the entire decade** — the
  Task-A2 Somali-Current cap (5.0) held with margin at the full 10-yr scale, the strongest possible validation
  of that fix (the per-step monitor is a far tighter physicality check than any snapshot spot-check). The
  verdict script (`core2_lw_ref_check.py`) **reuses the A1 reducers verbatim** (`load_weights`, `read_climate`,
  `decel`, `lin_slope_per_year`) — the same falsifiable drift instrument over 10 yr, only the VEL cap bumped
  3.0→5.0 — and adds the climate observable itself: the 10-yr-mean MLD computed **on the actual snapshot
  States via `ale.live_geometry` + `obs_compare.mld_density_threshold`** (the EXACT D1 code path, NOT a static-Z
  monthly-mean proxy). Two findings that gate D1: (i) the upper ocean has **essentially equilibrated** by yr 6
  — T0700 half-drift −0.018→−0.0007 °C/yr, decade lin-slope −0.0071 (a 5-yr spin-up + 10-yr ref reaches a near-
  stationary base, vs the 3-yr foundation's −0.024); (ii) the **MLD is stationary** across the decade (35.1–37.8
  m, mean 36.6 m), so "d(10-yr-mean MLD)/d(c_k)" is well-posed — a drifting observable would make the long-mean
  target ambiguous. **Sizing gotcha:** the frozen-ice adjoint burst is ~60 s steady-state at N=48 (1 d) after a
  ~200 s first-burst JIT compile (~32 s at N=24) ⇒ K=200 ≈ 3.4 h **per target**. The two-target `for`-loop in
  one 3.5-h sbatch would time-cut the second target — **split into independent single-target jobs** (matches the
  "parallelism = many INDEPENDENT burst jobs" design) with a 5-h wall each. (`scripts/paper/core2_lw_ref_check.py`.)

- **Task D1 — the adjoint↔TLM `--mode` switch (forward-mode mirror) + a free transpose validation.** The
  ensemble-averaged-sensitivity harness is **direction-agnostic**: both modes seed the SAME reference snapshots,
  same forcing-at-seed-date, same N-window, same MAD filter + streaming average — only the AD transform + the
  parameter/observable SHAPE flip. `--mode adjoint` (reverse, `jax.grad`): param = [nod2D] FIELD, observable =
  SCALAR ⇒ map `d(scalar)/d(param_i)` ("where to tune"), scalar = Σ map. `--mode tlm` (forward, `jax.jvp`): param
  = SCALAR (one global knob, broadcast to the field inside `params_from` so the unit tangent = a uniform bump),
  observable = [nod2D] FIELD ⇒ map `d(field_j)/d(param)` ("spatial fingerprint"), scalar = area/vol-weighted mean
  of the map. **The two scalars are the SAME directional derivative `dO/d(uniform θ)` — a built-in transpose
  check:** on 4 shared seeds the adjoint Σ-map (+1.52493) and the TLM weighted-mean (+1.51373) agreed to **0.73%**
  (per-burst ~1%). **Implementation:** ONE `window_mean(theta, …)` serves both (mode-set `acc0`: scalar vs field;
  the observable's reduction); only the outer transform differs (`jax.grad` vs `jax.jvp(…, (jnp.ones_like(θ),))`).
  A unified per-node weight `u` (1 for adjoint → Σ; normalized `w_scalar` for tlm → weighted mean) reduces a map
  to its scalar in both. **Forward-mode is leaner+faster:** no tape ⇒ `checkpoint=False` (plain scan) ⇒ peak **22
  GB vs 39 GB** and ~14 s/burst vs ~31 s (N=24) — only a longer first compile. **Map magnitudes are transposes:**
  TLM `|map|max`~5000 (a single convection node's MLD is very sensitive to `c_k`) vs adjoint ~0.03 (each node is a
  sliver of the GLOBAL-mean sensitivity) — large local TLM values with a moderate +1.5 global mean is correct,
  not a blow-up. The chaotic `N_blow` horizon + the need for ensemble-averaging apply to BOTH (TLM and adjoint
  share the linearization's singular values) — forward vs reverse changes only WHICH dimension is cheap.
  (`scripts/paper/core2_lw_avgadj.py --mode`, validation smoke job 25762879.)

- **Task D1 — the production climate-sensitivity RESULTS + the K=200 transpose nuance (per-burst exact,
  ensemble-mean noisy).** The Part-D payoff on the 10-yr reference (K=200, N=48, ~18-d cadence, MAD-filtered;
  jobs 25758787/8 adjoint, 25763255 tlm): **`d(10-yr-mean MLD)/d(c_k) = +1.458 ± 0.498 m`** (+slow/right —
  more mixing DEEPENS the climate-mean MLD, the D2a sign at the CLIMATE horizon; converged n_stable 134/144,
  full-domain finite map) and **`d(mean 0-100m T)/d(k_gm) = +4.12e-7 ± 1.50e-7 °C`** (positive, ~36% SE —
  small at a 1-d window, the adjoint↔EKI boundary). Both `AVG_ADJ_SENS_*_OK`. **The KEY ensemble lesson:**
  the adjoint↔TLM transpose is exact PER BURST (the K=4 matched-seed smoke agreed to 0.73%), but the K=200
  ENSEMBLE means diverge — adjoint filtered-mean **+1.458** vs TLM **+1.749** (**20%**) — while the robust
  MEDIANS agree (+1.70 vs +1.59, **6.7%**). **Why:** the TLM *response* field `d(MLD_j)/d(c_k)` is heavy-
  tailed (sharp deep-convection nodes, `|map|max ~1.7e6`), so its area-weighted MEAN is tail-dominated and
  noisy (SE **±4.92** vs the adjoint's smooth global-mean-sensitivity SE **±0.50**); the two means are
  statistically CONSISTENT (the 0.29 gap ≪ the TLM SE) but the **median is the right ensemble comparator**,
  not the mean. Takeaway: when comparing a reverse-mode (smooth, well-determined) reduction against a forward-
  mode (sharp, tail-dominated) one, validate the transpose on MATCHED SEEDS per-burst and compare ensemble
  MEDIANS — don't expect the heavy-tailed mean to land on the smooth one. The TLM filter still dropped 56/200
  (same June/Dec amplifiers) and kept-mean≈median, so it IS well-behaved — just intrinsically noisier per the
  physics, not a bug. **Plotting (`scripts/paper/fig_avgadj.py`):** one script renders any `lw_avgadj_*_map.npz`
  (map panel + across-burst convergence panel); the heavy-tailed TLM map needs a **signed-log (`SymLogNorm`)**
  colour (`--symlog auto` ⇒ on for tlm, off for adjoint) or the few extreme nodes wash out the broad
  fingerprint — adjoint maps stay linear (99th-pct symmetric clip). Figures: `fig_avgadj_climate.png` (both
  adjoint targets), `fig_avgadj_tlm.png` (the TLM fingerprint), `fig_avgadj_transpose.png` (adjoint-smooth vs
  TLM-sharp side-by-side — the transpose made visual). (`scripts/{core2_lw_avgadj,fig_avgadj}.py`.)

## GMD model-description paper (usability: portable restart, streaming output, YAML config)

- **Task A1 — device-count-portable sharded restart (`RESTART_PORTABLE_OK`).** `write_restart`/`read_restart`
  in `zarr_output.py` round-trip the **FULL** prognostic `State` across **any** device count (save P=4 → load
  P=2 AND P=2 → P=4, both bit-identity). **Two load-bearing details:** (1) enumerate leaves via
  **`dataclasses.fields(State)`** (pytree-flatten-equivalent), NOT a `*_old` name-glob — a glob silently drops
  the momentum AB2 slot `uv_rhsAB`, `sigma11/12/22`, `tke`; the test asserts every leaf has its own on-disk
  dataset (no-silent-drop) with a per-leaf distinct signature so a dropped/zeroed/mis-scattered entity is
  caught. (2) the **read is field-by-field streaming** — `reconstruct_global` ONE leaf to a dense global
  (partition-INDEPENDENT, the gid/owned maps already dedupe redundantly-owned elem/edge), `device_put`-shard to
  the new partition, free, next leaf ⇒ host peak ≈ one global field (~4 GB at NG5), never the ~40-leaf State
  co-resident (~80 GB). Portability is free because the on-disk format is gid-keyed/global-canonical: re-gathering
  from the COMPLETE global field fills the NEW partition's halo lanes correctly (a halo lane is just another gid
  in `myList`). Tested on CPU fake-devices with a new **halo-free block partition** (`partit.synth_block_partition`)
  — restart never exchanges halos, so it needs no real `dist_<NP>` files (and no `ragged_all_to_all`, which is
  GPU-only).

- **Task A2 — streaming time-mean/variance output (`STREAM_OUTPUT_OK`).** `OnlineStats` (Welford single-pass) in
  `zarr_output.py` accumulates per-grid-point mean+variance over a `{name: array}` dict **without storing every
  step** ⇒ a multi-year run yields the mean state AND the EKE map (`eke_from_stats` = ½⟨u'²+v'²⟩ from the velocity
  variance) at the cost of 2 extra copies per tracked field. It is a **registered pytree** (leaves = `count` scalar
  + the mean/M2 dicts) so it flows through `lax.scan` (accumulate INSIDE the sharded step loop) or a Python chunk
  loop — every op is element-wise ⇒ sharding-preserving. **Masked/dry-lane finiteness for free:** the Welford
  denominator `count` is ≥1 after the first sample, so a constant-0 dry lane has mean 0 / var 0, no `0/0` (the AD
  masked-lane rule, here for the forward accumulator). Matched offline `numpy.var`/`mean` to ≤1e-12. Snapshots are
  a thin cadence wrapper (`snapshot_due`/`write_snapshot`) over the existing gather-free `write_state_zarr`.

- **Task A3 — unified `RunConfig` + single-YAML (`RUN_CONFIG_OK`).** `run_config.py` composes the existing
  per-physics sub-configs (Ale/GM/Kpp/Tke/Ice) + the **promoted** run-dependent scalars (the viscosity γ's,
  dt + a cold→prod ramp, the tracer scheme selector) + the run-orchestration spec, all from ONE YAML. **The
  bit-identity mechanism:** every promoted constant becomes a `NamedTuple` (`ViscConfig`/`TracerConfig`) whose
  DEFAULTS equal the `config.py` module constants, threaded as a `visc_cfg=None`→`_DEFAULT_VISC` kwarg through
  `momentum.visc_filt_bidiff` → `step` → `step_jit`(static_argnames) → `integrate`/`run`/`run_step(s)_sharded`
  — so `None`/defaults trace to the SAME float literals ⇒ byte-identical (43 momentum + 90 step/ale gates
  unmoved; `RunConfig.defaults()` step == bare `step()` over 5 steps). **Two gotchas worth recording:** (1) the
  NG5 `visc_gamma1=0.2` promotion is INVISIBLE on the pi mesh — the flow-aware term `max(gamma0, gamma1·|Δu|)`
  is dominated by the `gamma0=0.003` floor at pi's tiny velocities, so γ1 only bites for `|Δu| ∈ ~(0.03, 0.35)`;
  test the threading by driving `visc_filt_bidiff` directly with a velocity field in that band, NOT a pi forward.
  (2) tracer **num_ord is hard-coded into the MFCT/QR4C reconstruction kernels**, so `TracerConfig` is a
  *validator*, not a switch: it raises `NotImplementedError` for any scheme ≠ the implemented `(0,1)` (NG5
  matches ⇒ zero tracer work; CORE2's scheme is OPEN — a mismatch is a kernel change, not a toggle). YAML schema
  is sparse (`null`=physics-off, `{}`=on-with-defaults, only NON-default overrides emitted); unknown top-level
  or sub-config keys raise (no silent typos). `configs/ng5.yaml` pinned; `configs/core2_full.yaml` = placeholder
  + TODO. pyyaml added to the env (A7 pins it in `pyproject.toml`).

- **Task A4 — per-step time-varying forcing on the SHARDED multi-step path (`SHARDED_FORCING_OK`).** The
  forcing itself was always built (`surface_forcing.py`, JRA55+L&Y bulk — it drove the CORE2 5+10 yr); the gap
  was that `run_steps_sharded` holds forcing CONSTANT across its `lax.scan` (the multi-node path only ever ran
  TIMING benchmarks, where the per-step cost is forcing-value-independent). NG5 is the first run that is BOTH
  multi-node AND needs a seasonal cycle ⇒ `run_steps_sharded_forced` folds the partitioned forcing
  (`partition_step_forcing` → `[P, n_steps, Lmax]`) to a `[n_steps, P*Lmax]` scan `xs` with
  `PartitionSpec(None,'p')` (`_fold_forcing_seq`: transpose P↔T then fold) so each `shard_map` device scans its
  `[n_steps, Lmax]` slice and step `i` gets its own forcing. Kept as a SEPARATE function ⇒ the constant
  timing/scaling path is byte-unchanged. **`bootstrap_ab2` flag** = the restart-continuation seam: a cold/
  post-dt-ramp chunk does step-1 `is_first_step=True`, a mid-run chunk carries AB2 forward (False) ⇒ chained ==
  continuous. **Two test gotchas:** (1) JRA55 is piecewise-constant over its 3-hourly window, so 3 consecutive
  30-min steps share one record (Tair identical) — a per-step gate over consecutive steps can't tell per-step
  from constant; use WELL-SEPARATED dates (Jan/Jul/Oct) injected as the stack. (2) over a few forced steps the
  sharded-vs-dense match is the FCT climate-close floor (~1.6e-4), NOT byte-id — the FCT limiter's upwind flips
  amplify the reassociation floor; gate on that PLUS a non-triviality check (the per-step result must differ
  from a constant-forcing run by ≫ the match, here 5e5 vs 1.6e-4) so the comparison can't pass trivially. The
  exact wiring is separately pinned by a pure-host fold-unit test (`seq[i]` == the independent fold of step i).

- **Task A6 — single-invocation config-driven run driver (`RUN_DRIVER_OK`).** `run.py` = load restart (or cold
  IC) → chunked forced steps → write portable restart → exit; no in-model orchestration (multi-segment campaigns
  = a SLURM `--dependency=afterok` chain, `scripts/chain_submit.sh`). **The headline gate is the RESTART SEAM:**
  a continuous N-step run == the same run done as two chained invocations (N/2 → `write_restart` → `read_restart`
  → N/2) to **5.7e-8** — essentially exact (the A1 restart is bit-faithful + AB2 continues). **The load-bearing
  detail** (the `core2_kpp_climate_run` precedent, now in the sharded driver): a restart-continuation chunk must
  NOT cold-start AB2 — `plan_chunks` sets `bootstrap_ab2=True` ONLY for step 0 of a cold run or the chunk that
  begins exactly at the dt-ramp boundary (a dt CHANGE invalidates the AB2 history formed at the old dt); every
  other chunk carries `uv_rhsAB`/`T_old` forward. **dt-ramp gotcha:** the SSH operator is dt-DEPENDENT (the
  implicit-solve matrix), so the driver rebuilds/caches a `sop` per dt — a stale operator at the new dt would be
  a silent error. **Forcing memory:** the per-step forcing is fed in fine (≈ few-day, `--chunk-steps`) chunks so
  a multi-year run never pre-stacks (~280 GB/day at NG5); `plan_chunks` also never straddles the ramp boundary.
  The chunk-to-chunk State handoff currently round-trips host (fine at CORE2/farc/dars; the NG5 on-device-chaining
  via `return_executable` is the noted B2 optimization — de-risk on the smaller meshes first per the user steer).

- **Task A5 — verify the RUNTIME forcing pipeline (CORE2 done; `NG5_FORCING_OK`).** Key correction (user): the
  port interpolates forcing **at runtime, exactly like FESOM** — `JRA55Reader.__init__(mesh, year)` builds the
  bilinear weights ONCE at setup (`sbc_ini`), then `.step()` reads disk + bilinear-interp + time-interp every
  step; SSS/runoff/chl readers too. **Nothing is pre-interpolated/pre-staged.** So A5 is NOT data prep — it is
  verifying the runtime setup initializes for a mesh (the only mesh-dependent step) + that the SOURCE files
  resolve (all do: JRA55 + PHC2_salx + CORE2_runoff + Sweeney chl on `/pool`). `scripts/debug/check_forcing.py MESH_DIR`
  builds the readers + does a 1-step finite-flux smoke; CORE2 (126 858 nodes) → all 10 fields finite, physical
  ranges. **The one per-mesh step is the JAX mesh EXPORT** (`load_mesh` needs the exported layout, not the raw
  FESOM `nod2d.out`); farc/dars/NG5 raw meshes are on `/pool/.../MESHES_FESOM2.1` but need the export (B0) before
  the same one-line check runs. `docs/FORCING_MESHES.md` records sources + per-mesh status.

- **Task A7 — release engineering (`RELEASE_OK`).** A **data-free** CI gate (`.github/workflows/ci.yml` +
  `test_release.py`): the repo's `data/` is a gitignored symlink to `/work`, so GitHub CI has no mesh — it runs
  only the gates that need none (the RunConfig bit-identity default invariant + YAML round-trip, OnlineStats
  correctness, the run-driver chunk/duration logic, the sharded-forcing FOLD wiring, the shipped configs
  parse+validate). The heavy bit-identity STEP gate + the forced/gradient suites stay on Levante
  (`run_suite.sbatch`) and SKIP cleanly in CI (their skipif sees the data absent). `pyproject.toml` pins pyyaml;
  `docs/USER_GUIDE.md` (YAML schema + restart/output + SLURM chaining + mesh/dt table) + a README quick-start
  pointer + a data-free `examples/tutorial.ipynb` (config → chunk-plan → OnlineStats, runs headless without data)
  round out the release surface.

- **Task A8 — standalone Python mesh preparation (`MESH_PREP_OK`).** For a SHIP-able model the C-export
  dependency (`load_mesh` reads 32 `.npy` arrays the C-FESOM `jax-mesh-export` branch dumps) is an adoption
  wall — users would need a C build to prepare a mesh. `scripts/prepare_mesh.py` is a pure-numpy port of
  `fesom_mesh.c`'s `compute_metrics`: raw FESOM ASCII (`nod2d/elem2d/aux3d.out` + `nlvls/elvls/edges/edge_tri.out`)
  to the EXACT C-export `.npy` layout, run ONCE offline (NOT online per-run — too heavy). **Verified byte-faithful
  against the C oracle (`data/mesh_core2`): all 32 arrays, ints EXACT, floats rel <=1.05e-13** (float64 order-of-
  ops). **The byte-identity gotchas that mattered** (all caught by diffing against the C export): (1) **TWO
  pi's** — the truncated `3.14159265358979` for RAD/OMEGA/CYCLIC, but `mesh_resolution` hardcodes FULL-precision
  pi; mixing them perturbs the last bits. (2) **off-by-one level in `area`** — the C accumulates `nz in [0,
  nlevels[e]-1)`, NOT `[0, nlevels[e])`; slicing one level too many blew `area`/`areasvol` up by ~1e10 (a full
  `elem_area`) while everything else was already ~1e-15 — the oracle caught it instantly. (3) **CSR ascending-
  element order** — `area`/`mesh_resolution` sum over a node's cells in ascending global element id; build the
  CSR by element-ascending iteration (or `np.add.at` on element-ordered pairs) so the float accumulation matches.
  (4) **two cyclic conventions** (`elem_center` min-anchored `>=/<` vs the simple `>/<` trim elsewhere) — at
  their own sites. (5) `nlevels`/edges are READ from `nlvls/elvls/edges/edge_tri.out` (already built by the FESOM
  partitioner), NOT reconstructed. **The lesson: a precise C-derivation spec + the existing C export as a
  per-array oracle makes even a 1350-line mesh-setup port converge fast** — 30/32 arrays matched on the first
  pass; only the area off-by-one + the verify-gate tolerance (relative, since arrays span 1e10->1e-20) needed a
  fix. The MFCT `edge_up_dn_tri` scan matched EXACTLY first try.

- **Task B0 — multi-node forced run + portable restart de-risk (farc green, dars `MACHINERY_SCALE_DARS_OK`).**
  The config-driven `run.py` driver works **single-node** at farc (638 k / 4 GPU) first try, but the dars
  (3.16 M, multi-node `jax.distributed`) B0 surfaced FIVE distinct scale/multi-process issues — each invisible
  single-node, each a genuine blocker for ANY multi-node run (incl. NG5), and the ladder caught them all cheaply
  on dars instead of the NG5 flagship. In peel order: **(1) GPU-0 setup OOM** — `core2_initial_state` built the
  global State on GPU 0 (~50-80 GB at 3.16 M); fix = host-build (`xp=np`, the Phase-8b B.3 pattern; value-
  identical). **(2) `np.asarray` on a cross-process global** — `write_restart`/resume pulled a multi-process
  sharded array to host (`spans non-addressable devices`); fix = fold on-device + let `_to_global_sharded` pass
  a device `jax.Array` through (reshard, not host-pull). **(3) multi-process reshape materialization** — the
  `[P,Lmax]↔[P*Lmax]` unfold/fold of a sharded array gathers the WHOLE global on one device (34.8 GiB OOM,
  `jit_reshape`); fix = keep the State **FOLDED `[P*Lmax]` end-to-end** (`run_steps_sharded_forced` gains
  `state_is_folded`/`return_folded`; the driver never unfolds). **(4) all_gather global materialization in the
  step** — `use_ragged=False` gathers global fields per device (34.8→still OOM); fix = the **ragged-halo path**
  (`use_ragged=True`, GPU-only, forward-only-safe — the scaling-benchmark path; cut it to 22 GiB). **(5)
  provisioning + stale store** — dars all-on physics needs **>8 GPUs** (dist_8 OOMed at 22 GiB; dist_16 cleared
  it — the per-device working set, not a global), and a prior dist_N's restart store shape-conflicts
  (`8*Lmax_8 != 16*Lmax_16`) ⇒ clear it first. **Through all five the restart seam stayed BIT-EXACT (5.68e-8)** —
  every failure was infrastructure (memory / multi-process I/O), never the physics. **The lesson: single-node
  GPU validation is necessary but NOT sufficient — multi-node has its own class of bugs (cross-process host
  pulls, sharded-reshape materialization, per-device provisioning) that only a real multi-node run surfaces;
  de-risk them on the cheap mesh (dars) before the flagship (NG5).** Still TODO: the device-count CHANGE
  (dist_16→dist_32) — this run was dist_16→dist_16 (same count); the cross-count restart is the headline A1
  feature at multi-node scale, a quick follow-up now that dars RUNS.

- **Task B0 follow-up — the multi-node device-count CHANGE restart (`RESTART_DEVCOUNT_OK`).** Chained dars seg0
  (cold, **dist_16**, 16 GPU, writes a portable restart) → seg1 (**dist_32**, 8 nodes = 32 GPU, `--dependency=
  afterok`) and seg1 resumed cleanly (`npes=32 devices=32`, step 8→16, exit 0). **Zero code change** — the only
  edit was the sbatch `--nodes`/`--partition`; `read_restart` is partition-INDEPENDENT by construction
  (reconstruct each field to the dense global by gid, then reshard onto whatever `new_part` is asked for; the
  on-disk format keys by gid/owned-index, not device count). Its guard checks the *global* mesh counts
  (nod2D/elem2D/edge2D), never the device count, so a dist_16-written store loads onto dist_32 with the halo
  lanes re-filled from the complete global ⇒ byte-identical to partitioning the original State on dist_32. **The
  lesson: a genuinely partition-portable restart needs NO per-target-count code — gid-keyed reconstruct +
  reshard makes save-N/resume-M just two different `new_part` values; the headline A1 feature cost one sbatch
  edit once the dars run existed.** This closes the dars B0 device-count story (dist_16↔dist_32 both directions
  exercisable); the seam stayed bit-exact through the count change too.

- **Task B2 prep — the NG5 stability gate is a gather-free in-job reduction, NOT a restart re-read
  (`fesom_jax/diagnostics.py` + `run_ng5.sbatch`).** The R0→R3 cold-spin-up ladder gates each rung on
  "finite / CFL-stable", and the cheapest robust signal is `state_diagnostics(state)`: per-leaf
  `jnp.sum(~isfinite)` + `jnp.max(jnp.abs(.))` over the FOLDED sharded final State ⇒ only a **scalar all-reduce**
  crosses the device mesh, never a gathered global field. So R0's verdict (`NG5_R0_FINITE`/`NONFINITE`) runs
  in the SAME 7.4 M / 64-GPU job (`--diagnostics`), no second allocation, no restart re-read — and the identical
  function works on a dense single-device State (the unit test) and a multi-process folded one. **Padding/dry
  lanes are 0 ⇒ finite ⇒ never a false NaN, and 0 never inflates a max**, so the blow-up detectors are mask-free;
  the masked min-layer-thickness>0 check (zstar stability, needs the layer mask) lives in the OFFLINE
  `scripts/debug/ng5_ladder_check.py`, which streams a saved restart one global field at a time. **A second lesson
  from the in-job multichunk test**: splitting a run into fine forcing-chunks (the A4/A6 NG5-memory contract)
  is NOT bit-identical to one big scan — the chunk-BOUNDARY step is a direct bootstrap-`one()` call instead of a
  scan iteration, so XLA reassociates it at the **same ~5.7e-8 float64 floor as the restart seam** (the seam's
  error was always this boundary step, NOT the bit-faithful disk round-trip). Physically meaningless (~3e-10 rel
  on T), but it sets the right tolerance and means "chunked == continuous" is a climate-close, not bit-exact,
  invariant. Cheap canary first (96 steps / 2 chunks on the real dist_64 sharding) before the full R0 — a rung
  failing ⇒ STOP + diagnose (NG5 ≈ 1,500 GPU-h/yr).

- **Task B2 — the NG5 forcing perf journey: MEASURE, don't guess (OOM fix → instrument → profile → local build,
  `RESTART`-grade discipline).** Three NG5-only failures the smaller meshes hid, each caught + fixed by data, not
  hypothesis. **(a) Forcing GPU-0 OOM.** The first R0 canary OOM'd in `jit_stack`: `SurfaceForcing.stack` built the
  per-chunk forcing `[n_steps, nod2D]` on GPU 0 (`jnp`) — ~2.65 GB/field at 7.4 M × 48 steps, plus the intermediate
  list of 48 per-date forcings co-resident — before `partition_step_forcing` could shard it. dars (chunk=8, 3.16 M)
  hid it. Fix = the **host-build pattern** (`step_forcing`/`stack` gain `xp=jnp` default vs `xp=np`; the driver uses
  `np`), the forcing analog of the host-IC fix; everything downstream was already host-then-sharded. **(b) The
  buffered-log blindness.** R0 then TIMED OUT at 90 min with a log showing ONLY the header — a walltime SIGKILL
  loses Python's block-buffered stdout ⇒ zero diagnostics. **You cannot diagnose what you cannot see**: fix =
  `PYTHONUNBUFFERED=1` + `flush=True` on every progress line, then ADD instrumentation (`run_from_config(...,
  progress=)`: per-setup-phase laps + per-chunk `host=`/`device=` split via a `block_until_ready`, gated off by
  default ⇒ suite/bit-identity untouched). **(c) The measured bottleneck.** With flushing: setup 3.5 min (cheap,
  NOT the suspect), steady per 48-step chunk **host 110 s + device 102 s, serial** ⇒ R0 ~112 min (the 90-min
  timeout). A **CPU compute-node profiler** (`scripts/bench/prof_ng5_forcing.py`, pure numpy ⇒ no GPUs) split the host:
  **84 % is `cf.stack` = the JRA interp at 4 s/step over all 7.4 M nodes, run REDUNDANTLY on every one of the 16
  nodes** (each owns ~472 k). **The fix (`fesom_jax/forcing_local.py`):** interpolate ONLY this process's local-
  partition nodes (a fresh reader on a `_SubMesh` of those nodes — independent state, dodging the JRA `_Field`
  mutable-coef trap) and scatter into `[P,n_steps,Lmax]` with only the local rows filled. **Bit-identical because**
  `_to_global_sharded`'s `make_array_from_callback` reads ONLY the local shards' slices of the host array — the
  non-local rows are never touched (so they can be anything; the local pad lanes must still be `_default_pad`=1.0 to
  byte-match `partition_step_forcing`). Validated: a CPU test (`build_local_forcing` == `stack`+`partition_step_
  forcing` byte-for-byte) AND a multi-node 96-step run whose diagnostics were **identical to every digit** of the
  global-forcing probe (`max|uv|=1.575`, `T[-2.076,30.06]`, `a_ice_max=0.1258`) — host **110 s → 14 s (~8×)**, NG5
  now device-bound. **The meta-lesson: at a new scale, instrument first (flushed, gated), measure the split on the
  CHEAPEST resource that reproduces it (the host forcing is pure numpy ⇒ a no-GPU compute node), then fix the part
  the data points at — the redundant ×16 global interp, not the setup I'd have guessed.** Separately the data also
  revealed the device floor (2.1 s/step ⇒ ~6,500 GPU-h/NG5-yr, ~4× the plan's line) — a paper-scope input, surfaced.

- **Task B2 — a NEW driver that re-implements setup SILENTLY DRIFTS from the validated path (2 cold-start gaps in
  `run_from_config`; the user caught one by EYE).** Looking at the NG5 R0 fields the user asked "where is the
  Southern Ocean ice?" — there was none. Root cause: `run_from_config.py` (the A6 model-paper driver) re-implemented
  the cold-start setup separately from the older, validated run scripts and dropped TWO steps every other entry
  point has. **(1) `ice.seed_ice` missing** — it cold-started the prognostic ice at `a_ice=0` instead of the C
  `fesom_ice_initial_state` (`a_ice=0.9` where SST<0, NH `m_ice=1`/SH `m_ice=2`). So in a 3-day Jan run there was
  no Antarctic (summer-hemisphere) ice; the Arctic grew from 0. EVERY validated run (`core2_kpp_climate_run`, all
  `core2_*_stability_run`, `bench_forward_scaling`, all ice tests) does `ice.seed_ice(core2_initial_state(...),
  mesh, sst)` — so the ladder was sound; only the new driver missed it. **(2) `boundary_node_p` missing** — the
  sharded mEVP fell back to `boundary_node_mask(LOCAL mesh)`, which marks PARTITION CUTS as coasts (the EVP zeros
  `u_ice`/`v_ice` at "boundary" nodes, `fesom_ice_evp.c:430-437`) ⇒ artificial ice-velocity walls on interior
  partition boundaries; the comment (`ice_evp.py:251`) says it MUST be the GLOBAL coastal mask partitioned in
  (what `bench_forward_scaling` passes). Fix: seed ice on cold start (`xp=np` host build) + pass
  `boundary_node_p = _shard_along_axis(boundary_node_mask(mesh), …)` when ice is on. Audited the REST of the driver
  against the references + the `run_steps_sharded_forced` kwargs and it's clean (IC incl `T_old/S_old=10/35` AB2
  history + zstar-cold thickness, forcing, the 7 physics cfgs, `params=None` = the off calibration slot, `stress=0`
  the bulk recomputes, AB2 bootstrap). **The lesson: step-level bit-tests miss IC/setup drift because they load
  state from C dumps (bypassing the cold-start wiring) — so a re-implemented driver needs a DIFF against a known-good
  end-to-end run, not just green unit tests. And domain-expert eyeballing of the actual fields (the user) catches
  what the gates don't. TODO: consolidate the cold-start setup into ONE shared helper used by `run_from_config` AND
  the run scripts, so it can't drift again.**

- **[model-paper / multi-year forcing / THE SAME consolidation gap, third instance] (`MULTIYEAR_FORCING_ROLLOVER_OK`)
  Multi-year forcing rollover WAS implemented and CORE-tested — but in the OLD per-step climate driver, not the new
  `run_from_config`.** The user asked "can we start a dars 3-yr run?" and I first claimed multi-year forcing wasn't
  implemented — WRONG (the user pushed: "I thought it was tested on CORE?"). It was: `core2_kpp_climate_run.py:303-308`
  rebuilds the forcing when the model year changes (`if y != cf_year: cf = build_surface_forcing(mesh, y)`), and the
  `kpp_climate_2yr` run logged `[forcing] rebuilt for year 1959` → a full 1959 of monthly means. The REAL gap: that
  rollover lives in the old per-step loop; the new consolidated **sharded/chunked `run_from_config`** builds the
  forcing ONCE for `cfg.forcing.start_year` and its chunk loop never rolled it — `_chunk_dates` correctly advances the
  CALENDAR into 1959/1960 (datetime), but `JRA55Reader` holds a single year's files (`{var}.{year}.nc`), so year-2 dates
  extrapolate off the end of the year-1 time axis (`step`: `rdate > hi` → clamped coeffs). Invisible because EVERY
  multi-year CORE validation went through the OLD driver; the new driver's e2e de-risk was sub-day (never crossed Jan 1).
  Same class as the `seed_ice`/`boundary_node_p` gaps above — the re-implemented driver dropped something the old ones had.
  **The fix, done the RIGHT way (the user's insight: "we spend a lot of time to interpolate on each device — all this
  knowledge is lost?").** The expensive part of "build forcing" — the per-node bilinear gather stencil (`idx4`/`dx4`/
  `dy4`/`denom`, built over every node) and the wind rotation — depends ONLY on (mesh, JRA grid), which is IDENTICAL
  every year. So DON'T rebuild it: `JRA55Reader.reopen_year(y)` (`jra55.py`) swaps only the 8 per-year `_Field` handles +
  1-D time axes and KEEPS the stencil/rotation (a full `build_surface_forcing(mesh, y)` rebuild — what the old loop did —
  would re-pay the per-node setup ~seconds–tens-of-seconds at NG5 scale every Jan 1 and discard the per-device interp
  knowledge; negligible vs ~55 h/yr compute, but wasteful and trivially avoidable). `SurfaceForcing.reopen_year` /
  `LocalForcing.reopen_year` delegate (SSS-restoring + chl are year-independent monthly climatologies; only JRA rolls).
  Wiring in `run_from_config`: `_year_boundaries(year, dt, start_step, n_steps)` (the absolute steps where the calendar
  year flips, exact incl. leap via `datetime`) → `plan_chunks(split_at=…)` so no chunk straddles a year; the loop reopens
  the ACTIVE reader (`local_forcing` if sharded-local, else `forcing`) when the chunk's year changes. A year boundary is
  FORCING-ONLY ⇒ same dt, AB2 continuous, NO re-bootstrap (unlike the dt-ramp split); `_write_restart_at` rolls the
  restart `calendar_date` too (same date fn). Tested: `reopen_year` keeps stencil IDENTITY + the data genuinely switches
  (real 1958→1959 Jan-1 wind differs, `test_jra55`); `_year_boundaries` + `plan_chunks` split (`test_run_entry`, pure).
  **The lesson (reinforced): when a NEW consolidated driver supersedes bespoke scripts, every multi-step behaviour the
  old scripts had (ice seed, coastal mask, year rollover) is a candidate gap — the validation ladder that ran through the
  OLD path won't catch it. And "rebuild" is often the wrong verb: separate the mesh/grid-invariant setup (build once) from
  the per-period data (swap cheaply) — here it turned a per-year full rebuild into a file-handle swap.**

- **[model-paper / dt-ramp calendar / a mid-run dt change breaks a single-dt step→date map] (`RAMP_CALENDAR_OK`,
  commit 4a434d8) The forcing CALENDAR must track the PIECEWISE model clock when dt changes mid-run.** The user wants
  the dars 3-yr run to bump dt 180→240 after year 1 ("the model is stable enough by then"). The dt-ramp already existed
  for COLD-START (small dt for the first ~100 steps, `plan_chunks` splits + re-bootstraps AB2 at `after_step`), but its
  CALENDAR half was missing: `_chunk_dates` mapped absolute step `n` → `base + n·dt` with a SINGLE dt. That's only exact
  up to the ramp; past it, using either dt absolutely mis-dates every step (at the year-1 ramp, step 175200+k is read as
  `base + (175200+k)·240` ≈ 120 days too late and growing) → years 2-3 forcing wrong. Fix: `_elapsed_seconds(n,dt0,ramp)`
  = `n·dt0` up to `after_step`, then `R·dt0 + (n-R)·dt1` (the real model clock), with `_step_at_elapsed` its inverse;
  `_chunk_dates` AND `_year_boundaries` take the base dt + ramp and use them (the 1959 boundary stays at the dt0 step
  175200, but 1960 moves to the dt1 step 306600 — the year-split must follow the ramp or the forcing reopens at the wrong
  step). The callers pass `cfg.dt`+`cfg.dt_ramp` for the calendar while STEPPING still uses `ch.dt` (the two are
  independent: `plan_chunks` already gives the right per-chunk dt). `dt_ramp=None` ⇒ `n·dt0` ⇒ **bit-identical** to before
  (the off-path invariant; all constant-dt runs unchanged). 3 model years with the ramp = 438000 steps (175200@180 +
  262800@240), not 525600. **Lesson: a "switch dt mid-run" feature is TWO things — the integrator's dt (handled) AND the
  forcing clock (easy to forget); when an absolute step index indexes a calendar, any non-constant dt breaks it.**

- **[model-paper / w_split / a JAX capability gap that blocks an exact Fortran match] (`WSPLIT_JAX_GAP`) The JAX port
  cannot do `w_split=true` — it only implements `use_wsplit=0`.** Setting up the dars 3-yr JAX-vs-Fortran comparison, the
  user asked to "turn on `w_split` for large meshes." But the JAX/C reference config was `use_wsplit=0`
  (`fesom_constants.h:56`; `config.py:103 USE_WSPLIT=False`), and the actual vertical-velocity split is an un-ported
  "Phase-5/CORE2 item" — `compute_wvel_split` (`ale.py:405`) early-returns `w_e=w, w_i=0` for `use_wsplit=0`, and the
  split terms were never added (`ale.py:396-404`). So JAX has NO `w_split=true` path. This directly conflicts with "all
  settings the same between JAX and Fortran": you can match w_split OR honor the user's `w_split=true`, not both. (Aside:
  the porting log notes the split *seeded a Fortran day-92 barotropic blow-up*, so OFF was the deliberately-stable
  reference.) Surfaced for the user to resolve (match at `w_split=.false.`, recommended; or accept it as one known
  difference). **Lesson: before claiming a "config-matched" cross-model comparison, audit which reference options the port
  actually IMPLEMENTS — a setting the baseline turns on may simply not exist in the port, and that's a result to report,
  not paper over.** Documented in `docs/memo/FORTRAN_dars_setup.md` (full JAX→FESOM2 namelist mapping; Fortran STAGED, not
  launched, pending the user's w_split call + the plan's "confirm Fortran config" gate).

- **[model-paper / scaling / a config that works single-node can HANG the compile multi-node] (`TKE_MULTINODE_HANG`)
  TKE+mEVP compiles+runs fine at farc/1-node but HANGS the XLA compile at dars/4-node — measured, not guessed.** The
  requested dars production config (TKE + mEVP + zstar) hung the first-chunk compile >66 min at dars/dist_16 (4 nodes,
  3.16 M nodes), 0 chunks — killed (don't burn budget on a failing rung). Bisected CHEAPLY on farc (`run_farc_diag.sbatch`,
  1 node/dist_4): TKE+EVP, KPP+mEVP, AND TKE+mEVP ALL compile + run FINITE in ~100 s. So it is NOT a TKE or mEVP code bug
  and NOT a single-node graph-size issue. The dars KPP+EVP perf runs (also dist_16/4-node) compiled fine, so it is
  **TKE-specific × multi-node** — almost certainly TKE's extra halo-exchange collectives (KPP is column-local, no halo)
  blowing up the XLA compile once the ragged halo is an INTER-node `ragged_all_to_all` at dars scale (cf. the known
  Phase-8b sharded-ragged slowness). Fallback that preserves the user's intent: **KPP + mEVP** (KPP works at dars; mEVP
  shares the EVP `lax.scan` that works at dars), de-risked by a timeout-guarded `run_dars_canary.sbatch` before committing
  the multi-day chain. **Lessons: (1) de-risk a NEW physics combo at the TARGET scale+node-count, not just small/1-node —
  the multi-node compiler path is a different beast; (2) a `timeout`-wrapped compile-canary turns a silent 66-min hang
  into a clean exit-124 you can branch on; (3) the cheap small-mesh bisection (farc) proved "not a code bug" in ~5 min of
  compute instead of hours of dars node-time — measure at the cheapest scale that can still answer the question.**

- **[model-paper / w_split / completed the port] (`WSPLIT_PORTED`) The implicit-advection half of w_split, behind
  `AleConfig.use_wsplit`.** `compute_wvel_split` already split `w→(w_e,w_i)`; the gap was the IMPLICIT consumers. Added,
  mirroring FESOM2 Fortran exactly: momentum `impl_vert_visc` gains the `w_i` upwind tridiagonal terms (`oce_ale.F90:2696-2742`,
  node→element 3-vertex mean, `w>0`=upward); a new `tracer_adv.adv_tra_vert_impl` (the standalone implicit `w_i` solve for the
  FCT low-order solution, `oce_adv_tra_ver.F90:90-239`); and the FCT antidiffusive flux now uses the FULL `w=w_e+w_i`
  (`oce_adv_tra_driver.F90:177-203`). Threaded `use_wsplit`/`wsplit_maxcfl` through `step.py` (momentum reads the LAGGED
  `st.w_i`; tracers the fresh one). **Off-path bit-identity is structural** — `w_i` is threaded only when `use_wsplit` is on, so
  `w_i=None` skips every new path; suite **714+49 green** (vs 710 baseline = +4 new w_split tests). It is a valid feature for
  genuinely fine meshes where the VERTICAL CFL is the limiter — but note it did NOT fix dars dt=180 (next lesson).

- **[model-paper / scaling / the dars-dt=180 blowup is the PORT LINEAGE'S intrinsic 2Δx instability, not the JAX]
  (`DARS_DT180_PORT_LINEAGE_2DX`) dars blows up at dt=180 (finite at dt=60) — and it is NOT the SSH, NOT a halo bug, NOT
  CFL/w_split. It is a known, UNRESOLVED instability the C/Kokkos/JAX ports all share.** Chased every hypothesis to ground:
  (1) **SSH verified a LITERAL port** of the C port `fesom_ssh.c` (maxiter 500, soltol 1e-5, warm-start `X=d_eta`,
  `rtol=soltol·RMS(rhs)`, MITgcm preconditioner, runtime-dt mass term — all identical; pi gates bit-validate). (2) **w_split
  had ZERO effect** (runs byte-class identical on/off) ⇒ not the vertical-velocity CFL. (3) **Partition-independent**:
  `dist_16`≡`dist_32` blew up identically. The answer was in the reference ports' OWN docs: the C port `MPI_PORT_REPORT.md`
  documents the multi-rank partition-boundary `eta`/`w` divergence, but the later Kokkos `DT1800_HANDOFF.md` **refuted the halo
  hypothesis** (256r vs 864r blow within 0.8% → "INTRINSIC, rank-independent") and pinned it on the port carrying **"~20–35%
  more grid-scale (2Δx) velocity energy than Fortran"** — a subtle operator-application difference vs Fortran, never resolved;
  the Kokkos workaround is a smaller dt (CORE2 1800→1200). The **dt ceiling scales with the finest cell**: CORE2 (~100 km) <1800,
  NG5 (uniform 5 km) =180 OK (user ran Kokkos NG5 for months), dars (1.2 km refined patch) <180. Confirmed end-to-end: the C
  port ITSELF `MPI_ABORT`s (CG-NaN `FESOM_DIE`) on dars dt=180. **Lessons: (1) before re-deriving a blowup, read the reference
  port's own bug docs — the answer (incl. the ruled-out halo) was already written down. (2) "the SSH must be a literal port" is
  checkable in minutes against the C source — verify, don't speculate (the user was right to insist). (3) a refined patch sets
  the global dt for the whole mesh; the cheap-eddy mesh you want is a UNIFORM one (FORCA20), not a regionally-refined one.**
  ⚠️ **Claim (2) above is CORRECTED by `DARS_DT180_TWO_MODES` (next) — w_split is NOT zero-effect; that read was
  trajectory-blind. The 2Δx-velocity FINAL-blocker conclusion still STANDS.**

- **[model-paper / w_split / dars dt=180 has TWO instabilities; the earlier "w_split zero effect" was trajectory-blind]
  (`DARS_DT180_TWO_MODES`) The `DARS_DT180_PORT_LINEAGE_2DX` claim "w_split had ZERO effect (byte-class identical on/off)"
  is REFUTED for the trajectory — it compared only the (both-NaN) end state. An instrumented probe (the new
  `--chunk-diagnostics`: gather-free per-level `max|T|` + `max|cfl_z|` after each small chunk; KPP+EVP+zstar = Kokkos's
  config, dist_16) shows w_split changes the blow-up in KIND and TIMING, even though it does NOT save the run.** w_split
  OFF (`dars_kpp_evp_dt180.yaml`): blows at step ~17 via a **SUBSURFACE tracer overshoot** — `|T|max` migrates to level
  20–22 (surface k=0 stays clean at 30.2) with `max_cfl_z≈5` (≫1) ⇒ an EXPLICIT vertical-advection CFL blow-up (the
  elevated 2Δx velocity → large `w` → `w·dt/dz` over the explicit QR4C vertical tracer at the thin 1.2 km mid-depth
  layers). w_split ON (`dars_kpp_evp_wsplit.yaml`, `ale:{use_wsplit:true}`): the tracer instability is **GONE** (T bounded
  [−2.1,30.1] through step 25) but the run now blows at step ~30 via **VELOCITY** (`max|uv|`→2.7e19) — the SAME signature
  as Kokkos (u=99, step 35) and the C port (CG-NaN). So the step-17 tracer death was a **JAX config gap (w_split off), NOT
  a port bug**; with w_split (= Kokkos's implicit vertical advection) **JAX tracks the lineage** — same velocity blow-up,
  similar step. `DARS_DT180_PORT_LINEAGE_2DX` STANDS as the FINAL (2Δx velocity) blocker. **Lessons: (1) instrument the
  TRAJECTORY (per-step, per-level, per-mechanism), not just the final state — "zero effect" hid two distinct failure modes;
  a blow-up that ends NaN either way can still be changed in kind+timing by a real fix. (2) `cfl_z` + per-level `max|T|`
  localize a blow-up to surface-vs-deep and advection-vs-mixing in one cheap gather-free reduction — the single most useful
  sharded stability probe; build it before guessing. (3) OPEN: w_split + intermediate dt (120/150) may UNBLOCK dars — the
  vertical CFL is cured, only the velocity 2Δx remains, which a smaller dt may clear.**

- **[model-paper / the CORE2 hindcast's "520 ms/step" was a DRIVER recompile bug, not model cost; and the
  multi-GPU forced path is non-deterministic] (`PERCHUNK_RECOMPILE_5X` + `MULTIGPU_FORCED_NONDETERMINISM`) The
  CORE2 1958–2020 hindcast smoke ran at 520–562 ms/step, ~6× an unrelated forward bench. Decomposition
  (`scripts/bench/core2_perf_decomp.py` — the PRODUCTION kernel with one physics slot toggled off at a time) showed the
  all-on step is only **96 ms** (bare 58.6 = the 58 ms bench base + ice 18 + GM 10 + zstar 6 + TKE 3 — every
  component cheap and linear). The 5.4× was **100 % XLA RECOMPILATION**: `run.py`'s chunk loop calls
  `run_steps_sharded_forced` per chunk, which builds `jax.jit(jax.shard_map(body))` with a FRESH `body` closure
  each call ⇒ jit cache MISS ⇒ a full ~27 s recompile EVERY chunk. FIX (commit 4778f50):
  `run_steps_sharded_forced(reuse_executable=True)` caches the compiled `jfn` in a module dict keyed by all
  compilation determinants (id(sm/sop/cfgs) + n_steps/dt/bootstrap/folded/ragged/npes/have_bn); `run.py` opts in
  via env `FESOM_REUSE_EXE` (default OFF ⇒ byte-identical) ⇒ ~6 s/chunk = 125 ms/step (4.5×, ~1.6 GPU-days). A
  persistent XLA disk cache (`JAX_COMPILATION_CACHE_DIR`) warms each chained job's first compile chunks. **The
  reuse "bit-identity test" FAILED (max|Δ|≈3e3) — but so does FRESH-vs-FRESH by the SAME per-leaf magnitude
  (ssh_rhs ~3.4k, uv_rhsAB ~2.6k, sigma ~1.8k, T ~0.36): the multi-GPU forced path is NON-DETERMINISTIC at
  roundoff (atomic scatter-adds in the halo / element→node / EVP reductions), amplified by stiff cold-start
  dynamics.** So reuse adds NO error beyond the model's own non-reproducibility. Lessons: (1) a chunked driver
  re-jitting a fresh closure per chunk is a silent 5× tax — the bench harness dodged it via `return_executable`,
  only the driver had the bug; ALWAYS confirm a chunked driver reuses its compiled executable. (2) On multi-GPU,
  "bit-identity" is the WRONG correctness test for a forced run — validate a perf change against FRESH-vs-FRESH
  noise, not zero. (3) The paper's "portable bit-identical restart" claim must soften to "resumes the trajectory"
  on GPU multi-device (bit-exact only on a deterministic single-device/CPU path, like any parallel ocean model).
  See [[fesom-jax-perchunk-recompile-and-nondeterminism]].**

- **[model-paper / hiding the per-chunk forcing build doesn't work in-loop; interp is 75 % but netCDF reads are
  thread-unsafe ⇒ on-device interp is the real (deferred) lever] (`FORCING_INTERP_75PCT_NETCDF_THREAD_WALL`)
  After the reuse fix the hindcast sits at 125 ms/step = 92 ms GPU + ~30 ms/step host forcing (forcing is
  interpolated at RUNTIME, FESOM-style, every step). Two in-loop attempts to hide the host part both FAILED:
  (a) periodic-blocking instead of per-chunk `block_until_ready` — no effect (the forcing is on the device's
  critical path; the step CONSUMES it); (b) prefetch next chunk's forcing while the device runs — slightly WORSE
  (the sharded `partition_step_forcing` device_put queues on the compute stream and backpressures). Reverted both.
  Measurement (`scripts/bench/core2_forcing_measure.py`): `forcing.stack` (JRA bilinear interp, host) = **1091 ms (75 %)**,
  `partition_step_forcing` (scatter + device_put) = 367 ms (25 %), per 48-step chunk. The multi-CORE plan (split
  nodes across threads via the `LocalForcing` sub-mesh machinery) hit a WALL: **netCDF4/HDF5 is NOT thread-safe**
  — concurrent `read_slice` across sub-readers corrupts (IndexError in the HDF5 read) even with independent file
  handles. Lessons: (1) `--local-forcing` is a multi-NODE optimization (distribute interp across nodes); on a
  SINGLE node one host owns all nodes so it can't reduce host work — and CORE2 anti-scales, so adding nodes to
  split the forcing costs more compute than it saves. (2) JAX implicit async does NOT overlap a per-chunk sharded
  `device_put` with compute — it backpressures on the compute stream. (3) netCDF4/HDF5 is thread-unsafe ⇒ no
  naive ThreadPool over readers (use processes, or read serially then thread only the numpy interp). (4) The clean
  throughput win is **on-device interpolation** (device_put the static bilinear weights once + the small raw JRA
  records, gather+weight on the GPU — eliminates the host interp AND shrinks the transfer ~4×) — a real `jra55`
  re-architecture, benefits CORE2/FORCA20/NG5, DEFERRED because R1 runs fine in the background at 4.5× and the
  paper's critical path is figures. (5) CORE2 split = 75 % interp / 25 % device_put (NG5 was 84/16 — smaller mesh
  ⇒ relatively bigger fixed device_put). See [[fesom-jax-perchunk-recompile-and-nondeterminism]].**

- **[model-paper / `ice: {}` is STANDARD EVP, not mEVP — the hindcast config silently had the wrong ice rheology]
  (`ICE_WHICHEVP_DEFAULT_STDEVP`) `configs/core2_full.yaml` had `ice: {}` (= `IceConfig()` defaults) with a comment
  "mEVP sea ice" — but `IceConfig.whichEVP` defaults to **0 = standard EVP**; mEVP is `whichEVP=1`. So R1 would
  have run standard EVP, contradicting the all-on intent. The VALIDATED JAMES all-on config is explicit
  (`core2_kpp_climate_run.py:268` = `IceConfig(whichEVP=1)`). Fixed to `ice: {whichEVP: 1}` (commit 0300158);
  `ice_dt` stays the 500 default as those validated runs used (NOT the ocean dt, despite the IceConfig docstring —
  the validated runs override neither). Re-smoke GREEN. Lessons: (1) `{}` = "defaults", and the DEFAULT ice
  rheology is standard EVP, not mEVP — every other paper config sets `ice: {whichEVP: 1}` explicitly with a
  warning comment; trust the VALIDATED config-construction code, not a hopeful YAML comment. (2) When matching a
  Fortran run, diff the JAX sub-config DEFAULTS against the namelist (here surfaced `ice_diff`=10 in JAX vs 0 in
  the FORCA20 namelist — flagged in `docs/CORE2_FORTRAN_SPEC.md`).**

- **[paper Phase 3 / unfolding the ushow monthly output back to node indexing — the fidelity-figure linchpin]
  (`USHOW_TO_NODES_OK`) The CORE2 hindcast streams monthly means in the *ushow* point-cloud layout
  (`write_ushow_sharded`): arrays folded over the sharded lane axis `[P*Lmax=128292]`, NOT the global node index
  — each ocean node on its OWNER lane (real lon/lat/value), every non-owned lane (halo+pad) the Antarctic sentinel
  `(0, -89.99)` + `FILL=1e38`; 3-D fields `[nz, lane]`. `paper_jax/scripts/common.py:{ushow_to_nodes,unfold_ushow,
  ushow_lane_map}` invert it by matching each owned lane's `(lon,lat)` to the mesh node coords. **The one
  load-bearing gotcha (the handoff's premise was WRONG): EXACT float64 matching recovers only ~3/126858 nodes** —
  the ushow lon/lat come from the model's `geo_coord_nod2D` (rad→deg) and are NOT bit-identical to
  `fesom.mesh.diag.nc` lon/lat (a different deg path). **Rounding to 6 decimals gives a clean full collision-free
  bijection** (verified 126858/126858, 0 collisions; even 4 decimals works ⇒ margin); the helper raises on any
  collision / unmatched lane / non-bijection so a bad assumption is loud, never silent. **Three facts that shape
  the API:** (1) lon/lat are byte-identical across every month of a run ⇒ compute the `(owned_lane, node_id)`
  permutation ONCE via `ushow_lane_map` and reuse it (`lane_map=`) for all 744 months (hoists the O(nod2) match
  out of the loop). (2) owned mask = `lat > -89.98` (sentinel -89.99 < real CORE2 min -78.5); equals value-based
  `< 1e37` ⇒ FILL never leaks. (3) **3-D fields carry 0.0 in sub-bottom cells** (`salt` min = 0.0), so 3-D
  reductions (OHC/drift/sections) MUST weight by `nod_area[nz,nod2]` / `nlevels_nod2D` (both 0 below bottom) — a
  naive `nanmean` over `[nz,nod2]` would average in the dry zeros. Returns 3-D as `[nz, nod2]` (matches `nod_area`
  + native storage), surface as `[nod2]`. Validated by 6 pure-numpy unit tests (synthetic fold, no `/work`:
  recovery, FILL-no-leak, lane-map reuse, + collision/no-match/count-mismatch raises) AND a real-data
  `validate_ushow.py` (USHOW_TO_NODES_OK: all nodes finite + bounded, global area-weighted SST 18.2 °C, 66% regrid
  coverage). The standalone-repo constraint forces coordinate matching (can't import the partition layout) — but
  it's empirically clean, so it's the right call.**

- **[paper/F2-F4-F7] CORE2 1958–2020 hindcast is FAITHFUL: regenerated fidelity figs on the real climatology window.**
  *Why it matters:* the provisional figs (built on 7 months of 1958, near the cold IC) read worse than the model
  actually is — the proper 1980–2009 window tells the real story. *How to apply:* F2 SST RMSE vs PHC3.0 dropped
  **1.23 → 0.60 °C** (SSS 0.40 psu) once reduced over 360 real months (`reduce_meanstate.py --y0 1980 --y1 2009`).
  F4 drift over all 757 months: full-depth Tbar **3.636 → 3.615 °C** across 63 yr (no JAX runaway), ocean volume
  1.321e18 m³, OHC does the classic CORE2 cold-start dip-and-recover, 1958-vs-2021 T(z) profiles overlie. F7 sea
  ice vs **OSI-SAF** (716 mid-month files, EASE2-250 → 6.25e8 m²/cell, `grid_ice_area`): seasonal phase correct
  both hemispheres, JAX modestly icier (NH-area seasonal RMSD 1.63, SH 0.75 ×10¹² m²) — a clean *statistical*
  match, never pointwise. Helpers `hemisphere_ice_area/extent`, `monthly_climatology`, `grid_ice_area` are pure +
  unit-tested (35 tests green total).

- **[paper/obs] OSI-SAF CDR is the located sea-ice obs; per-hemisphere daily EASE2-250 in % — integrate, don't regrid, for area.**
  *Why it matters:* observed ice AREA is a robust scalar integral immune to grid mismatch; only the Mar/Sep *maps*
  need regridding. *How to apply:* `ice_conc` (%) NaN over land; area = Σ(conc/100)·625 km², extent = Σ(conc>15%)·
  625 km² (EASE2 is equal-area → constant cell). Sample the file nearest day-15 per (hem,yr,mo); monthly-mean vs
  mid-month differs a few %, immaterial for the seasonal-cycle claim. Maps: accumulate native conc, mean, then one
  `regrid_to_latlon` per hemisphere, overlay NH/SH disjoint bands.

- **[paper/standalone] No Fortran CORE2 oracle (R2) on disk yet, and we keep plots standalone (not nereus) — both are deliberate, reversible choices.**
  *Why it matters:* the figures currently compare JAX↔obs only; every reduce already carries NaN `*_fortran`
  columns + `has_fortran=0` so the oracle panels auto-light when `/work/.../fesom2_core2` lands. *How to apply:*
  R2 = a CORE2 Fortran hindcast matched to the JAX config (zstar/cvmix-TKE/mEVP/opt_visc=7/CORE2-forcing) — must be
  launched externally (only the FORCA20 Fortran run exists today, different mesh). Plotting stays pure numpy/xarray
  for GMD reproducibility (no `fesom_jax` import); `nereus` (`~/PYTHON/nereus`, own mamba env, tested) is the natural
  upgrade for nicer mesh-native maps **if/when** we want it — user's call to defer.

- **[paper/R2] CORE2 Fortran oracle (R2) drafted = clone FORCA20, change 3 things; STAGE RUN ARTIFACTS ON /work NOT /home.**
  *Why it matters:* the matched-Fortran hindcast is the parity reference; getting the config right = clone the
  already-validated FORCA20 namelists and touch only what CORE2 changes. *How to apply:* deltas are (1) `namelist.config`
  dt=1800 (`step_per_day=48`, **constant — no ramp**, unlike FORCA20's dt-ramp) + CORE2 mesh + ranks; (2) `namelist.oce`
  **GM/Redi ON** (`Fer_GM=Redi=.true.`, coarse 1° needs eddy param; FORCA20 had off); (3) `namelist.ice` `ice_diff=10.0`
  (match JAX `IceConfig`; FORCA20=0.0). dyn/tra/cvmix/forcing/icepack copy verbatim (re-verified: opt_visc=7, tracer
  (0,1), JRA55-do-v1.4.0, TKE c_k/c_eps/α, surf_relax_S=1.929e-6). CORE2 mesh + `dist_256` pre-exist in
  `/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2/`. **Clock semantics (verified on FORCA20):** `fesom.clock` line-2
  year = NEXT start year ⇒ self-chain `run_length=9 y`, stop when next-start≥2021 = 7 chunks × 9 yr = 1958-2020 exactly.
  Staged complete at `/work/ab0995/a270088/fesom2_core2/` (run dir, NOT `/home` — run configs/outputs live on `/work`).

- **Partition-independent canonical Zarr output (`ushow_output.write_global_zarr`) — the FESOM3 `mod_io_zarr` layout, single-process.**
  *Lesson:* the monthly/daily output (`write_ushow_sharded`) writes the **folded `[P*Lmax]`** lane layout — partition-DEPENDENT, FILL=1e38 halo/pad lanes, needs an unfold (`paper_jax` `ushow_to_nodes`) to analyse. The clean alternative the user wanted (matching what they built in the FESOM3 Fortran port) = **canonical global-node order + user chunking, partition-independent**. *How to apply:* `write_global_zarr` scatters each node's unique-OWNER lane (the existing `_folded_gid_owned_nod` gid/owned maps) into a dense global `[nod2D]`/`[nod2D,nz]` via `g[gid[owned]] = host[owned]`, then writes through `write_ushow_zarr` (now with `chunk_horiz`/`chunk_vert` knobs). Result: byte-identical at any device count (verified `synth npes 4 ≡ 8`), `xr.open_zarr` gives `[nz,nod2]` directly (no unfold; ushow/pyfesom2 read as-is), reduce scripts can drop the `ushow_to_nodes` linchpin. **Perf (real CORE2 month 1980_01, 7 fields ~100 MB):** host-gather+scatter ~0.3 s, the rest is zarr compression/IO that the folded writer pays anyway ⇒ ~same cost, amortised over ~138 s of compute between monthly writes ⇒ negligible. Cross-checked max|Δ|=0 vs an independent coord-match unfold. **⚠️ SINGLE-PROCESS only** — it host-gathers owned lanes (fine ≤ FORCA20, a 3-D field ≲ 1 GB; raises `NotImplementedError` if `process_count>1`). Multi-node (NG5) needs the **no-single-node-gather** redistribute-to-writers path: one forward `ragged_all_to_all` folded-shard → canonical-chunk across writer ranks (the analogue of FESOM3 `mod_io_decomp`'s `MPI_Alltoallv`; output needs no autodiff ⇒ the ragged-a2a grad-transpose bug doesn't apply).

- **Canonical output + partition-independent restart wired as the DEFAULT (`output_layout='global'`), folded kept as the NG5/OOM option.**
  *Lesson:* monthly/daily output (`run.py` `_emit_ushow`) and the restart (`zarr_output.write_restart`) now default to the partition-independent canonical layout; the folded gather-free writer is opt-in via `output_layout='folded'` (CLI `--output-layout folded`). *How to apply:* (1) one `output_layout` knob drives all three write sites (daily/monthly/restart); a **fail-fast startup guard** in `run_from_config` raises if `output_layout='global'` AND `jax.process_count()>1` (multi-node can't host-gather) with the hint to use `'folded'` — better than OOMing hours in. (2) Restart canonical = `write_state_zarr(..., layout='global')` → `_write_state_global` scatters each State leaf's owned lanes by gid into dense `[n_global,…]` (full f64 dtype, ALL kinds nod+elem via `_kind_of`, no fold/FILL/gid-maps), `layout='canonical_global'` in attrs. (3) The keystone that kept it low-risk: **`reconstruct_global` auto-detects the layout** (canonical → return `z[:]`; folded → the gid-scatter) ⇒ `read_restart` and every existing restart test work UNCHANGED for both formats, and old folded restarts on disk still resume. (4) Partition-independence is now a *tested property*: same global State saved from P=4 vs P=2 ⇒ byte-identical canonical arrays (`RESTART_PARTITION_INDEPENDENT_OK`), and the restart-seam continuous==chained test passes through the canonical path. Snapshots (`write_snapshot`) left folded (out of scope). 27 + restart-seam tests green.

- **Multi-process canonical output: `all_gather` beats no-gather `redistribute` at every scale (ragged_all_to_all is slow).**
  *Lesson:* built two multi-process partition-independent writers (`fesom_jax/canonical_redist.py`) so the canonical `write_global_zarr` works multi-node, not just single-process: **`all_gather`** (gather the field to every device, each process writes a disjoint chunk range) and **`redistribute`** (the true no-single-node-gather path — one forward `lax.ragged_all_to_all` ships owned lanes to their chunk-owner device, mirroring FESOM3 `mod_io_decomp`'s `MPI_Alltoallv`). Both are correct + byte-identical canonical (validated value==gid on 4-GPU single-node AND 2-node/8-GPU multi-process). **Throughput (CORE2 monthly payload ~200 MB):** folded ~0.4 s ≫ host_gather ~1.0 s > all_gather ~1.5 s ≫ **redistribute ~9 s** — and the redistribute cost is the `ragged_all_to_all` primitive itself (~7.7 s collective-only; caching the shard_map and a `mode='drop'` scatter did NOT help — XLA:CPU has no ragged_all_to_all at all, and the GPU one is ~6× slower than all_gather even at 2 nodes where its no-replication should win). *How to apply:* `write_global_zarr(method=...)`: `auto` ⇒ host_gather (1 proc) / **all_gather** (multi-proc); `redistribute` is the explicit OOM-only fallback (when all_gather's per-device replicated field is too big, i.e. NG5-scale). Default `output_layout='global'` everywhere now (CLI), single→host_gather, multi→all_gather, logged (no silent fallback). The canonical **restart** is multi-process too (`_write_state_global` all_gathers each leaf, scatters to canonical global, writes its disjoint chunk range; handles BOTH nod+elem State kinds; a P-independent chunk grid ⇒ byte-identical at any device count) — validated 2-node/8-GPU (30 nod + 10 elem leaves bit-exact via reconstruct). So multi-node restarts are canonical/partition-independent too; `restart_layout = output_layout` in run.py. Folded stays the fastest/`--output-layout folded` escape (and the OOM fallback). **gotcha:** an all_gather inside shard_map needs `out_specs=P()` + `check_vma=False` (else it re-tiles the gathered array).

- **[paper/R2] CORE2 Fortran oracle (R2) COMPLETE + wired into F2/F4/F7 — strong on-par result; one real S-drift caveat.**
  *Why it matters:* R2 (the matched Fortran CORE2 hindcast, 1958-2020, all 22 streams × 63 yr) is the parity reference;
  the matched-Fortran comparison both validates the port and surfaces where it differs. *How to apply:* the timeout-resilient
  self-chaining job (`/work/.../fesom2_core2/run_core2_r2.sbatch`) finished in 3 jobs (clock semantics: line-2 = next-start yr,
  stop ≥2021). Reductions read Fortran per-year/per-var NetCDF (`{var}.fesom.YYYY.nc`, `[time=12,nz1,nod2]` on the SAME
  native node order as `fesom.mesh.diag.nc` ⇒ node-for-node comparable to the unfolded JAX ushow; `_FillValue`→NaN under
  xarray, pad 3-D to 48 levels to match `area3d`; record i = month i+1). Shared readers in `paper_jax/scripts/common.py`:
  `fortran_iter_months` / `fortran_var_years` / `align_index` (mid-month decimal-year match onto the JAX time axis — JAX has
  757 months incl. an extra 2021-01, Fortran 756, so align, don't assume equal length). **Results:** SST RMSE vs PHC JAX 0.604
  / Fortran 0.606 (|JAX−Fortran| 0.058 °C); Tbar JAX 3.636→3.615 / Fortran →3.621 (Δ 0.006 °C); NH ice area Mar JAX 15.89 /
  Fortran 15.92 — the port reproduces Fortran across mean-state, drift, and ice. **⚠️ one real divergence:** JAX global-mean
  SALINITY drifts +0.0045 psu/63 yr while Fortran stays flat (T/OHC track) — both from identical PHC IC, same vol-weighted
  reduction ⇒ a real salt-budget difference (likely a missing global freshwater/virtual-salt normalization in the port); see
  the 2026-06-30 handoff. Keep `common.py` for diagnostics on BOTH sides (identical reduction ⇒ fair comparison), use nereus
  only for plotting.

- **[paper] "scatter looks strange" = 1°-bin-average speckle; fix = native-triangulation render via nereus. STANDALONE means no `fesom_jax`, not no deps.**
  *Why it matters:* the old F2 maps were `np.histogram2d` bin-averages to a 1° grid; the ~1° CORE2 mesh ALIASES against the 1°
  target ⇒ many empty bins = white speckle dots (looked like a scatter plot). *How to apply:* render node fields on the native
  Delaunay triangulation with **nereus** (`nr.plot(field, lon, lat, projection="rob"/"np"/"sp", method="linear", ax=, interpolator=)`,
  env `/work/.../mambaforge/envs/nereus`, v0.4.1) — `method="linear"` is gap-free in the mesh hull (avoid the default
  `idw`/80 km radius which re-introduces holes on a coarse mesh). The reduces now store **node-level** fields (no regrid) +
  node lon/lat; the figs do the interpolation at plot time. The user clarified STANDALONE = the paper pipeline must not import
  the MODEL (`fesom_jax`) — ordinary libraries (nereus, cartopy, cmocean) ARE allowed deps. The nereus env reads the ushow
  zarr (zarr3) + NetCDF and passes the whole suite ⇒ it now drives the entire pipeline (Makefile `PYTHON`). Helpers added:
  `common.regular_to_nodes` (gridded obs→nodes, periodic-padded + nearest-fill) and `node_weighted_rms`. Panel diet: F4 6→4,
  F7 8→4, F2 → a 2×3 bias triplet [JAX−PHC | Fortran−PHC | JAX−Fortran] (last column ≈ white = port reproduces Fortran).

- **[paper/R2] The CORE2 salinity drift = a sublimation term breaking the zstar freshwater-budget closure — RESOLVED 2026-07-01.**
  *Symptom:* JAX global vol-mean S drifts +0.0044 psu/63 yr while the matched Fortran stays flat; near-uniform +0.031 psu
  surface offset. *The diagnostic that cracked it (do this FIRST for any "conservation" drift):* **plot global-mean SSH**
  (area-weighted `hbar`/`ssh`) alongside the drifting field. JAX ⟨SSH⟩ drops **−0.442 m/63 yr**, Fortran flat; the predicted
  drop if S̄ rose purely by volume loss (`−H·ΔS/S̄`, H=3627 m) is **−0.444 m** — a 0.5% match ⇒ the drift is **volume-driven
  (freshwater budget non-closure), NOT a salt source**. Salt content is conserved; the ocean loses ~7 mm/yr of freshwater,
  concentrating S̄=salt/V. This one plot rules out the salinity floor, brine/rsf, restoring, and every salt-source hypothesis
  at once. *Root cause:* under zstar `oce_fluxes` removes the global-mean freshwater flux via `flux = evaporation −
  ice_sublimation + prec + runoff − ice/snow-growth`; **sublimation (ice→atmos, not ocean FW) must cancel, which requires
  `evaporation` to be the BUNDLED `evap_ow + subli`** — Fortran bundles it in `therm_ice` (`ice_thermo_oce.F90:651`
  `evap=evap+subli`) BEFORE storing `evaporation(i)` (`:324`); the C-port bundles it too (`fesom_ice_thermo.c:407`
  `*evap=_evap+_subli`). The JAX regressed: `ice_thermo.py` surfaced `evaporation=evap` = **open-water only** (the value at
  the `evap*(1-A)` line, before the bundle), so the balance computed `evap_ow − subli`, leaving `−⟨subli·A⟩` uncancelled ⇒ a
  net freshwater deficit injected every step, applied *uniformly* by the global-mean correction (hence the near-uniform
  surface offset; ⟨subli·A⟩≈2.2e-10 m/s ≈ 3.6 cm/yr over ice = realistic). *Fix:* `ThermoOut.evaporation = evap + subli`
  (1 line, `ice_thermo.py`); only consumer is `ice_coupling.fresh_water_balance_zstar` (zstar+ice path), linfs untouched.
  Verified by running the REAL balance fn: buggy ⟨water_flux_bal⟩=+1.0e-9 m/s (=−⟨subli⟩), fixed=−2e-25 (machine zero).
  *Lesson (grep-before-you-run):* when a port splits a bundled Fortran field into components for a downstream reconstruction
  (`A − B` to recover `A−B`), verify you took the field at the SAME point in the Fortran control flow the consumer reads it —
  therm_ice bundles evap+subli at its LAST line, so the stored `evaporation` is the bundled one, NOT the open-water `_evap`
  named earlier. A one-sided cancellation term (in the removed global mean but not the applied local flux, or vice-versa) is
  the generic signature of a slow, near-uniform, volume-driven tracer drift. *The C-port precedent (per the user) was the
  right place to look* — not for the SAME bug (that was runoff double-count, already avoided) but because the C-port has the
  CORRECT bundling, proving the JAX regressed.

- **[paper/R2] The CORE2 JAX−Fortran SST wavenumber-1 zonal pattern = DIURNAL ALIASING in the monthly-mean output (a diagnostic
  artifact, NOT a model bug) — RESOLVED 2026-07-01.**
  *Symptom:* JAX−Fortran annual-mean SST (`meanstate.nc:sst_jaxfor`) is a clean global k=1 zonal harmonic (R²=0.88 tropics,
  peak −79°W / trough +101°E, hemispheres opposite-signed, zero global mean). Looks like a coordinate/rotated-grid or
  wind-rotation effect. *The mandate + all the "obvious" suspects were RED HERRINGS — verified byte-identical to Fortran:* the
  wind g2r rotation (`jra55._vector_g2r` + Euler 50/15/−90 matrix == C `fesom_vector_g2r`/`build_rotation_matrix`), the forcing
  time mid-shift (`NM_NC_TMID=0`, both apply it), the solar-zenith open-water albedo (both `open_water_albedo=0` ⇒ constant
  `albw`), and the geographic forcing-interp coords (C-exported). *Root cause:* the JAX **monthly-mean output** sampled ONE
  chunk-final `state_p` snapshot per chunk (`run.py`), and the hindcast used **48-step = exactly 1-day chunks**, so every sample
  fell at the SAME UTC time (~00:00) ⇒ the mean of ~30 same-time-of-day snapshots ALIASES the diurnal SST cycle into a k=1
  pattern (warm on the afternoon side / cool on the night side; at 00:00 UTC → warm Americas/West, cool Asia/East — exactly the
  observed sign). Fortran writes a TRUE every-timestep monthly mean (`namelist.io 'sst',1,'m'`) ⇒ no alias. So the "SST
  difference" is the JAX output's diurnal alias, not a physics divergence — the model SST is on-par. *The diagnostic that
  cracked it (do FIRST for any zonal-k=1 "difference"): is the field a snapshot or a true time-mean, and at what cadence?*
  *Fix (GENERAL, dt-independent):* accumulate the output fields over EVERY step inside the chunk scan (`run_steps_sharded_forced`
  gains opt-in `sample_fn`; the scan carries a running sum → the driver ÷ by the step count = a true time-mean, matching
  Fortran) + split chunks at DATE-based period boundaries (`_period_boundaries` day/month, dt-INDEPENDENT via `_step_at_elapsed`)
  so each chunk's sum belongs to one period, keyed by the chunk's FIRST step (`_MeanStream` — keying by the final step
  misattributes a boundary-ending chunk one period late). *NOT the config hack (chunk_steps=16 for 3 samples/day) — that only
  works at dt=1800 (48 steps/day); the every-step mean is correct at any mesh/dt.* Verified: sample_fn=None byte-identical
  (`test_forcing_sharded` 4 pass); the accumulation is an exact per-step sum AND does not perturb the State at npes=1 AND npes=2
  (real dist_2). Rerun R1 (with the salinity fix) + regenerate `meanstate.nc` ⇒ the k=1 should vanish. *Lesson:* a
  time-sampled diagnostic (snapshot at a step-count cadence) can imprint a longitude-organized pattern that mimics a
  coordinate/physics bug; compare LIKE-for-LIKE (both true time-means) before blaming the model — and make the mean dt-general,
  not tied to one config's steps-per-day.

- **[model-paper] ushow multi-file animation silently fails on a specific variable ("Could not find variable 'm_ice' in zarr
  file N") when the store has no consolidated Zarr metadata — FIX, CONFIRMED by the user 2026-07-02.**
  *Symptom:* the user globbed the per-month canonical output (`ushow "monthly/1958_0*"`) expecting an animation across
  months — ushow loaded the store (variable buttons appeared) but every frame past the first failed to read `m_ice`
  specifically, with no animation. *First checked, ruled out:* whether fesom_jax's writer was actually missing/inconsistent
  data — it wasn't: `ls` + `.zattrs`/`.zarray` diffs showed all 9 months had byte-identical schema (`a_ice lat lon m_ice
  salt ssh temp u v`, same dims/dtype/chunking), and raw (unsorted, `ls -f`) directory order was IDENTICAL across every
  file too — so this is not a write-side bug, and not a cross-file directory-order inconsistency either. *The tell:* ushow's
  own startup log prints `Zarr: No consolidated metadata, will read individual .zarray files` — a routine fallback message,
  but circumstantial evidence it has a less-robust code path when no `.zmetadata` manifest exists, and our writers
  (`zarr_output.py`/`ushow_output.py`/`canonical_redist.py`) never called `zarr.consolidate_metadata()`. *Fix:* added
  `zarr.consolidate_metadata(out_dir)` at the end of all four writer paths that produce ushow-viewable output —
  `write_ushow_zarr` (unconditional, single-process), `write_ushow_sharded` (rank-0-only, after the data barrier),
  `write_all_gather`/`write_redistribute` in `canonical_redist.py` (same, `pid==0`-guarded post-barrier) — purely additive
  (a `.zmetadata` manifest aggregating existing `.zarray`/`.zattrs`; never touches chunk data), so it does not change how
  `common.jax_canonical_months`/`reconstruct_global` read the store (plain `zarr.open_group`, unaffected either way).
  Verified via `test_ushow_output.py`/`test_canonical_redist.py` (3 passed + the 2 previously `NDEV<4`-skipped multi-process
  tests re-run with `XLA_FLAGS=--xla_force_host_platform_device_count=4`, both pass) — no regression, and the two
  multi-process writers used by FORCA20's multi-node output are now genuinely exercised. Backfilled onto all
  already-written live production output (`core2_hindcast_v2` 341/341 months, `forca20_hindcast_v2` 3/3 months + 121/121
  days; the single most-recently-modified dir per root was skipped each time as a margin against the still-running writer
  process). *Confirmed:* the user retested interactively — animation now advances through frames correctly, the fix
  worked. *Lesson:* when a third-party VIEWER (not this codebase) fails on a SPECIFIC variable/frame while the underlying
  data is byte-identical across every file, check what optional-but-preferred on-disk convention the viewer itself
  announces (a log line, a warning, a fallback message) before assuming a data bug — and when you can't drive the
  viewer's UI yourself, say so plainly and wait for confirmation rather than claiming a fix works.

- **[model-paper] ushow shows no dates during (now-working) multi-file animation, and can't build a time series either —
  because the writer never wrote an actual `time` COORDINATE, only a date STRING in a metadata attr — FIX 2026-07-02.**
  *Symptom:* immediate follow-up to the bullet above — once animation worked, frames had no date label, and the user
  separately noted this also blocks building a time series (both need a real time axis to key off). *Root cause:*
  `ushow_output.py` stashed the period's date as `attrs["calendar_date"]`/`attrs["calendar_month"]` — a plain STRING on
  the store's root `.zattrs`, not a `time`-dimensioned array variable. Telling clue: the module already defined
  `TIME_DIM = "time"  # in ushow's TIME_NAMES` — a `grep TIME_DIM` found exactly that ONE definition and no use anywhere
  else, i.e. it was designed for but never wired up. *Fix:* added `ushow_output.cf_days(date)` (days since 1970-01-01,
  `calendar=proleptic_gregorian` — matches plain `datetime.date` arithmetic, negative offsets pre-1970 decode fine) +
  `_put_time(root, time_days)` (writes a `time[1]` array with `_ARRAY_DIMENSIONS=["time"]` + CF `units`/`calendar`
  attrs), and a `time_days=` kwarg threaded through all 4 writer entry points down to `_put_time`/`_create_store`.
  `run.py` computes the value per period: `_write_daily` uses the exact calendar day, `_write_monthly` uses day 15
  (mid-month — matches the SAME convention `paper_jax/scripts/common.py` already uses for its decimal-year x-axis, so
  the two pipelines now agree). *Verified:* 7/7 tests green (the same 5 writer/consolidate tests + `test_daily_output`/
  `test_monthly_output`, both now list `'time'` in the written fields); backfilled onto live output by PARSING each
  store's EXISTING `calendar_date`/`calendar_month` attr (no need to touch run state) — 354/354 + 3/3 + 129/129 stores,
  0 errors; spot-checked 3 backfilled stores decode back to the exact expected date via `TIME_EPOCH + timedelta(days=t)`.
  *Lesson:* an unused "anticipated" constant (a dimension name, a config key, a schema field defined but never actually
  populated) is a strong signal the feature was planned but the wiring got dropped — `grep` for a suspicious constant's
  OTHER uses before assuming it's actually implemented just because it's *defined*.

- **[model-paper] ushow's "ultimate test" (per-point time series) STILL failed after the `time[1]` coordinate fix — two
  more root causes, found only after the user pointed me at ushow's SOURCE (`~/ushow`, a sibling repo) — FIX 2026-07-02.**
  *Symptom:* animation worked (prior bullet), but clicking a point for a value-vs-time plot produced nothing sensible.
  *Why binary-strings-only investigation stalled here:* I'd confirmed via `uterm` (ushow's terminal sibling, far easier
  to introspect than the X11 GUI) that "Total virtual time steps: N across N zarr files" was correct, and the on-canvas
  HUD showed "time 1/N" — but couldn't tell if that was a real date or a plain index without seeing rendered output, and
  `strings`-based reverse engineering of the binary was giving plausible-but-unverifiable theories. The user's source
  pointer turned hours of inference into minutes of `grep`. *Root cause 1:* `zarr_get_dim_info`/`zarr_read_timeseries_fileset`
  (`file_zarr.c`) only read the `time` coordinate's VALUES when the DATA VARIABLE's own `_ARRAY_DIMENSIONS` names a dim
  matching `TIME_NAMES` (`var->time_dim_id>=0`) — confirmed against `ushow/tests/test_file_zarr.c`'s own fixtures, which
  ALWAYS declare `_ARRAY_DIMENSIONS:["time","ncells"]` on the data variable itself, time leading; a bare store-level
  `time[1]` array with no matching data-variable dim is silently ignored. *Fix:* every data field gains a `[1,...]`
  leading axis (`_put`/`write_ushow_sharded`/`_create_store` in `canonical_redist.py` — every low-level write-indexing
  call site needed the new leading time index prepended, e.g. `z[0, r0:r1]` not `z[r0:r1]`); lon/lat stay node-only
  (never time-varying). *Root cause 2, a genuine USHOW BUG (not a convention mismatch):* even with dim #1 fixed, the
  HUD showed the WRONG date (1970-01-01, the epoch) on every frame. Traced to `file_zarr.c`'s
  `zarr_get_dim_info_fileset` (~line 1475): its value-reader only has branches for `dtype=='i'&&size==8` (int64) or a
  literal `dtype=='d'` — but `parse_dtype` (same file) maps EVERY float width (4-byte or 8-byte) to `'f'`, NEVER `'d'`
  (dead branch) — so a float64 `time` array matches neither branch, `got_values` stays 0, and it silently falls back to
  a plain `0,1,2,...` index (which happens to format as the epoch date when passed through the otherwise-correct
  `format_time_from_units`). Ran ushow's OWN test suite (`cd ~/ushow/tests && make WITH_ZARR=1 test_file_zarr &&
  ./test_file_zarr` — needs DKRZ's spack c-blosc/lz4 paths) — 34/34 pass, but `zarr_get_dim_info_fileset_basic` only
  asserts the dimension SIZE, never checks the VALUES are correct — a real gap in ushow's own coverage. *Fix:* write
  `time` as int64, not float64 (`cf_days()` already returns a whole day-count, so nothing is lost). *The verification
  technique that cracked it:* `timeout cmd 2>&1` shows NOTHING on success for either `ushow`/`uterm` — both fully-buffer
  stdout unless attached to a real terminal (not just any redirect) — use `script -qc "cmd" out.txt` (or Python's `pty`
  module, which additionally lets you INJECT keypresses, e.g. `os.write(master, b"j")` to step frames) to see real
  output. This is how I got `uterm` to print `Found zarr variable: sst [time=1, nod2=126858]` and
  `time 1/3 1958-01-15` — a direct, unambiguous pass/fail signal for a GUI feature I could otherwise never observe
  headlessly. *Verified end-to-end through the ACTUAL production `write_ushow_zarr` function* (not a manual data patch):
  built a 3-file synthetic set, confirmed frame 1 → `1958-01-15` and frame 3 → `1958-03-15` via a real injected
  keypress. *Migration of the already-written ~20GB of production output* (reshape every field + fix the time dtype)
  was written and tested 100% byte-identical on scratch copies (2D+3D fields, two mesh sizes) before touching live
  data — deliberately gated on the user's explicit go-ahead, since unlike the two previous fixes (purely additive
  metadata sidecars) this rewrites actual array shapes in place. *Lesson:* (1) when a closed/hard-to-introspect tool
  misbehaves and binary-strings guessing stalls, ask if source is available before continuing to infer — a sibling
  repo can turn hours into minutes; (2) a tool's OWN test suite passing does not mean the specific behavior you depend
  on is covered — read the actual assertions, not just the pass/fail count; (3) `timeout` alone is not a reliable
  headless-testing tool for interactive terminal/GUI programs — get a real pty.

- **[model-paper] the time-schema migration itself had a bug, caught by a spot-check on an ARBITRARY (not the newest)
  store after "0 errors" — FIX 2026-07-02.** *Symptom:* first migration pass reported 639/639 migrated, 0 script
  errors; a follow-up check of `core2_hindcast_v2/monthly/1990_03` (picked arbitrarily) raised `KeyError: 'time'`.
  *Root cause:* the migration script assumed every store either already had a `time` coordinate (float64, needing a
  dtype fix) or would gain one via the reshape step — but it silently reshaped ANY data field lacking a leading
  `"time"` dim regardless of whether a backing `time` coordinate array actually existed. 164 stores (written by the
  STILL-RUNNING `core2_hindcast_v2`/`forca20_hindcast_v2` chain jobs, which only re-import Python on their NEXT
  `sbatch` resubmit — never mid-job — so they kept writing with STALE pre-fix code for hours after the fix landed on
  disk) had never gotten a `time` coordinate from any earlier backfill, and came out of the migration WORSE than
  before: a dangling `_ARRAY_DIMENSIONS=["time",...]` reference with no backing array. *Fix:* extended the migration
  to also SYNTHESIZE a missing `time` coordinate (from the store's own `calendar_date`/`calendar_month` attr, same
  formula the original backfill used) when `"time" not in g`, not just dtype-convert an existing one. Verified on a
  copy, reran live (idempotent — 2:50 vs the first pass's 18:32, since most stores were already-correct no-ops), then
  ran a FULL verification sweep (every store, not a sample) — 647/650 pass, the 3 "failures" exactly match the 3
  dirs each run intentionally skips as the live-writer safety margin (confirmed against the migration log's own
  "skipped most-recent X" line) — expected, not a defect. *Lesson:* a migration/backfill script's "0 errors" summary
  only means its OWN error handling didn't trip — it says nothing about whether its ASSUMPTIONS about prior state
  were correct. When a resource is fed by a still-running process that can't have picked up an intervening code fix,
  don't trust "I ran the backfill once, therefore it's covered" — re-verify against live current state, and spot-check
  an ARBITRARY item (not just the newest or most suspicious one) — this bug was invisible from the newest end.

- **[model-paper] restarts get an ARCHIVAL stream — immutable, one folder per firing, user-controlled cadence,
  REPLACING the old rolling `--checkpoint-every` restart entirely — mirrors FESOM3, 2026-07-02.** *The ask:* the
  user was unhappy the restart was one directory, always overwritten, no history, no controllable frequency —
  "restarts should have separate folder for each of the restarts, like in ../fesom3 and I should be able to
  continue from any restart in the past... restarts should have frequency, that I can control." *Investigated the
  actual reference instead of guessing:* `/home/a/a270088/fesom3/src/io/mod_io_restart.F90` writes an IMMUTABLE
  `fesom.<YYYY>.<DDD>.<SSSSS>` folder (year/doy/sec-of-day, zero-padded ⇒ lexical sort == chronological) per
  checkpoint, NEVER overwritten, kept forever by default (`restart_keep=0`); cadence is a `restart_length`/
  `restart_length_unit` pair (y/m/d/h/s/off); resume ALWAYS follows a `restart.latest` POINTER FILE (one line, a
  folder name) — never sorts/globs the directory (a stray crashed `.tmp` or a later-but-unpointed folder can't be
  mistaken for current); the last step of a run is ALWAYS checkpointed regardless of cadence. *Design fork the user
  resolved directly:* I initially proposed keeping the frequent rolling checkpoint SEPARATE from a new coarser
  archival stream (crash-safety vs branching, two concerns). User: "we are fine losing longer computations...
  it's for the user to decide - if they want they can do more frequent restarts" — ONE unified, user-configurable-
  frequency mechanism, exactly matching FESOM3 (which doesn't have two restart concepts either). So the archival
  stream REPLACED `--checkpoint-every`/`restart_out` in both production sbatch scripts (CORE2=yearly,
  FORCA20=monthly); the library-level rolling-checkpoint code in `run.py` is untouched/still available, just unused
  by these two chains now. *Implementation:* `RunConfig.restart_archive_out/_period/_length` (general N, not just
  "every one"); `run.py`'s `_archive_tag`/`_archive_boundaries` mirror FESOM3's naming/cadence exactly, added to
  the existing chunk-split machinery (same mechanism as the daily/monthly output period splits, so a chunk never
  straddles a firing); the chunk loop fires on every boundary crossed PLUS unconditionally at the run's end
  (mirrors "last step always checkpointed" — a CLEAN chain resubmit never loses anything, only an actual crash
  falls back to the last calendar boundary); `zarr_output.write_restart_latest`/`resolve_latest_restart` mirror the
  pointer file (atomic write-then-rename). *A sbatch-script gotcha caught before shipping:* the natural resume
  snippet (`from fesom_jax import zarr_output; ...resolve_latest_restart(...)`) works but silently drags in
  `import jax`, which noisily fails CUDA init (a `cuInit(0) failed` traceback) in the sbatch BATCH-script context
  (outside `srun`'s GPU-scoped allocation) just to read one line of text — reimplemented the pointer read in PURE
  BASH (`cat "$ARCHIVE/restart.latest"`) instead. Caught by testing the exact snippet in isolation BEFORE trusting
  it in production — a lesson on its own: always test sbatch-level shell logic standalone, not just the underlying
  Python function it wraps. *Verified 3 ways before touching the live chains:* 22 pytest tests (unit boundary/tag
  math incl. general-N filtering; an integration test spying on `write_restart` to lock the exact firing steps);
  the isolated bash-snippet test that caught the CUDA-import bug; a full CLI-level end-to-end smoke test running
  the ACTUAL `run_from_config.py` through a real 2-job resume cycle (cold→step60 with a day-boundary + final write,
  then a genuine `restart.latest`-resolved resume to step72) — 3 immutable dirs, none overwritten, exactly as
  designed. *Cost transparency surfaced BEFORE wiring in:* FORCA20 monthly × 36 months ≈ 972 GB total (27 GB/
  restart, kept forever) — a real number, explicitly given to the user and accepted, not silently defaulted.
  *Lesson:* when a user references another model/tool as the target convention ("like in X"), go read X's actual
  source if it's available rather than inventing a plausible-sounding design — it resolved naming, cadence-unit,
  and resume-semantics questions I would otherwise have had to guess or re-ask about, and confirmed the "last step
  always checkpointed" behavior I'd have designed anyway but couldn't have been sure was the RIGHT convention to
  match without checking.

- **2026-07-03 — orphaned rolling `restart_out` (stale YAML default, no CLI override) crashed FORCA20's cold-start
  canary 25 min in, right after a clean 30-day run.** When the archival-restart mechanism replaced the old rolling
  `--checkpoint-every`/`restart_out` stream in the production sbatch scripts (previous lesson above), I removed the
  CHAIN's dependence on `restart_out` but never checked whether the underlying library code (`run.py`) still
  unconditionally WRITES to `cfg.restart_out` at the end of every job — it does (`target = out_dir if out_dir is not
  None else cfg.restart_out`, then an unconditional final `_write_restart_at` regardless of `checkpoint_every`).
  Neither `run_core2_hindcast.sbatch` nor `run_forca20_hindcast.sbatch` ever passed `--restart-out` on the CLI
  (despite `core2_full.yaml`'s own comment claiming "the hindcast sbatch overrides mesh/partition/steps/restart via
  CLI" — an intent that was never actually implemented), so both fell through to each YAML's literal default:
  `forca20_tke_mevp_prod.yaml` pointed at `runs/forca20_3yr/restart` — the OLD, stalled, explicitly-superseded chain's
  directory (see its own header: "do NOT resume it"). That old directory's zarr arrays were shaped for `dist_32`
  (the YAML's own default partition); the sbatch script overrides `PART=dist_16` via `--partition`, so the new run's
  local shard shape didn't match the pre-existing array on disk — `root.require_dataset` raised `TypeError: shape
  do not match existing array`, one rank crashed mid-write, and the other 3 ranks hung on the JAX distributed
  shutdown barrier until DEADLINE_EXCEEDED killed the whole job. CORE2's `core2_full.yaml` had the identical latent
  bug (`restart_out: runs/core2/restart`, a directory distinct from the actual `runs/core2_hindcast_v2` in use) but
  it hadn't crashed yet only because CORE2's partition (`dist_4`) never changes across jobs, so the shape happened
  to keep matching — it was silently overwriting a stray directory every checkpoint, not equivalent-safe, just
  not-yet-triggered. *Caught by:* actually reading the crash traceback instead of just noting the exit code — the
  literal path in `"writing restart -> …"` immediately named a directory that had no business being touched by this
  run. *Fix:* both sbatch scripts now pass `--restart-out "$OUT/restart"` explicitly (mirrors how they already pin
  down `--restart-archive-out "$ARCHIVE"` — the CLI override, not the YAML default, should be the actual source of
  truth for any per-TAG path in a shared config template); updated both YAML defaults too so they're not misleading
  bystanders. *Lesson:* when a new mechanism (the restart archive) supersedes an old one (rolling `restart_out`) in
  the automation layer, check whether the OLD mechanism's write path is still LIVE at the library level even though
  the chain no longer reads it back — "unused for resume" is not the same as "inert." A stale default that's merely
  unread is latent; the moment ANY run parameter it was silently coupled to (here: partition/device-count) changes,
  it can crash the job or silently clobber an unrelated directory. Grep for every place a YAML default could still
  be reached un-overridden whenever a chain's CLI-argument surface changes.

- **2026-07-03 — review fix set: the zstar `a_ice_old` freshwater leak (the SECOND budget bug in the same
  10 lines), the PGF ρ−2ρ0 double subtraction, and WHY the dump gates couldn't see either.** The full-codebase
  review (`docs/CODE_REVIEW-20260703.md`) found `ice_step.py` passing start-of-step `state.a_ice` as the balance's
  `a_ice_old` where the C uses the THERMO-ENTRY (post-FCT/cut_off) concentration: the C's `values_old` backup runs
  inside the thermo loop AFTER advection and BEFORE overwriting (`fesom_ice_thermo.c:497-506`) — the JAX comment
  ("the PREVIOUS step's concentration") had misread the C's loop structure. With the right `a_co`, the balance's
  `prec_snow·(1−a_old)` cancels thermo's `snow·(1−A)` term-for-term and post-balance `⟨water_flux⟩ ≡ 0` exactly;
  with `state.a_ice`, `⟨prec_snow·(A_entry−A_state)⟩` (~1e-13 m/s at step 1) leaks into the volume budget every
  step — the same class as the 5a61b0 sublimation leak, in the same block. Also fixed: `step.py` fed the
  Shchepetkin PGF `density − DENSITY_0` although `eos.compute_pressure_bv` already returns ρ−ρ0 (the C's
  `density_m_rho0`) — analytically a no-op (the kernel is offset-invariant, which is why the JZ.7 gate passed at
  1e-14) but ~30× coarser stencil ulp and a trap for any future non-invariant consumer; fixing it perturbs zstar
  trajectories at the rounding floor (relaunch-only change). Plus three guards: `eos.jm_components` sqrt(S)
  double-where (S≡0 below bottom ⇒ NaN cotangents in d/dS adjoints), an entry raise in
  `run_steps_sharded(return_grad_fn=True, use_ragged=True)` (the broken ragged transpose was silently reachable),
  and the missing `IceConfig.ref_sss` field (34.7, `oce_modules.F90:37`) that made `ref_sss_local=0` a trace-time
  AttributeError. *Caught by:* a systematic review re-reading the C loop structure — NOT by the test suite,
  and that is the lesson. *Why the gates were blind:* the z2_cdump forcing gate compares water_flux PER NODE at a
  ~1e-8 bulk-ulp floor; a ~1e-13 near-uniform budget leak is invisible there, and the unit test for the balance
  re-derived the implementation's own formula (self-consistent by construction). Both budget bugs (sublimation +
  a_ice_old) lived exactly in the gap between per-kernel dump gates and global invariants. *Fix for the gap:* a
  new CONSERVATION gate (`test_ice_step.py::test_zstar_freshwater_balance_closes_globally`) asserts the assembled
  step's post-balance `⟨water_flux⟩/⟨|water_flux|⟩ < 1e-8` on the real CORE2 state PLUS a clamp-perturbed state
  (a_ice=1.4 → cut_off 1.0) with a computed negative control proving the old wiring would fail it by ≥4 orders.
  *Lesson:* per-node dump oracles validate KERNELS; they cannot see GLOBAL-budget wiring bugs whose per-node
  signature is a near-uniform offset below the reduction-class floor. Every conservation statement the model
  makes ("the balance zeroes the global mean", "the tracer sum is preserved") deserves its own assembled-step
  invariant gate, asserted against the mathematical property — not against the implementation's own re-derivation.

- **[bench/paper§5] Single-device + CPU baselines measured (jobs 26037292/26037293); npes==1 needs
  `synth_serial`, not `read_partition`.** `read_partition` refuses npes==1 by design — `bench_forward_scaling.py`
  now builds the identity partition for `--npes 1` (zero halo, exchange=identity). *Results (CORE2 FULL
  ice+kpp+gm+JRA1958, warm-executable protocol):* **1×A100 = 243.2 ms/step (20.3 SYPD, peak 8.41 GiB)** ⇒
  4-GPU parallel efficiency 70% vs the 86.7 ms ragged number; **1 CPU node (128 cores, JAX_PLATFORMS=cpu,
  single process, XLA threads) = 5755 ms/step (0.86 SYPD)** — ~29× slower than Kokkos-CPU's 128-rank MPI on
  the SAME node (200 ms/step) ⇒ JAX-CPU is a functionality/accessibility mode, never a performance mode; and
  it cannot scale out (XLA collectives are not production-grade on the CPU backend — no multi-node CPU
  shard_map). New sbatch: `bench_core2_cpu.sbatch`, `bench_core2_1gpu.sbatch`; `paper_jax` reduce_scaling now
  labels `plat=cpu` rows "JAX (CPU)" so the GPU figure/table ignore them.

- **[bench/HPC] Levante's `gpu` partition MIXES A100-40GB (cell09) and A100-80GB (cell13) — ALWAYS pin
  `#SBATCH --constraint=a100_80` for anything memory- or bandwidth-sensitive.** Job 26037293 (the 1-GPU
  CORE2 baseline) silently landed on a 40GB card (l40366) — different HBM bandwidth (1.6 vs 2.0 TB/s), so
  its 243 ms/step is NOT comparable with the a100_80 multi-GPU rows (re-measured as job 26038900); and the
  gradient bench (26038321/26038491) OOM'd unexplainably until `nvidia-smi` in the log exposed the 40GB
  card — the adjoint's per-step checkpoint carry stack (~2 GiB × N steps for the full CORE2 State) simply
  cannot fit there. Audit trick: `sacct -j <id> -o NodeList` + `sinfo -n <node> -o "%f"` reveals the card
  after the fact — all historical paper-scaling rows checked and confirmed a100_80. Second lesson from the
  same saga: `XLA_PYTHON_CLIENT_MEM_FRACTION≥0.9` starves the CUDA module loader ("Failed to load
  in-memory CUBIN ... CUDA_ERROR_OUT_OF_MEMORY") for programs this large — 0.85 with default preallocation
  is the working setting; PREALLOCATE=false fragments and kills the multi-ten-GiB carry-stack alloc.

- **[bench/AD] Gradient cost MEASURED (job 26046417, attempt 5): reverse-mode = 4.7× forward,
  0.83 GiB/step of window; the fix for the CUBIN-load OOM was DISABLING XLA COMMAND BUFFERS.**
  CORE2 full (ice+kpp+gm), 1×A100-80GB, per-step ckpt + remat_blocks: fwd 196 ms/step (== the
  bench-harness 194 ms — dense integrate() path cross-checks), value_and_grad 909-923 ms/step
  (ratio 4.63/4.70 at n=12/24, n-independent), peaks 29.1→39.1 GiB (slope 0.83 GiB/step + ~19 GiB
  base) ⇒ day-scale windows fit one 80GB card. The grad executable only LOADS with
  `XLA_FLAGS="--xla_gpu_enable_command_buffer="` (CUDA-graph instantiation of the huge backward
  program eats device memory at load — same mitigation run_perf.sbatch uses) + MEM_FRACTION=0.80.
  Still failing: seg=-1 (O(√N) two-level) executable does not load even so — the nested-scan
  backward program is bigger still; workaround unknown, single data point, don't over-generalize.
  Grad-bench forward compile via integrate() is ~280-340 s (vs 25 s for run_steps_sharded) — the
  dense path compiles much slower; budget accordingly.

- **[bench/paper§5] dars timing-validity A/B (job 26046701, dars_ab.26046701.log): the historical
  dt=180 bench state IS non-finite by step 25 (117.8M NaN in T — the DARS_INSTABILITY_FINDING
  suspicion, now measured), but it does NOT invalidate the timings: dt=60 verified-finite re-time =
  504.9 ms/step vs 521.2 on the blown-up state (3%), and the production-like zstar+wsplit+mEVP(+KPP)
  at dt=120 = 522.5 ms/step (4%, peak 26.9 vs 22.1 GiB).** So per-step cost is insensitive to state
  health at this scale and the paper's dars rows stand; SYPD however must use dt_prod=120 (w_split),
  not 240 — fixed in paper_jax reduce_scaling/make_numbers (dars 1.28→0.64 SYPD, 300→600 GPU-h/yr).
  Bonus finding from the same batch: forca20/dist_32 with KPP compiles+runs fine (326 ms/step, finite)
  ⇒ the production forca20 dist_32 compile-hang is TKE-multinode-specific, not partition-specific;
  and NG5 anti-scales at 128 GPU (763.7 vs 583.8 ms at 64) — the JAX communication wall, while
  Kokkos-CUDA still improves there. All new benches now print a [bench-finite] gate line.

- **[bench/paper§5] dars confirms the 128-GPU communication wall (jobs 26047719/26047720, dt=60
  verified-finite): 283.3 ms/step at 64 GPUs (32→64 doubling keeps only 56% efficiency) but
  423.3 ms at 128 — anti-scaling, exactly like NG5 (583.8→763.7 ms), while Kokkos-CUDA still
  improves on both meshes at 128 (dars 236.5→210.7 ms, NG5 491.6→373.8 ms).** Two meshes of very
  different size (3.16M vs 7.4M nodes) now turn over at the same GPU count, i.e. the wall tracks
  the per-GPU workload (~25–58k surface nodes/GPU) — the same halo-exchange wall CORE2 hits at
  8 GPUs. Paper updated (fig10 + §5 strong-scaling + §7); dars SYPD peaks at 64 GPUs (1.16).

- **[fidelity/paper§4] CORE2 v3 hindcast COMPLETE (2026-07-05, 63/63 yr) and the zstar freshwater
  fix is confirmed at full length: vol-mean S drift = +5×10⁻⁵ psu over 62 compared years (Fortran
  reference: +2×10⁻⁵ — same order; pre-fix JAX was +4×10⁻³, i.e. 100× down), end-state vol-mean T
  gap 0.0001 °C, OHC gap 0.5 ZJ.** Mean state 1980–2009: SST RMSE vs PHC 0.605 (JAX) / 0.606
  (Fortran) with |JAX−Fortran| RMS 0.004 °C; SSS 0.380/0.380/0.002 psu — the port-vs-original
  difference is two orders below the shared model-vs-obs bias. NOTE the analysis cap: the user
  distrusts the final (2020) JRA55-do forcing year, so ALL comparisons end at 2019
  (paper_jax paths.LAST_COMPARE_YEAR; the paper says 1958–2019/62-year/744 months). Formatting
  gotcha: post-fix drifts are ~1e-5, so any `%.4f` annotation renders "0.0000" — the fig04/macros
  now use scientific notation, and the salinity panel uses a fixed ±0.01 psu window instead of
  matplotlib's offset-notation autoscale (unreadable per user feedback).

- **[fidelity/paper§4] Fortran reference runs write what their namelist says and nothing more:
  the 3-yr FORCA20 reference (fesom2_forca20) has YEARLY MEANS ONLY, so a "show me a snapshot"
  figure required re-integrating January 1959 in a copy dir (fesom2_forca20_snap) from the
  end-of-1958 restart with daily sst/ssh/unod/vnod.** Recipe that worked: copy namelists, cut
  io_list to the needed fields at 1,'d', run_length=1,'m', hand-write fesom.clock
  ("86160 365 1958 / 0.0 1 1959"), and SYMLINK the 253 GB fesom.1958.{oce,ice}.restart dirs
  (read-only use — verified "restart from record 12 of 12" through the symlink; never copy, never
  point ResultPath at the original). Cost: ~8 min init + ~1.5 min/simulated day on the original
  32-node/4096-rank layout, ~half of it the daily 3-D netCDF writes. When planning reference runs,
  decide the snapshot-output question BEFORE the run, not after.

- **[usability] External input-data paths are now resolved in ONE place (`fesom_jax/paths.py`),
  never hardcoded in the reader that opens them: explicit arg / run-YAML `forcing:` key → env var
  → Levante default.** The six inputs (JRA55-do forcing, PHC IC, SSS restoring, runoff, chl, mesh/
  partition root — plus the repo-relative cached-IC dir) each have an env var (`FESOM_JRA_DIR`,
  `FESOM_PHC_PATH`, `FESOM_SSS_PATH`, `FESOM_RUNOFF_PATH`, `FESOM_CHL_PATH`, `FESOM_MESH_ROOT`,
  `FESOM_IC_DIR`) and the four forcing files also a YAML key inside `forcing:` (strict-keyed like
  every other config block). Two rules that made this a no-risk refactor: (1) the env is read at
  CALL time (`paths.resolve`), not at import, so exporting a var in a job script after
  `import fesom_jax` still works; (2) all four `build_surface_forcing` path kwargs became `None`
  ("resolve yourself") instead of an import-time default, so the default path on Levante is
  byte-identical while a caller can override any single one. Keep the legacy `DEFAULT_*` module
  names alive as import-time snapshots — test gates (`Path(jra55.DEFAULT_JRA_DIR).is_dir()`) and
  scripts import them. `paths.require()` fails with the env var, the YAML key, and `docs/DATA.md`
  named in the message: an actionable error is the difference between "unusable off Levante" and
  "usable off Levante" far more than the plumbing itself.

- **[usability][mesh] The CORE2 mesh on `/pool` is NOT immutable — its level files were silently
  regenerated under us.** `nlvls.out`/`elvls.out` (the per-node/per-element level counts) were
  rewritten upstream on **2026-07-03**, a month after this project's C mesh export (2026-06-06).
  They differ at exactly **2 nodes and 4 elements**, all in the Ross Sea (~154 °W, 77 °S), where the
  newer files are shallower (580 m → 280 m at the worst point). Everything fesom-jax has ever
  produced — the 63-yr hindcast, every paper figure — used the OLD levels, and because `/pool`
  overwrote them, **our dense `.npy` export is now the only surviving copy**. Three consequences,
  each of which cost real time to work out:
  (1) `test_prepare_mesh.py` had been failing since 2026-07-03 and looked like a code regression —
  it was not: it derives a mesh from today's `/pool` and diffs it against an export built from the
  old one. Fixed by *pinning* the level files it feeds `prepare_mesh` to the ones the export
  encodes, plus a new explicit tripwire (`test_upstream_levels_delta_is_as_documented`) that
  asserts the divergence is exactly the known 2/4 points — so the next upstream edit fails loudly
  instead of silently changing the bathymetry.
  (2) The Zenodo data package must ship the OLD levels (reconstructed from the `.npy` export, not
  copied from `/pool`), or the published package would not reproduce the published results.
  (3) Both versions are structurally valid — the FESOM invariant (no element deeper than its
  shallowest node) holds for each — so nothing "detects" the swap for you. **Lesson: a mesh under
  `/pool` is a dependency with no version number. Check `ls -l` mtimes on the raw `.out` files
  before trusting that a reference run and a fresh export share a bathymetry.**

- **[usability] The pi mesh (5 MB) now ships INSIDE the package** (`fesom_jax/data/mesh_pi`,
  `DEFAULT_PI_MESH_DIR`, `package-data` in `pyproject.toml`). Before this, `data/` was a gitignored
  symlink to `/work`, so a fresh `git clone` had **zero** meshes and the README's quick-start failed
  on its first line for everyone except the author — while the README claimed the CORE2 mesh was
  "small, in-repo" (it is 204 MB, on `/work`). 5 MB of committed mesh is the difference between a
  repo that only its author can run and one that `pip install` + a laptop CPU can run in 2 minutes
  (forward model *and* gradient — `examples/01_pi_quickstart.ipynb`). If a model's data does not
  ship, its documentation is fiction; test the quick-start as a stranger would, from a clean clone.

- **[usability] A module named after the wrong thing is a bug in the documentation layer**
  (`core2_forcing.py` → `surface_forcing.py`; the builder and the dataclass shed their `Core…`
  names for `build_surface_forcing` / `SurfaceForcing` — spelled out in the shim's docstring, and
  deliberately not repeated here so that grepping the tree for the old symbols stays a clean
  "are we done?" gate). The old name was wrong twice: it named a **mesh** (CORE2)
  although the module is entirely mesh-agnostic (pi/farc/dars/CORE2/NG5), and "CORE2 forcing" is a
  real but *different* dataset (CORE-II, Large & Yeager) while this driver reads **JRA55-do**. To an
  ocean modeller that is not a cosmetic wart — it asserts a false fact about the science. Renamed to
  say what it is: the *driver* that assembles per-step surface forcing from the input datasets, as
  distinct from `forcing.py`, which holds the differentiable *kernels* (L&Y09 bulk, wind stress) it
  calls. Two things made the rename safe to do on a live tree: (1) a **deprecation shim** at the old
  path re-exporting the new module — old names bound to the *same* objects, plus a PEP-562
  `__getattr__` fallback — so the long-running SLURM chains keep importing `core2_forcing` unchanged
  while emitting a `DeprecationWarning`; (2) `\bcore2_forcing\b` as the migration regex, since `_` is
  a word character, so the word boundary **automatically** spared the three lookalikes that must NOT
  move: the published Zenodo archive `core2_forcing_1958.zip` (renaming it breaks `fetch_data.py`
  against already-uploaded files), the `core2_forcing_measure.py` benchmark filename, and the
  `core2_forcing_missing` pytest gate (which really is about CORE2 *fixture data*). **Lesson: when a
  rename is mostly `sed`, the risk is not the code — it is the identifiers that merely look like the
  one you are renaming. Prove it: invert the substitution across the tree and diff against `HEAD`;
  every file must round-trip to a byte-identical original, or the diff is not the pure rename you
  claimed.** (Ironically `CORE2_runoff.nc` — a real file on disk, inside the JRA55-do directory —
  keeps its name; not every "core2" in the forcing path is a mistake.)
