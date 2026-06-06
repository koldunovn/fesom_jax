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
  remat suffices to N≥200. (`scripts/phase3_grad_memory.py`, GPU job 25378918, Task 3.1.)

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
  (`core2_forcing.ice_ic_aice`/`compute_surface_fluxes`, Task 5.6.)

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
  setting `water_flux=−flx_fw`; then `core2_forcing.compute_surface_fluxes`'s ice-on branch
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
  latter at high latitudes.* (`scripts/core2_stability_run.py`, Task 5.7.)

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
  (`scripts/core2_stability_gpu.sh`, Task 5.7.)

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
  shown to be STRONG (not mild) under real forcing.* (`scripts/core2_grad_gate.py` [1]/[4],
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
  (`scripts/core2_grad_gate.py` [3], job 25394380, Task 5.8.)

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
  (the C comment `:44-45` says so), and Phase-5 `core2_forcing` already taps `uvnode[:,0]` for
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
  call.* (`scripts/core2_ice_stability_run.py`, job 25396309, Task 6.7.)

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
  mixing seam) is well-conditioned separately.* (`scripts/core2_ice_grad_gate.py` [1]/[3],
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
