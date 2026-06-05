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
