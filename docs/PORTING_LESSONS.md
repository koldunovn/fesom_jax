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
