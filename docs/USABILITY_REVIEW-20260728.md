# fesom-jax: a three-system usability review

*(External review received 2026-08-03; conducted 27–28 July 2026 against `main@0383a17`.
Fixes are tracked on branch `fix/usability-review` — see the git log for per-finding status.
Companion document: `CODE_REVIEW-20260703.md`, the static numerics audit.)*

**Repository:** <https://github.com/koldunovn/fesom_jax> @ `0383a17` · **Reviewed:** 27–28 July 2026

## 1. Introduction

`fesom-jax` ports the FESOM2 unstructured-grid ocean and sea-ice model to JAX, differentiable
end to end, for hybrid machine-learning work. Its numerics were already audited statically
(`docs/CODE_REVIEW-20260703.md`); what had not been done was running it as an outside user.

We installed and ran it independently on three systems:

| System | Hardware | Coverage |
|---|---|---|
| **Levante** (DKRZ) | 4× A100-80GB **NVLink** | full: 1-yr CORE2, config matrix, restarts, gradients |
| **Albedo** (AWI) | 4× A100-80GB **PCIe** | full, plus GPU test suite and scaling |
| **MacBook** M3 Pro | CPU only, no dataset | install, docs, pi-mesh path, test suite |

**Goal.** Follow the project's own documentation, run a common set of setups, judge whether
the output is a physically plausible ocean, and record every problem the documentation did not
prepare us for.

**Plan.** Install per `README.md`; fetch the published data; run the pi-mesh quick-start, the
three notebooks, a short and then a one-year CORE2 integration, a physics-configuration
matrix, restart portability across device counts, gradients at CORE2 scale, and the project's
test suite.

**Headline.** The model works — both HPCs integrated a full simulated year of the 1° global
ocean with no non-finite values and a correct seasonal cycle, and every advertised physics
configuration ran. **All blocking problems are in packaging and documentation, none in the
numerics.**

## 2. Methods

Each system was reviewed independently by an agent (Claude Code) working from the project's
documentation and a short common brief, without sight of the other systems' results; findings
were merged afterwards. Agreement across three machines separates genuine defects from local
accidents, and the disagreements proved as informative as the agreements. Physical
plausibility was judged against observations (global mean SST ≈18 °C, SSS ≈34.7 psu, Arctic
ice maximum in March and Antarctic in September, deep mixed layers in the Labrador and
Greenland Seas), using an equation of state written independently of the model's own.

## 3. Results

### 3.1 Reproduction across systems

| # | Problem | Levante | Albedo | Laptop |
|---|---|---|---|---|
| **P1** | `pi_demo.yaml` cannot run through its documented driver | ✔ | ✔ | ✔ |
| **P2** | `forcing: null` silently ignored | ✔ | ✔ | ✔ |
| **P3** | `--ic-dir` relative default overrides `$FESOM_IC_DIR` | ✔ | ✔ | ✔ |
| **P4** | Installed versions ≠ `docs/ENV.md` | ✔ | ✔ | ✔ |
| **P5** | Verification suite largely inert off Levante | ✔ | ✔ | ✔ |
| **P6** | Notebook 03 not self-contained | ✔ | ✔ | ✔ |
| **P7** | CORE2 `jax.grad` exhausts an 80 GB GPU | ✔ | ✔ | n/a |
| **P8** | `ice: null` produces sub-freezing seawater | ✔ | ✔ | n/a |
| **P9** | Data fetcher pulls a version without partitions | ✗ | ✔ | n/a |
| **P10** | Performance table holds only on NVLink | ✗ (0.092 s/step) | ✔ (0.659 s/step) | n/a |
| **P11** | Bit-identity "standing invariant" fails on GPU | untested | ✔ | ✗ (CPU) |
| **P12** | JAX aborts on a 256-core login node | ✗ | ✔ | ✗ |

Six problems reproduced everywhere; those are unambiguous defects. Two of the
system-specific ones deserve comment.

**P9 was invisible on Levante for a methodological reason:** Levante used the site's local
copies of the input data rather than the project's `fetch_data.py`, so the broken fetcher was
never exercised — precisely the failure mode of testing on the development machine.

**P10 is a hardware dependence, not a discrepancy.** Levante *confirmed* the README's
performance claim; Albedo measured 7.6× slower. Both are correct, and the deciding factor —
the GPU interconnect — is absent from the published table (§3.3).

### 3.2 Blocking problems

**P1 · The first command in the Quick start does not run.** `pi_demo.yaml` is the only
configuration advertised as needing no data, and the README calls the config-driven run "the
release path". It fails three times over: `mesh: null` is documented but unimplemented
(`load_mesh` has a pi default that `run_from_config.py:171` defeats by passing `None`
explicitly); the cold start is hardcoded to the CORE2 initial condition although the package
ships `data/ic_pi/`; and the forcing build is unconditional. `grep -rn "pi_demo" fesom_jax/`
returns nothing — the config has no test coverage.
*Fix:* fall back to `DEFAULT_PI_MESH_DIR` when `cfg.mesh` is `None`; default `--ic-dir` to the
packaged `ic_pi` for the pi mesh; skip `build_surface_forcing` when `cfg.forcing is None`; add
a test that runs `pi_demo.yaml`.

**P2 · `forcing: null` is not merely ignored — following the error message changes the
physics.** The documented rule is `null = that physics OFF`, but `run_from_config.py:199`
calls `build_surface_forcing` unconditionally, reinterpreting `null` as "no path override" and
falling through to the Levante default path. Albedo measured the consequence of the suggested
workaround: a **1.65 °C SST divergence**. A silently wrong run is worse than a crash.
*Fix:* branch on `cfg.forcing is None` and pass a zero forcing stack; `run.py:398` needs the
same, as it currently raises rather than supporting an unforced path.

**P9 · The data fetcher downloads the wrong dataset version.** `fetch_data.py` hardcodes
Zenodo record `21324319` (v1.0), which has no `core2_partitions.zip`, and exits 1 *before* the
10.4 GB forcing is fetched — leaving the user with 370 MB and no way forward. Record
`21328211` (v1.1) contains the partitions.
*Fix:* point `ZENODO_RECORD` at the concept DOI so it always resolves to the newest version,
and fetch the large archive first.

**P7 · The advertised gradient does not fit at CORE2 scale.** This is the most consequential
finding, and it needs the mechanism spelled out.

Differentiability is the project's reason for existing: `jax.grad` answers "how much does
this ocean metric change if I adjust this physics parameter?" exactly and in one pass, which
is what makes sensitivity maps, parameter calibration and trained ML closures possible.
Reverse-mode differentiation does this in two passes — forward, then backward — and the
backward pass needs the intermediate values from the forward one. **They must be held in
memory, so cost grows with the number of timesteps differentiated through.** A CORE2 step is
expensive to store: 126 858 nodes × 48 levels, and within each step sixteen substeps
including KPP, GM/Redi, FCT advection, a conjugate-gradient SSH solve and a 120-iteration
sea-ice loop.

README §2 demonstrates the gradient on the 3 140-node `pi` mesh and states it "is identical at
CORE2 scale, just with that mesh and `dt`". CORE2 is 40× larger. Measured on an 80 GB A100:
Albedo **~37.7 GiB per step**, so **2 steps** fit; Levante 1–2 depending on physics. At
dt = 1800 s that is **one hour of simulated ocean**.

The consequence is a scissors. The same README states the gradient is meaningful over
"hours→days" — shorter and the signal has not developed, longer and chaos destroys it. So the
window physics requires and the window memory permits **barely overlap**, and every capability
advertised in §Differentiable capabilities — mixed-layer sensitivity, `k_gm`/`c_k` calibration,
the learned TKE closure — sits in the gap.

A remedy exists and works: `integrate(remat_segments=N)` stores every N-th step and recomputes
the rest, trading compute for memory. Levante ran a 6-step window with `remat_segments=2`, and
the laptop confirmed at pi scale that it returns a **bit-identical** gradient — not an
approximation. But `remat_segments` and `remat_blocks` appear **nowhere in the README**, so a
user following the documentation hits the wall with no indication that a door exists beside it.
*Fix:* replace "identical at CORE2 scale" with the measured scaling, and document
`remat_segments` beside the §2 example — including that it is not monotonic (`=2` worked on
Levante where `=3` did not), so guidance on choosing it is needed.

**P12 · JAX aborts on a many-core login node.** Running the README's quick start — the pi
mesh, no data, the first thing anyone does — the process does not merely warn but **aborts**,
during XLA compilation, with a C++ stack trace:

```
F0728 env.cc:93] Check failed: ret == 0 (11 vs. 0)
   Thread tf_foreach creation via pthread_create() failed.
```

The mechanism: Albedo's login node reports **256 cores**, and XLA's CPU backend sizes its
compile-time thread pools from that count. The per-user process limit is `ulimit -u 2048`, so
the pools overshoot it, `pthread_create` returns EAGAIN, and XLA treats a failed thread
creation as a fatal `CHECK` rather than degrading to fewer threads.

This is XLA plus site configuration rather than a fesom-jax bug, but the documentation is
where it should be caught. `README.md` and `docs/ENV.md` prepare a user for exactly one
login-node symptom — the benign `cuInit(0) / CUDA error 303` fallback — so an abort with an
unfamiliar stack trace reads as a broken installation, and the natural response is to
reinstall, which changes nothing. Any fat login node is a candidate.
*Fix:* one line in `docs/ENV.md` naming the symptom and the workaround, `taskset -c 0-15
python …`, which restricts the visible core count and runs normally.

### 3.3 Documentation that misleads

**P10 · The performance table has an unstated hardware precondition.** README §Performance
gives CORE2 on one node (4 GPU) as **0.087 s/step, 56.9 SYPD** — simulated years per day of
wall clock, the number an ocean modeller uses to plan an experiment.

| | measured | vs claim |
|---|---|---|
| Levante (A100 **NVLink**) | 0.092 s/step | within 6 % — **claim confirmed** |
| Albedo (A100 **PCIe**) | 0.659 s/step | **7.6× slower** — ~7.5 SYPD, not 56.9 |

Both measurements are correct; the difference is the interconnect. The model exchanges halo
data between GPUs at **every timestep**, so inter-GPU bandwidth is on the critical path.
NVLink runs at hundreds of GB/s between devices, PCIe at an order of magnitude less. The
table's caption says only "A100", which names the compute and omits the thing that actually
decides the number.

The practical damage is twofold. Someone sizing a campaign from the table budgets a day of
wall clock for what will take a week. And Albedo found CORE2 *anti-scales* on PCIe — **4 GPUs
were 2.1× slower than 1** — so the shipped `partition: dist_4` default is not merely
suboptimal there but actively worse than running on a single GPU. A user trusting the defaults
gets the worst available configuration and no signal that anything is wrong.
*Fix:* state the interconnect in the Performance table and publish `nvidia-smi topo -m`
alongside the SYPD numbers; and reconsider the default partition, which is a free ~2× for
everyone not on NVLink.

**P11 · The "standing invariant" fails on GPU.** `docs/USER_GUIDE.md` §8 guarantees
unconditionally that `RunConfig.defaults()` is bit-identical to the bare model. Same commit,
same environment: **551 passed / 0 failed on CPU, 548 passed / 9 failed on GPU** — the nine
being exactness gates, including `test_defaults_bit_identical_step`, the invariant itself.
*Fix:* qualify the guarantee as CPU-only, or relax the GPU gates to a stated tolerance. An
invariant that fails on the production backend is worse than a documented approximation.

**P8 · `ice: null` produces impossible seawater, silently.** Sea ice is presented as an
ordinary toggle and notebook 03 invites running without it. On a forced global run it drives
the polar ocean far below freezing — Levante measured −24.6 °C over 15 million km² (22 % of
ocean nodes), Albedo −12.8 °C — while the run exits 0, produces no NaNs, and `--diagnostics`
reports `FINITE`. This is inherited FESOM behaviour, not a porting defect, but nothing warns
the user.
*Fix:* warn when a non-trivial ocean area sits below freezing; mark `ice: null` as a
diagnostic rather than a supported physics setting.

**P5 · Most of the verification suite cannot run outside Levante.** 54 of 78 test files read
unpublished Fortran/C reference dumps; the laptop counted 334 skips, 116 of them needing those
dumps, and one test hardcodes an absolute Levante path. The suite still exits green, so a
newcomer concludes the model is verified locally when the gates never ran. The laptop also
found three sharded files that **exit 0 having run zero tests**.
*Fix:* print a closing summary ("N gates skipped: no reference data") and state what fraction
of the suite an external user can run.

**P4 · Installed versions drift from the documented set.** Unbounded pins (`jax>=0.4.34`) gave
JAX 0.11.0 against a recorded 0.10.1 on all three systems. Nothing broke — Levante ran 836
tests and a full model year on it, and an A/B against 0.10.1 showed no performance difference.
It matters because several caveats in the docs are version-specific.
*Fix:* ship a constraints file reproducing the recorded set; date-stamp those caveats.

**P6 · Notebook 03 is not self-contained.** Its multi-device section needs
`XLA_FLAGS=--xla_force_host_platform_device_count=4` set *before* the kernel starts; the
notebook explains this in a markdown cell placed *after* the code that needs it, so under
`nbconvert --execute` it fails with an XLA broadcasting error mentioning neither devices nor
partitions. With the flag set it passes and reproduces its own claim (max |ΔT| = 2.4e-10 °C).
*Fix:* set the variable in the first code cell, or raise an error naming the device count.

**Restarts — portable, but the tolerance is overstated.** Restarts written on 4 GPUs resumed
on 4, 2 and 1 on both HPCs; the capability is real. But divergence exceeds what the docs
imply: Albedo measured median 3.3e-9 °C and maximum 0.315 °C against a documented "~1e-12"
floor and a "byte-identical" claim elsewhere; Levante saw rms ~1e-3 °C. Both are consistent
with chaotic divergence seeded by reduction order — a documentation problem, not a defect.
*Fix:* state the realistic tolerance and mechanism; drop "byte-identical" for cross-device
restarts.

### 3.4 Minor issues

| | Problem | Fix |
|---|---|---|
| P3 | `--ic-dir` default is relative and always overrides `$FESOM_IC_DIR`, whose absence the error message then blames | `default=None`; let `paths.require` resolve it |
| P13 | `pip install -e ".[cuda,dev,viz]"` installs no Jupyter, so the documented notebooks cannot run | add jupyter to the `viz` extra |
| P14 | README's `mamba activate` exits 0 without activating | use `conda activate`, as `USER_GUIDE.md` does |
| P15 | README §2b calls `window_mean_mld_field`, absent from the repository | define it or mark the block illustrative |
| P16 | README §2 prints `k_gm`/`redi_kmax` as "eddy sensitivities"; both are structurally 0.0 since GM is off there | annotate `[OFF in this run]`, as notebook 01 does |
| P17 | Output is written float32 although the model integrates in float64 | document it; offer float64 for analysis |
| P18 | `--progress` prints raw `perf_counter()`, so its first line is node uptime, not setup time | subtract the start time, as the neighbouring `_lap` does |
| P19 | Sharded tests need ~2–3× the walltime `run_suite_shard.sbatch` allocates | raise it; state expected runtime |
| P20 | Every SLURM directive and path in the repo is Levante-specific | note that the scripts are examples, not portable |
| P21 | A wrong Jupyter kernel surfaces as `ModuleNotFoundError: No module named 'jax'` | register a named kernel; pass `--ExecutePreprocessor.kernel_name` |

### 3.5 What worked

On both HPCs a full simulated year of CORE2 ran clean: Arctic and Antarctic ice correctly
antiphased with realistic magnitudes (NH maximum ~17×10⁶ km² in March, minimum ~7.8 in
September), deep mixed layers in the Labrador, Irminger and Greenland Seas, global mean SST
18.1 °C and SSS 34.60 psu, no non-finite values. All eight physics configurations ran.
Forward- and reverse-mode gradients agreed to 2×10⁻¹⁶, and four devices reproduced one to
2.4×10⁻¹⁰ °C. The physics-gating error messages are unusually good, naming the reason, the
physics and the fix.

## 4. Conclusion

The model is in good shape; its packaging is not. Six problems reproduced on all three systems
and every one lies in the on-ramp — the documented first command, the data fetcher, the
configuration driver, the recorded environment. The most consequential is **P7**:
differentiability is the project's reason for existing, and the documented path walls off at
one or two timesteps while the option that relieves it goes unmentioned.

The two system-specific results argue for the method. **P9** — a data fetcher that cannot
fetch the data — was invisible on the machine where the code was developed, because that
machine did not need it. **P10** shows a headline performance claim exactly right on one
cluster and 7.6× optimistic on another, with the deciding factor absent from the table.
Neither would have surfaced from testing more thoroughly in one place.

**Priorities:** fix the fetcher (P9), since without it nobody obtains the data; fix
`forcing: null` (P2), since a silently wrong run is worse than a crash; and correct the CORE2
gradient claim (P7).
