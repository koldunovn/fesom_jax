# Adjoint guide: what to use, what is verified, what is open

State of 2026-11-16 (`main` a46150b). Start here. The other adjoint documents are design notes and history,
and they stop around 2026-10-11:

- `ADJOINT_HORIZON.md`: why the adjoint grows, and the first freezes;
- `ADJOINT_API.md`: the API and the presets up to `long_window_combined`;
- `ADJOINT_TKE.md`: the TKE diagnosis;
- `LIMITER_GRADIENTS.md`: the FCT limiter decision.

Everything after that date is described in this guide, in the docstrings of `AdjointConfig`
(`fesom_jax/adjoint_stab.py`) and `IceConfig` (`fesom_jax/ice.py`), and in `PORTING_LESSONS.md`. The lesson
entries behind each number are listed in §9.

All ratios below are adjoint gradient / central finite difference (FD) of the same model, at the FD step sizes
given in the lesson entries.

## 1. Why the defaults do not survive long windows

`AdjointConfig()` differentiates every term. `IceConfig()` has `adjoint_mode="exact"` and
`adjoint_damping="none"`, so the rheology's momentum subcycles (EVP by default, `whichEVP=0`; mEVP with
`whichEVP=1`) are unrolled exactly. Both are correct derivatives of the discrete model, and neither is usable
over long windows:

- **Ocean:** the turbulence closure's response to the ocean state makes the reverse-pass norm grow by 1.21-1.55
  per 1800 s step. On a GPU the result is not even reproducible: a 1-day `exact()` gradient took values of both
  signs in five identical sweeps, because the forward's atomic scatter-adds differ at round-off and the
  backward amplifies the difference.
- **Sea ice:** the exact mEVP tangent has a near-rigid stress mode in a few elements. It flips sign every
  subcycle with |λ| 6-18, and the result is NaN.

Every switch below acts in the backward pass only (`stop_gradient` of the forward value, or a custom VJP).
The forward stays bitwise unchanged, and with all switches off the traced program is the plain model. Each
switch removes one derivative path, so it is a choice about which sensitivity to give up. That is why every
setting below comes with an FD check.

## 2. Status by mesh, cost and window

**Abbreviations used in the table**
- **"drift ice":** `IceConfig(whichEVP=1, adjoint_mode="exact", adjoint_damping="frozen_sweep")` with the
  default `adjoint_theta=0.97` and `adjoint_alpha_kappa=1`.
- **"year ice":** drift ice plus `adjoint_sweep_guard=10`, `adjoint_thin_mass=100` and
  `adjoint_cuts=("fct_flux", "strength_conc")`.
- **Ice costs** use the Arctic drift box unless the row says otherwise.
- **MLD costs** use the window mean over box `natl_subpolar` (50-60N, 40-20W).
- **The TKE constants** are ln c_k, ln c_eps, ln cd and ln alpha, plus ln k_gm where listed.

### CORE2 (1°, dt 1800 s, double precision, one GPU unless said)

| cost and controls | window | ocean `AdjointConfig` | ice | adjoint / FD |
|---|---|---|---|---|
| ice drift; ln P*, ln C_d | 2-30 d (Jan, Jul; Mar to 10 d) | `long_window()` | drift ice | P* 0.95-1.02, C_d 0.99-1.00 |
| ice drift; ln P*, ln C_d | 90 d, Jan 2013 | `long_window()` + `freeze_fct_limiter` | drift ice | P* 1.04-1.05, C_d 0.97-0.99 |
| ice drift; ln P*, ln C_d | 180 / 365 d, Jan 2013 | `long_window()` + `freeze_fct_limiter` | drift ice | P* 1.03 / 1.05, C_d 0.96 / 0.97-0.98 |
| MLD; four TKE constants | 30, 60, 90 d, Jan and Jul 2013 | `long_window_tke()` | prescribed | all within 7.7 %; July box T within 2.2 %; **January box T fails** (11-500 %) |
| ice drift, MLD and box T together, GM live | 30-90 d | `long_window_combined()` | drift ice | ice within 3.1 % (Antarctic P* 8.2 %); MLD within 8.4 %; January box T within 5.1 % at 90 d, **fails at 30 and 60 d** |
| MLD; four TKE constants + k_gm | 365 d | `year_window_combined()` ("Y3") | year ice | Jan 2013: 2.9-5.8 %. Jul 2013: cd +15 %, alpha +14 %. Jul 2007: alpha −9.5 %, k_gm −12 % (the others within 8 %) |
| same | 180 d (Jan 2013, Jul 2013, Jul 2007) | Y3 + `tke_pr_ice_cut=0.15` ("Y4") | year ice | within 4.1 % / 3.1 % / 8.3 %; **Jan 1995 bursts** (§7) |
| same | 365 d, Jan and Jul 2013 | Y4 + `tke_ml_raw_cut=-1e-7` ("Y4raw7") | year ice | Jan: c_k +2.7, c_eps −2.9, cd +0.2, alpha −0.2, k_gm +7.0 %. Jul: +7.6, −8.0, +9.0, +2.1, −4.4 %. These are the best one-year numbers so far; measured on the fast sharded path (§5) against that configuration's own FD |

Y4 and Y4raw7 are not presets. Build them with `_replace` or with a `--setting` spec (§3).

### farc (4 km Arctic, 638k nodes, dt 900 s, sharded over 4 GPUs)

| cost and controls | window | ocean | ice | adjoint / FD |
|---|---|---|---|---|
| ice drift; ln P*, ln C_d | 30 d to 1 yr from Jan 2013 | `long_window()` + `freeze_fct_limiter` | year ice with **θ 0.95** and `adjoint_sweep_guard=1e3` | P* 1.01-1.03; C_d 0.977 at 30 d, then a constant 0.90-0.91 at 90, 180 and 365 d |
| NH ice volume; P*, C_d, atmospheric forcing (`Params.forcing_adj`) | 1-90 d from 2 Jan 2013 | "ml6": the `long_window_tke` masks with `tke_pr_cut` and `tke_ml_cut` at 1e-6 | θ 0.95, thin mass 100, guard; cuts `fct_pin` + `fct_bound_coast` (no `strength_conc`) | 30 d: P* 1.09, C_d 1.10. 90 d: P* 1.15, C_d 1.04. Air temperature 0.99-1.01. **Wind controls unreliable** (u_air 0.85-1.85) |
| same | 151 d, 2 Jan to 1 Jun 2012 | ml6 + `tke_floor_cut`, later `freeze_mixing` (see the lesson entry) | as above + `adjoint_warm_ts` | **open**: a steady spurious P* accrual in late May (§7) |

About farc:
- **θ 0.97 blows up at 30-90 days on farc**, although it is the CORE2 default. The 4 km mesh has less margin,
  and θ 0.95 is its limit.
- **The C_d shortfall on farc is the sum of what the cuts cost**: about 3-4 points for `strength_conc`, and
  1.4 points for going from θ 0.9 to 0.95. It does not grow with the window, so it is not an instability.

## 3. Recipes

```python
from fesom_jax.adjoint_stab import AdjointConfig
from fesom_jax.ice import IceConfig

# CORE2, 30-90 days, everything live (TKE constants, GM, sea-ice parameters):
adj = AdjointConfig.long_window_combined()
ice = IceConfig(whichEVP=1, adjoint_mode="exact", adjoint_damping="frozen_sweep")

# CORE2, one year (best measured: "Y4raw7"):
adj = AdjointConfig.year_window_combined()._replace(tke_pr_ice_cut=0.15, tke_ml_raw_cut=-1e-7)
ice = IceConfig(whichEVP=1, adjoint_mode="exact", adjoint_damping="frozen_sweep",
                adjoint_sweep_guard=10, adjoint_thin_mass=100,
                adjoint_cuts=("fct_flux", "strength_conc"))

# farc, sea-ice drift, up to a year:
adj = AdjointConfig.long_window()._replace(freeze_fct_limiter=True)
ice = IceConfig(whichEVP=1, adjoint_mode="exact", adjoint_damping="frozen_sweep",
                adjoint_theta=0.95, adjoint_sweep_guard=1e3, adjoint_thin_mass=100,
                adjoint_cuts=("fct_flux", "strength_conc"))
```

Pass `adjoint_cfg=adj, ice_cfg=ice` to `step` / `integrate`, and run the window with
`fesom_jax.adjoint.chunked_grad`. It parks chunk-boundary states on the host, so device memory stays at one
chunk, and it returns the cotangent norm at every boundary, also per field. `chunked_grad_multi` runs several
costs over one forward.

**From the command line.** The CORE2 / farc drivers take the same settings:
- `scripts/adjoint/core2_adj.py` (ocean costs);
- `scripts/adjoint/ice_cbar.py`, with actions `fwd`, `fdbatch`, `grad`, `tkegrad` and `cutcensus` (ice
  costs, and TKE constants with the ice live).

The ocean setting is a `--setting` spec parsed by `tke_diag.parse_setting`, for example Y3:

    tke_pr_cut=1e-6+tke_pr_ratio=100+tke_pr_raw_cut=-1e-6+tke_ml_cut=1e-8+freeze_fct_limiter+tke_ml_ratio=100+tke_ml_raw_cut=-1e-6+tke_pr_lat_cut=70

The ice setting is given with `--ice-mode exact --ice-damping frozen_sweep --ice-theta 0.97
--ice-alpha-kappa 1 --ice-sweep-guard 10 --ice-thin-mass 100 --ice-cuts fct_flux+strength_conc`.

**Cheaper ice options when the ice parameters themselves are not controls:**
- `adjoint_mode="rheology"` differentiates thermodynamics, advection and the ocean-ice coupling, with the ice
  velocity dark. It works for EVP and mEVP.
- `"prescribed"` gives the ice trajectory, with differentiable fluxes.
- `"frozen"` freezes the whole ice step.
- If only C_d is needed: `adjoint_damping="local"` with `adjoint_kappa=0`.

**Time step:** use dt 1800 s on CORE2 for the adjoint. At dt 2700 the 30-45-day checks pass, but a one-year
reverse sweep grows ×10 per 15 days backwards from about day 255 (in the ocean: S, T, tke, w_e and velocity).

## 4. The switches

All switches are fields of `AdjointConfig` or `IceConfig`; the docstrings and comments there carry the measured
reason for each.

**`AdjointConfig`: freezes (bool)**

| switch | what the backward drops | status |
|---|---|---|
| `freeze_mixing` | the state dependence of Kv, Av | in `long_window()`. Kills the TKE-constant gradients |
| `freeze_gm_slope` | the N² dependence of the GM/Redi slope | in `long_window()`. Costs January box-T accuracy (33 % at 90 d) |
| `freeze_fct_limiter` | the Zalesak limiter factors of the ocean tracer FCT | in the combined presets. Removes salinity bursts on near-tied deep cells; ice gradients move ≤ 0.05 % |
| `freeze_stratification`, `freeze_convection`, `freeze_tke_state`, `freeze_vert_visc`, `freeze_k33`, `freeze_advection`, `freeze_mixing_state` | see the docstrings | diagnostic or special purpose. `freeze_pgf` is diagnostic only |

**`AdjointConfig`: masked TKE / GM cuts (None = off)**

| switch | what the backward drops | status |
|---|---|---|
| `tke_pr_cut`, `tke_pr_ratio`, `tke_pr_raw_cut` | the Prandtl-number derivative at near-neutral, smoothing-cancellation and inverted cells | in `long_window_tke()` |
| `tke_ml_cut`, `tke_ml_ratio`, `tke_ml_raw_cut` | the same masks on the mixing length's N² derivative | `tke_ml_cut` in `long_window_tke()`, the other two in `long_window_combined()`. Needed for GM live |
| `tke_pr_lat_cut` | the Prandtl derivative poleward of φ* | in `year_window_combined()` (70). **Valid only for costs far from φ***, never for Arctic costs |
| `tke_pr_ice_cut` | the Prandtl derivative under sea ice (concentration > a*) | Y4 (0.15) |
| `tke_pr_tu_cut` | the Prandtl derivative in the diffusive regime (Turner angle) | tried as the state-based alternative to the latitude cut. Fails a January MLD check (11-16 %) |
| `tke_floor_cut` | the `tke_old` derivative where TKE sits near its floor | needed for the farc 2012 volume window (1e-5) |
| `tke_n2_cut`, `tke_n2_band` | the buoyancy-N² derivative at near-neutral cells | opt-in. `tke.n2` everywhere spoils the 2013 gradients |
| `gm_n2_ratio`, `gm_n2_raw_cut`, `gm_slope_cut` | masked GM-input cuts, and a steep-slope cut | opt-in. Explored as alternatives to `freeze_gm_slope`; not adopted |
| `conv_surrogate` | adds a smoothed convective switch to the backward | opt-in, January box-T study, not adopted |
| `diag_cuts` | named diagnostic cut points (`KNOWN_CUTS`) | diagnosis only |

**`IceConfig` (derivative-only; `whichEVP=1` for the sweep options)**

| switch | what the backward drops | status |
|---|---|---|
| `adjoint_damping="frozen_sweep"`, `adjoint_theta`, `adjoint_alpha_kappa` | replaces the unrolled mEVP tangent by one sweep linearised at the final iterate, with raised pseudo-time coefficients and the plastic softening scaled by θ | the ice adjoint. θ 0.97 on CORE2, 0.95 on farc |
| `adjoint_sweep_guard` | a step's mEVP tangent when its transposed sweeps grow past G (the step is reported) | year windows. GPU round-off decides whether a near-rigid step bursts |
| `adjoint_thin_mass` | the mEVP load derivative on thin ice (kg/m²) | year windows. Polynya bursts on farc |
| `adjoint_cuts`: `fct_flux` | the ice FCT's antidiffusive correction (low-order tangent) | drift costs |
| `adjoint_cuts`: `strength_conc` | the concentration factor exp(−C(1−A)) of the ice strength | drift costs, year windows (summer pack loop) |
| `adjoint_cuts`: `fct_pin`, `fct_bound_coast` | limiter-bound derivatives at ties and on the coast and its first ring | the ice-volume cost, which needs the limiter derivative that `fct_flux` removes |
| `adjoint_cuts`: `fct_limiter`, `fct_bound`, `fct_hilo`, `fct_hilo_coast` | narrower or wider variants of the above | measured alternatives (see the `IceConfig` comments) |
| `adjoint_cuts`: `advect`, `fct`, `ssh`, `ocean_vel`, `ice2ocean`, `ocean_ts`, `water_flux` | single coupling paths | diagnostic |
| `adjoint_warm_ts` | the ocean surface T, S the ice thermodynamics sees, where T − T_f > ΔT | the farc 2012 volume window (0.2 K) |

## 5. Sharded and fast paths

There are two sharded adjoint chunks, and giving both to `core2_adj.build_model` raises an error.

- **`build_model(dist=N)`** (`integrate_sharded.sharded_chunk_program`) is the farc path.
  - Command line: `ice_cbar.py grad --dist 4`.
  - One farc step's VJP needs about 100 GiB on one device. Sharded over 4 GH200 it takes 42-52 GiB per GPU
    and 0.9-1.2 s per reverse step.
  - One node with 4 × A100 80 GB also works, with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`.
  - To plot a sharded cotangent, sum the owned and halo copies (`core2_adj.dense_cotangent`).
- **`build_model(devices=N)`** (`fesom_jax.adjoint_sharded`) is the CORE2 fast adjoint.
  - Command line: `--devices 4 --fast ice2+se+sp`.
  - It runs ice every second step, the split-explicit SSH (substeps follow dt: M 40 at dt 1800) and single
    precision.
  - A one-year gradient takes about 1.04 h on 4 GH200 (0.213 s per step, without compile).
  - Its 2-step gradient equals the dense one to 3e-9 to 5e-7 on CPU. Its year gradients pass against its own
    FD (the Y4raw7 row in §2).
  - Plan and checks: `docs/plans/20261022-core2-fast-adjoint.md`.

The dense float64 single-device step remains the reference. Check any other path against it before use.

## 6. The coupled ocean (`fesom_jax.coupled`)

- **`ocean_interval` is a forward-only, single-precision AOT executable.** For gradients, build the coupled
  interval from the dense `fesom_jax.step.step` in float64, with `jax.checkpoint` per step (and `remat_blocks`
  inside the step if memory is short).
- **The coupled ice block honours `IceConfig.adjoint_mode`, `adjoint_damping` and `adjoint_sweeps`**, as the
  stand-alone ice step does. Set them explicitly. The default exact mode makes the coupled reverse sweep grow
  ×1e18 per coupling interval.
- **`frozen_sweep` closes over the mesh in its `custom_vjp`.** The jitted ocean interval must close over the
  concrete mesh. Passing the mesh as a jit argument fails with "No constant handler for DynamicJaxprTracer".
- **The coupled flux block's reverse pass is finite since the melt-pond square roots got a "double where"
  (`_sqrt0` in `coupled/flux.py`).** Before that fix, sqrt'(0) at zero pond volume left an Inf residual on 94k
  of 127k CORE2 nodes, and the backward pass multiplied it by a zero cotangent.
- **General pattern:** a NaN that appears only in the reverse pass is a 0 × Inf on lanes a `jnp.where`
  discards. Sanitise the operand on those lanes, and keep the primal bitwise.

## 7. Open problems

1. **January box temperature (CORE2):** the window-mean box T from January misses the FD at 30 and 60 days
   with every setting tried. It is linked to the convective switch.
2. **July-start year windows with Y3:** cd and alpha come out 14-15 % too large, built up between July and
   January. Y4raw7 brings both under 10 %, but it has been measured only on the fast path and only for 2013
   starts.
3. **The 1995 Aegean event:** with Y4, windows from January 1995 burst at a deep shelf node (26.1E 36.6N,
   230-490 m, TKE at its floor). `tke_ml_raw_cut=-1e-7` removes it on a 30-day test bed, but the 180-day and
   365-day reruns are not done.
4. **The farc 2012 spring volume adjoint:** a steady spurious P* accrual in late May, about 20× the FD over
   8 days. It runs through u_ice → ice advection → thermodynamics, and none of the existing cuts fixes it
   without losing the real C_d response.
5. **Bursts from GPU round-off:** a year sweep on a GPU can contain a burst from round-off alone, which a
   repeat of the same run does not have. The sweep guard handles the mEVP case. For the ocean, read the
   per-field traces.

## 8. How to check a gradient

- **Repeat before reporting.** On a GPU, two processes differ at about 1e-4 relative from round-off alone.
- **Measure the FD noise first.** Restart the base from a shared boundary with nothing perturbed and compare.
  On farc in summer that noise reached 58 % of the P* FD at h 1e-2. The FD step must beat the noise, stay
  linear, and keep the perturbed forward alive.
- **Read the growth rate per step and the per-field traces** (`adjoint_stab.amplification`,
  `amplification_by_field`), not the scalar gradient alone. TKE dominates the total norm and hides T/S
  bursts.
- **Find where an error builds up with a windowed FD.** A forward resumed from boundary c with the control
  perturbed from there on gives exactly the gradient the reverse sweep has accumulated at c
  (`ice_cbar.py tkegrad --grad-trace`; `ice_cbar.py fwd` from a stored boundary). Use it when one window
  contains several faults.
- **A clean trace screens for bursts, not for a bias.** Every new cut gets a 1-day full-model check against
  the exact adjoint before any long window. A cut that is clean on the offline one-step map can still break the
  full model.
- **Localise a burst by dumping the cotangent and seeding with it** (`ice_cbar grad --dump-cot`). A mode on
  five nodes is invisible to power iteration from noise.
- **Count what a cut covers** before trusting it: `ice_cbar.py cutcensus` reports, forward-only, the fraction
  of interfaces each cut touches. Cuts that hit 33-46 % of a cost box's interfaces cost real sensitivity.
- **Use both checks:** dot tests (`adjoint_check.dot_test`) check the transpose; FD sweeps
  (`adjoint_check.fd_sweep`) check the linearisation.

## 9. Where the evidence is (`docs/PORTING_LESSONS.md`)

Search for these headings:

| topic | headings |
|---|---|
| TKE adjoint, C1-C7 | "(TKE adjoint, step C1)" to "(TKE adjoint, steps C6-C7)", 2026-10-06 to 2026-10-10 |
| Sea-ice adjoint on CORE2, I2-I25 | "(sea-ice adjoint, task I…)", 2026-10-08 to 2026-10-10 |
| Combined preset, GM live, January box T | "(combined adjoint, branch adjoint-combined / adjoint-gm4 / adjoint-gm5 …)", 2026-10-10 and 2026-10-11 |
| One-year preset | "(combined adjoint, branch adjoint-yr-y3) — the one-year preset", 2026-10-17 |
| farc ice drift to a year | "(sea-ice adjoint, task 26, farc)", 2026-10-11 to 2026-10-16 |
| farc ice volume | "(sea-ice adjoint, farc)", 2026-10-17 to 2026-10-20 |
| Sharded adjoint chunk on CORE2 | "(CORE2 fast adjoint, Task 5)", 2026-10-22 |
| Ice-parameter calibration (thermodynamic ice adjoint, L-BFGS, observation costs) | "(sea-ice calibration, task …)" and "(ice-parameter calibration, Task …)", 2026-10-10 to 2026-10-20 |
| Control transforms (raw, linear, log) | 2026-10-14 entries |
