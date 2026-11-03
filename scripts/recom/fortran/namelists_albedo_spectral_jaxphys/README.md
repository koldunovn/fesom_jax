The physics-matched twin set of `../namelists_albedo_spectral/` (the spectral fork, 37 tracers) for the fork's
full-step gate G4b (`fesom_jax/tests/test_recom_step_fork.py`, `configs/albedo/pi_recom_spectral_g4b.yaml`): the
same REcoM setup, inputs and forcing with the physics brought as close to fesom-jax as the fork's FESOM2 base
(cda35117, 2025-10) allows. Changes from `../namelists_albedo_spectral/`:

| file | change | why |
|---|---|---|
| oce | `Fer_GM .false.`, `Redi .false.` (both `.true.`) | the fork's GM/Redi is the FESOM 1.4-style `init_Redi_GM` (no ODM95 taper, depth scaling or Redi tapering: the `oce_dyn` group has none of those keys), which fesom-jax does not have; off on both sides (`gm: null`). With both off `Ki` stays zero, so `diff_part_hor_redi` adds nothing |
| oce | `mix_scheme 'cvmix_TKE'` (`'KPP'`) | fesom-jax's classical TKE (`tke: {}`) |
| cvmix | added, `= ../namelists_jaxphys/namelist.cvmix` | the TKE constants of the 692f804 twin (`tke_cd 3.75`); without the file the fork's own default is `tke_cd 1.0` |
| ice | `whichEVP 1` (0), `ice_diff 10.0` (0.0) | fesom-jax `IceConfig(whichEVP=1)`, the CORE2 ice diffusion |
| tra | `use_momix .false.` (`.true.`), `K_hor 0.` (3000.) | fesom-jax has no Monin-Obukhov mixing; `K_hor` only feeds `Ki` (Redi) and would be dead anyway |

Kept from the user's set (equal to fesom-jax already): `opt_visc 7` with gamma 0.003/0.1/0.285, `use_wsplit .false.`,
mEVP 120 subcycles with alpha = beta = 250, `ice_gamma_fct 0.5`, Sweeney chlorophyll, CORE2 runoff, PHC2 SSS
restoring, `use_instabmix .true.`, `Kv0_const`, `K_ver 1e-5`, MFCT/QR4C/FCT with weights `0., 1.`. The fork's
`&ice_therm` has no `h0_s/hmin/armin/iclasses/...` keys; its defaults equal the values the 692f804 twin set.
Regenerate: `cp -r ../namelists_albedo_spectral . ; cp ../namelists_jaxphys/namelist.cvmix . ;` the five sed edits
in the table.
