# FESOM2 bugs that fesom_jax reproduces on purpose

fesom_jax follows FESOM2 line by line. Where FESOM2 has a bug, fesom_jax does what the code intends by default.
A bug is reproduced only where it is needed to compare with a Fortran FESOM2 run, and then only behind a switch whose
name says so. These switches are not physics. Leave them off unless the run is a comparison with Fortran FESOM2.

| switch (default) | FESOM2 version, file and lines | what FESOM2 does | effect | used by |
|---|---|---|---|---|
| `gm.reproduce_fesom_bug_kgm_surface_level` (false) | 0aaa2d73, `oce_fer_gm.F90` `init_Redi_GM`: :223 and :304-306 against :349, :439, :443; `init_Redi_GM_acc` :582, :615-616, :634 likewise; `oce_setup_step.F90:1042` | writes the surface value of the GM coefficient `fer_k` at the maximum `ulevels` of the node's elements, but builds the vertical profile from `fer_k(ulevels_nod2D(n), n)`, which holds `fer_K = 500 m2/s` at the first call and then its own value times `zscaling(ulevels_nod2D)` at every call (never restarted, so each run segment starts from 500 again). The Redi branch (:330-334) avoids it | only with ice-shelf cavities, at the nodes where `ulevels_nod2D /= ulevels_nod2D_max` (CORE3: 7,237 nodes, 1,104 of them at the shelf front): there the GM coefficient is `500·zscaling(z)` at the front and decays to 0 under the shelves within tens of steps, ignoring K_GM_max, the resolution scaling and the hemispheric ceiling | `configs/core3_cavity_jaxcpl.yaml`, `configs/core3_cpl.yaml` (comparison with jaxcpl). Reported upstream (FESOM/fesom2). |

Not a switch: `ocean2ice` with `ice_ave_steps > 1` passes `u_w/(k+1)` instead of the running mean of the ocean surface
velocity (`ice_oce_coupling.F90`, the comment in `fesom_jax.ice_coupling.ocean2ice_avg`). fesom_jax copies it
unconditionally, because whether to fix it is a FESOM2 decision.
