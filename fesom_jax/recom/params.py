"""REcoM's parameters: the defaults of ``recom_config`` (``recom_modules.F90:418-1000``) overridden by a
``namelist.recom`` (parent plan Task 6).

Two objects, the split a jitted/differentiated caller needs:

* :class:`RecomSwitches` -- every LOGICAL, INTEGER and CHARACTER entry of the REcoM namelist groups: a
  frozen, hashable dataclass, a static argument (it selects code paths, never a leaf).
* :class:`RecomParams` -- every REAL entry: a NamedTuple, hence a pytree whose leaves are Python floats by
  default; pass one with ``jnp`` scalars in place of floats to differentiate with respect to a parameter.

:func:`load_namelist` reads a namelist file into both; :func:`check_supported` raises
``NotImplementedError`` for every switch value outside the ported ``bin_2p3z2d`` configuration (parent plan
"Out of scope"). :func:`derived` returns the quantities REcoM computes once from the parameters
(``initialize_tiny_thresholds``, ``recom_init.F90:888-1004``).

Literal values: REcoM compiles with ``-r8`` (``REcoM_src/CMakeLists.txt:31``), so every unsuffixed real
literal is a double; the Python floats below are the exact decimal values. Two defaults are Fortran
literals that parse as zero (``visc_ref_water = 0.d00158``, ``depth_scaling1 = 0.d015``: mantissa ``0.``,
exponent ``158``/``15``); the ``bin_2p3z2d`` namelist sets both, so it does not matter there, but the
defaults here are the values Fortran holds (0). Constants of ``recom_config`` that are not namelist
entries (``tiny``, ``tiny_chl``, ``SecondsPerDay``, ``Pa2atm``, ``redO2C``, ``recom_modules.F90:488-494``) are
module constants here, not parameters.

Namelist keys are case-insensitive (Fortran); unknown groups or keys raise ``KeyError``. The ciso group
``&paciso`` (module ``REcoM_ciso``, ``recom_modules.F90:2361-2381``) is read and ignored: ``ciso`` must be
``.false.``.
"""

from __future__ import annotations

import dataclasses
import re
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple, Tuple, Union

# recom_config constants (recom_modules.F90:488-494); -r8 makes the literals double.
ZERO = 0.0
TINY = 2.23e-16
TINY_CHL = 1.0e-5
SECONDS_PER_DAY = 86400.0
PA2ATM = 101325.0
REDO2C = 1.453

NAMELIST_GROUPS = {
    "parecomsetup": ('enable_3zoo2det', 'enable_coccos',),
    "pavariables": ('use_REcoM', 'REcoM_restart', 'bgc_num', 'diags3d_num', 'bgc_base_num', 'VDet', 'VDet_zoo2',
            'VPhy', 'VDia', 'VCocco', 'VPhaeo', 'allow_var_sinking', 'biostep', 'REcoM_Geider_limiter',
            'REcoM_Grazing_Variable_Preference', 'REcoM_Grazing_Variable_Efficiency', 'Grazing_detritus',
            'het_resp_noredfield', 'diatom_mucus', 'O2dep_remin', 'use_ballasting', 'use_density_scaling',
            'use_viscosity_scaling', 'OmegaC_diss', 'CO2lim', 'Diags', 'constant_CO2', 'UseFeDust', 'UseDustClim',
            'UseDustClimAlbani', 'use_photodamage', 'HetRespFlux_plus', 'REcoMDataPath', 'restore_alkalinity',
            'useRivers', 'useRivFe', 'useErosion', 'NitrogenSS', 'useAeolianN', 'firstyearoffesomcycle',
            'lastyearoffesomcycle', 'numofCO2cycles', 'currentCO2cycle', 'Nmocsy', 'recom_debug', 'ciso',
            'benthos_num', 'use_MEDUSA', 'sedflx_num', 'bottflx_num', 'add_loopback', 'lb_tscale', 'use_atbox',
            'use_virt_tracers',),
    "pasinking": ('Vdet_a', 'Vcalc',),
    "painitialization_N": ('cPhyN', 'cHetN', 'cZoo2N',),
    "paArrhenius": ('recom_Tref', 'C2K', 'Ae', 'Tmax_phaeo', 'Topt_phaeo', 'uopt_phaeo', 'beta_phaeo', 'ord_d',
            'expon_d', 'ord_phy', 'expon_phy', 'ord_cocco', 'expon_cocco', 'ord_phaeo', 'expon_phaeo', 'reminSi',
            'k_o2_remin',),
    "palimiter_function": ('NMinSlope', 'SiMinSlope', 'NCmin', 'NCmin_d', 'NCmin_c', 'NCmin_p', 'SiCmin', 'k_Fe',
            'k_Fe_d', 'k_Fe_c', 'k_Fe_p', 'k_si', 'P_cm', 'P_cm_d', 'P_cm_c', 'P_cm_p',),
    "palight_calculations": ('k_w', 'a_chl',),
    "paphotosynthesis": ('alfa', 'alfa_d', 'alfa_c', 'alfa_p', 'parFrac',),
    "paassimilation": ('V_cm_fact', 'V_cm_fact_d', 'V_cm_fact_c', 'V_cm_fact_p', 'NMaxSlope', 'SiMaxSlope', 'NCmax',
            'NCmax_d', 'NCmax_c', 'NCmax_p', 'SiCmax', 'NCuptakeRatio', 'NCUptakeRatio_d', 'NCUptakeRatio_c',
            'NCUptakeRatio_p', 'SiCUptakeRatio', 'k_din', 'k_din_d', 'k_din_c', 'k_din_p', 'Chl2N_max',
            'Chl2N_max_d', 'Chl2N_max_c', 'Chl2N_max_p', 'res_phy', 'res_phy_d', 'res_phy_c', 'res_phy_p',
            'biosynth', 'biosynthSi',),
    "pairon_chem": ('totalligand', 'ligandStabConst', 'fe_2ligands', 'fe_compl_nica',),
    "pazooplankton": ('graz_max', 'epsilonr', 'res_het', 'Redfield', 'loss_het', 'pzDia', 'sDiaNsq', 'pzPhy',
            'sPhyNsq', 'pzCocco', 'sCoccoNsq', 'pzPhaeo', 'sPhaeoNsq', 'pzMicZoo', 'sMicZooNsq', 'tiny_het',),
    "pasecondzooplankton": ('graz_max2', 'epsilon2', 'res_zoo2', 'loss_zoo2', 'fecal_rate_n', 'fecal_rate_c',
            'fecal_rate_n_mes', 'fecal_rate_c_mes', 'pzDia2', 'sDiaNsq2', 'pzPhy2', 'sPhyNsq2', 'pzCocco2',
            'sCoccoNsq2', 'pzPhaeo2', 'sPhaeoNsq2', 'pzHet', 'sHetNsq', 'pzMicZoo2', 'sMicZooNsq2', 't1_zoo2',
            't2_zoo2', 't3_zoo2', 't4_zoo2',),
    "pathirdzooplankton": ('graz_max3', 'epsilon3', 'loss_miczoo', 'res_miczoo', 'pzDia3', 'sDiaNsq3', 'pzPhy3',
            'sPhyNsq3', 'pzCocco3', 'sCoccoNsq3', 'pzPhaeo3', 'sPhaeoNsq3',),
    "pagrazingdetritus": ('pzDet', 'sDetNsq', 'pzDetZ2', 'sDetZ2Nsq', 'pzDet2', 'sDetNsq2', 'pzDetZ22',
            'sDetZ2Nsq2',),
    "paaggregation": ('agg_PD', 'agg_PP',),
    "padin_rho_N": ('rho_N',),
    "padic_rho_C1": ('rho_C1',),
    "paphytoplankton_N": ('lossN', 'lossN_d', 'lossN_c', 'lossN_p',),
    "paphytoplankton_C": ('lossC', 'lossC_d', 'lossC_c', 'lossC_p',),
    "paphytoplankton_ChlA": ('deg_Chl', 'deg_Chl_d', 'deg_Chl_c', 'deg_Chl_p',),
    "padetritus_N": ('gfin', 'grazEff2', 'grazEff3', 'reminN',),
    "padetritus_C": ('reminC', 'rho_c2',),
    "paheterotrophs": ('lossN_z', 'lossC_z',),
    "paseczooloss": ('lossN_z2', 'lossC_z2',),
    "pathirdzooloss": ('lossN_z3', 'lossC_z3',),
    "paco2lim": ('Cunits', 'a_co2_phy', 'a_co2_dia', 'a_co2_cocco', 'a_co2_phaeo', 'a_co2_calc', 'b_co2_phy',
            'b_co2_dia', 'b_co2_cocco', 'b_co2_phaeo', 'b_co2_calc', 'c_co2_phy', 'c_co2_dia', 'c_co2_cocco',
            'c_co2_phaeo', 'c_co2_calc', 'd_co2_phy', 'd_co2_dia', 'd_co2_cocco', 'd_co2_phaeo', 'd_co2_calc',),
    "pairon": ('Fe2N', 'Fe2N_benthos', 'kScavFe', 'kScavFe2', 'dust_sol', 'RiverFeConc',),
    "pacalc": ('calc_prod_ratio', 'calc_diss_guts', 'calc_diss_rate', 'calc_diss_rate2', 'calc_diss_omegac',
            'calc_diss_exp',),
    "pabenthos_decay_rate": ('decayRateBenN', 'decayRateBenC', 'decayRateBenSi', 'q_NC_Denit',),
    "paco2_flux_param": ('permil', 'permeg', 'Xacc', 'CO2_for_spinup',),
    "paalkalinity_restoring": ('surf_relax_Alk',),
    "paballasting": ('rho_POC', 'rho_PON', 'rho_CaCO3', 'rho_opal', 'rho_ref_part', 'rho_ref_water',
            'visc_ref_water', 'w_ref1', 'w_ref2', 'depth_scaling1', 'depth_scaling2', 'max_sinking_velocity',),
    "nam_rsbc": ('fe_data_source', 'nm_fe_data_file', 'nm_aen_data_file', 'nm_river_data_file',
            'nm_erosion_data_file', 'nm_co2_data_file',),
    "paciso": ('ciso_init', 'ciso_14', 'ciso_organic_14', 'lambda_14', 'delta_CO2_13', 'big_delta_CO2_14',
            'atbox_spinup', 'cosmic_14_init',),
}


class RecomParams(NamedTuple):
    """Every REAL entry of the REcoM namelist groups, default = ``recom_config`` (file:line given)."""
    # &pavariables
    VDet: float = 20.0                  # recom_modules.F90:512
    VDet_zoo2: float = 200.0            # recom_modules.F90:513
    VPhy: float = 0.0                   # recom_modules.F90:515
    VDia: float = 0.0                   # recom_modules.F90:516
    VCocco: float = 0.0                 # recom_modules.F90:517
    VPhaeo: float = 0.0                 # recom_modules.F90:518
    lb_tscale: float = 1.0              # recom_modules.F90:577
    # &pasinking
    Vdet_a: float = 0.0288              # recom_modules.F90:612
    Vcalc: float = 0.0216               # recom_modules.F90:613
    # &painitialization_N
    cPhyN: float = 0.2                  # recom_modules.F90:618
    cHetN: float = 0.2                  # recom_modules.F90:619
    cZoo2N: float = 0.2                 # recom_modules.F90:620
    # &paArrhenius
    recom_Tref: float = 288.15          # recom_modules.F90:625
    C2K: float = 273.15                 # recom_modules.F90:626
    Ae: float = 4500.0                  # recom_modules.F90:627
    Tmax_phaeo: float = 16.0            # recom_modules.F90:630
    Topt_phaeo: float = 7.5272          # recom_modules.F90:631
    uopt_phaeo: float = 0.7328          # recom_modules.F90:632
    beta_phaeo: float = 0.7829          # recom_modules.F90:633
    ord_d: float = -0.2216              # recom_modules.F90:636
    expon_d: float = 0.0406             # recom_modules.F90:637
    ord_phy: float = -1.2154            # recom_modules.F90:638
    expon_phy: float = 0.0599           # recom_modules.F90:639
    ord_cocco: float = -0.231           # recom_modules.F90:640
    expon_cocco: float = 0.0327         # recom_modules.F90:641
    ord_phaeo: float = -0.231           # recom_modules.F90:642
    expon_phaeo: float = 0.0327         # recom_modules.F90:643
    reminSi: float = 0.02               # recom_modules.F90:645
    k_o2_remin: float = 15.0            # recom_modules.F90:648
    # &palimiter_function
    NMinSlope: float = 50.0             # recom_modules.F90:655
    SiMinSlope: float = 1000.0          # recom_modules.F90:656
    NCmin: float = 0.04                 # recom_modules.F90:657
    NCmin_d: float = 0.04               # recom_modules.F90:658
    NCmin_c: float = 0.04               # recom_modules.F90:659
    NCmin_p: float = 0.04               # recom_modules.F90:660
    SiCmin: float = 0.04                # recom_modules.F90:661
    k_Fe: float = 0.04                  # recom_modules.F90:662
    k_Fe_d: float = 0.12                # recom_modules.F90:663
    k_Fe_c: float = 0.04                # recom_modules.F90:664
    k_Fe_p: float = 0.09                # recom_modules.F90:665
    k_si: float = 4.0                   # recom_modules.F90:666
    P_cm: float = 3.0                   # recom_modules.F90:667
    P_cm_d: float = 3.5                 # recom_modules.F90:668
    P_cm_c: float = 3.3                 # recom_modules.F90:669
    P_cm_p: float = 3.4                 # recom_modules.F90:670
    # &palight_calculations
    k_w: float = 0.04                   # recom_modules.F90:676
    a_chl: float = 0.03                 # recom_modules.F90:678
    # &paphotosynthesis
    alfa: float = 0.14                  # recom_modules.F90:682
    alfa_d: float = 0.19                # recom_modules.F90:683
    alfa_c: float = 0.1                 # recom_modules.F90:684
    alfa_p: float = 0.1                 # recom_modules.F90:685
    parFrac: float = 0.43               # recom_modules.F90:686
    # &paassimilation
    V_cm_fact: float = 0.7              # recom_modules.F90:691
    V_cm_fact_d: float = 0.7            # recom_modules.F90:692
    V_cm_fact_c: float = 0.7            # recom_modules.F90:693
    V_cm_fact_p: float = 0.7            # recom_modules.F90:694
    NMaxSlope: float = 1000.0           # recom_modules.F90:695
    SiMaxSlope: float = 1000.0          # recom_modules.F90:696
    NCmax: float = 0.2                  # recom_modules.F90:697
    NCmax_d: float = 0.2                # recom_modules.F90:698
    NCmax_c: float = 0.15               # recom_modules.F90:699
    NCmax_p: float = 0.1                # recom_modules.F90:700
    SiCmax: float = 0.8                 # recom_modules.F90:701
    NCuptakeRatio: float = 0.2          # recom_modules.F90:702
    NCUptakeRatio_d: float = 0.2        # recom_modules.F90:703
    NCUptakeRatio_c: float = 0.2        # recom_modules.F90:704
    NCUptakeRatio_p: float = 0.2        # recom_modules.F90:705
    SiCUptakeRatio: float = 0.2         # recom_modules.F90:706
    k_din: float = 0.55                 # recom_modules.F90:707
    k_din_d: float = 1.0                # recom_modules.F90:708
    k_din_c: float = 0.55               # recom_modules.F90:709
    k_din_p: float = 0.55               # recom_modules.F90:710
    Chl2N_max: float = 3.15             # recom_modules.F90:711
    Chl2N_max_d: float = 4.2            # recom_modules.F90:712
    Chl2N_max_c: float = 3.5            # recom_modules.F90:713
    Chl2N_max_p: float = 3.5            # recom_modules.F90:714
    res_phy: float = 0.01               # recom_modules.F90:715
    res_phy_d: float = 0.01             # recom_modules.F90:716
    res_phy_c: float = 0.0075           # recom_modules.F90:717
    res_phy_p: float = 0.008            # recom_modules.F90:718
    biosynth: float = 2.33              # recom_modules.F90:719
    biosynthSi: float = 0.0             # recom_modules.F90:720
    # &pairon_chem
    totalligand: float = 1.0            # recom_modules.F90:730
    ligandStabConst: float = 100.0      # recom_modules.F90:732
    # &pazooplankton
    graz_max: float = 2.4               # recom_modules.F90:739
    epsilonr: float = 0.35              # recom_modules.F90:740
    res_het: float = 0.01               # recom_modules.F90:742
    Redfield: float = 6.625             # recom_modules.F90:743
    loss_het: float = 0.05              # recom_modules.F90:745
    pzDia: float = 0.5                  # recom_modules.F90:746
    sDiaNsq: float = 0.0                # recom_modules.F90:747
    pzPhy: float = 1.0                  # recom_modules.F90:748
    sPhyNsq: float = 0.0                # recom_modules.F90:749
    pzCocco: float = 0.5                # recom_modules.F90:750
    sCoccoNsq: float = 0.0              # recom_modules.F90:751
    pzPhaeo: float = 1.0                # recom_modules.F90:752
    sPhaeoNsq: float = 0.0              # recom_modules.F90:753
    pzMicZoo: float = 1.0               # recom_modules.F90:754
    sMicZooNsq: float = 0.0             # recom_modules.F90:755
    tiny_het: float = 1e-05             # recom_modules.F90:759
    # &pasecondzooplankton
    graz_max2: float = 0.1              # recom_modules.F90:764
    epsilon2: float = 0.0144            # recom_modules.F90:765
    res_zoo2: float = 0.0107            # recom_modules.F90:767
    loss_zoo2: float = 0.003            # recom_modules.F90:769
    fecal_rate_n: float = 0.13          # recom_modules.F90:770
    fecal_rate_c: float = 0.295         # recom_modules.F90:771
    fecal_rate_n_mes: float = 0.25      # recom_modules.F90:772
    fecal_rate_c_mes: float = 0.32      # recom_modules.F90:773
    pzDia2: float = 1.0                 # recom_modules.F90:774
    sDiaNsq2: float = 0.0               # recom_modules.F90:775
    pzPhy2: float = 0.5                 # recom_modules.F90:776
    sPhyNsq2: float = 0.0               # recom_modules.F90:777
    pzCocco2: float = 0.2               # recom_modules.F90:778
    sCoccoNsq2: float = 0.0             # recom_modules.F90:779
    pzPhaeo2: float = 0.5               # recom_modules.F90:780
    sPhaeoNsq2: float = 0.0             # recom_modules.F90:781
    pzHet: float = 0.8                  # recom_modules.F90:782
    sHetNsq: float = 0.0                # recom_modules.F90:783
    pzMicZoo2: float = 0.8              # recom_modules.F90:784
    sMicZooNsq2: float = 0.0            # recom_modules.F90:785
    t1_zoo2: float = 28145.0            # recom_modules.F90:786
    t2_zoo2: float = 272.5              # recom_modules.F90:787
    t3_zoo2: float = 105234.0           # recom_modules.F90:788
    t4_zoo2: float = 274.15             # recom_modules.F90:789
    # &pathirdzooplankton
    graz_max3: float = 0.46             # recom_modules.F90:797
    epsilon3: float = 0.64              # recom_modules.F90:799
    loss_miczoo: float = 0.01           # recom_modules.F90:801
    res_miczoo: float = 0.01            # recom_modules.F90:803
    pzDia3: float = 0.5                 # recom_modules.F90:804
    sDiaNsq3: float = 0.0               # recom_modules.F90:805
    pzPhy3: float = 1.0                 # recom_modules.F90:806
    sPhyNsq3: float = 0.0               # recom_modules.F90:807
    pzCocco3: float = 0.0               # recom_modules.F90:810
    sCoccoNsq3: float = 0.0             # recom_modules.F90:811
    pzPhaeo3: float = 1.0               # recom_modules.F90:813
    sPhaeoNsq3: float = 0.0             # recom_modules.F90:814
    # &pagrazingdetritus
    pzDet: float = 1.0                  # recom_modules.F90:820
    sDetNsq: float = 0.0                # recom_modules.F90:821
    pzDetZ2: float = 1.0                # recom_modules.F90:822
    sDetZ2Nsq: float = 0.0              # recom_modules.F90:823
    pzDet2: float = 1.0                 # recom_modules.F90:824
    sDetNsq2: float = 0.0               # recom_modules.F90:825
    pzDetZ22: float = 1.0               # recom_modules.F90:826
    sDetZ2Nsq2: float = 0.0             # recom_modules.F90:827
    # &paaggregation
    agg_PD: float = 0.165               # recom_modules.F90:833
    agg_PP: float = 0.015               # recom_modules.F90:835
    # &padin_rho_N
    rho_N: float = 0.11                 # recom_modules.F90:841
    # &padic_rho_C1
    rho_C1: float = 0.1                 # recom_modules.F90:846
    # &paphytoplankton_N
    lossN: float = 0.05                 # recom_modules.F90:850
    lossN_d: float = 0.05               # recom_modules.F90:851
    lossN_c: float = 0.05               # recom_modules.F90:852
    lossN_p: float = 0.05               # recom_modules.F90:853
    # &paphytoplankton_C
    lossC: float = 0.1                  # recom_modules.F90:857
    lossC_d: float = 0.1                # recom_modules.F90:858
    lossC_c: float = 0.1                # recom_modules.F90:859
    lossC_p: float = 0.1                # recom_modules.F90:860
    # &paphytoplankton_ChlA
    deg_Chl: float = 0.25               # recom_modules.F90:864
    deg_Chl_d: float = 0.25             # recom_modules.F90:865
    deg_Chl_c: float = 0.2              # recom_modules.F90:866
    deg_Chl_p: float = 0.25             # recom_modules.F90:867
    # &padetritus_N
    gfin: float = 0.3                   # recom_modules.F90:872
    grazEff2: float = 0.8               # recom_modules.F90:874
    grazEff3: float = 0.8               # recom_modules.F90:876
    reminN: float = 0.165               # recom_modules.F90:878
    # &padetritus_C
    reminC: float = 0.15                # recom_modules.F90:883
    rho_c2: float = 0.1                 # recom_modules.F90:885
    # &paheterotrophs
    lossN_z: float = 0.15               # recom_modules.F90:889
    lossC_z: float = 0.15               # recom_modules.F90:890
    # &paseczooloss
    lossN_z2: float = 0.02              # recom_modules.F90:894
    lossC_z2: float = 0.02              # recom_modules.F90:895
    # &pathirdzooloss
    lossN_z3: float = 0.05              # recom_modules.F90:899
    lossC_z3: float = 0.05              # recom_modules.F90:900
    # &paco2lim
    Cunits: float = 976.5625            # recom_modules.F90:906
    a_co2_phy: float = 1.162            # recom_modules.F90:907
    a_co2_dia: float = 1.04             # recom_modules.F90:908
    a_co2_cocco: float = 1.109          # recom_modules.F90:909
    a_co2_phaeo: float = 1.162          # recom_modules.F90:910
    a_co2_calc: float = 1.102           # recom_modules.F90:911
    b_co2_phy: float = 48.88            # recom_modules.F90:912
    b_co2_dia: float = 28.9             # recom_modules.F90:913
    b_co2_cocco: float = 37.67          # recom_modules.F90:914
    b_co2_phaeo: float = 48.88          # recom_modules.F90:915
    b_co2_calc: float = 42.38           # recom_modules.F90:916
    c_co2_phy: float = 0.2255           # recom_modules.F90:917
    c_co2_dia: float = 0.8778           # recom_modules.F90:918
    c_co2_cocco: float = 0.3912         # recom_modules.F90:919
    c_co2_phaeo: float = 0.2255         # recom_modules.F90:920
    c_co2_calc: float = 0.7079          # recom_modules.F90:921
    d_co2_phy: float = 10230000.0       # recom_modules.F90:922
    d_co2_dia: float = 2640000.0        # recom_modules.F90:923
    d_co2_cocco: float = 9450000.0      # recom_modules.F90:924
    d_co2_phaeo: float = 10230000.0     # recom_modules.F90:925
    d_co2_calc: float = 13430000.0      # recom_modules.F90:926
    # &pairon
    Fe2N: float = 0.033                 # recom_modules.F90:933
    Fe2N_benthos: float = 0.15          # recom_modules.F90:936
    kScavFe: float = 0.07               # recom_modules.F90:937
    kScavFe2: float = 0.0               # recom_modules.F90:940
    dust_sol: float = 0.02              # recom_modules.F90:941
    RiverFeConc: float = 100.0          # recom_modules.F90:942
    # &pacalc
    calc_prod_ratio: float = 0.02       # recom_modules.F90:946
    calc_diss_guts: float = 0.0         # recom_modules.F90:947
    calc_diss_rate: float = 0.005714    # recom_modules.F90:948
    calc_diss_rate2: float = 0.005714   # recom_modules.F90:949
    calc_diss_omegac: float = 0.197     # recom_modules.F90:951
    calc_diss_exp: float = 1.0          # recom_modules.F90:953
    # &pabenthos_decay_rate
    decayRateBenN: float = 0.005        # recom_modules.F90:958
    decayRateBenC: float = 0.005        # recom_modules.F90:959
    decayRateBenSi: float = 0.005       # recom_modules.F90:960
    q_NC_Denit: float = 0.86            # recom_modules.F90:961
    # &paco2_flux_param
    permil: float = 9.76e-07            # recom_modules.F90:966
    permeg: float = 1e-06               # recom_modules.F90:967
    Xacc: float = 1e-12                 # recom_modules.F90:968
    CO2_for_spinup: float = 278.0       # recom_modules.F90:970
    # &paalkalinity_restoring
    surf_relax_Alk: float = 3.2e-07     # recom_modules.F90:974
    # &paballasting
    rho_POC: float = 1033.0             # recom_modules.F90:978
    rho_PON: float = 1033.0             # recom_modules.F90:979
    rho_CaCO3: float = 2830.0           # recom_modules.F90:981
    rho_opal: float = 2090.0            # recom_modules.F90:982
    rho_ref_part: float = 1230.0        # recom_modules.F90:984
    rho_ref_water: float = 1027.0       # recom_modules.F90:986
    visc_ref_water: float = 0.0         # recom_modules.F90:988: Fortran `0.d00158` = 0 x 10^158
    w_ref1: float = 10.0                # recom_modules.F90:989
    w_ref2: float = 200.0               # recom_modules.F90:990
    depth_scaling1: float = 0.0         # recom_modules.F90:992: Fortran `0.d015` = 0 x 10^15
    depth_scaling2: float = 0.0         # recom_modules.F90:994
    max_sinking_velocity: float = 250.0 # recom_modules.F90:997


@dataclasses.dataclass(frozen=True)
class RecomSwitches:
    """Every LOGICAL / INTEGER / CHARACTER entry of the REcoM namelist groups (static, hashable)."""
    # &parecomsetup
    enable_3zoo2det: bool = False  # recom_modules.F90:497
    enable_coccos: bool = False  # recom_modules.F90:498
    # &pavariables
    use_REcoM: bool = True  # recom_modules.F90:503
    REcoM_restart: bool = False  # recom_modules.F90:504
    bgc_num: int = 36  # recom_modules.F90:508
    diags3d_num: int = 31  # recom_modules.F90:510
    bgc_base_num: int = 22  # recom_modules.F90:509
    allow_var_sinking: bool = True  # recom_modules.F90:519
    biostep: int = 1  # recom_modules.F90:520
    REcoM_Geider_limiter: bool = False  # recom_modules.F90:523
    REcoM_Grazing_Variable_Preference: bool = True  # recom_modules.F90:526
    REcoM_Grazing_Variable_Efficiency: bool = True  # recom_modules.F90:527
    Grazing_detritus: bool = False  # recom_modules.F90:529
    het_resp_noredfield: bool = True  # recom_modules.F90:530
    diatom_mucus: bool = True  # recom_modules.F90:531
    O2dep_remin: bool = True  # recom_modules.F90:534
    use_ballasting: bool = True  # recom_modules.F90:535
    use_density_scaling: bool = True  # recom_modules.F90:536
    use_viscosity_scaling: bool = True  # recom_modules.F90:537
    OmegaC_diss: bool = True  # recom_modules.F90:540
    CO2lim: bool = True  # recom_modules.F90:541
    Diags: bool = True  # recom_modules.F90:544
    constant_CO2: bool = True  # recom_modules.F90:545
    UseFeDust: bool = True  # recom_modules.F90:546
    UseDustClim: bool = True  # recom_modules.F90:547
    UseDustClimAlbani: bool = True  # recom_modules.F90:549
    use_photodamage: bool = False  # recom_modules.F90:550
    HetRespFlux_plus: bool = True  # recom_modules.F90:552
    REcoMDataPath: str = '/albedo/work/projects/MarESys/ogurses/input/mesh_CORE2_finaltopo_mean/'  # recom_modules.F90:553
    restore_alkalinity: bool = True  # recom_modules.F90:555
    useRivers: bool = False  # recom_modules.F90:556
    useRivFe: bool = False  # recom_modules.F90:557
    useErosion: bool = False  # recom_modules.F90:558
    NitrogenSS: bool = False  # recom_modules.F90:563
    useAeolianN: bool = False  # recom_modules.F90:564
    firstyearoffesomcycle: int = 1958  # recom_modules.F90:566
    lastyearoffesomcycle: int = 2023  # recom_modules.F90:567
    numofCO2cycles: int = 1  # recom_modules.F90:568
    currentCO2cycle: int = 1  # recom_modules.F90:569
    Nmocsy: int = 1  # recom_modules.F90:570
    recom_debug: bool = False  # recom_modules.F90:571
    ciso: bool = False  # recom_modules.F90:572
    benthos_num: int = 4  # recom_modules.F90:573
    use_MEDUSA: bool = False  # recom_modules.F90:574
    sedflx_num: int = 0  # recom_modules.F90:575
    bottflx_num: int = 4  # recom_modules.F90:580
    add_loopback: bool = False  # recom_modules.F90:576
    use_atbox: bool = False  # recom_modules.F90:581
    use_virt_tracers: bool = False  # recom_modules.F90:584
    # &pairon_chem
    fe_2ligands: bool = False  # recom_modules.F90:733
    fe_compl_nica: bool = False  # recom_modules.F90:735
    # &nam_rsbc (FESOM2 gen_surface_forcing.F90:1651, defaults blank)
    fe_data_source: str = ''
    nm_fe_data_file: str = ''
    nm_aen_data_file: str = ''
    nm_river_data_file: str = ''
    nm_erosion_data_file: str = ''
    nm_co2_data_file: str = ''


# The switch values of the ported configuration (bin_2p3z2d). A switch listed here with any other value
# raises NotImplementedError in check_supported; the reason names what is missing.
SUPPORTED: dict = {
    "enable_3zoo2det": (True, "only the 2phy-3zoo-2det ecosystem of bin_2p3z2d is ported"),
    "enable_coccos": (False, "coccolithophores/Phaeocystis are out of scope"),
    "use_REcoM": (True, "use_REcoM = .false. means no REcoM: leave RunConfig.recom unset"),
    "bgc_num": (31, "the bin_2p3z2d tracer set has 31 BGC tracers"),
    "bgc_base_num": (22, "the bin_2p3z2d tracer set has 22 base tracers"),
    "benthos_num": (4, "4 benthic pools without ciso"),
    "biostep": (1, "biostep > 1 (BGC sub-stepping) is not ported; the any(abs(sms) <= tiny) reset is then a "
                   "no-op (recom_sms.F90:259)"),
    "Nmocsy": (1, "mocsy is called per point (Nmocsy = 1)"),
    "REcoM_Geider_limiter": (False, "the Geider limiter is not ported"),
    "REcoM_Grazing_Variable_Preference": (True, "fixed grazing preferences are not ported"),
    "REcoM_Grazing_Variable_Efficiency": (True, "a fixed grazing efficiency is not ported"),
    "Grazing_detritus": (True, "the no-detritus-grazing branches are not ported"),
    "het_resp_noredfield": (True, "the Redfield heterotroph respiration is not ported"),
    "diatom_mucus": (True, "aggregation without diatom mucus is not ported"),
    "O2dep_remin": (True, "O2-independent remineralisation is not ported"),
    "use_photodamage": (True, "chlorophyll degradation without photodamage is not ported"),
    "CO2lim": (True, "photosynthesis without CO2 limitation is not ported"),
    "OmegaC_diss": (False, "the saturation-state calcite dissolution is not ported"),
    "use_ballasting": (True, "sinking without ballasting is not ported"),
    "use_density_scaling": (True, "ballasting without density scaling is not ported"),
    "use_viscosity_scaling": (True, "ballasting without viscosity scaling is not ported"),
    "restore_alkalinity": (True, "a run without alkalinity restoring is not ported"),
    "useRivers": (False, "river input is out of scope"),
    "useRivFe": (False, "river iron is out of scope"),
    "useErosion": (False, "erosion input is out of scope"),
    "NitrogenSS": (False, "external nitrogen sources and sinks are out of scope"),
    "useAeolianN": (False, "aeolian nitrogen deposition is out of scope"),
    "ciso": (False, "carbon isotopes are out of scope"),
    "use_MEDUSA": (False, "the MEDUSA sediment model is out of scope"),
    "sedflx_num": (0, "MEDUSA sediment fluxes are out of scope"),
    "add_loopback": (False, "the MEDUSA loopback is out of scope"),
    "use_atbox": (False, "the atmospheric CO2 box is out of scope"),
    "fe_2ligands": (False, "two-ligand iron chemistry is not ported"),
    "fe_compl_nica": (False, "the NICA iron complexation is not ported"),
    "fe_data_source": ("Albani", "only the Albani monthly dust climatology is ported"),
}
# Accepted with any value, no effect on the ported code: output/diagnostic and bookkeeping flags, the
# flags REcoM 692f804 declares but never reads (UseFeDust, UseDustClim, UseDustClimAlbani,
# HetRespFlux_plus), the file names and the CO2 cycle years (read by the input layer, Task 13, and with constant_CO2
# off by fesom_jax.run_config.RunConfig.bgc_co2), use_virt_tracers (FESOM2 reads it
# only for linfs virtual tracer fluxes, ice_oce_coupling.F90:450; fesom-jax REcoM runs zstar),
# allow_var_sinking (sinking lane, recom_sinking.F90:122).
NO_EFFECT = ("REcoM_restart", "recom_debug", "Diags", "diags3d_num", "UseFeDust", "UseDustClim",
             "UseDustClimAlbani", "HetRespFlux_plus", "REcoMDataPath", "firstyearoffesomcycle",
             "lastyearoffesomcycle", "numofCO2cycles", "currentCO2cycle", "bottflx_num", "use_virt_tracers",
             "allow_var_sinking", "nm_fe_data_file", "nm_aen_data_file", "nm_river_data_file",
             "nm_erosion_data_file", "nm_co2_data_file")

_PARAM_FIELDS = {f.lower(): f for f in RecomParams._fields}
_SWITCH_FIELDS = {f.name.lower(): f.name for f in dataclasses.fields(RecomSwitches)}
_GROUP_KEYS = {g.lower(): {k.lower() for k in keys} for g, keys in NAMELIST_GROUPS.items()}
_GROUP_KEYS["nam_rsbc"] = {"fe_data_source", "nm_fe_data_file", "nm_aen_data_file", "nm_river_data_file",
                           "nm_erosion_data_file", "nm_co2_data_file"}
_IGNORED_GROUPS = {"paciso": _GROUP_KEYS.pop("paciso")}
assert set(_SWITCH_FIELDS) | set(_PARAM_FIELDS) == set().union(*_GROUP_KEYS.values())


# --------------------------------------------------------------------------------------
# A Fortran namelist reader (the subset REcoM's namelist.recom uses)
# --------------------------------------------------------------------------------------
def _strip_comment(line: str) -> str:
    out, quote = [], None
    for ch in line:
        if quote:
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch == "!":
            break
        out.append(ch)
    return "".join(out)


def _parse_value(tok: str):
    t = tok.strip()
    if len(t) >= 2 and t[0] == t[-1] and t[0] in "'\"":
        return t[1:-1]
    low = t.lower()
    if low in (".true.", ".t.", "t", "true"):
        return True
    if low in (".false.", ".f.", "f", "false"):
        return False
    if re.fullmatch(r"[+-]?\d+", t):
        return int(t)
    if re.fullmatch(r"[+-]?(\d+\.?\d*|\.\d+)([edq][+-]?\d+)?", low):
        return float(re.sub("[dq]", "e", low))
    return t                                   # e.g. a template placeholder (@RECOM_RESTART@)


def read_namelist(path: Union[str, Path]) -> dict:
    """``{group: {key: value}}`` (lower-case names) of a Fortran namelist file: ``&group`` ... ``/``
    blocks, one or more ``key = value`` per line (comma-separated), ``!`` comments, logicals, integers,
    reals with ``d`` exponents, quoted strings and indexed array elements (``a(2) = x`` -> key ``a(2)``).
    A value that is none of these is kept as its raw string (template placeholders)."""
    groups: dict = {}
    cur = None
    for raw in Path(path).read_text().splitlines():
        line = _strip_comment(raw).strip()
        if not line:
            continue
        if cur is None:
            m = re.match(r"&(\w+)\s*(.*)$", line)
            if not m:
                raise ValueError(f"{path}: text outside a namelist group: {raw!r}")
            cur = m.group(1).lower()
            if cur in groups:
                raise ValueError(f"{path}: namelist group &{cur} appears twice")
            groups[cur] = {}
            line = m.group(2).strip()
            if not line:
                continue
        closes = line.endswith("/") and (line[:-1].count("'") + line[:-1].count('"')) % 2 == 0
        body = line[:-1].strip() if closes else line
        matched = False
        for m in re.finditer(r"([A-Za-z_]\w*(?:\(\s*\d+\s*\))?)\s*=\s*('[^']*'|\"[^\"]*\"|[^,]+?)\s*(?:,|$)", body):
            matched = True
            key = re.sub(r"\s+", "", m.group(1)).lower()
            if key in groups[cur]:
                raise ValueError(f"{path}: &{cur} sets {key} twice")
            groups[cur][key] = _parse_value(m.group(2))
        if body and not matched:
            raise ValueError(f"{path}: cannot read {raw!r} (only key = value entries are supported)")
        if closes:
            cur = None
    if cur is not None:
        raise ValueError(f"{path}: namelist group &{cur} is not closed with '/'")
    return groups


def from_namelist_dict(groups: dict) -> Tuple[RecomSwitches, RecomParams]:
    """Apply ``{group: {key: value}}`` (as :func:`read_namelist` returns it) to the defaults.
    Unknown groups or keys raise ``KeyError``; a value of the wrong kind raises ``TypeError``."""
    sw_kw, par_kw = {}, {}
    for g, entries in groups.items():
        g = g.lower()
        if g in _IGNORED_GROUPS:
            bad = {k.split("(")[0] for k in entries} - _IGNORED_GROUPS[g]
            if bad:
                raise KeyError(f"unknown key(s) in &{g}: {sorted(bad)}")
            continue
        if g not in _GROUP_KEYS:
            raise KeyError(f"unknown REcoM namelist group &{g} (known: {sorted(_GROUP_KEYS)})")
        for k, v in entries.items():
            k = k.lower()
            if k not in _GROUP_KEYS[g]:
                raise KeyError(f"unknown key {k!r} in &{g}")
            if k in _PARAM_FIELDS:
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    raise TypeError(f"&{g} {k} = {v!r}: a REAL is expected")
                par_kw[_PARAM_FIELDS[k]] = float(v)
            else:
                name = _SWITCH_FIELDS[k]
                want = type(getattr(RecomSwitches, name))
                if want is bool and not isinstance(v, bool):
                    if name in NO_EFFECT:           # e.g. an unfilled template placeholder
                        v = str(v)
                    else:
                        raise TypeError(f"&{g} {k} = {v!r}: a LOGICAL is expected")
                elif want is int and (isinstance(v, bool) or not isinstance(v, int)):
                    raise TypeError(f"&{g} {k} = {v!r}: an INTEGER is expected")
                elif want is str and not isinstance(v, str):
                    raise TypeError(f"&{g} {k} = {v!r}: a CHARACTER value is expected")
                sw_kw[name] = v
    return RecomSwitches(**sw_kw), RecomParams(**par_kw)


@lru_cache(maxsize=16)
def _load_cached(path: str, mtime: float) -> Tuple[RecomSwitches, RecomParams]:
    return from_namelist_dict(read_namelist(path))


def load_namelist(path: Union[str, Path], check: bool = True) -> Tuple[RecomSwitches, RecomParams]:
    """``(switches, params)`` from a ``namelist.recom`` (defaults for everything it does not set);
    ``check`` runs :func:`check_supported`."""
    p = Path(path).expanduser().resolve()
    sw, par = _load_cached(str(p), p.stat().st_mtime)
    if check:
        check_supported(sw)
    return sw, par


def check_supported(sw: RecomSwitches) -> RecomSwitches:
    """Raise ``NotImplementedError`` for any switch outside the ported ``bin_2p3z2d`` configuration."""
    bad = []
    for name, (want, why) in SUPPORTED.items():
        if getattr(sw, name) != want:
            bad.append(f"{name} = {getattr(sw, name)!r} ({why})")
    if bad:
        raise NotImplementedError("REcoM switch(es) outside the ported bin_2p3z2d configuration "
                                  "(fesom_jax/recom/params.py SUPPORTED): " + "; ".join(bad))
    return sw


class Derived(NamedTuple):
    """Quantities REcoM computes once from the parameters (``initialize_tiny_thresholds``,
    ``recom_init.F90:941-973``): the per-species floors of the post-SMS clip."""
    tiny_N: float
    tiny_C: float
    tiny_N_d: float
    tiny_C_d: float
    tiny_Si: float


def derived(p: RecomParams) -> Derived:
    """``recom_init.F90:941-973`` (cocco/Phaeocystis floors omitted: out of scope)."""
    tiny_N = TINY_CHL / p.Chl2N_max
    tiny_C = tiny_N / p.NCmax
    tiny_N_d = TINY_CHL / p.Chl2N_max_d
    tiny_C_d = tiny_N_d / p.NCmax_d
    tiny_Si = tiny_C_d * p.SiCmax
    return Derived(tiny_N, tiny_C, tiny_N_d, tiny_C_d, tiny_Si)
