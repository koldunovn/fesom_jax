The broadband (RECOM_WAVEBANDS=OFF) twin of `../namelists_albedo_spectral/`: the same 4p3z2d setup without the
CDOM tracer (36 BGC tracers, no 1037), for the fork's `install_nowb*/bin/fesom.x`. The `&spectral` and
`&nam_rspecsbc` groups are kept in namelist.recom; a non-spectral build does not read them. Regenerate with
`cp -r ../namelists_albedo_spectral . ; sed bgc_num 37->36 ; delete the 1037 line of namelist.tra`.
