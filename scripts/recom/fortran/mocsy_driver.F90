! Standalone mocsy driver for the fesom-jax REcoM carbonate gate G1
! (docs/plans/20261025-recom-lane-mocsy.md, step M2).
!
! Calls exactly the mocsy entry points REcoM 692f804 calls, with REcoM's option strings:
!   * vars_sprac  (recom_carbonate_chemistry.F90:240, the interior water-column call)
!   * pistonvel, flxco2, o2flux (recom_forcing.F90:195, 222, 278, the surface calls)
! one point at a time (Nmocsy = 1, as REcoM), from a grid of inputs.
!
! The pH-solver iteration count is not an output of vars_sprac. To record it, the driver
! repeats the solver set-up of vars_sprac (vars.F90:385-605: p80, the clamps, rho, the
! mol/m3 -> mol/kg division, constants) and calls solve_at_general itself; the module counter
! mphsolvers::niter_atgen then holds the count of that solve. The driver checks that this
! replica's pH is bitwise the vars_sprac pH (column "replica_ok").
!
! Input  (stream, little endian): int32 n, int32 nin, then real64 inp(nin, n)
! Output (stream, little endian): int32 n, int32 nout, then real64 out(nout, n)
! Column order: the NAMES_IN / NAMES_OUT lists in scripts/recom/make_mocsy_g1.py.
!
! Usage: mocsy_driver.x <input.bin> <output.bin>
program mocsy_driver
    use msingledouble, only: r8
    use mvars, only: vars_sprac
    use gasx, only: flxco2, pistonvel, o2flux
    use mconstants, only: constants
    use mp80, only: p80
    use mrho, only: rho
    use mphsolvers, only: solve_at_general, niter_atgen
    implicit none

    integer, parameter :: nin = 14, nout = 70
    integer :: n, nin_file, i, j, ios
    real(r8), allocatable :: inp(:, :), out(:, :)
    character(len=512) :: fin, fout

    ! one-point arrays (Nmocsy = 1)
    real(r8), dimension(1) :: temp, sal, alk, dic, sil, phos, patm, depth, lat, lon
    real(r8), dimension(1) :: wind, fice, xco2, o2, ppo
    real(r8), dimension(1) :: ph, pco2, fco2, co2, hco3, co3, omegaa, omegac, kspc, betad
    real(r8), dimension(1) :: rhosw, p, tempis
    real(r8), dimension(1) :: co2flux, co2ex, dpco2, k0s, kw660, o2ex
    real(r8), dimension(1) :: sph, spco2, sfco2, sco2, shco3, sco3, somegaa, somegac
    real(r8), dimension(1) :: sbetad, srhosw, sp, stempis
    real(r8), dimension(1) :: ak0, ak1, ak2, akb, akw, aks, akf, akspc, akspa
    real(r8), dimension(1) :: ak1p, ak2p, ak3p, aksi, ast, aft, abt
    integer :: niter_main, niter_last, niter_surf, niter_surf_last
    real(r8) :: ph_rep, sph_rep

    call get_command_argument(1, fin)
    call get_command_argument(2, fout)

    open(10, file=trim(fin), access='stream', form='unformatted', status='old', iostat=ios)
    if (ios /= 0) stop 'cannot open input'
    read(10) n, nin_file
    if (nin_file /= nin) stop 'input column count mismatch'
    allocate(inp(nin, n), out(nout, n))
    read(10) inp
    close(10)
    out = 0.0_r8

    do i = 1, n
        temp(1) = inp(1, i)
        sal(1) = inp(2, i)
        alk(1) = inp(3, i)
        dic(1) = inp(4, i)
        sil(1) = inp(5, i)
        phos(1) = inp(6, i)
        patm(1) = inp(7, i)
        depth(1) = inp(8, i)
        lat(1) = inp(9, i)
        lon(1) = inp(10, i)
        wind(1) = inp(11, i)
        fice(1) = inp(12, i)
        xco2(1) = inp(13, i)
        o2(1) = inp(14, i)

        ! ---- interior: recom_carbonate_chemistry.F90:240-247 ----
        call vars_sprac(ph, pco2, fco2, co2, hco3, co3, omegaa, omegac, kspc, betad, &
                rhosw, p, tempis, temp, sal, alk, dic, sil, phos, patm, depth, lat, 1, &
                optCON='mol/m3', optT='Tpot   ', optP='m ', optB='u74', &
                optK1K2='l  ', optKf='dg', optGAS='Pinsitu', optS='Sprc')
        niter_last = niter_atgen   ! the second BetaD solve (DIC + dx/2), vars.F90:669
        call solver_replica(depth, rhosw, niter_main, ph_rep)

        ! ---- surface: recom_forcing.F90:195, 222-226, 278 ----
        call pistonvel(wind, fice, 1, kw660)
        sfco2 = 0.0_r8   ! flxco2 builds its zero depth as fco2*0.0 (gasx.F90:323)
        call flxco2(co2flux, co2ex, dpco2, sph, spco2, sfco2, sco2, shco3, sco3, somegaa, &
                somegac, sbetad, srhosw, sp, stempis, k0s, temp, sal, alk, dic, sil, phos, &
                kw660, xco2, patm, 10.0_r8, 1, lon, lat, &
                optCON='mol/m3', optT='Tpot   ', optP='m ', optB='u74', optK1K2='l  ', &
                optKf='dg', optGAS='Pinsitu', optS='Sprc')
        niter_surf_last = niter_atgen
        ppo = patm
        call o2flux(temp, sal, kw660, ppo, o2, 1, o2ex)
        call solver_replica((/0.0_r8/), srhosw, niter_surf, sph_rep)

        ! constants at the interior point (diagnostic, to localise a failing output)
        call constants(ak0, ak1, ak2, akb, akw, aks, akf, akspc, akspa, ak1p, ak2p, ak3p, &
                aksi, ast, aft, abt, temp, sal, patm, depth, lat, 1, &
                'Tpot   ', 'm ', 'u74', 'l  ', 'dg', 'Pinsitu')

        j = 0
        call put(ph(1)); call put(pco2(1)); call put(fco2(1)); call put(co2(1))
        call put(hco3(1)); call put(co3(1)); call put(omegaa(1)); call put(omegac(1))
        call put(kspc(1)); call put(betad(1)); call put(rhosw(1)); call put(p(1))
        call put(tempis(1))
        call put(real(niter_main, r8)); call put(real(niter_last, r8))
        if (ph_rep == ph(1)) then
            call put(1.0_r8)
        else
            call put(0.0_r8)
        end if
        ! 16 so far; surface block
        call put(kw660(1)); call put(co2flux(1)); call put(co2ex(1)); call put(dpco2(1))
        call put(sph(1)); call put(spco2(1)); call put(sfco2(1)); call put(sco2(1))
        call put(shco3(1)); call put(sco3(1)); call put(somegaa(1)); call put(somegac(1))
        call put(sbetad(1)); call put(srhosw(1)); call put(sp(1)); call put(stempis(1))
        call put(k0s(1)); call put(o2ex(1))
        call put(real(niter_surf, r8)); call put(real(niter_surf_last, r8))
        if (sph_rep == sph(1)) then
            call put(1.0_r8)
        else
            call put(0.0_r8)
        end if
        ! 37 so far; interior constants
        call put(ak0(1)); call put(ak1(1)); call put(ak2(1)); call put(akb(1))
        call put(akw(1)); call put(aks(1)); call put(akf(1)); call put(akspc(1))
        call put(akspa(1)); call put(ak1p(1)); call put(ak2p(1)); call put(ak3p(1))
        call put(aksi(1)); call put(ast(1)); call put(aft(1)); call put(abt(1))
        ! 53 so far; pad the rest (room for later columns)
        do while (j < nout)
            call put(0.0_r8)
        end do
    end do

    open(11, file=trim(fout), access='stream', form='unformatted', status='replace')
    write(11) n, nout
    write(11) out
    close(11)

contains

    subroutine put(x)
        real(r8), intent(in) :: x
        j = j + 1
        out(j, i) = x
    end subroutine put

    ! The solver set-up of vars_sprac (vars.F90:385-605) for optP='m', optT='Tpot',
    ! optCON='mol/m3', optS='Sprc', then solve_at_general (varsolver.F90:153) with the
    ! module iteration counter; returns the count and the pH it implies.
    subroutine solver_replica(dep, rhosw_in, niter, ph_out)
        real(r8), intent(in) :: dep(1), rhosw_in(1)
        integer, intent(out) :: niter
        real(r8), intent(out) :: ph_out
        real(r8) :: ssal, salk, sdic, ssil, sphos, drho, ta, tc, pt, sit, h
        real(r8), dimension(1) :: sal1

        niter = -1
        ph_out = 1.e20_r8
        if (.not. (dic(1) > 0. .and. dic(1) < 1.0e+4)) return
        ssal = max(sal(1), 0.0_r8)
        sphos = max(phos(1), 0.0_r8)
        ssil = max(sil(1), 0.0_r8)
        sdic = max(dic(1), 0.0_r8)
        salk = max(alk(1), 0.0_r8)
        drho = rhosw_in(1)
        ta = salk / drho
        tc = sdic / drho
        pt = sphos / drho
        sit = ssil / drho
        sal1(1) = ssal
        call constants(ak0, ak1, ak2, akb, akw, aks, akf, akspc, akspa, ak1p, ak2p, ak3p, &
                aksi, ast, aft, abt, temp, sal1, patm, dep, lat, 1, &
                'Tpot   ', 'm ', 'u74', 'l  ', 'dg', 'Pinsitu')
        h = solve_at_general(ta, tc, abt(1), pt, sit, ast(1), aft(1), ak0(1), ak1(1), &
                ak2(1), akb(1), akw(1), aks(1), akf(1), ak1p(1), ak2p(1), ak3p(1), aksi(1))
        niter = niter_atgen
        if (h > 0.d0) ph_out = -1. * log10(h)
    end subroutine solver_replica

end program mocsy_driver
