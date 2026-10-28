"""The page for REcoM's authors from make_figures.py's output (any Python 3).

    python scripts/recom/page/build_page.py --fig <make_figures out> --out <page dir> [--preview]

Writes ``<page dir>/index.html`` and copies the figures and videos next to it (published paths = file names).
``--preview`` adds the banner that the figures come from the one-year test run.
"""
from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path

# speed measured on Levante (model years per simulated day, 45-minute step, CORE2 + REcoM, steady stepping)
SPEED = [("1 node, 4 × NVIDIA GH200", "123", "227"), ("4 nodes, 16 × NVIDIA A100", "222", "258")]


def fmt(x, d=1, sign=False):
    return (f"{x:+.{d}f}" if sign else f"{x:.{d}f}").replace("-", "−")


def page(s: dict, preview: bool) -> str:
    p, figs = s["period"], s["figures"]
    y0, y1 = s["map_years"]
    period = f"{y0}" if y0 == y1 else f"{y0}–{y1}"
    cy0, cy1 = s["chl_obs_years"]
    my0, my1 = s["clim_years"]
    mclim = f"{my0}" if my0 == my1 else f"{my0}–{my1}"
    pi = "test" in s["label"]
    st = s["surface_stats"]
    ratio = 10 ** p["chl_log_bias"]
    banner = ("""<p class="banner"><strong>Preview.</strong> These figures come from a one-year test run (1990, cold start,
    preindustrial carbon: 278 ppm and preindustrial DIC). The 2000–2019 hindcast with observed atmospheric CO<sub>2</sub>
    is running now, and this page will switch to it.</p>""" if preview else "")
    carbon_note = ("""The test year runs with preindustrial carbon (278 ppm, GLODAPv2 preindustrial DIC), so its fluxes
    are not comparable with today's ocean sink.""" if pi else """For comparison, the Global Carbon Budget puts the ocean
    sink at about 2.5 Pg C yr<sup>−1</sup> for 2010–2019. This run had no preindustrial spin-up and has no control
    run, so its net flux mixes the uptake of anthropogenic carbon with the drift from the initial state.""")
    dic_ref = "preindustrial" if pi else "present-day (normalised to 2002)"
    spin = s["years"][0] - 1
    ya, yb = s["years"]
    a0, a1 = s["annual"][str(ya)], s["annual"][str(yb)]
    runs = ("a one-year CORE2 test run" if pi else
            f"a {ya}–{yb} CORE2 hindcast with observed atmospheric CO<sub>2</sub>")
    drift = ("" if ya == yb else f"""<p>Global NPP falls from {fmt(a0['npp'])} Pg C yr<sup>−1</sup> in {ya} to
  {fmt(a1['npp'])} in {yb}, and export at 100 m from {fmt(a0['export100'], 2)} to {fmt(a1['export100'], 2)}:
  {yb - ya + 1} years after the cold start, the ecosystem has not reached a steady state.</p>""")
    return f"""<title>REcoM in fesom-jax</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,600&family=Source+Sans+3:wght@400;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: one reading column for text, figures break out to a wider measure; a chlorophyll-green accent, carbon blue as second voice */
:root {{
  --paper: #f5f8f6; --ink: #14211e; --muted: #55665f; --rule: #d3ded9; --card: #ffffff;
  --chl: #17795a; --carbon: #2a5a9b; --warn-bg: #fff4dc; --warn-ink: #6a4a00;
  --display: "Newsreader", "Iowan Old Style", Georgia, serif;
  --body: "Source Sans 3", "Segoe UI", system-ui, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --paper: #0f1614; --ink: #e2ebe7; --muted: #9db0a8; --rule: #26352f; --card: #f7f9f8;
  --chl: #4cc493; --carbon: #86aee8; --warn-bg: #3a2f12; --warn-ink: #f3d68f; color-scheme: dark }} }}
:root[data-theme="dark"] {{
  --paper: #0f1614; --ink: #e2ebe7; --muted: #9db0a8; --rule: #26352f; --card: #f7f9f8;
  --chl: #4cc493; --carbon: #86aee8; --warn-bg: #3a2f12; --warn-ink: #f3d68f; color-scheme: dark }}
body {{ background: var(--paper); color: var(--ink); font: 17px/1.55 var(--body); }}
.wrap {{ max-width: 1180px; margin: 0 auto; padding-left: max(16px, env(safe-area-inset-left, 0px));
  padding-right: max(16px, env(safe-area-inset-right, 0px)); padding-block: 40px 72px; display: grid; gap: 56px; }}
.prose {{ max-width: 68ch; }}
.prose p {{ margin: 0 0 0.8em; }}
h1, h2 {{ font-family: var(--display); font-weight: 600; text-wrap: balance; margin: 0; line-height: 1.1; }}
h1 {{ font-size: clamp(2.4rem, 6vw, 4rem); letter-spacing: -0.01em; }}
h2 {{ font-size: clamp(1.6rem, 3.4vw, 2.2rem); margin-bottom: 0.4em; }}
.eyebrow {{ font: 500 0.78rem/1 var(--mono); letter-spacing: 0.08em; text-transform: uppercase; color: var(--chl); margin: 0 0 14px; }}
.lede {{ font-size: 1.2rem; color: var(--ink); max-width: 62ch; margin: 18px 0 0; }}
.banner {{ background: var(--warn-bg); color: var(--warn-ink); padding: 12px 16px; border-radius: 6px; max-width: 80ch; margin: 22px 0 0; }}
section {{ display: grid; gap: 18px; }}
figure {{ margin: 0; display: grid; gap: 8px; }}
figure .plate {{ background: var(--card); border: 1px solid var(--rule); border-radius: 4px; overflow: hidden; }}
figure img, figure video {{ display: block; width: 100%; height: auto; max-width: 100%; }}
figcaption {{ color: var(--muted); font-size: 0.93rem; max-width: 90ch; }}
.numbers {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1px; background: var(--rule);
  border-block: 1px solid var(--rule); }}
.num {{ background: var(--paper); padding: 16px 18px 14px; display: grid; gap: 4px; align-content: start; min-width: 0; }}
.num b {{ font: 600 2.1rem/1 var(--display); font-variant-numeric: tabular-nums; }}
.num span {{ font-size: 0.9rem; color: var(--muted); }}
.num .unit {{ font: 400 0.8rem var(--mono); color: var(--muted); }}
.num.carbon b {{ color: var(--carbon); }} .num.chl b {{ color: var(--chl); }}
.pair {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 24px; }}
.pair > * {{ min-width: 0; }}
table {{ border-collapse: collapse; font-variant-numeric: tabular-nums; min-width: 420px; }}
th, td {{ text-align: left; padding: 8px 14px 8px 0; border-bottom: 1px solid var(--rule); vertical-align: top; }}
th {{ font: 600 0.8rem var(--mono); letter-spacing: 0.05em; text-transform: uppercase; color: var(--muted); }}
td.n {{ font: 500 1.05rem var(--mono); }}
.scroll {{ overflow-x: auto; }}
dl {{ display: grid; grid-template-columns: minmax(150px, 220px) 1fr; gap: 10px 22px; margin: 0; max-width: 100ch; }}
dt {{ font-weight: 600; }} dd {{ margin: 0; min-width: 0; }}
ul.plain {{ margin: 0; padding-left: 1.1em; max-width: 80ch; display: grid; gap: 6px; }}
code {{ font: 0.88em var(--mono); }}
a {{ color: var(--chl); }}
a:focus-visible {{ outline: 2px solid var(--chl); outline-offset: 2px; }}
@media (max-width: 560px) {{ dl {{ grid-template-columns: 1fr; gap: 2px 0; }} dd {{ margin-bottom: 10px; }} body {{ font-size: 16px; }} }}
@media (prefers-reduced-motion: reduce) {{ video {{ animation: none; }} }}
</style>

<div class="wrap">
<header>
  <p class="eyebrow">FESOM2 + REcoM-2 · bin_2p3z2d · CORE2 mesh · JRA55-do</p>
  <h1>REcoM on GPUs</h1>
  <p class="lede">REcoM-2 now runs inside fesom-jax, the JAX version of FESOM2: the ocean–sea-ice model and the
  ecosystem step together on GPUs, with the same equations and parameters as FESOM2-REcoM. Below is {runs},
  next to satellite chlorophyll and the WOA and GLODAP climatologies.</p>
  {banner}
</header>

<section aria-labelledby="season">
  <figure>
    <div class="plate"><video src="{figs['chl_season']}" autoplay muted loop playsinline controls
      aria-label="Monthly surface chlorophyll, REcoM next to GlobColour"></video></div>
    <figcaption id="season">The seasonal cycle of surface chlorophyll a. Left: REcoM, monthly means of {mclim}.
    Right: GlobColour merged satellite product, monthly climatology {cy0}–{cy1}. The model is shown only where the
    satellite has data in that month.</figcaption>
  </figure>
  <div class="numbers">
    <div class="num chl"><b>{fmt(p['npp'])}</b><span>net primary production</span><span class="unit">Pg C yr⁻¹, {period}</span></div>
    <div class="num chl"><b>{fmt(p['export100'], 2)}</b><span>POC export at 100 m, e-ratio {fmt(p['e_ratio'], 2)}</span><span class="unit">Pg C yr⁻¹</span></div>
    <div class="num carbon"><b>{fmt(p['co2flux'], 2)}</b><span>air–sea CO₂ flux into the ocean</span><span class="unit">Pg C yr⁻¹</span></div>
    <div class="num"><b>123</b><span>model years per day on one GPU node, double precision</span><span class="unit">CORE2 + REcoM</span></div>
  </div>
</section>

<section aria-labelledby="chl">
  <div class="prose"><h2 id="chl">Chlorophyll against the satellites</h2>
  <p>Annual-mean surface chlorophyll over the months where both the model and GlobColour have values. On a log
  scale the patterns correlate with r = {fmt(p['chl_log_corr'], 2)}; the area-weighted geometric mean of
  model / satellite is {fmt(ratio, 2)}.</p></div>
  <figure><div class="plate"><img src="{figs['chl_maps']}" alt="Annual-mean surface chlorophyll: REcoM, GlobColour, ratio"></div>
    <figcaption>Left: REcoM ({mclim}). Middle: GlobColour ({cy0}–{cy1}). Right: model / satellite, red where the
    model has more chlorophyll.</figcaption></figure>
  <figure><div class="plate"><img src="{figs['chl_zonal']}" alt="Zonal-mean chlorophyll by month"></div>
    <figcaption>Zonal-mean surface chlorophyll through the year: the spring bloom moving poleward in each hemisphere.</figcaption></figure>
  <figure><div class="plate"><img src="{figs['chl_regions']}" alt="Seasonal chlorophyll in four regions"></div>
    <figcaption>Seasonal cycles in four regions, area means over the cells the satellite sees in each month.</figcaption></figure>
</section>

<section aria-labelledby="npp">
  <div class="prose"><h2 id="npp">Primary production and export</h2>
  <p>Global NPP is {fmt(p['npp'])} Pg C yr<sup>−1</sup> and POC export at 100 m {fmt(p['export100'], 2)} Pg C
  yr<sup>−1</sup> ({period}). Satellite-based NPP estimates are around 50 Pg C yr<sup>−1</sup>, with a wide spread
  between algorithms; export estimates at 100 m range from about 4 to 12 Pg C yr<sup>−1</sup>.</p></div>
  <figure><div class="plate"><img src="{figs['npp_export']}" alt="Maps of NPP and export at 100 m"></div>
    <figcaption>Annual means, {period}.</figcaption></figure>
</section>

<section aria-labelledby="carbon">
  <div class="prose"><h2 id="carbon">Carbon</h2>
  <p>The ocean takes up {fmt(p['co2flux'], 2)} Pg C yr<sup>−1</sup> ({period}); the area-mean surface pCO<sub>2</sub>
  is {fmt(p['pco2surf'])} µatm. {carbon_note}</p></div>
  <figure><div class="plate"><video src="{figs['run']}" autoplay muted loop playsinline controls
      aria-label="Every month of the run: chlorophyll and air-sea CO2 flux"></video></div>
    <figcaption>Every month of the run: surface chlorophyll (left) and the air–sea CO<sub>2</sub> flux (right, blue =
    outgassing, red = uptake).</figcaption></figure>
  <figure><div class="plate"><img src="{figs['co2']}" alt="Maps of air-sea CO2 flux and surface pCO2"></div>
    <figcaption>Annual-mean air–sea CO<sub>2</sub> flux and surface ocean pCO<sub>2</sub>, {period}.</figcaption></figure>
  <figure><div class="plate"><img src="{figs['carbon']}" alt="Surface DIC and alkalinity against GLODAPv2"></div>
    <figcaption>Surface DIC against the GLODAPv2 {dic_ref} field (mean difference {fmt(st['DIC']['bias'], 1, True)}
    mmol m<sup>−3</sup>) and alkalinity against GLODAPv2 ({fmt(st['Alk']['bias'], 1, True)} mmol m<sup>−3</sup>).
    Both started from these fields.</figcaption></figure>
</section>

<section aria-labelledby="nut">
  <div class="prose"><h2 id="nut">Nutrients and oxygen</h2>
  <p>Surface nitrate and silicate against WOA13 and oxygen at 400 m against WOA18, annual means {period}. The run
  started from these climatologies, so the differences show where the model moves away from them.</p></div>
  <figure><div class="plate"><img src="{figs['nutrients']}" alt="Surface nitrate and silicate against WOA13"></div>
    <figcaption>Surface DIN: mean difference {fmt(st['DIN']['bias'], 2, True)}, rms {fmt(st['DIN']['rmse'], 2)} mmol
    m<sup>−3</sup>. Surface DSi: {fmt(st['DSi']['bias'], 2, True)}, rms {fmt(st['DSi']['rmse'], 2)} mmol
    m<sup>−3</sup>.</figcaption></figure>
  <figure><div class="plate"><img src="{figs['o2']}" alt="Oxygen at 400 m against WOA18"></div>
    <figcaption>Oxygen near 400 m, where the oxygen minimum zones sit.</figcaption></figure>
  <figure><div class="plate"><img src="{figs['sections']}" alt="Zonal-mean nitrate and oxygen sections"></div>
    <figcaption>Global zonal means of nitrate and oxygen (top) and their difference from WOA (bottom).</figcaption></figure>
</section>

<section aria-labelledby="ts">
  <div class="prose"><h2 id="ts">Global numbers through the run</h2>
  {drift}</div>
  <figure><div class="plate"><img src="{figs['timeseries']}" alt="Monthly global NPP, export, CO2 flux and inventories"></div>
    <figcaption>Monthly global NPP, export, air–sea CO<sub>2</sub> flux, and the change of the global inventories of
    DIC, alkalinity, DIN, DSi, DFe and O<sub>2</sub> since the first month.</figcaption></figure>
</section>

<section aria-labelledby="setup">
  <div class="prose"><h2 id="setup">The setup</h2></div>
  <dl>
    <dt>REcoM</dt><dd>REcoM-2, the bin_2p3z2d configuration of FESOM2: small phytoplankton and diatoms, three
    zooplankton, two detritus classes, calcite, benthos, mocsy carbonate chemistry, alkalinity restoring; 31 tracers.
    REcoM version 692f804 (the one FESOM2 main pins) with the namelist of FESOM2's bin_2p3z2d configuration.</dd>
    <dt>Ocean–sea-ice</dt><dd>FESOM2 physics as in fesom-jax: z* coordinate, GM/Redi, TKE mixing, mEVP sea ice.
    CORE2 mesh: 126,858 surface nodes, 47 layers. 45-minute time step for physics and biology alike.</dd>
    <dt>Forcing</dt><dd>JRA55-do, including sea-level pressure for the gas exchange. Albani monthly dust for iron.
    No rivers, no erosion, no aeolian nitrogen.</dd>
    <dt>Start</dt><dd>Physics from a 1958–{spin} fesom-jax hindcast. Biology cold-started from WOA13 nitrate and silicate,
    WOA18 oxygen, GLODAPv2 DIC and alkalinity, and the PISCES iron field, as in the FESOM2 configuration.</dd>
    <dt>Atmospheric CO₂</dt><dd>The monthly Global Carbon Budget 2023 series, read for each model year.</dd>
    <dt>Precision</dt><dd>Double, as FESOM2-REcoM.</dd>
  </dl>
</section>

<section aria-labelledby="close">
  <div class="prose"><h2 id="close">How close is this to FESOM2-REcoM</h2>
  <p>The port follows the Fortran routine by routine. We ran FESOM2-REcoM and fesom-jax from the same state on the
  small pi test mesh and compared them:</p></div>
  <ul class="plain">
    <li>After one time step, all 31 tracers agree to 2 × 10<sup>−8</sup> relative. Temperature in the two models
    already differs by 4 × 10<sup>−8</sup> after that step, from round-off alone.</li>
    <li>After 30 days, the global inventories agree to 1 × 10<sup>−9</sup>.</li>
    <li>Tracer mass is conserved to 5 × 10<sup>−15</sup> per step.</li>
    <li>Several GPUs give the same result as one, to 4 × 10<sup>−13</sup>.</li>
    <li>On CORE2, a year in single precision gives the same annual-mean NPP, export and CO<sub>2</sub> flux as double
    precision to three digits.</li>
  </ul>
  <div class="prose"><p>Where we knowingly differ: the CO<sub>2</sub> series is read every model year (FESOM2 reads it
  at the start of each run segment, which is the same in yearly segments), and single precision is offered as an
  option, which FESOM2 does not allow with REcoM.</p></div>
</section>

<section aria-labelledby="speed">
  <div class="prose"><h2 id="speed">Speed</h2>
  <p>Model years per day, CORE2 with REcoM. A 20-year hindcast takes about four hours on one GPU node.</p></div>
  <div class="scroll"><table>
    <thead><tr><th>Hardware</th><th>Double</th><th>Single</th></tr></thead>
    <tbody>{''.join(f'<tr><td>{html.escape(h)}</td><td class="n">{d}</td><td class="n">{s_}</td></tr>' for h, d, s_ in SPEED)}</tbody>
  </table></div>
  <div class="prose"><p>The ocean–sea-ice model alone reaches about 1,000 model years per day on one GPU node, in
  single precision.</p></div>
</section>

<section aria-labelledby="run">
  <div class="prose"><h2 id="run">What you can run now</h2></div>
  <ul class="plain">
    <li>CORE2 hindcasts with REcoM from any year of our 1958–2019 physics hindcast, with constant or observed
    atmospheric CO<sub>2</sub>.</li>
    <li>Double or single precision (single is about 1.8 times faster).</li>
    <li>One GPU node or several.</li>
    <li>Derivatives of any model output with respect to REcoM parameters (for example, how export depends on the
    sinking or grazing parameters), checked against finite differences on the pi mesh; not yet tried on CORE2.</li>
    <li>Not yet ported: coccolithophores and Phaeocystis, rivers, erosion, aeolian nitrogen, carbon isotopes, the
    atmospheric CO<sub>2</sub> box, the Geider limiter, MEDUSA sediments.</li>
  </ul>
</section>
</div>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()
    s = json.loads((args.fig / "summary.json").read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    for f in s["figures"].values():
        shutil.copy2(args.fig / f, args.out / f)
    (args.out / "index.html").write_text(page(s, args.preview))
    print("wrote", args.out / "index.html", "+", len(s["figures"]), "figures")


if __name__ == "__main__":
    main()
