"""JAX (this port) vs Kokkos-CUDA (M5.24) — SYPD across the mesh/scale ladder.

SYPD = dt / (365 * s_per_step).  Each mesh at its PRODUCTION dt (Kokkos convention):
  CORE2 1800, farc 900 (measured at production dt directly);
  dars/NG5 240 (measured at the cold-stable dt=180, x1.03 CG correction for dt=240).
All JAX numbers are the FULL model (real JRA+PHC+ice), ragged halo, forward, A100.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# s_per_step (seconds). JAX = measured this campaign; Kokkos = M5.24 SCALING_M524.md.
DATA = {
    "CORE2 (127k×48)": dict(dt=1800, cg=1.0, c="tab:blue",
        jax={1: 0.0867, 2: 0.1227},
        kok={1: 0.117, 2: 0.095, 4: 0.112, 8: 0.111}),
    "farc (638k×48)": dict(dt=900, cg=1.0, c="tab:green",
        jax={1: 0.2984},
        kok={1: 0.309, 2: 0.244, 4: 0.210, 8: 0.190, 16: 0.177, 32: 0.256}),
    "dars (3.16M×57)": dict(dt=240, cg=1.03, c="tab:orange",
        jax={2: 0.934, 4: 0.513},
        kok={2: 0.814, 4: 0.475, 8: 0.344, 16: 0.237, 32: 0.211}),
    "NG5 (7.4M×70)": dict(dt=240, cg=1.03, c="tab:red",
        jax={8: 0.840, 16: 0.584},
        kok={2: 2.335, 4: 1.273, 8: 0.810, 16: 0.492, 32: 0.374}),
}


def sypd(dt, cg, s):
    return dt / (365.0 * s * cg)


fig, ax = plt.subplots(figsize=(9, 6.2))
for name, d in DATA.items():
    kn = sorted(d["kok"]); ks = [sypd(d["dt"], d["cg"], d["kok"][n]) for n in kn]
    jn = sorted(d["jax"]); js = [sypd(d["dt"], d["cg"], d["jax"][n]) for n in jn]
    ax.plot(kn, ks, "--s", color=d["c"], mfc="none", lw=1.6, ms=7, alpha=0.85)
    ax.plot(jn, js, "-o", color=d["c"], lw=2.4, ms=8, label=name)
    for n, s in zip(jn, js):                       # annotate JAX points
        ax.annotate(f"{s:.2f}", (n, s), textcoords="offset points", xytext=(6, 6),
                    fontsize=8, color=d["c"], fontweight="bold")

ax.axhline(1.0, color="grey", ls=":", lw=1, alpha=0.7)
ax.text(1.05, 1.04, "1 SYPD (production band)", fontsize=8, color="grey")
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xticks([1, 2, 4, 8, 16, 32]); ax.set_xticklabels([1, 2, 4, 8, 16, 32])
ax.set_xlabel("nodes  (4× A100 / node)"); ax.set_ylabel("SYPD  (simulated years / day)")
ax.set_title("FESOM2→JAX vs Kokkos-CUDA — full-model throughput (SYPD)\n"
             "solid ● = JAX (this port, ragged)   dashed □ = Kokkos M5.24 (hand-tuned CUDA)",
             fontsize=11)
ax.grid(True, which="both", ls=":", alpha=0.4)
# mesh-color legend + a separate JAX/Kokkos style legend
leg1 = ax.legend(title="mesh (JAX points labeled)", loc="upper right", fontsize=9)
from matplotlib.lines import Line2D
style = [Line2D([0], [0], color="k", lw=2.4, marker="o", label="JAX (this port)"),
         Line2D([0], [0], color="k", lw=1.6, ls="--", marker="s", mfc="none", label="Kokkos M5.24")]
ax.add_artist(leg1); ax.legend(handles=style, loc="lower left", fontsize=9)

out = "docs/figures/jax_vs_kokkos_sypd.png"
import os; os.makedirs("docs/figures", exist_ok=True)
fig.tight_layout(); fig.savefig(out, dpi=140)
print("wrote", out)

# also print the table
print(f"\n{'mesh':16s} {'nodes':>6s} {'JAX SYPD':>9s} {'Kokkos SYPD':>12s} {'gap':>7s}")
for name, d in DATA.items():
    for n in sorted(d["jax"]):
        js = sypd(d["dt"], d["cg"], d["jax"][n])
        ks = sypd(d["dt"], d["cg"], d["kok"][n]) if n in d["kok"] else None
        gap = f"{(d['kok'][n]/d['jax'][n]-1)*100:+.0f}%" if n in d["kok"] else "  n/a"
        kstr = f"{ks:12.2f}" if ks else f"{'n/a':>12s}"
        print(f"{name:16s} {n:6d} {js:9.2f} {kstr} {gap:>7s}")
