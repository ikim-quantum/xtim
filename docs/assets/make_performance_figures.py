"""Regenerate the figures in docs/performance.md (light and dark PNGs) from the measured numbers below.

usage: python docs/assets/make_performance_figures.py   (needs matplotlib)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

HERE = os.path.dirname(os.path.abspath(__file__))

TOOLS = ["xtim 4.1.1", "tsim 1.0.0", "Clifft 0.13.0", "SymFT 2026.10.8", "Stim 1.16.0"]
FAIL = "did not run"

# ns per attempted shot on the clifft-bench workloads, per-shot detector and observable records (see performance.md).
# One entry per tool in TOOLS order: a number, FAIL, or None (not applicable).
NO_PS = [
    ("distillation_15to1_rm15", [14.4, 48.5, 17.2, 618.8, None]),
    ("msc_d3_inject_cultivate", [42.2, 82.2, 400.2, 464.8, None]),
    ("pure_surface_d7_r7", [84.0, 114.5, 655.7, 1283.7, 904.0]),
    ("reed_muller_measured_sampling", [44.1, 37.7, 21.5, 454.3, None]),
    ("msc_d5_inject_cultivate", [497.5, 2028.3, 15977.3, 38108.2, None]),
    ("msc_full_d5", [1894.3, 3899.9, 27054.7, 41160.0, None]),
    ("msc_proxy_d7_unverified", [2654.1, 4362.4, 14372.0, 22151.4, None]),
    ("bt27_direct_x (noiseless)", [168.6, FAIL, 4232.4, FAIL, None]),
    ("bt81_direct_x (noiseless)", [170.5, FAIL, 1617.7, FAIL, None]),
]
ALL_PS = [
    ("distillation_15to1_rm15", [14.9, 65.1, 43.4, 759.0, None]),
    ("msc_d3_inject_cultivate", [32.2, 67.9, 447.0, 451.6, None]),
    ("pure_surface_d7_r7", [71.3, 109.3, 628.4, 277.4, 963.2]),
    ("msc_d5_inject_cultivate", [74.3, 111.1, 4863.4, 5050.4, None]),
    ("msc_full_d5 (selective mask)", [1360.2, 1691.1, 27053.0, "no subset post-selection", None]),
    ("msc_proxy_d7_unverified", [161.3, 204.1, 5365.1, 1399.2, None]),
]

# Seconds to estimate the post-selected logical error rate of d3 cultivation to +-10 % (1 sigma).
TTP = [  # (route, p=1e-3 s, p=5e-4 s)
    ("xtim check", 5.1, 18.1),
    ("tsim check", 11.8, 53.0),
    ("xtim fidelity", 33.1, 171.0),
    ("Clifft EXP_VAL", 39.7, 163.0),
    ("Clifft check", 78.8, 347.0),
    ("Magna fidelity", 750.0, 2520.0),
]


def dur(s):
    return f"{s:.3g} s" if s < 90 else f"{s / 60:.3g} min"


THEMES = {
    "light": dict(bg="white", fg="#1d2430", muted="#5a6573", grid="#e3e6eb",
                  tools=["#16735f", "#b85c1a", "#2457a6", "#8a5cb8", "#6b7280"], p1="#2457a6", p2="#8a5cb8"),
    "dark": dict(bg="#0d1117", fg="#e3e8ef", muted="#9aa5b3", grid="#2b333e",
                 tools=["#3fbf9f", "#e08a4a", "#7fa8ec", "#c39ae8", "#a3aab5"], p1="#7fa8ec", p2="#c39ae8"),
}
MARKERS = ["D", "s", "o", "^", "v"]


def rate(x):
    for unit, f in (("G", 1e9), ("M", 1e6), ("k", 1e3)):
        if x >= f:
            return f"{x / f:.3g}{unit}"
    return f"{x:.3g}"


def throughput(rows, title, fname, theme):
    t = THEMES[theme]
    used = [j for j in range(len(TOOLS)) if any(v[j] is not None for _, v in rows)]
    fig, ax = plt.subplots(figsize=(10, 0.95 * len(rows) * len(used) / 4 + 2.2), facecolor=t["bg"])
    lo, hi = 1e4, 3e8
    ax.set_xlim(lo, hi)
    step = 0.8 / len(used)
    for i, (label, vals) in enumerate(rows):
        base = len(rows) - 1 - i
        for k, j in enumerate(used):
            v, color = vals[j], t["tools"][j]
            if v is None:
                continue
            y = base + 0.4 - k * step
            if isinstance(v, str):
                ax.text(lo * 1.3, y, f"{TOOLS[j]}: {v}", color=t["muted"], va="center", fontsize=8.5, style="italic")
                continue
            r = 1e9 / v
            ax.hlines(y, lo, r, color=color, alpha=0.28, lw=3)
            ax.plot(r, y, MARKERS[j], color=color, ms=7)
            ax.text(r * 1.25, y, rate(r) + "/s", color=color, va="center", fontsize=8.5, fontweight="bold")
    ax.set_yticks([len(rows) - 1 - i + 0.4 - (len(used) - 1) * step / 2 for i in range(len(rows))])
    ax.set_yticklabels([r[0] for r in rows], fontsize=9.5, color=t["fg"], family="monospace")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: rate(x)))
    ax.set_xlabel("Attempted shots per second (higher is better)", color=t["fg"], fontsize=11)
    ax.set_title(title, color=t["fg"], fontsize=12, loc="left", pad=28)
    ax.tick_params(axis="x", colors=t["muted"])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=t["grid"], lw=0.8)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_facecolor(t["bg"])
    for j in used:
        ax.plot([], [], MARKERS[j], color=t["tools"][j], label=TOOLS[j], ms=7)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(used), frameon=False, labelcolor=t["fg"], fontsize=9.5)
    fig.text(0.01, 0.01, "One pinned CPU core · median of two runs · p = 10⁻³ unless noiseless · per-shot records · 2026-11",
             color=t["muted"], fontsize=8.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(HERE, f"{fname}-{theme}.png"), dpi=130, facecolor=t["bg"])
    plt.close(fig)


def time_to_precision(theme):
    t = THEMES[theme]
    fig, ax = plt.subplots(figsize=(9.6, 5.2), facecolor=t["bg"])
    lo = 2
    ax.set_xlim(lo, 7e3)
    for i, (label, a, b) in enumerate(TTP):
        base = len(TTP) - 1 - i
        for k, (v, color) in enumerate(((a, t["p1"]), (b, t["p2"]))):
            y = base + 0.2 - 0.4 * k
            ax.hlines(y, lo, v, color=color, alpha=0.3, lw=3)
            ax.plot(v, y, "o", color=color, ms=9)
            ax.text(v * 1.25, y, dur(v), color=color, va="center", fontsize=10, fontweight="bold")
    ax.set_yticks(range(len(TTP)))
    ax.set_yticklabels([r[0] for r in TTP][::-1], fontsize=10, color=t["fg"], family="monospace")
    ax.set_xscale("log")
    ax.set_xticks([10, 60, 600, 3600])
    ax.set_xticklabels(["10 s", "1 min", "10 min", "1 h"])
    ax.minorticks_off()
    ax.set_xlabel("Time to the logical error rate of d3 cultivation, ±10 % (1σ)", color=t["fg"], fontsize=12)
    ax.tick_params(axis="x", colors=t["muted"])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=t["grid"], lw=0.8)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_facecolor(t["bg"])
    for name, color in (("p = 10⁻³", t["p1"]), ("p = 5·10⁻⁴", t["p2"])):
        ax.plot([], [], "o", color=color, label=name, ms=8)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, labelcolor=t["fg"], fontsize=11)
    fig.text(0.02, 0.01, "One pinned CPU core · simulation time only (compile excluded) · lower is better",
             color=t["muted"], fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(HERE, f"time-to-precision-{theme}.png"), dpi=130, facecolor=t["bg"])
    plt.close(fig)


if __name__ == "__main__":
    for theme in THEMES:
        throughput(NO_PS, "Without post-selection", "throughput-unselected", theme)
        throughput(ALL_PS, "Post-selected on every detector", "throughput-selected", theme)
        time_to_precision(theme)
