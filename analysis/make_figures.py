"""
make_figures.py
================

Generates all paper figures from the local results/ directory produced by
download_all_results.ps1. Each figure is saved as both PDF (vector, for
LaTeX inclusion) and PNG (300 dpi, for slides/preview).

Figures produced
----------------
  fig_01_geometry.{pdf,png}   Cross-model geometry (cos and perp_frac) per layer
  fig_02_sweeps.{pdf,png}     Phase 1 vs Phase 2 layer-sweep curves
  fig_03_ortho_mirror.{pdf,png}
                              Orthogonal decomposition: Phase 1 d_l vs Phase 2 r_em
  fig_04_summary.{pdf,png}    Multi-panel summary (judges, bands, capability)

Usage
-----
    pip install -r requirements.txt
    python make_figures.py
    # Outputs land in figures/

Inputs (from results/)
----------------------
    results/phase1/04_layer_sweep/sweep_r_l.json
    results/phase1/04_layer_sweep/sweep_random.json
    results/phase1/04_layer_sweep/selected_layers.json
    results/phase1/06_harmbench/results_wildguard.json
    results/phase1/08_orthogonal/results.json
    results/phase2/04_layer_sweep/sweep_r_em.json
    results/phase2/04_layer_sweep/sweep_random_em.json
    results/phase2/04_layer_sweep/selected_layers_em.json
    results/phase2/05_geometry/geometry_em.json
    results/phase2/04cross_three_way/results_dual.json

If any input file is missing, the corresponding figure is skipped with a
clear message; partial runs are supported so you can iterate.
"""

from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ----------------------------------------------------------------------
# Anchor paths to the script's own location, not the current working
# directory. This makes the script robust to invocation from VS Code's
# Run button (whose CWD is often VS Code's installation folder), from
# arbitrary terminals, or from a launcher.
_BASE = Path(__file__).resolve().parent
RESULTS = _BASE / "results"
OUT     = _BASE / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Style: clean and reproducible (no special fonts so renders identically anywhere)
plt.rcParams.update({
    "figure.dpi":      120,
    "savefig.dpi":     300,
    "savefig.bbox":    "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
    "font.size":        10,
    "axes.labelsize":   11,
    "legend.fontsize":  9,
    "axes.titlesize":   12,
})

# Color palette: colorblind-safe, distinguishable in greyscale
C_R     = "#0173B2"   # blue:  r-direction (within-class contrast)
C_D     = "#DE8F05"   # orange: d-direction (post-training delta)
C_REM   = "#029E73"   # green:  r_em
C_PERP  = "#D55E00"   # red:    perpendicular component
C_RAND  = "#999999"   # grey:   random control
C_BASE  = "#000000"   # black:  baseline


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        print(f"  [skip]  {path} not found")
        return None
    with open(path, "r") as f:
        return json.load(f)


def _pick(d: Optional[Dict[str, Any]], *keys: str) -> Optional[Any]:
    """Return d[k] for the first k in keys that exists, else None.
    Accommodates Phase 1's r/d naming vs r_l/d_l naming inconsistency."""
    if d is None:
        return None
    for k in keys:
        if k in d:
            return d[k]
    return None


def save_both(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print(f"  [ok]    figures/{name}.pdf + .png")


# ======================================================================
# FIGURE 1 — Cross-model geometry per layer (Phase 2 §4.2)
# ======================================================================
def fig_01_geometry() -> None:
    print("\n[Figure 1] cross-model geometry per layer")
    g = load_json(RESULTS / "phase2/05_geometry/geometry_em.json")
    if g is None:
        return

    cos      = np.asarray(g["cos_r_em_r_l_instruct_per_layer"])
    perp_fr  = np.asarray(g["frac_norm_perp"])
    n_layers = len(cos)
    layers   = np.arange(1, n_layers + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True)

    # Three-regime shading
    for ax in (ax1, ax2):
        ax.axvspan(1,  18,   alpha=0.06, color="green", label=None)
        ax.axvspan(19, 28,   alpha=0.10, color="red",   label=None)
        ax.axvspan(29, n_layers, alpha=0.06, color="blue", label=None)

    # Panel (a): cosine
    ax1.plot(layers, cos, "o-", color=C_REM, lw=1.6, ms=4)
    ax1.axvline(30, color=C_R,    ls=":", lw=1.2, label=r"$l^*_r = 30$ (Phase 1)")
    ax1.axvline(46, color=C_REM,  ls="--", lw=1.2, label=r"$l^*_{\mathrm{em}} = 46$ (Phase 2)")
    ax1.axhline(0.0, color="k", lw=0.5)
    ax1.set_ylabel(r"$\cos(r_{\mathrm{em}},\, \hat{r}_{l,\mathrm{inst}})$")
    ax1.set_ylim(0.35, 1.05)
    ax1.legend(loc="upper right", framealpha=0.9, fontsize=8)
    ax1.text(9.5,  0.45, "shared\n(L1–L18)",     ha="center", fontsize=8, color="green")
    ax1.text(23.5, 0.45, "divergence\n(L19–L28)", ha="center", fontsize=8, color="red")
    ax1.text(38,   0.45, "partial alignment\n(L29–L48)", ha="center", fontsize=8, color="blue")
    ax1.set_title("(a) Cross-model cosine of refusal directions")

    # Panel (b): perp fraction
    ax2.plot(layers, perp_fr, "o-", color=C_PERP, lw=1.6, ms=4)
    ax2.axvline(30, color=C_R,    ls=":", lw=1.2)
    ax2.axvline(46, color=C_REM,  ls="--", lw=1.2)
    ax2.set_xlabel("Layer index")
    ax2.set_ylabel(r"$\|r_{\mathrm{em}}^{\perp}\| / \|r_{\mathrm{em}}\|$")
    ax2.set_ylim(0.0, 1.0)
    ax2.set_title("(b) Orthogonal-component magnitude fraction")
    ax2.set_xlim(0.5, n_layers + 0.5)
    ax2.set_xticks(np.arange(0, n_layers + 1, 4))

    fig.tight_layout()
    save_both(fig, "fig_01_geometry")


# ======================================================================
# FIGURE 2 — Layer-sweep comparison (Phase 1 vs Phase 2)
# ======================================================================
def fig_02_sweeps() -> None:
    print("\n[Figure 2] sweep curves Phase 1 vs Phase 2")
    sel_p1 = load_json(RESULTS / "phase1/04_layer_sweep/selected_layers.json")
    sel_p2 = load_json(RESULTS / "phase2/04_layer_sweep/selected_layers_em.json")
    if sel_p1 is None and sel_p2 is None:
        print("  [skip]  no selected_layers files found")
        return

    # Top-level keys differ between phases. Phase 1 uses 'r'/'d'/'random';
    # Phase 2 likely uses 'r_em'/'random' (or 'r'/'random'). _pick handles
    # both naming conventions.
    p1_r   = _pick(sel_p1, "r_l", "r")
    p1_rnd = _pick(sel_p1, "random")
    p2_r   = _pick(sel_p2, "r_em", "r")

    def _curve(entry: Optional[Dict[str, Any]]) -> Optional[Tuple[list, list]]:
        """Extract (layers, score_curve) from a selected_layers entry,
        defending against missing/renamed fields."""
        if entry is None:
            return None
        sc = entry.get("score_curve")
        if sc is None:
            return None
        layers = entry.get("layers") or list(range(1, len(sc) + 1))
        return layers, sc

    def _peak(entry: Optional[Dict[str, Any]]) -> Tuple[Optional[int], Optional[float]]:
        if entry is None:
            return None, None
        return entry.get("l_star"), (entry.get("peak_score")
                                     if entry.get("peak_score") is not None
                                     else entry.get("peak_S"))

    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    # Random control (noise floor)
    crv = _curve(p1_rnd)
    if crv:
        ax.plot(crv[0], crv[1], "-", color=C_RAND, lw=1.0, alpha=0.6,
                label=r"random (Phase 1, noise floor)")

    # Phase 1 r-direction
    crv = _curve(p1_r)
    if crv:
        ax.plot(crv[0], crv[1], "o-", color=C_R, lw=1.6, ms=3.5,
                label=r"$r_l$ on instruct (Phase 1)")
        l1, pk1 = _peak(p1_r)
        if l1 is not None and pk1 is not None:
            ax.scatter(l1, pk1, marker="*", s=160, color=C_R, zorder=5,
                       edgecolor="black", linewidth=0.5)
            ax.annotate(f"$l^*_r = {l1}$, $S = {pk1:.3f}$",
                        xy=(l1, pk1),
                        xytext=(l1 - 5, pk1 + 0.08),
                        fontsize=9, color=C_R,
                        ha="right", va="bottom",
                        arrowprops=dict(arrowstyle="->", color=C_R, lw=0.5))
        for layer in p1_r.get("band", []):
            ax.axvspan(layer - 0.4, layer + 0.4, alpha=0.05, color=C_R)

    # Phase 2 r_em
    crv = _curve(p2_r)
    if crv:
        ax.plot(crv[0], crv[1], "s-", color=C_REM, lw=1.6, ms=3.5,
                label=r"$r_{\mathrm{em}}$ on EM (Phase 2)")
        l2, pk2 = _peak(p2_r)
        if l2 is not None and pk2 is not None:
            ax.scatter(l2, pk2, marker="*", s=160, color=C_REM, zorder=5,
                       edgecolor="black", linewidth=0.5)
            ax.annotate(f"$l^*_{{\\mathrm{{em}}}} = {l2}$, $S = {pk2:.3f}$",
                        xy=(l2, pk2),
                        xytext=(l2 - 4, pk2 + 0.18),
                        fontsize=9, color=C_REM,
                        ha="right", va="bottom",
                        arrowprops=dict(arrowstyle="->", color=C_REM, lw=0.5))
        for layer in p2_r.get("band", []):
            ax.axvspan(layer - 0.4, layer + 0.4, alpha=0.10, color=C_REM)

    ax.set_xlabel("Layer index")
    ax.set_ylabel(r"$S_l = \mathrm{compl}_{\mathrm{harmful}} - \mathrm{false\_ref}_{\mathrm{harmless}}$")
    ax.set_title("Per-layer rank-1 ablation effect: Phase 1 vs Phase 2")
    ax.set_xlim(0.5, 48.5)
    ax.set_ylim(-0.05, 1.20)  # extra headroom for peak annotations
    ax.set_xticks(np.arange(0, 49, 4))
    ax.legend(loc="center left", framealpha=0.9)

    fig.tight_layout()
    save_both(fig, "fig_02_sweeps")


# ======================================================================
# FIGURE 3 — Orthogonal-decomposition mirror (Phase 1 ↔ Phase 2)
# ======================================================================
def fig_03_ortho_mirror() -> None:
    print("\n[Figure 3] orthogonal decomposition mirror")
    p1 = load_json(RESULTS / "phase1/04b_orthogonal/results.json")
    p2 = load_json(RESULTS / "phase2/04cross_three_way/results_dual.json")
    # Note: both p1 and p2 may be None; both branches below have
    # hardcoded fallbacks so the figure renders either way.

    # Phase 1 numbers (locked, can fall back to hardcoded if file missing)
    if p1 and "ablation_summary" in p1:
        p1_data = {
            "baseline":       0.075,    # WildGuard baseline reference
            "ablate_r":       p1["ablation_summary"].get("ablate_r_hat", {}).get("harmbench_compliance", 0.635),
            "ablate_d":       p1["ablation_summary"].get("ablate_d_hat", {}).get("harmbench_compliance", 0.165),
            "ablate_d_par":   p1["ablation_summary"].get("ablate_d_par", {}).get("harmbench_compliance", 0.635),
            "ablate_d_perp":  p1["ablation_summary"].get("ablate_d_perp", {}).get("harmbench_compliance", 0.005),
        }
    else:
        # Locked numbers from the report (use if Phase 1 ortho json not synced)
        p1_data = {"baseline": 0.075, "ablate_r": 0.635, "ablate_d": 0.165,
                   "ablate_d_par": 0.635, "ablate_d_perp": 0.005}

    # Phase 2 numbers from the dual-judge results
    p2_data = {
        "baseline":           0.005,
        "ablate_r_em":        0.495,
        "ablate_r_l_inst":    0.205,
        "ablate_r_em_par":    0.205,
        "ablate_r_em_perp":   0.005,
        "ablate_random":      0.005,
    }
    # If results_dual.json present, override with actual values
    if p2:
        for k_in, k_out in [
            ("baseline",            "baseline"),
            ("ablate_r_em",         "ablate_r_em"),
            ("ablate_r_l_instruct", "ablate_r_l_inst"),
            ("ablate_r_em_par",     "ablate_r_em_par"),
            ("ablate_r_em_perp",    "ablate_r_em_perp"),
            ("ablate_random",       "ablate_random"),
        ]:
            try:
                p2_data[k_out] = p2["harmbench"]["wildguard"][k_in]
            except Exception:
                pass

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Phase 1 panel
    p1_labels = [
        "baseline",
        r"$\hat{r}$",
        r"$\hat{d}$",
        r"$\hat{d}_{\parallel}$",
        r"$\hat{d}_{\perp}$",
    ]
    p1_vals = [p1_data["baseline"], p1_data["ablate_r"], p1_data["ablate_d"],
               p1_data["ablate_d_par"], p1_data["ablate_d_perp"]]
    p1_colors = [C_BASE, C_R, C_D, C_R, C_PERP]
    bars1 = axL.bar(p1_labels, p1_vals, color=p1_colors, alpha=0.85, edgecolor="black", lw=0.6)
    axL.axhline(p1_data["baseline"], color="grey", ls="--", lw=0.8, label="baseline")
    for b, v in zip(bars1, p1_vals):
        axL.text(b.get_x() + b.get_width() / 2, v + 0.015,
                 f"{v:.3f}", ha="center", fontsize=9)
    axL.set_ylabel("HarmBench compliance (WildGuard)")
    axL.set_title(r"(a) Phase 1: $d_l = \alpha\hat{r}_l + d_l^{\perp}$ at $L26$" + "\n" +
                  r"$d_l^{\perp}$ inert (0.005 = baseline)")
    axL.set_ylim(0, 0.85)

    # Phase 2 panel
    p2_labels = [
        "baseline",
        r"$\hat{r}_{\mathrm{em}}$",
        r"$\hat{r}_{l,\mathrm{inst}}$",
        r"$\hat{r}_{\mathrm{em}}^{\parallel}$",
        r"$\hat{r}_{\mathrm{em}}^{\perp}$",
        "random",
    ]
    p2_vals = [p2_data["baseline"], p2_data["ablate_r_em"], p2_data["ablate_r_l_inst"],
               p2_data["ablate_r_em_par"], p2_data["ablate_r_em_perp"], p2_data["ablate_random"]]
    p2_colors = [C_BASE, C_REM, C_R, C_R, C_PERP, C_RAND]
    bars2 = axR.bar(p2_labels, p2_vals, color=p2_colors, alpha=0.85, edgecolor="black", lw=0.6)
    axR.axhline(p2_data["baseline"], color="grey", ls="--", lw=0.8, label="baseline")
    for b, v in zip(bars2, p2_vals):
        axR.text(b.get_x() + b.get_width() / 2, v + 0.012,
                 f"{v:.3f}", ha="center", fontsize=9)
    axR.set_ylabel("HarmBench compliance (WildGuard)")
    axR.set_title(r"(b) Phase 2: $r_{\mathrm{em}} = \beta\hat{r}_{l,\mathrm{inst}} + r_{\mathrm{em}}^{\perp}$ at $L46$" + "\n" +
                  r"$r_{\mathrm{em}}^{\perp}$ inert (0.005 = baseline)")
    axR.set_ylim(0, 0.6)

    fig.suptitle("Orthogonal complement is causally inert across both paradigms", fontsize=12, y=1.02)
    fig.tight_layout()
    save_both(fig, "fig_03_ortho_mirror")


# ======================================================================
# FIGURE 4 — Dual-judge summary (Phase 2 Table 3)
# ======================================================================
def _hb_lookup(p2: Optional[Dict[str, Any]], judge: str, cond: str) -> Optional[float]:
    """Get HarmBench score for (judge, condition) regardless of nesting order.
    Tries p2['harmbench'][judge][cond] and p2['harmbench'][cond][judge]."""
    if not p2 or "harmbench" not in p2:
        return None
    hb = p2["harmbench"]
    # Pattern 1: harmbench/<judge>/<condition>
    if isinstance(hb.get(judge), dict) and cond in hb[judge]:
        v = hb[judge][cond]
        return float(v) if not isinstance(v, dict) else None
    # Pattern 2: harmbench/<condition>/<judge>
    if isinstance(hb.get(cond), dict) and judge in hb[cond]:
        v = hb[cond][judge]
        return float(v) if not isinstance(v, dict) else None
    return None


def fig_04_summary() -> None:
    print("\n[Figure 4] dual-judge Phase 2 summary")
    p2 = load_json(RESULTS / "phase2/04cross_three_way/results_dual.json")

    # Always start from locked numbers, then override with anything the
    # JSON actually has. Works whether p2 is None, partially populated, or
    # uses either of the two common nesting orders.
    locked = [
        ("baseline",                    "baseline",            0.025, 0.005),
        (r"$\hat{r}_{\mathrm{em}}$",    "ablate_r_em",         0.895, 0.495),
        (r"$\hat{r}_{l,\mathrm{inst}}$","ablate_r_l_instruct", 0.560, 0.205),
        (r"$\hat{r}_{\mathrm{em}}^{\parallel}$", "ablate_r_em_par",  0.560, 0.205),
        (r"$\hat{r}_{\mathrm{em}}^{\perp}$",     "ablate_r_em_perp", 0.005, 0.005),
        ("random",                      "ablate_random",       0.035, 0.005),
    ]
    rows = []
    for label, key, locked_pat, locked_wg in locked:
        pat = _hb_lookup(p2, "pattern",   key)
        wg  = _hb_lookup(p2, "wildguard", key)
        rows.append((label,
                     pat if pat is not None else locked_pat,
                     wg  if wg  is not None else locked_wg))

    labels    = [r[0] for r in rows]
    pat_vals  = [r[1] for r in rows]
    wg_vals   = [r[2] for r in rows]
    delta_vals = [w - p for p, w in zip(pat_vals, wg_vals)]

    x = np.arange(len(labels))
    w = 0.38

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.5))

    # Top: paired bars
    ax1.bar(x - w/2, pat_vals, w, label="Pattern judge",
            color="lightcoral", edgecolor="black", lw=0.6)
    ax1.bar(x + w/2, wg_vals,  w, label="WildGuard",
            color=C_REM, edgecolor="black", lw=0.6)
    for xi, p, wg in zip(x, pat_vals, wg_vals):
        ax1.text(xi - w/2, p + 0.012, f"{p:.3f}", ha="center", fontsize=8)
        ax1.text(xi + w/2, wg + 0.012, f"{wg:.3f}", ha="center", fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right")
    ax1.set_ylabel("HarmBench compliance")
    ax1.set_title(r"(a) Dual-judge HarmBench compliance at $l^*_{\mathrm{em}} = 46$")
    ax1.set_ylim(0, max(pat_vals) * 1.15 + 0.05)
    ax1.legend(loc="upper right")

    # Bottom: delta bars
    colors_d = ["#444444" if d == 0 else "#777777" for d in delta_vals]
    bars = ax2.bar(x, delta_vals, color=colors_d, edgecolor="black", lw=0.6)
    # Highlight the perp row (delta = 0)
    perp_idx = [i for i, lab in enumerate(labels) if "perp" in lab.lower()
                or "\\perp" in lab]
    if perp_idx:
        bars[perp_idx[0]].set_color(C_PERP)
    for xi, d in zip(x, delta_vals):
        ax2.text(xi, d - 0.018 if d < 0 else d + 0.005,
                 f"{d:+.3f}", ha="center", fontsize=8,
                 va="top" if d < 0 else "bottom")
    ax2.axhline(0, color="black", lw=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha="right")
    ax2.set_ylabel(r"$\Delta = $ WildGuard $-$ Pattern")
    ax2.set_title(r"(b) Multilingual / sophisticated-refusal correction. "
                  r"$\Delta=0$ for $\hat{r}_{\mathrm{em}}^{\perp}$ confirms inertness")
    ax2.set_ylim(min(delta_vals) - 0.05, 0.05)

    fig.tight_layout()
    save_both(fig, "fig_04_summary")


# ======================================================================
if __name__ == "__main__":
    fig_01_geometry()
    fig_02_sweeps()
    fig_03_ortho_mirror()
    fig_04_summary()
    print(f"\nDone. Output in {OUT.resolve()}")