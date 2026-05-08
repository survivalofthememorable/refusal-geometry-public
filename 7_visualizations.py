"""
Step 7: Generate all visualizations.

Produces:
  - 01_norms.png:               r_l and d_l norms vs layer
  - 02_cosine_per_layer.png:    cos(r_l, d_l) vs layer
  - 03_cross_projection.png:    fraction of d along r and vice versa
  - 04_layer_sweep_curves.png:  causal effect S_l vs layer for r/d/random
  - 05_layer_sweep_components.png: compliance_harmful and false_refusal_harmless per layer
  - 06_band_overlap.png:        Venn-style or bar chart of bands
  - 07_harmbench_summary.png:   bar chart of compliance per condition with CIs
  - 08_capability.png:          capability preservation bar chart
  - 09_summary_panel.png:       4-panel headline figure for the paper

CPU-only, fast. Uses matplotlib.
"""

import modal
from modal_app import app, image, volume, TIMEOUT_SHORT, VOL_MOUNT, VOLUMES
from config import CONFIG


@app.function(
    image=image,
    volumes=VOLUMES,
    timeout=TIMEOUT_SHORT,
)
def make_plots():
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from pathlib import Path
    from common import setup_logger, load_json

    cfg = CONFIG
    paths = cfg.paths

    # Reload volume BEFORE opening any file handle in /vol/. See note in step 1.
    volume.reload()

    plots_dir = Path(paths["plots_dir"])
    plots_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("step7", paths["log"])
    logger.info("=" * 60)
    logger.info("STEP 7: visualizations")
    logger.info("=" * 60)

    # Style
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    geometry = load_json(paths["geometry"])
    selected = load_json(paths["selected_layers"])
    sweep_r = load_json(paths["sweep_r"])
    sweep_d = load_json(paths["sweep_d"])
    sweep_random = load_json(paths["sweep_random"]) if Path(paths["sweep_random"]).exists() else None
    harmbench = load_json(paths["harmbench_results"])

    layers = geometry["layers"]
    L = len(layers)
    l_star_r = selected["r"]["l_star"]
    l_star_d = selected["d"]["l_star"]

    def draw_l_star(ax, l_r, l_d):
        ax.axvline(l_r, color="C0", linestyle="--", alpha=0.5, label=f"$l^*_r={l_r}$")
        ax.axvline(l_d, color="C1", linestyle="--", alpha=0.5, label=f"$l^*_d={l_d}$")

    # ---- 01 norms ----
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(layers, geometry["r_norms"], "C0-o", markersize=4, label=r"$\|r_l\|$ (refusal)")
    ax.plot(layers, geometry["d_norms"], "C1-s", markersize=4, label=r"$\|d_l\|$ (post-training delta)")
    draw_l_star(ax, l_star_r, l_star_d)
    ax.set_xlabel("Layer $l$")
    ax.set_ylabel("Norm")
    ax.set_title("Direction norms across layers")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "01_norms.png")
    plt.close(fig)

    # ---- 02 cosine per layer ----
    fig, ax = plt.subplots(figsize=(7, 4))
    cos = geometry["cos_r_d_per_layer"]
    ax.plot(layers, cos, "C2-o", markersize=4)
    ax.axhline(0, color="k", linewidth=0.5)
    draw_l_star(ax, l_star_r, l_star_d)
    ax.set_xlabel("Layer $l$")
    ax.set_ylabel(r"$\cos(r_l, d_l)$")
    ax.set_title("Cosine similarity between refusal direction and post-training delta")
    ax.set_ylim(-1.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "02_cosine_per_layer.png")
    plt.close(fig)

    # ---- 03 cross projection ----
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(layers, geometry["frac_d_along_r"], "C0-o", markersize=4, label=r"$|d \cdot \hat{r}| / \|d\|$")
    ax.plot(layers, geometry["frac_r_along_d"], "C1-s", markersize=4, label=r"$|r \cdot \hat{d}| / \|r\|$")
    draw_l_star(ax, l_star_r, l_star_d)
    ax.set_xlabel("Layer $l$")
    ax.set_ylabel("Fraction along the other direction")
    ax.set_title("How much each direction projects onto the other")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "03_cross_projection.png")
    plt.close(fig)

    # ---- 04 layer sweep curves S_l ----
    fig, ax = plt.subplots(figsize=(8, 4.5))

    def get_curve(sweep):
        per_layer = sweep["per_layer"]
        ls = sorted(int(k) for k in per_layer.keys())
        S = [per_layer[str(l)].get("score_S", float("nan")) for l in ls]
        return ls, S

    ls_r, S_r = get_curve(sweep_r)
    ls_d, S_d = get_curve(sweep_d)
    ax.plot(ls_r, S_r, "C0-o", markersize=4, label="ablate $r_l$")
    ax.plot(ls_d, S_d, "C1-s", markersize=4, label="ablate $d_l$")
    if sweep_random is not None:
        ls_rand, S_rand = get_curve(sweep_random)
        ax.plot(ls_rand, S_rand, "C7-^", markersize=4, alpha=0.6, label="random direction (control)")
    draw_l_star(ax, l_star_r, l_star_d)
    ax.set_xlabel("Layer $l$")
    ax.set_ylabel(r"$S_l$ = compliance$_{\mathrm{harmful}}$ - false-refusal$_{\mathrm{harmless}}$")
    ax.set_title("Causal effect of single-layer ablation (validation set)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "04_layer_sweep_curves.png")
    plt.close(fig)

    # ---- 05 components ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    def comp_curve(sweep, key):
        per_layer = sweep["per_layer"]
        ls = sorted(int(k) for k in per_layer.keys())
        return ls, [per_layer[str(l)].get(key, float("nan")) for l in ls]

    ls_r, ch_r = comp_curve(sweep_r, "compliance_harmful")
    ls_d, ch_d = comp_curve(sweep_d, "compliance_harmful")
    axes[0].plot(ls_r, ch_r, "C0-o", markersize=4, label="ablate $r_l$")
    axes[0].plot(ls_d, ch_d, "C1-s", markersize=4, label="ablate $d_l$")
    axes[0].set_xlabel("Layer $l$")
    axes[0].set_ylabel("Compliance on harmful (validation)")
    axes[0].set_title("Refusal disabled?")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-0.05, 1.05)

    ls_r, fr_r = comp_curve(sweep_r, "false_refusal_harmless")
    ls_d, fr_d = comp_curve(sweep_d, "false_refusal_harmless")
    axes[1].plot(ls_r, fr_r, "C0-o", markersize=4, label="ablate $r_l$")
    axes[1].plot(ls_d, fr_d, "C1-s", markersize=4, label="ablate $d_l$")
    axes[1].set_xlabel("Layer $l$")
    axes[1].set_title("Harmless capability damaged?")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-0.05, 1.05)

    fig.tight_layout()
    fig.savefig(plots_dir / "05_layer_sweep_components.png")
    plt.close(fig)

    # ---- 06 band overlap ----
    fig, ax = plt.subplots(figsize=(8, 3))
    in_r = [1 if l in geometry["band_r"] else 0 for l in layers]
    in_d = [1 if l in geometry["band_d"] else 0 for l in layers]
    ax.bar(layers, in_r, color="C0", alpha=0.5, label="band of $r_l$")
    ax.bar(layers, [-x for x in in_d], color="C1", alpha=0.5, label="band of $d_l$")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlabel("Layer $l$")
    ax.set_yticks([])
    ax.set_title(f"Layer bands (IoU = {geometry['iou']:.2f})")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(plots_dir / "06_band_overlap.png")
    plt.close(fig)

    # ---- 07 harmbench summary ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    hb = harmbench["harmbench"]
    conds = ["baseline", "ablate_r", "ablate_d", "ablate_random"]
    conds = [c for c in conds if c in hb]
    means = [hb[c]["mean"] for c in conds]
    lows = [hb[c]["ci_low"] for c in conds]
    highs = [hb[c]["ci_high"] for c in conds]
    err_low = [m - lo for m, lo in zip(means, lows)]
    err_high = [hi - m for m, hi in zip(means, highs)]
    colors = {"baseline": "gray", "ablate_r": "C0", "ablate_d": "C1", "ablate_random": "C7"}
    bar_colors = [colors[c] for c in conds]

    ax.bar(conds, means, yerr=[err_low, err_high], color=bar_colors, capsize=5)
    ax.set_ylabel("HarmBench compliance (StrongREJECT)")
    ax.set_ylim(0, 1)
    ax.set_title("Causal ablation effect on HarmBench")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(plots_dir / "07_harmbench_summary.png")
    plt.close(fig)

    # ---- 08 capability ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cap = harmbench["capability"]
    conds = [c for c in ["baseline", "ablate_r", "ablate_d"] if c in cap]
    means = [cap[c]["mean"] for c in conds]
    lows = [cap[c]["ci_low"] for c in conds]
    highs = [cap[c]["ci_high"] for c in conds]
    err_low = [m - lo for m, lo in zip(means, lows)]
    err_high = [hi - m for m, hi in zip(means, highs)]
    bar_colors = [colors[c] for c in conds]
    ax.bar(conds, means, yerr=[err_low, err_high], color=bar_colors, capsize=5)
    ax.set_ylabel("Alpaca compliance (capability proxy)")
    ax.set_ylim(0, 1)
    ax.set_title("Capability preservation under ablation")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(plots_dir / "08_capability.png")
    plt.close(fig)

    # ---- 09 four-panel summary ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (a) layer sweep
    ax = axes[0, 0]
    ax.plot(ls_r, S_r, "C0-o", markersize=4, label=r"ablate $r_l$")
    ax.plot(ls_d, S_d, "C1-s", markersize=4, label=r"ablate $d_l$")
    if sweep_random is not None:
        ax.plot(ls_rand, S_rand, "C7-^", markersize=4, alpha=0.6, label="random")
    draw_l_star(ax, l_star_r, l_star_d)
    ax.set_xlabel("Layer $l$")
    ax.set_ylabel(r"$S_l$")
    ax.set_title("(a) Causal effect by layer")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (b) cosine
    ax = axes[0, 1]
    ax.plot(layers, cos, "C2-o", markersize=4)
    ax.axhline(0, color="k", linewidth=0.5)
    draw_l_star(ax, l_star_r, l_star_d)
    ax.set_xlabel("Layer $l$")
    ax.set_ylabel(r"$\cos(r_l, d_l)$")
    ax.set_title("(b) Geometric agreement")
    ax.set_ylim(-1.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (c) HarmBench bars
    ax = axes[1, 0]
    conds_hb = [c for c in ["baseline", "ablate_r", "ablate_d", "ablate_random"] if c in hb]
    means = [hb[c]["mean"] for c in conds_hb]
    err_low = [hb[c]["mean"] - hb[c]["ci_low"] for c in conds_hb]
    err_high = [hb[c]["ci_high"] - hb[c]["mean"] for c in conds_hb]
    ax.bar(conds_hb, means, yerr=[err_low, err_high], color=[colors[c] for c in conds_hb], capsize=5)
    ax.set_ylabel("HarmBench compliance")
    ax.set_ylim(0, 1)
    ax.set_title("(c) HarmBench ablation")
    ax.grid(True, alpha=0.3, axis="y")

    # (d) capability bars
    ax = axes[1, 1]
    conds_cap = [c for c in ["baseline", "ablate_r", "ablate_d"] if c in cap]
    means_cap = [cap[c]["mean"] for c in conds_cap]
    el_cap = [cap[c]["mean"] - cap[c]["ci_low"] for c in conds_cap]
    eh_cap = [cap[c]["ci_high"] - cap[c]["mean"] for c in conds_cap]
    ax.bar(conds_cap, means_cap, yerr=[el_cap, eh_cap], color=[colors[c] for c in conds_cap], capsize=5)
    ax.set_ylabel("Alpaca compliance")
    ax.set_ylim(0, 1)
    ax.set_title("(d) Capability preservation")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Refusal direction vs post-training delta in Qwen2.5-14B", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(plots_dir / "09_summary_panel.png")
    plt.close(fig)

    volume.commit()
    logger.info(f"All plots saved to {plots_dir}")


@app.local_entrypoint()
def main():
    make_plots.remote()
