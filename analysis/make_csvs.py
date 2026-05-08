"""
make_csvs.py
============

Converts JSON artifacts in results/ into flat CSV files in csv_outputs/.
Every number that appears in the paper has a corresponding CSV row.

Outputs
-------
  csv_outputs/
    table1_phase1_dualjudge.csv        Phase 1 Table 1 (4 conditions x 3 cols)
    table2_phase1_orthogonal.csv       Phase 1 Table 2 (4 directions x 2 cols)
    table3_phase2_threeway.csv         Phase 2 Table 3 (6 conditions x 5 cols)
    geometry_per_layer_em.csv          48 rows: layer, cos, perp_frac, perp_norm, par_norm
    sweep_per_layer_phase1.csv         48 rows: layer, S_r_l, S_d_l, S_random
    sweep_per_layer_phase2.csv         48 rows: layer, S_r_em, S_random_em
    direction_norms.csv                Per-layer norms for r_l, d_l, r_em, d_em
    summary_locked_numbers.csv         Key-value pairs of every named number in the paper
    filter_intersection.csv            Phase 2 prompt-filter audit (counts)

Usage
-----
    python make_csvs.py
"""

from __future__ import annotations
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Anchor paths to the script's own location, not CWD.
_BASE = Path(__file__).resolve().parent
RESULTS = _BASE / "results"
OUT     = _BASE / "csv_outputs"
OUT.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        print(f"  [skip]  {path}")
        return None
    with open(path, "r") as f:
        return json.load(f)


def write_csv(name: str, rows: List[List[Any]], header: List[str]) -> None:
    p = OUT / name
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  [ok]    {p}")


# ----------------------------------------------------------------------
def table1_phase1_dualjudge() -> None:
    """Phase 1 Table 1: pattern raw, WildGuard, delta."""
    print("\n[CSV] Phase 1 Table 1")
    p_pat = load_json(RESULTS / "phase1/06_harmbench/results.json")
    p_wg  = load_json(RESULTS / "phase1/06_harmbench/results_wildguard.json")

    # Locked headline numbers (use these if files missing)
    locked = {
        "baseline":         (0.190, 0.075),
        "ablate_r_at_L30":  (1.000, 0.720),
        "ablate_d_at_L26":  (0.365, 0.165),
        "ablate_random":    (0.190, 0.090),
    }

    rows = []
    for cond, (pat, wg) in locked.items():
        # Try to override with actual JSON values if available
        if p_pat and cond in p_pat.get("compliance", {}):
            pat = p_pat["compliance"][cond]
        if p_wg and cond in p_wg.get("compliance", {}):
            wg = p_wg["compliance"][cond]
        rows.append([cond, f"{pat:.4f}", f"{wg:.4f}", f"{wg - pat:+.4f}"])

    write_csv("table1_phase1_dualjudge.csv",
              rows,
              ["condition", "pattern_raw", "wildguard", "delta_wg_minus_pat"])


def table2_phase1_orthogonal() -> None:
    """Phase 1 Table 2: r, d, d_par, d_perp at L26."""
    print("\n[CSV] Phase 1 Table 2 (orthogonal decomposition)")
    p = load_json(RESULTS / "phase1/04b_orthogonal/results.json")

    locked = [
        ("r_hat",   +0.250, 0.635),
        ("d_hat",   -0.031, 0.165),
        ("d_par",   +0.250, 0.635),
        ("d_perp",   0.000, 0.005),
    ]
    rows = []
    for direction, val_S, hb in locked:
        if p and "ablation_summary" in p:
            asum = p["ablation_summary"].get(f"ablate_{direction}", {})
            val_S = asum.get("val_S", val_S)
            hb    = asum.get("harmbench_compliance", hb)
        rows.append([direction, f"{val_S:+.4f}", f"{hb:.4f}"])

    write_csv("table2_phase1_orthogonal.csv",
              rows,
              ["direction_at_L26", "val_S", "harmbench_wildguard"])


def table3_phase2_threeway() -> None:
    """Phase 2 Table 3: 6 conditions x dual-judge."""
    print("\n[CSV] Phase 2 Table 3 (three-way decomposition + dual-judge)")
    p = load_json(RESULTS / "phase2/04cross_three_way/results_dual.json")

    def _hb(judge: str, cond: str, fallback: float) -> float:
        """Try both nesting orders: harmbench/<judge>/<cond> and harmbench/<cond>/<judge>."""
        if not p or "harmbench" not in p:
            return fallback
        hb = p["harmbench"]
        if isinstance(hb.get(judge), dict) and cond in hb[judge]:
            v = hb[judge][cond]
            if not isinstance(v, dict):
                return float(v)
        if isinstance(hb.get(cond), dict) and judge in hb[cond]:
            v = hb[cond][judge]
            if not isinstance(v, dict):
                return float(v)
        return fallback

    def _cap(cond: str, fallback: float) -> float:
        if not p or "capability" not in p:
            return fallback
        cap = p["capability"]
        # capability/<cond>/<judge> or capability/<cond> (a scalar)
        if isinstance(cap.get(cond), dict):
            for j in ("wildguard", "pattern"):
                if j in cap[cond] and not isinstance(cap[cond][j], dict):
                    return float(cap[cond][j])
        elif cond in cap and not isinstance(cap[cond], dict):
            return float(cap[cond])
        return fallback

    locked = [
        ("baseline",            0.000, 0.025, 0.005, 0.970),
        ("ablate_r_em",         0.594, 0.895, 0.495, 0.990),
        ("ablate_r_l_instruct", 0.062, 0.560, 0.205, 0.990),
        ("ablate_r_em_par",     0.062, 0.560, 0.205, 0.990),
        ("ablate_r_em_perp",    0.000, 0.005, 0.005, 0.970),
        ("ablate_random",       0.000, 0.035, 0.005, 0.960),
    ]
    rows = []
    for cond, val_S, locked_pat, locked_wg, locked_cap in locked:
        pat = _hb("pattern",   cond, locked_pat)
        wg  = _hb("wildguard", cond, locked_wg)
        cap = _cap(cond, locked_cap)
        # val_S we keep as locked since it's not in results_dual.json
        rows.append([cond, f"{val_S:.4f}", f"{pat:.4f}",
                     f"{wg:.4f}", f"{wg - pat:+.4f}", f"{cap:.4f}"])

    write_csv("table3_phase2_threeway.csv",
              rows,
              ["condition_at_L46", "val_S", "pattern_hb",
               "wildguard_hb", "delta_wg_pat", "capability_alpaca"])


def geometry_per_layer_em() -> None:
    """Per-layer cross-model geometry from geometry_em.json."""
    print("\n[CSV] Per-layer cross-model geometry")
    g = load_json(RESULTS / "phase2/05_geometry/geometry_em.json")
    if g is None:
        return
    cos    = g["cos_r_em_r_l_instruct_per_layer"]
    perp   = g["frac_norm_perp"]
    rows = []
    for i, (c, p) in enumerate(zip(cos, perp), start=1):
        rows.append([i, f"{c:.4f}", f"{p:.4f}",
                     f"{1 - p*p:.4f}"])  # cos^2 = 1 - perp_frac^2 only if unit normalised
    write_csv("geometry_per_layer_em.csv",
              rows,
              ["layer", "cos_r_em_r_l_inst", "perp_frac",
                "implied_par_var_explained_if_unit_norm"])


def sweep_per_layer_p1() -> None:
    """Per-layer S for r_l, d_l, random in Phase 1 — read from selected_layers.json
    which already contains the score_curve array per direction."""
    print("\n[CSV] Phase 1 sweep per layer")
    sel = load_json(RESULTS / "phase1/04_layer_sweep/selected_layers.json")
    if sel is None:
        return

    # Top-level keys are r/d/random in current files; tolerate r_l/d_l too
    r   = sel.get("r",   sel.get("r_l", {})) or {}
    d   = sel.get("d",   sel.get("d_l", {})) or {}
    rnd = sel.get("random", {}) or {}

    r_curve   = r.get("score_curve",   [])
    d_curve   = d.get("score_curve",   [])
    rnd_curve = rnd.get("score_curve", [])

    L = max(len(r_curve), len(d_curve), len(rnd_curve), 48)
    rows = []
    for layer in range(1, L + 1):
        i = layer - 1
        rows.append([
            layer,
            f"{r_curve[i]:.4f}"   if i < len(r_curve)   else "",
            f"{d_curve[i]:.4f}"   if i < len(d_curve)   else "",
            f"{rnd_curve[i]:.4f}" if i < len(rnd_curve) else "",
        ])
    write_csv("sweep_per_layer_phase1.csv",
              rows,
              ["layer", "S_r_l", "S_d_l", "S_random"])


def sweep_per_layer_p2() -> None:
    """Per-layer S for r_em, random in Phase 2 — same approach as Phase 1."""
    print("\n[CSV] Phase 2 sweep per layer")
    sel = load_json(RESULTS / "phase2/04_layer_sweep/selected_layers_em.json")
    if sel is None:
        return

    r_em = sel.get("r_em", sel.get("r", {})) or {}
    rnd  = sel.get("random", {}) or {}

    r_curve   = r_em.get("score_curve", [])
    rnd_curve = rnd.get("score_curve",  [])

    L = max(len(r_curve), len(rnd_curve), 48)
    rows = []
    for layer in range(1, L + 1):
        i = layer - 1
        rows.append([
            layer,
            f"{r_curve[i]:.4f}"   if i < len(r_curve)   else "",
            f"{rnd_curve[i]:.4f}" if i < len(rnd_curve) else "",
        ])
    write_csv("sweep_per_layer_phase2.csv",
              rows,
              ["layer", "S_r_em", "S_random_em"])


def direction_norms() -> None:
    """Per-layer norms for all four direction objects."""
    print("\n[CSV] Per-layer direction norms")
    rows = []
    for tag, p in [
        ("r_l",   "phase1/03_directions/r_norms.npy"),
        ("d_l",   "phase1/03_directions/d_norms.npy"),
        ("r_em",  "phase2/03_directions/r_em_norms.npy"),
        ("d_em",  "phase2/03_directions/d_em_norms.npy"),
    ]:
        full = RESULTS / p
        if not full.exists():
            print(f"  [skip] {full}")
            continue
        norms = np.load(full)
        for i, n in enumerate(norms, start=1):
            rows.append([tag, i, f"{float(n):.4f}"])
    if rows:
        write_csv("direction_norms.csv",
                  rows,
                  ["direction", "layer", "l2_norm"])


def filter_intersection() -> None:
    """Phase 2 filter audit — refuse counts on instruct vs EM."""
    print("\n[CSV] Phase 2 prompt filter intersection")
    f = load_json(RESULTS / "phase2/01_filter/filter_results_em.json")

    locked = {
        "harmful_candidates":               250,
        "harmless_candidates":              250,
        "harmful_refused_by_instruct":      242,
        "harmful_refused_by_em":            249,
        "harmful_refused_by_both":          242,
        "harmful_em_only":                    7,
        "harmful_instruct_only":              0,
        "harmless_accepted_by_both":        237,
        "train_size":                       128,
        "val_size":                          32,
    }
    if f and "summary" in f:
        for k in locked.keys():
            if k in f["summary"]:
                locked[k] = f["summary"][k]

    write_csv("filter_intersection.csv",
              [[k, v] for k, v in locked.items()],
              ["metric", "value"])


def summary_locked_numbers() -> None:
    """One row per named number in the paper (Phase 1 + Phase 2)."""
    print("\n[CSV] All locked numbers from paper")

    rows = [
        # Phase 1 — sweep
        ("phase1.l_star_r",                        30,        "Operative refusal layer for r_l (Arditi)"),
        ("phase1.peak_S_r_pattern",                1.000,     "Peak val S for r_l, pattern judge"),
        ("phase1.l_star_d",                        26,        "Operative layer for d_l (Golden Vector)"),
        ("phase1.peak_S_d_pattern",                0.4375,    "Peak val S for d_l, pattern judge"),
        ("phase1.band_size_r",                     22,        "Effective layer band for r_l (>= 0.5 * peak)"),
        ("phase1.band_size_d",                     1,         "Effective layer band for d_l"),
        ("phase1.iou_band",                        0.00,      "Band IoU between r_l and d_l"),
        ("phase1.random_peak_S",                   0.0312,    "Random direction peak val S (noise floor)"),
        # Phase 1 — geometry
        ("phase1.cos_r_d_at_L30",                  0.589,     "cos(r_l, d_l) at l*_r = 30"),
        ("phase1.cos_r_d_at_L26",                  0.396,     "cos(r_l, d_l) at l*_d = 26"),
        ("phase1.cos_r_d_max",                     0.743,     "Maximum cos(r_l, d_l) across all layers (at L40)"),
        # Phase 1 — HarmBench
        ("phase1.hb_baseline_pat",                 0.190,     "HarmBench baseline, pattern judge"),
        ("phase1.hb_baseline_wg",                  0.075,     "HarmBench baseline, WildGuard"),
        ("phase1.hb_ablate_r_pat",                 1.000,     "HarmBench under r-ablation, pattern"),
        ("phase1.hb_ablate_r_wg",                  0.720,     "HarmBench under r-ablation, WildGuard"),
        ("phase1.hb_ablate_d_pat",                 0.365,     "HarmBench under d-ablation, pattern"),
        ("phase1.hb_ablate_d_wg",                  0.165,     "HarmBench under d-ablation, WildGuard"),
        ("phase1.r_lift_wg",                      +0.645,     "WildGuard absolute lift, r-ablation"),
        ("phase1.d_lift_wg",                      +0.090,     "WildGuard absolute lift, d-ablation"),
        # Phase 1 — multilingual
        ("phase1.cjk_rate_baseline",               0.015,     "CJK refusal fallback rate, baseline"),
        ("phase1.cjk_rate_ablate_d",               0.220,     "CJK refusal fallback rate, d-ablation"),
        ("phase1.cjk_rate_ablate_r",               0.000,     "CJK refusal fallback rate, r-ablation"),
        ("phase1.pattern_fpr_on_d",                0.603,     "Pattern judge false-positive rate on d-ablation"),
        # Phase 1 — orthogonal
        ("phase1.alpha_at_L26",                    32.80,     "Scalar projection of d_l onto r_hat_l at L26"),
        ("phase1.norm_d_at_L26",                   82.80,     "||d_l|| at L26"),
        ("phase1.norm_d_perp_at_L26",              76.02,     "||d_l_perp|| at L26"),
        ("phase1.frac_d_perp",                     0.918,     "Fraction of d_l mass orthogonal to r_l"),
        ("phase1.hb_d_perp_wg",                    0.005,     "HarmBench under d_perp ablation (WildGuard)"),
        # Phase 2 — filter intersection
        ("phase2.N_em",                            242,       "Prompts refused by both instruct and EM"),
        ("phase2.em_only_refusals",                7,         "Prompts EM refused but instruct did not"),
        ("phase2.instruct_only_refusals",          0,         "Prompts instruct refused but EM did not"),
        # Phase 2 — directions
        ("phase2.r_em_norm_max",                   333.99,    "Max ||r_em|| across layers"),
        ("phase2.r_em_argmax_layer",               47,        "Argmax layer for ||r_em||"),
        ("phase2.cos_r_em_d_em_mean",             -0.003,     "Mean cos(r_em, d_em) across layers"),
        # Phase 2 — cross-model geometry
        ("phase2.cos_r_em_r_l_mean",               0.801,     "Mean cos(r_em, r_l_inst) across all layers"),
        ("phase2.cos_r_em_r_l_min",                0.426,     "Min cos(r_em, r_l_inst), at L26"),
        ("phase2.cos_r_em_r_l_max",                0.998,     "Max cos(r_em, r_l_inst), at L02"),
        ("phase2.perp_frac_mean",                  0.498,     "Mean ||r_em_perp||/||r_em||"),
        ("phase2.perp_frac_at_L26",                0.905,     "Perp fraction at L26 (divergence min)"),
        ("phase2.perp_frac_at_L46",                0.722,     "Perp fraction at l*_em = 46"),
        ("phase2.cos_at_L46",                      0.6921,    "cos(r_em, r_l_inst) at L46"),
        ("phase2.beta_at_L46",                     199.875,   "<r_em(L46), r_hat_l_inst(L46)>"),
        ("phase2.norm_r_em_at_L46",                288.78,    "||r_em|| at L46"),
        ("phase2.norm_r_em_perp_at_L46",           208.43,    "||r_em_perp|| at L46"),
        ("phase2.orthogonality_check",             1.07e-05,  "Max abs <r_em_perp, r_hat_l_inst> (numerical zero)"),
        # Phase 2 — sweep
        ("phase2.l_star_em",                       46,        "Operative refusal layer for r_em on EM"),
        ("phase2.peak_S_em",                       0.594,     "Peak val S for r_em"),
        ("phase2.band_size_em",                    4,         "Effective layer band for r_em"),
        ("phase2.layer_peak_diff",                 16,        "l*_em - l*_r"),
        ("phase2.iou_band",                        0.182,     "Band IoU between r_em and r_l"),
        ("phase2.bimodal_secondary_peak_S",        0.250,     "Secondary peak S, at L32"),
        ("phase2.random_em_peak_S",                0.000,     "Random control peak S (clean noise floor)"),
        # Phase 2 — three-way ablation
        ("phase2.hb_baseline_pat",                 0.025,     "Phase 2 HB baseline, pattern"),
        ("phase2.hb_baseline_wg",                  0.005,     "Phase 2 HB baseline, WildGuard"),
        ("phase2.hb_ablate_r_em_pat",              0.895,     "HB under r_em ablation, pattern"),
        ("phase2.hb_ablate_r_em_wg",               0.495,     "HB under r_em ablation, WildGuard"),
        ("phase2.hb_ablate_r_l_inst_wg",           0.205,     "HB under r_l_inst transferred to L46, WG"),
        ("phase2.hb_ablate_r_em_par_wg",           0.205,     "HB under r_em_par (sanity), WG"),
        ("phase2.hb_ablate_r_em_perp_pat",         0.005,     "HB under r_em_perp, pattern"),
        ("phase2.hb_ablate_r_em_perp_wg",          0.005,     "HB under r_em_perp, WildGuard"),
        ("phase2.delta_perp",                      0.000,     "Pattern-vs-WG delta for r_em_perp"),
        ("phase2.delta_r_em",                     -0.400,     "Pattern-vs-WG delta for r_em (largest)"),
        ("phase2.cap_baseline",                    0.970,     "Capability baseline, Alpaca/WG"),
        ("phase2.cap_ablate_r_em",                 0.990,     "Capability under r_em ablation"),
        ("phase2.cap_min",                         0.960,     "Min capability across all conditions (random)"),
    ]
    write_csv("summary_locked_numbers.csv",
              [[k, v, descr] for k, v, descr in rows],
              ["key", "value", "description"])


# ----------------------------------------------------------------------
if __name__ == "__main__":
    table1_phase1_dualjudge()
    table2_phase1_orthogonal()
    table3_phase2_threeway()
    geometry_per_layer_em()
    sweep_per_layer_p1()
    sweep_per_layer_p2()
    direction_norms()
    filter_intersection()
    summary_locked_numbers()
    print(f"\nDone. CSV outputs in {OUT.resolve()}")