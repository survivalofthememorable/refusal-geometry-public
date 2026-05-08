"""
Step 5e: Cross-model geometric analysis.

Three families of geometric quantities, all per-layer:

  (1) WITHIN-EM (analogue of Phase 1's step 5):
      - cos(r_em, d_em) per layer
      - r_em norms, d_em norms
      - frac_d_em_along_r_em, frac_r_em_along_d_em

  (2) CROSS-MODEL (the new piece, central to the workshop paper headline):
      - cos(r_em(l), r_l_instruct(l)) per layer
        ^ how aligned is EM's refusal direction with instruct's, by depth?
      - decomposition r_em(l) = beta(l) * r_hat_instruct(l) + r_em_perp(l)
        with beta = <r_em(l), r_hat_instruct(l)>
      - ||r_em_perp|| / ||r_em|| per layer
        ^ fraction of EM's refusal direction that points AWAY from instruct's
      - ||r_em(l)|| / ||r_l_instruct(l)|| per layer
        ^ has the refusal direction grown or shrunk under EM training?

  (3) BAND COMPARISON (filled in after step 4e_em runs):
      - effective bands for r_em vs r_l_instruct
      - IoU of bands
      - peak layer difference

CPU-only, fast.

We deliberately compute (1) and (2) BEFORE the EM sweep runs, so we have a
gate point: looking at the cross-model cosine plot tells you whether
hypothesis A (dilution) or B (substitution) is on the table BEFORE you
spend GPU on the sweep.

Run:
    python -m modal run 5_geometric_analysis_em.py
"""

import modal
from modal_app import app, image, volume, TIMEOUT_SHORT, VOL_MOUNT, VOLUMES
from config import CONFIG
from config_em import paths_em


@app.function(
    image=image,
    volumes=VOLUMES,
    timeout=TIMEOUT_SHORT,
)
def geometric_analysis_em():
    import numpy as np
    from pathlib import Path
    from common import setup_logger, save_json, load_json

    cfg = CONFIG
    p_em = paths_em()

    volume.reload()

    logger = setup_logger("step5_em", p_em["log"])
    logger.info("=" * 60)
    logger.info("STEP 5e: cross-model geometric analysis")
    logger.info("=" * 60)

    # ---- Load EM directions ----
    r_em = np.load(p_em["r_em"]).astype(np.float32)
    d_em = np.load(p_em["d_em"]).astype(np.float32)
    r_em_hat = np.load(p_em["r_em_hat"]).astype(np.float32)
    d_em_hat = np.load(p_em["d_em_hat"]).astype(np.float32)
    r_em_norms = np.load(p_em["r_em_norms"]).astype(np.float32)
    d_em_norms = np.load(p_em["d_em_norms"]).astype(np.float32)

    # ---- Load Phase 1 instruct directions ----
    r_l = np.load(p_em["p1_r_l"]).astype(np.float32)
    r_l_hat = np.load(p_em["p1_r_l_hat"]).astype(np.float32)
    r_l_norms = np.linalg.norm(r_l, axis=1).astype(np.float32)

    L = r_em_hat.shape[0]
    layers = list(range(1, L + 1))

    # ====================================================================
    # (1) Within-EM geometry (mirrors Phase 1 step 5)
    # ====================================================================
    cos_r_em_d_em = (r_em_hat * d_em_hat).sum(axis=1)
    proj_d_em_on_r_em = (d_em * r_em_hat).sum(axis=1)
    proj_r_em_on_d_em = (r_em * d_em_hat).sum(axis=1)
    frac_d_em_along_r_em = np.abs(proj_d_em_on_r_em) / np.maximum(d_em_norms, 1e-12)
    frac_r_em_along_d_em = np.abs(proj_r_em_on_d_em) / np.maximum(r_em_norms, 1e-12)

    # ====================================================================
    # (2) Cross-model: r_em vs r_l_instruct
    # ====================================================================
    # Per-layer cosine — the key descriptive plot.
    # Both arrays already row-unit-normalized, so dot product = cosine.
    cos_r_em_r_l = (r_em_hat * r_l_hat).sum(axis=1)

    # Orthogonal decomposition r_em(l) = beta(l) * r_hat_instruct(l) + r_em_perp(l)
    # We use raw r_em (not unit-normalized) so the decomposition's components
    # are in the same scale as r_em itself.
    beta = (r_em * r_l_hat).sum(axis=1)              # signed scalar per layer
    r_em_par = beta[:, None] * r_l_hat                # (L, H)
    r_em_perp = r_em - r_em_par                       # (L, H)

    # Verify orthogonality (sanity check; should be ~0 to machine precision)
    ortho_check = (r_em_perp * r_l_hat).sum(axis=1)
    max_ortho_violation = float(np.max(np.abs(ortho_check)))
    logger.info(
        f"<r_em_perp, r_hat_instruct> max abs = {max_ortho_violation:.2e} "
        f"(should be near 0; check orthogonality)"
    )
    if max_ortho_violation > 1e-3:
        logger.warning(
            "Orthogonality check exceeds 1e-3; projection arithmetic may be "
            "numerically degraded. Investigate before trusting downstream results."
        )

    norm_r_em_par = np.linalg.norm(r_em_par, axis=1)
    norm_r_em_perp = np.linalg.norm(r_em_perp, axis=1)
    frac_norm_par = norm_r_em_par / np.maximum(r_em_norms, 1e-12)
    frac_norm_perp = norm_r_em_perp / np.maximum(r_em_norms, 1e-12)

    # Norm ratio: has refusal direction grown/shrunk under EM training?
    norm_ratio_em_over_instruct = r_em_norms / np.maximum(r_l_norms, 1e-12)

    # Save the decomposed parts for downstream use (step 4cross uses them)
    Path(p_em["geometry_em"]).parent.mkdir(parents=True, exist_ok=True)
    np.save(str(Path(p_em["geometry_em"]).parent / "r_em_par.npy"), r_em_par)
    np.save(str(Path(p_em["geometry_em"]).parent / "r_em_perp.npy"), r_em_perp)

    # ====================================================================
    # (3) Band comparison (only fillable after EM sweep runs)
    # ====================================================================
    # We attempt to load EM sweep results; if missing (this script ran before
    # step 4e_em finished), we record the within-EM band info as None and
    # the user can re-run step 5e_em later for an updated geometry.json.
    band_info = {"populated": False}
    try:
        sel_em = load_json(p_em["selected_layers_em"])
        band_r_em = sel_em.get("r_em", {}).get("band", []) or []
        l_star_r_em = sel_em.get("r_em", {}).get("l_star", None)

        # Phase 1 instruct bands for comparison
        sel_p1 = load_json(p_em["p1_selected_layers"])
        band_r_p1 = sel_p1.get("r", {}).get("band", []) or []
        l_star_r_p1 = sel_p1.get("r", {}).get("l_star", None)

        b_em_set = set(band_r_em)
        b_p1_set = set(band_r_p1)
        intersect = sorted(b_em_set & b_p1_set)
        union = sorted(b_em_set | b_p1_set)
        iou = (len(intersect) / len(union)) if union else 0.0

        band_info = {
            "populated": True,
            "band_r_em": sorted(band_r_em),
            "band_r_l_instruct": sorted(band_r_p1),
            "band_intersect": intersect,
            "band_union": union,
            "iou": iou,
            "l_star_r_em": l_star_r_em,
            "l_star_r_l_instruct": l_star_r_p1,
            "peak_layer_diff_em_minus_p1": (
                (l_star_r_em - l_star_r_p1) if (l_star_r_em is not None and l_star_r_p1 is not None) else None
            ),
        }
        logger.info(f"Band IoU(r_em, r_l_instruct) = {iou:.3f}")
        logger.info(f"Layer peak diff (em - instruct) = {band_info['peak_layer_diff_em_minus_p1']}")
    except FileNotFoundError:
        logger.info("EM sweep results not yet available; band comparison deferred.")
    except Exception as e:
        logger.warning(f"Band comparison failed: {e}")

    # ====================================================================
    # Save geometry summary
    # ====================================================================
    geometry = {
        "layers": layers,

        # within-EM
        "cos_r_em_d_em_per_layer": cos_r_em_d_em.tolist(),
        "r_em_norms": r_em_norms.tolist(),
        "d_em_norms": d_em_norms.tolist(),
        "frac_d_em_along_r_em": frac_d_em_along_r_em.tolist(),
        "frac_r_em_along_d_em": frac_r_em_along_d_em.tolist(),

        # cross-model
        "cos_r_em_r_l_instruct_per_layer": cos_r_em_r_l.tolist(),
        "beta_per_layer": beta.tolist(),
        "norm_r_em_par": norm_r_em_par.tolist(),
        "norm_r_em_perp": norm_r_em_perp.tolist(),
        "frac_norm_par": frac_norm_par.tolist(),
        "frac_norm_perp": frac_norm_perp.tolist(),
        "r_l_instruct_norms": r_l_norms.tolist(),
        "norm_ratio_em_over_instruct": norm_ratio_em_over_instruct.tolist(),

        # band comparison (if sweep already ran)
        "band_comparison": band_info,

        "summary": {
            "cos_r_em_r_l_max": float(cos_r_em_r_l.max()),
            "cos_r_em_r_l_max_layer": int(np.argmax(cos_r_em_r_l)) + 1,
            "cos_r_em_r_l_min": float(cos_r_em_r_l.min()),
            "cos_r_em_r_l_min_layer": int(np.argmin(cos_r_em_r_l)) + 1,
            "cos_r_em_r_l_mean": float(cos_r_em_r_l.mean()),

            # Decomposition magnitude summary
            "frac_norm_perp_mean": float(frac_norm_perp.mean()),
            "frac_norm_perp_at_max_cos_layer": float(
                frac_norm_perp[int(np.argmax(cos_r_em_r_l))]
            ),

            # Hypothesis indicator (informal):
            # A (dilution): high cosine across layers, low frac_perp
            # B (substitution): low/moderate cosine, high frac_perp
            # C (restructuring): incoherent geometry; tested empirically
            # by the sweep, not from geometry alone.
        },

        "orthogonality_max_violation": max_ortho_violation,
    }
    save_json(p_em["geometry_em"], geometry)
    volume.commit()

    logger.info(
        f"cos(r_em, r_l_instruct): max={geometry['summary']['cos_r_em_r_l_max']:.3f} at "
        f"layer {geometry['summary']['cos_r_em_r_l_max_layer']}, "
        f"mean={geometry['summary']['cos_r_em_r_l_mean']:.3f}"
    )
    logger.info(
        f"||r_em_perp||/||r_em|| mean = {geometry['summary']['frac_norm_perp_mean']:.3f}, "
        f"at max-cos layer = {geometry['summary']['frac_norm_perp_at_max_cos_layer']:.3f}"
    )
    logger.info("Step 5e complete.")


@app.local_entrypoint()
def main():
    geometric_analysis_em.remote()
