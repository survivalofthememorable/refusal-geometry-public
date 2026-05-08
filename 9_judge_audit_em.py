"""
Step 9-em: Add pattern-judge scores to the cross-ablation completions.

Why this exists:
  step 4cross_three_way scored completions inline with WildGuard. To present
  the "pattern judge vs WildGuard" comparison in the paper for EM (mirroring
  the Phase 1 table), we ALSO need pattern-judge scores on the same
  completions. Pattern judging is local-CPU and instantaneous; we add it
  as a post-processing pass.

  Also flags any judge disagreements for inspection (analogue to the
  Phase 1 multilingual fallback / wrapped-refusal audit).

CPU-only, no GPU needed (pattern judge is regex). Run after step 4cross.

Outputs:
  - 04cross_three_way/completions_dual_scored.json
    (mirrors completions.json structure but with both compliance_score_pattern
    and compliance_score_wildguard fields; the original
    'compliance_score' in completions.json was the WildGuard score)
  - 04cross_three_way/results_dual.json (summary by both judges)

Run:
    python -m modal run 9_judge_audit_em.py
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
def add_pattern_scores():
    import json
    import numpy as np
    from pathlib import Path
    from common import setup_logger, save_json, load_json, judge_string_match

    cfg = CONFIG
    p_em = paths_em()

    volume.reload()
    logger = setup_logger("step9_em", p_em["log"])
    logger.info("=" * 60)
    logger.info("STEP 9-em: dual-score cross completions (pattern + wildguard)")
    logger.info("=" * 60)

    # ---- Load cross completions (these already have WildGuard scores) ----
    if not Path(p_em["cross_completions"]).exists():
        raise RuntimeError(
            f"Cross completions not found at {p_em['cross_completions']}. "
            f"Run step 4cross_three_way first."
        )
    completions = load_json(p_em["cross_completions"])

    # ---- Walk the structure and add pattern-judge scores ----
    # We need the prompt text for each completion's id. For val we have it
    # in prompt_lists_em. For HarmBench and capability, we need to reconstruct
    # the prompt set the same deterministic way step 4cross did.
    splits = load_json(p_em["prompt_lists_em"])
    val_harmful_lookup = {r["id"]: r["text"] for r in splits["harmful_val"]}
    val_harmless_lookup = {r["id"]: r["text"] for r in splits["harmless_val"]}

    # Recover HarmBench prompts deterministically (same logic as step 4cross)
    from datasets import load_dataset
    try:
        hb = load_dataset("walledai/HarmBench", "standard", split="train")
    except Exception:
        hb = load_dataset("cais/HarmBench", split="train")
    hb_field = "prompt" if "prompt" in hb.column_names else "Behavior"
    rng = np.random.default_rng(cfg.seed)
    hb_indices = rng.choice(len(hb), size=min(cfg.n_harmbench_eval, len(hb)), replace=False)
    hb_lookup = {f"harmbench_{int(i)}": hb[int(i)][hb_field] for i in hb_indices}

    # Recover capability prompts deterministically (same logic)
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    alpaca_filt = [
        {"instruction": ex["instruction"], "idx": i}
        for i, ex in enumerate(alpaca)
        if not ex.get("input") and len(ex["instruction"].split()) >= 5
    ]
    rng2 = np.random.default_rng(cfg.seed + 1)
    cap_indices = rng2.choice(len(alpaca_filt), size=cfg.n_alpaca_capability, replace=False)
    cap_lookup = {
        f"alpaca_cap_{int(alpaca_filt[int(i)]['idx'])}": alpaca_filt[int(i)]["instruction"]
        for i in cap_indices
    }

    def add_pattern(item, prompt_lookup):
        # The original 'compliance_score' was the WildGuard score (set in step 4cross).
        # Rename to make it explicit; add pattern-judge alongside.
        wg = item.get("compliance_score")
        prompt_text = prompt_lookup.get(item["id"], "")
        if not prompt_text:
            # If we can't find the prompt, the pattern judge still works on
            # the completion alone (it's keyed on response content, not prompt).
            # But log it.
            logger.warning(f"No prompt text for {item['id']}; using empty prompt")
        pat = judge_string_match(prompt_text, item["completion"])
        return {
            "id": item["id"],
            "completion": item["completion"],
            "compliance_score_wildguard": wg,
            "compliance_score_pattern": pat,
        }

    # Walk val
    new_completions = {"val": {}, "harmbench": {}, "capability": {}}
    for cond, splits_dict in completions["val"].items():
        new_completions["val"][cond] = {
            "harmful": [add_pattern(it, val_harmful_lookup) for it in splits_dict["harmful"]],
            "harmless": [add_pattern(it, val_harmless_lookup) for it in splits_dict["harmless"]],
        }

    for cond, items in completions["harmbench"].items():
        new_completions["harmbench"][cond] = [add_pattern(it, hb_lookup) for it in items]

    for cond, items in completions["capability"].items():
        new_completions["capability"][cond] = [add_pattern(it, cap_lookup) for it in items]

    # ---- Save dual-scored completions ----
    dual_path = str(Path(p_em["cross_completions"]).parent / "completions_dual_scored.json")
    save_json(dual_path, new_completions)
    logger.info(f"Saved dual-scored completions to {dual_path}")

    # ---- Build dual-judge summary ----
    def safe_mean_skip_nan(vals):
        clean = [v for v in vals
                 if not (isinstance(v, float) and v != v)]
        return float(sum(clean) / len(clean)) if clean else float("nan"), len(clean)

    dual_summary = {"val": {}, "harmbench": {}, "capability": {}}

    for cond, splits_dict in new_completions["val"].items():
        h_pat = [r["compliance_score_pattern"] for r in splits_dict["harmful"]]
        h_wg  = [r["compliance_score_wildguard"] for r in splits_dict["harmful"]]
        l_pat = [r["compliance_score_pattern"] for r in splits_dict["harmless"]]
        l_wg  = [r["compliance_score_wildguard"] for r in splits_dict["harmless"]]
        ch_pat, n_h_pat = safe_mean_skip_nan(h_pat)
        ch_wg,  n_h_wg  = safe_mean_skip_nan(h_wg)
        lr_pat, n_l_pat = safe_mean_skip_nan([1 - s for s in l_pat])
        lr_wg,  n_l_wg  = safe_mean_skip_nan([1 - s for s in l_wg])
        dual_summary["val"][cond] = {
            "pattern": {"compliance_harmful": ch_pat, "false_refusal_harmless": lr_pat,
                        "score_S": ch_pat - lr_pat,
                        "n_harmful": n_h_pat, "n_harmless": n_l_pat},
            "wildguard": {"compliance_harmful": ch_wg, "false_refusal_harmless": lr_wg,
                          "score_S": ch_wg - lr_wg,
                          "n_harmful": n_h_wg, "n_harmless": n_l_wg},
        }

    for cond, items in new_completions["harmbench"].items():
        pat_scores = [r["compliance_score_pattern"] for r in items]
        wg_scores = [r["compliance_score_wildguard"] for r in items]
        m_pat, n_pat = safe_mean_skip_nan(pat_scores)
        m_wg, n_wg = safe_mean_skip_nan(wg_scores)
        dual_summary["harmbench"][cond] = {
            "pattern_compliance": m_pat, "n_pattern": n_pat,
            "wildguard_compliance": m_wg, "n_wildguard": n_wg,
            "delta_wg_minus_pattern": (m_wg - m_pat) if not (np.isnan(m_wg) or np.isnan(m_pat)) else float("nan"),
        }
        logger.info(
            f"[harmbench/{cond}] pattern={m_pat:.3f} wildguard={m_wg:.3f} "
            f"delta={m_wg - m_pat:+.3f}"
        )

    for cond, items in new_completions["capability"].items():
        pat_scores = [r["compliance_score_pattern"] for r in items]
        wg_scores = [r["compliance_score_wildguard"] for r in items]
        m_pat, n_pat = safe_mean_skip_nan(pat_scores)
        m_wg, n_wg = safe_mean_skip_nan(wg_scores)
        dual_summary["capability"][cond] = {
            "pattern_compliance": m_pat, "n_pattern": n_pat,
            "wildguard_compliance": m_wg, "n_wildguard": n_wg,
        }
        logger.info(f"[capability/{cond}] pattern={m_pat:.3f} wildguard={m_wg:.3f}")

    dual_results_path = str(Path(p_em["cross_results"]).parent / "results_dual.json")
    save_json(dual_results_path, dual_summary)
    volume.commit()
    logger.info(f"Saved dual-judge summary to {dual_results_path}")
    logger.info("Step 9-em complete.")


@app.local_entrypoint()
def main():
    add_pattern_scores.remote()
