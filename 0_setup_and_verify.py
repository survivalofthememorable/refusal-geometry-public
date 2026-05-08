"""
Step 0: Environment + dataset preparation.

What this does:
  1. Verifies model architecture and tokenizer for both base and instruct.
  2. Documents hidden_states semantics by inspection.
  3. Loads AdvBench harmful prompts and Alpaca harmless prompts.
  4. Samples N candidates for filtering (Step 1).
  5. Saves a metadata.json with everything needed for reproducibility.

Outputs:
  - /vol/phase1/metadata.json
  - /vol/phase1/00_verify_report.json
  - /vol/phase1/01_candidate_prompts.json
"""

import json
from pathlib import Path

import modal
from modal_app import app, image, volume, hf_secret, GPU_A100, TIMEOUT_SHORT, VOL_MOUNT, VOLUMES
from config import CONFIG, save_config_to_metadata


@app.function(
    image=image,
    gpu=GPU_A100,
    volumes=VOLUMES,
    secrets=[hf_secret],
    timeout=TIMEOUT_SHORT,
)
def setup_and_verify():
    """Run on Modal: verify env, prepare datasets."""
    import numpy as np
    import pandas as pd
    from datasets import load_dataset
    from common import (
        setup_logger, seed_all, save_json, file_hash, text_hash,
        load_model_and_tokenizer, verify_architecture, verify_hidden_states_semantics,
    )

    cfg = CONFIG
    paths = cfg.paths

    # Volume must be reloaded BEFORE opening any file handles in /vol,
    # otherwise volume.reload() / volume.commit() can fail with
    # "there are open files preventing the operation".
    Path(paths["root"]).mkdir(parents=True, exist_ok=True)
    logger = setup_logger("step0", paths["log"])
    logger.info("=" * 60)
    logger.info("STEP 0: setup and verify")
    logger.info("=" * 60)

    seed_all(cfg.seed)

    # ---- 0.1 Verify both models ----
    verify_report = {}

    for label, mid, rev in [
        ("instruct", cfg.instruct_model_id, cfg.instruct_revision),
        ("base", cfg.base_model_id, cfg.base_revision),
    ]:
        logger.info(f"--- Verifying {label}: {mid} ---")
        model, tok = load_model_and_tokenizer(mid, rev, cfg.forward_dtype, "auto", logger)
        verify_architecture(model, cfg.expected_n_layers, cfg.expected_hidden_size, logger)
        hs_info = verify_hidden_states_semantics(model, tok, logger)

        verify_report[label] = {
            "model_id": mid,
            "revision": rev,
            "n_layers": model.config.num_hidden_layers,
            "hidden_size": model.config.hidden_size,
            "vocab_size": model.config.vocab_size,
            "tokenizer_class": tok.__class__.__name__,
            "tokenizer_pad_token_id": tok.pad_token_id,
            "tokenizer_eos_token_id": tok.eos_token_id,
            "hidden_states_info": hs_info,
        }

        # Free GPU mem before next model.
        del model
        del tok
        import torch, gc
        gc.collect()
        torch.cuda.empty_cache()

    # Confirm tokenizers are identical between base and instruct.
    # Same tokenizer_class + same vocab_size + same special tokens is necessary.
    assert verify_report["instruct"]["vocab_size"] == verify_report["base"]["vocab_size"], \
        "Vocab sizes differ between base and instruct — cannot subtract activations meaningfully"
    logger.info("Tokenizer/vocab identity confirmed between base and instruct.")

    save_json(paths["verify_report"], verify_report)
    logger.info(f"Verify report saved to {paths['verify_report']}")

    # ---- 0.2 Load and sample harmful prompts (AdvBench) ----
    logger.info("Loading AdvBench harmful prompts from /root/advbench.csv ...")
    advbench = pd.read_csv("/root/advbench.csv")
    logger.info(f"AdvBench has {len(advbench)} prompts")

    rng = np.random.default_rng(cfg.seed)
    harmful_indices = rng.choice(len(advbench), size=cfg.n_harmful_initial, replace=False)
    harmful_prompts = [
        {
            "id": f"advbench_{int(i)}",
            "source": "advbench",
            "text": str(advbench.iloc[int(i)]["goal"]),
            "source_index": int(i),
        }
        for i in harmful_indices
    ]
    logger.info(f"Sampled {len(harmful_prompts)} harmful candidates")

    # ---- 0.3 Load and sample harmless prompts (Alpaca) ----
    logger.info("Loading Alpaca harmless prompts ...")
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    # Filter to instruction-only (no input field) for cleaner contrast with AdvBench.
    alpaca_filtered = [
        {"instruction": ex["instruction"], "idx": i}
        for i, ex in enumerate(alpaca)
        if (not ex.get("input")) and len(ex["instruction"].split()) >= 5
    ]
    logger.info(f"Alpaca filtered to {len(alpaca_filtered)} instruction-only prompts")

    harmless_indices = rng.choice(len(alpaca_filtered), size=cfg.n_harmless_initial, replace=False)
    harmless_prompts = [
        {
            "id": f"alpaca_{int(alpaca_filtered[int(i)]['idx'])}",
            "source": "alpaca",
            "text": alpaca_filtered[int(i)]["instruction"],
            "source_index": int(alpaca_filtered[int(i)]["idx"]),
        }
        for i in harmless_indices
    ]
    logger.info(f"Sampled {len(harmless_prompts)} harmless candidates")

    # ---- 0.4 Save candidate prompts ----
    candidate_prompts = {
        "harmful_candidates": harmful_prompts,
        "harmless_candidates": harmless_prompts,
        "harmful_source_hash": file_hash("/root/advbench.csv"),
        "alpaca_dataset": "tatsu-lab/alpaca",
        "n_harmful_candidates": len(harmful_prompts),
        "n_harmless_candidates": len(harmless_prompts),
    }
    save_json(paths["candidate_prompts"], candidate_prompts)
    logger.info(f"Candidate prompts saved to {paths['candidate_prompts']}")

    # ---- 0.5 Save full metadata ----
    save_config_to_metadata(
        cfg,
        paths["metadata"],
        extra={
            "verify_report": verify_report,
            "step_completed": "step0",
        },
    )
    logger.info(f"Metadata saved to {paths['metadata']}")

    # Commit the volume so the next script sees these files.
    volume.commit()
    logger.info("Volume committed. Step 0 complete.")


@app.local_entrypoint()
def main():
    setup_and_verify.remote()
