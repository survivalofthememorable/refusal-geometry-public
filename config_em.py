"""
Phase 2 config extension.

This module extends the Phase 1 Config dataclass with EM-specific fields and
a paths_em property. Phase 1 config remains untouched; Phase 2 scripts import
from THIS file instead of the original config.py.

Strategy:
  - EM-specific fields live as new attributes on the same Config dataclass
    (we monkey-patch via setattr on the singleton CONFIG, which is simpler
    than subclassing and keeps imports identical to Phase 1 scripts).
  - paths_em is added as a separate dict accessor; it's a NEW property
    living under /vol/phase2/, never shadowing /vol/phase1/.

Why /vol/phase2/ and not /vol/phase1/02_em_*: because the EM run is a
separate experiment that reads (read-only) from Phase 1's outputs. Keeping
its outputs in a separate root makes it impossible to corrupt Phase 1 by
accident, and makes "delete Phase 2 and restart" a single rm -rf.

Constants enforced here:
  - EM_PROMPT_FLOOR = 32  (Arditi 2024 Appendix C: stable r-direction at N>=32)
  - All paths derived from CONFIG.em_output_root, with the same shape as
    Phase 1 paths so plotting / downstream tools can be parameterised.

Required env config (set before running any Phase 2 script):
  - cfg.em_model_id  — HuggingFace model id for FullFT-Medical
  - cfg.em_revision  — revision/commit (default "main"; pin before publish)

If the EM model is local-only (not on HF Hub), set em_model_id to the local
path and em_revision to "main"; HuggingFace transformers handles paths
transparently as long as the directory has config.json + tokenizer files.
"""

from pathlib import Path
from typing import Dict
from config import CONFIG  # Phase 1 singleton

# ---- Arditi (2024) Appendix C ----
# Their robustness analysis demonstrates stable refusal-direction extraction
# at N=32 paired prompts. We use this as the floor; we cap at 128 to stay
# comparable to Phase 1's extraction set size.
EM_PROMPT_FLOOR = 32
EM_PROMPT_CAP = 128


# ---- EM model identification ----
# *** SET THESE BEFORE RUNNING ***
# Either a HuggingFace Hub model id or a local path. The default sentinel
# below will trigger an explicit error in step 0 if not changed.
EM_MODEL_ID_DEFAULT = "ModelOrganismsForEM/Qwen2.5-14B-Instruct_full-ft"
EM_REVISION_DEFAULT = "main"

# Allow override from environment so we don't have to edit this file when
# switching between local and Hub paths.
import os as _os
_EM_ID_ENV = _os.environ.get("EM_MODEL_ID", "").strip()
_EM_REV_ENV = _os.environ.get("EM_REVISION", "").strip()


# ---- Patch Phase 1 CONFIG with EM fields ----
# We use setattr so existing Phase 1 scripts are unaffected; Phase 2 scripts
# read these via the same CONFIG singleton.
setattr(CONFIG, "em_model_id", _EM_ID_ENV or EM_MODEL_ID_DEFAULT)
setattr(CONFIG, "em_revision", _EM_REV_ENV or EM_REVISION_DEFAULT)
setattr(CONFIG, "em_output_root", "/vol/phase2")
setattr(CONFIG, "em_prompt_floor", EM_PROMPT_FLOOR)
setattr(CONFIG, "em_prompt_cap", EM_PROMPT_CAP)


def assert_em_model_set():
    """Fail loudly if the EM model id has not been set.

    Call this at the top of any Phase 2 script that loads the EM model.
    The default sentinel value would otherwise cause a confusing HF error
    minutes into the run; this raises immediately with a helpful message.
    """
    if CONFIG.em_model_id.startswith("TODO_") or not CONFIG.em_model_id:
        raise RuntimeError(
            "EM model id is not set. Either edit EM_MODEL_ID_DEFAULT in "
            "config_em.py, or set the EM_MODEL_ID environment variable, "
            "e.g.:\n"
            "    set EM_MODEL_ID=path/to/fullft-medical    (Windows cmd)\n"
            "    $env:EM_MODEL_ID = 'path/to/fullft-medical'    (PowerShell)\n"
            "    export EM_MODEL_ID=path/to/fullft-medical    (bash)\n"
            "Modal forwards env vars to functions automatically; you don't "
            "need to do anything special inside the container."
        )


def paths_em() -> Dict[str, str]:
    """Phase 2 path layout. Mirrors Phase 1 layout under /vol/phase2/.

    Lives as a function rather than a property because Phase 1's CONFIG
    is a frozen-in-place dataclass and we don't want to mutate its
    @property handlers after construction.
    """
    root = Path(CONFIG.em_output_root)
    p1_paths = CONFIG.paths   # Phase 1 paths, read-only

    return {
        # Phase 2 root + log
        "root": str(root),
        "log": str(root / "phase2.log"),
        "metadata": str(root / "metadata.json"),

        # Step 0e (verification)
        "verify_report": str(root / "00_verify_report.json"),

        # Step 1e (filter on EM)
        "candidate_prompts": p1_paths["candidate_prompts"],   # READ-ONLY: reuse Phase 1's candidates
        "filter_results_em": str(root / "01_filter_results_em.json"),
        "prompt_lists_em": str(root / "01_prompt_lists_em.json"),

        # Step 2e (EM activations)
        "act_dir": str(root / "02_activations"),
        "act_em_harmful_train": str(root / "02_activations" / "em_harmful_train.npy"),
        "act_em_harmless_train": str(root / "02_activations" / "em_harmless_train.npy"),
        "act_em_harmful_val": str(root / "02_activations" / "em_harmful_val.npy"),
        "act_em_harmless_val": str(root / "02_activations" / "em_harmless_val.npy"),
        "extraction_progress": str(root / "02_activations" / "_progress.json"),

        # Step 3e (EM directions)
        "r_em": str(root / "03_directions" / "r_em.npy"),
        "r_em_hat": str(root / "03_directions" / "r_em_hat.npy"),
        "d_em": str(root / "03_directions" / "d_em.npy"),
        "d_em_hat": str(root / "03_directions" / "d_em_hat.npy"),
        "r_em_norms": str(root / "03_directions" / "r_em_norms.npy"),
        "d_em_norms": str(root / "03_directions" / "d_em_norms.npy"),
        "directions_metadata": str(root / "03_directions" / "metadata.json"),

        # Step 4e (sweep on EM)
        "sweep_dir": str(root / "04_layer_sweep"),
        "sweep_r_em": str(root / "04_layer_sweep" / "sweep_r_em.json"),
        "sweep_random_em": str(root / "04_layer_sweep" / "sweep_random_em.json"),
        "selected_layers_em": str(root / "04_layer_sweep" / "selected_layers_em.json"),

        # OPTIONAL Tier 3: d_em sweep (appendix only)
        "sweep_d_em": str(root / "04_layer_sweep" / "sweep_d_em.json"),

        # Step 5e (cross-model geometric analysis)
        "geometry_em": str(root / "05_geometry" / "geometry_em.json"),

        # Step 4cross (three-way decomposition at l*_em)
        "cross_dir": str(root / "04cross_three_way"),
        "cross_results": str(root / "04cross_three_way" / "results.json"),
        "cross_completions": str(root / "04cross_three_way" / "completions.json"),
        "cross_directions": str(root / "04cross_three_way" / "directions.npz"),

        # Step 9b-em (WildGuard rescoring of cross completions)
        "cross_completions_wildguard": str(root / "04cross_three_way" / "completions_wildguard.json"),
        "cross_results_wildguard": str(root / "04cross_three_way" / "results_wildguard.json"),

        # Read-only references to Phase 1 data we depend on
        "p1_act_instruct_harmful_train": p1_paths["act_instruct_harmful_train"],
        "p1_act_instruct_harmless_train": p1_paths["act_instruct_harmless_train"],
        "p1_act_instruct_harmful_val": p1_paths["act_instruct_harmful_val"],
        "p1_act_instruct_harmless_val": p1_paths["act_instruct_harmless_val"],
        "p1_r_l": p1_paths["r_l"],
        "p1_r_l_hat": p1_paths["r_l_hat"],
        "p1_d_l": p1_paths["d_l"],
        "p1_d_l_hat": p1_paths["d_l_hat"],
        "p1_filter_results": p1_paths["filter_results"],
        "p1_prompt_lists": p1_paths["prompt_lists"],
        "p1_selected_layers": p1_paths["selected_layers"],
    }
