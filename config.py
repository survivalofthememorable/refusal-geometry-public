"""
Central config. Change values here, not in step scripts.
Every script imports CONFIG from this file.
"""

from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, Any
import json


@dataclass
class Config:
    # ---- Models ----
    # Pin to specific revisions for reproducibility.
    # Update revisions only deliberately, not silently.
    instruct_model_id: str = "Qwen/Qwen2.5-14B-Instruct"
    instruct_revision: str = "main"  # TODO: pin to specific commit hash before publishing
    base_model_id: str = "Qwen/Qwen2.5-14B"
    base_revision: str = "main"      # TODO: pin to specific commit hash before publishing

    # ---- Architecture (verified at runtime) ----
    expected_n_layers: int = 48
    expected_hidden_size: int = 5120

    # ---- Datasets ----
    n_harmful_train: int = 128
    n_harmless_train: int = 128
    n_harmful_val: int = 32
    n_harmless_val: int = 32
    n_harmbench_eval: int = 200
    n_alpaca_capability: int = 100

    # Initial sampling: oversample so refusal-filter dropouts don't leave us short.
    n_harmful_initial: int = 250
    n_harmless_initial: int = 250

    # ---- Activation extraction ----
    activation_dtype: str = "float32"   # store activations in float32
    forward_dtype: str = "bfloat16"     # forward pass in bfloat16
    batch_size_extraction: int = 1      # one prompt at a time, simpler and safer

    # ---- Layer sweep ----
    sweep_max_new_tokens: int = 100
    sweep_temperature: float = 0.0      # greedy
    sweep_strongreject_threshold: float = 0.5  # compliance > 0.5 = "complied"
    sweep_layers_to_skip: list = field(default_factory=lambda: [])  # skip layers if known irrelevant
    band_threshold_frac: float = 0.5    # band = layers where score >= 0.5 * peak

    # ---- HarmBench eval ----
    eval_max_new_tokens: int = 256
    eval_n_bootstrap: int = 1000

    # ---- Judge ----
    # Default is "string_match": robust pattern-based refusal classifier with
    # ZERO external dependencies. Fast, free, no model downloads, no gating.
    # Achieves ~90-95% accuracy on Qwen2.5-Instruct outputs (model produces
    # very stereotyped refusal openings). Good enough for layer selection.
    #
    # Other options (require HF model access, may need approval forms):
    #   "wildguard"     - allenai/wildguard. Apache-2.0 but gated; approve
    #                     form at https://huggingface.co/allenai/wildguard.
    #                     Set judge_model="allenai/wildguard".
    #   "llama_guard"   - meta-llama/Llama-Guard-3-8B. Gated by Meta;
    #                     approval can take hours to days.
    #                     Set judge_model="meta-llama/Llama-Guard-3-8B".
    judge_kind: str = "wildguard"
    judge_model: str = "allenai/wildguard"  # unused for string_match

    # ---- Reproducibility ----
    seed: int = 42

    # ---- Paths (relative to volume mount) ----
    output_root: str = "/vol/phase1"

    @property
    def paths(self) -> Dict[str, str]:
        root = Path(self.output_root)
        return {
            "root": str(root),
            "metadata": str(root / "metadata.json"),
            "log": str(root / "pipeline.log"),

            # Step 0
            "verify_report": str(root / "00_verify_report.json"),

            # Step 1 (filtering)
            "candidate_prompts": str(root / "01_candidate_prompts.json"),
            "filter_results": str(root / "01_filter_results.json"),
            "prompt_lists": str(root / "01_prompt_lists.json"),

            # Step 2 (activations)
            "act_dir": str(root / "02_activations"),
            "act_instruct_harmful_train": str(root / "02_activations" / "instruct_harmful_train.npy"),
            "act_instruct_harmless_train": str(root / "02_activations" / "instruct_harmless_train.npy"),
            "act_base_harmful_train": str(root / "02_activations" / "base_harmful_train.npy"),
            "act_instruct_harmful_val": str(root / "02_activations" / "instruct_harmful_val.npy"),
            "act_instruct_harmless_val": str(root / "02_activations" / "instruct_harmless_val.npy"),
            "extraction_progress": str(root / "02_activations" / "_progress.json"),

            # Step 3 (directions)
            "r_l": str(root / "03_directions" / "r_l.npy"),
            "r_l_hat": str(root / "03_directions" / "r_l_hat.npy"),
            "d_l": str(root / "03_directions" / "d_l.npy"),
            "d_l_hat": str(root / "03_directions" / "d_l_hat.npy"),
            "r_norms": str(root / "03_directions" / "r_norms.npy"),
            "d_norms": str(root / "03_directions" / "d_norms.npy"),
            "directions_metadata": str(root / "03_directions" / "metadata.json"),

            # Step 4 (layer sweep)
            "sweep_dir": str(root / "04_layer_sweep"),
            "sweep_r": str(root / "04_layer_sweep" / "sweep_r.json"),
            "sweep_d": str(root / "04_layer_sweep" / "sweep_d.json"),
            "sweep_random": str(root / "04_layer_sweep" / "sweep_random.json"),
            "sweep_progress": str(root / "04_layer_sweep" / "_progress.json"),
            "selected_layers": str(root / "04_layer_sweep" / "selected_layers.json"),

            # Step 5 (geometry)
            "geometry": str(root / "05_geometry" / "geometry.json"),

            # Step 6 (HarmBench)
            "harmbench_results": str(root / "06_harmbench" / "results.json"),
            "harmbench_completions": str(root / "06_harmbench" / "completions.json"),
            "capability_results": str(root / "06_harmbench" / "capability.json"),

            # Step 7 (plots)
            "plots_dir": str(root / "07_plots"),
        }


CONFIG = Config()


def save_config_to_metadata(cfg: Config, path: str, extra: Dict[str, Any] = None):
    """Persist config + environment info for full reproducibility."""
    import torch
    import transformers
    import sys
    import platform

    meta = {
        "config": asdict(cfg),
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    if extra:
        meta["extra"] = extra

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, default=str)