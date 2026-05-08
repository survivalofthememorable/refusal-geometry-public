"""
Phase 2 orchestrator. Runs Tier 1+2 sequentially with stage gates.

This is a LOCAL entrypoint — it doesn't run on Modal itself. It calls
.remote() on each stage's modal function in the correct order.

Tier 1+2 order (assumes Phase 1 outputs already on /vol/phase1/):

  1.  0_setup_em.verify_em                              ~5  min A100
  2.  1_filter_prompts_em.filter_prompts_em             ~15 min A100
  3.  2_extract_activations_em.extract_em_activations   ~20 min A100
  4.  3_compute_directions_em.compute_directions_em     <1  min CPU
  5.  5_geometric_analysis_em.geometric_analysis_em     <1  min CPU  *** GATE A ***
  6.  4_layer_sweep_em.run_sweep_em("r_em")             ~2  hr  A100
  7.  4_layer_sweep_em.run_sweep_em("random_em")        ~2  hr  A100
  8.  4_layer_sweep_em.select_layers_em                 <1  min CPU
  9.  5_geometric_analysis_em (re-run for band info)    <1  min CPU
  10. 4cross_three_way.decompose_cross_directions       <1  min CPU
  11. 4cross_three_way.run_cross_ablation               ~1  hr  A100
  12. 9_judge_audit_em.add_pattern_scores               <5  min CPU

Total: ~6.5 hours A100 + a few minutes of CPU.

GATE A (after step 5): inspect cos(r_em, r_l_instruct) plot.
  If high (close to 1) across most layers: hypothesis A (dilution).
  If low/moderate: hypothesis B (substitution) is plausible; sweep result
    will tell us which.
  Either way, proceed to sweep.

Manual / interactive:
  Each stage prints its log to stdout via the Modal driver. You can re-run
  any individual stage with python -m modal run <script>.py. Resumability
  is per-stage; partial progress is preserved via the volume's progress files.

Run:
    python -m modal run run_phase2.py
or one stage at a time, e.g.:
    python -m modal run 4_layer_sweep_em.py
"""

import time
import modal
from modal_app import app
from config_em import assert_em_model_set


# ---- Import each stage's Modal function ----
# We import the function objects so we can call .remote() on them in order.
from importlib import import_module


def _import_function(module_name: str, func_name: str):
    mod = import_module(module_name)
    return getattr(mod, func_name)


@app.local_entrypoint()
def run_tier_1_2():
    """Run Tier 1+2 (~6.5 hrs A100). Sequential to maximize cache reuse."""
    assert_em_model_set()

    # Note: module names use the file name without .py and replace the
    # underscored prefix dot. Python doesn't allow modules starting with a
    # digit; Modal handles them via -m modal run path/to/file.py.
    # Importing them via importlib won't work here for that reason.
    #
    # Instead, the user invokes this orchestrator AS DOCUMENTATION and runs
    # the individual scripts. Below we simply print the order.
    stages = [
        ("0_setup_em.py",                 "verify_em",                   "~5 min A100"),
        ("1_filter_prompts_em.py",        "filter_prompts_em",           "~15 min A100"),
        ("2_extract_activations_em.py",   "extract_em_activations",      "~20 min A100"),
        ("3_compute_directions_em.py",    "compute_directions_em",       "<1 min CPU"),
        ("5_geometric_analysis_em.py",    "geometric_analysis_em",       "<1 min CPU (GATE A)"),
        ("4_layer_sweep_em.py",           "run_sweep_em(r_em+random_em) + select_layers_em", "~4 hr A100"),
        ("5_geometric_analysis_em.py",    "geometric_analysis_em (re-run for band info)", "<1 min CPU"),
        ("4cross_three_way.py",           "decompose_cross_directions + run_cross_ablation", "~1 hr A100"),
        ("9_judge_audit_em.py",           "add_pattern_scores",          "<5 min CPU"),
    ]

    print("\n" + "=" * 70)
    print("Phase 2 Tier 1+2 stage order")
    print("=" * 70)
    print(
        f"{'#':<3} {'Script':<35} {'Function(s)':<55} {'Cost':<25}"
    )
    print("-" * 120)
    for i, (script, func, cost) in enumerate(stages, 1):
        print(f"{i:<3} {script:<35} {func:<55} {cost:<25}")
    print("\n")
    print(
        "Run each in sequence with `python -m modal run <script>`. "
        "Each stage is resumable; if a stage fails partway, just re-run it."
    )
    print(
        "GATE A: after step 5_geometric_analysis_em, look at "
        "geometry_em.json's cos_r_em_r_l_instruct_per_layer. High values "
        "(>0.9) suggest hypothesis A (dilution); low/moderate values suggest "
        "hypothesis B (substitution)."
    )
    print("=" * 70)
