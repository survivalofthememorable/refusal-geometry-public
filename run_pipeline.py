"""
Orchestrator: run all phases in order.

Usage from the directory containing the step scripts:
  modal run run_pipeline.py

You can also run individual steps:
  modal run 0_setup_and_verify.py
  modal run 1_filter_prompts.py
  ...

Each step is independently resumable. If a step crashes mid-run,
re-running it picks up where it left off via progress files in the volume.
"""

import modal
from modal_app import app


# Import all step functions so Modal sees them.
# Each step file defines its own Modal app, but Modal can chain them.
# Easier path: run each step manually with `modal run <step>.py`.

# This file is here mainly as documentation and to give you a single
# command to run if everything is healthy.

if __name__ == "__main__":
    print("""
This orchestrator is intentionally a documentation stub.
Run each step explicitly so failures are localized:

  modal run 0_setup_and_verify.py
  modal run 1_filter_prompts.py
  modal run 2_extract_activations.py
  modal run 3_compute_directions.py
  modal run 4_layer_sweep.py
  modal run 5_geometric_analysis.py
  modal run 6_harmbench_eval.py
  modal run 7_visualizations.py

Then locally:
  python 8_download_results.py

Each step is resumable. If 4_layer_sweep.py is interrupted at layer 23,
re-running it will resume from layer 24.
""")
