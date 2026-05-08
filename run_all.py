"""
End-to-end orchestrator for the Phase 1 pipeline.

Runs every step in order via `modal run`, so each step gets its own
container, its own resumable progress files, and its own clean logs.

Usage (from the directory containing the step scripts):
    python run_all.py              # run everything from step 0
    python run_all.py --start 4    # resume from step 4 (e.g., after fixing a bug)
    python run_all.py --only 4 5   # run only steps 4 and 5
    python run_all.py --skip-download  # don't run step 8 at the end

Exit behavior:
- If any step fails, the script stops and exits non-zero.
- Each step is itself resumable; re-running this script with --start N
  picks up cleanly because progress files persist on the Modal volume.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


# Ordered list of (step_number, script_name, human_label, runs_locally)
STEPS = [
    (0, "0_setup_and_verify.py",   "Setup & verify",          False),
    (1, "1_filter_prompts.py",     "Filter prompts",          False),
    (2, "2_extract_activations.py","Extract activations",     False),
    (3, "3_compute_directions.py", "Compute directions",      False),
    (4, "4_layer_sweep.py",        "Layer sweep",             False),
    (5, "5_geometric_analysis.py", "Geometric analysis",      False),
    (6, "6_harmbench_eval.py",     "HarmBench evaluation",    False),
    (7, "7_visualizations.py",     "Generate plots",          False),
    (8, "8_download_results.py",   "Download to local",       True),
]


def fmt_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def run_step(step_num: int, script: str, label: str, runs_locally: bool) -> int:
    """Run one step. Returns the subprocess return code."""
    here = Path(__file__).resolve().parent
    script_path = here / script

    if not script_path.exists():
        print(f"[ERROR] Step {step_num}: {script} not found at {script_path}")
        return 127

    if runs_locally:
        # Step 8 is a local Python script (uses Modal CLI internally).
        cmd = [sys.executable, str(script_path)]
    else:
        # Modal step: invoke via `modal run`.
        cmd = [sys.executable, "-m", "modal", "run", str(script_path)]

    print()
    print("=" * 72)
    print(f"  Step {step_num}: {label}")
    print(f"  Command: {' '.join(cmd)}")
    print("=" * 72)
    sys.stdout.flush()

    t0 = time.time()
    # Stream output live; do not capture it (so user sees Modal's progress)
    result = subprocess.run(cmd, cwd=str(here))
    duration = time.time() - t0

    if result.returncode == 0:
        print(f"[OK]   Step {step_num} finished in {fmt_duration(duration)}")
    else:
        print(f"[FAIL] Step {step_num} exited with code {result.returncode} after {fmt_duration(duration)}")
    return result.returncode


def parse_args():
    p = argparse.ArgumentParser(description="Run the full Phase 1 pipeline")
    p.add_argument(
        "--start", type=int, default=0,
        help="Step number to start from (default: 0)",
    )
    p.add_argument(
        "--end", type=int, default=8,
        help="Last step number to run, inclusive (default: 8)",
    )
    p.add_argument(
        "--only", type=int, nargs="+", default=None,
        help="Run only these specific step numbers, in order.",
    )
    p.add_argument(
        "--skip-download", action="store_true",
        help="Don't run step 8 (local download). Useful when running from a server.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print which steps would run without executing them.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.only is not None:
        selected = [s for s in STEPS if s[0] in args.only]
        # Preserve --only's input order (in case user wants step 4 before step 2 for some reason)
        selected.sort(key=lambda s: args.only.index(s[0]))
    else:
        selected = [s for s in STEPS if args.start <= s[0] <= args.end]

    if args.skip_download:
        selected = [s for s in selected if s[0] != 8]

    if not selected:
        print("No steps selected. Nothing to do.")
        return 0

    print()
    print("Pipeline plan:")
    for step_num, script, label, runs_locally in selected:
        loc = "local" if runs_locally else "modal"
        print(f"  [{step_num}] {label}  ({loc}: {script})")

    if args.dry_run:
        print("\n--dry-run set; not executing.")
        return 0

    overall_t0 = time.time()
    for step_num, script, label, runs_locally in selected:
        rc = run_step(step_num, script, label, runs_locally)
        if rc != 0:
            print()
            print(f"Pipeline halted at step {step_num}.")
            print(f"To resume after fixing the issue: python run_all.py --start {step_num}")
            return rc

    print()
    print("=" * 72)
    print(f"  All steps finished in {fmt_duration(time.time() - overall_t0)}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())