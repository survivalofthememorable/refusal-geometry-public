"""
Step 8: Sync everything from the Modal volume to local disk.

Run this from your local machine after the pipeline completes:
  python 8_download_results.py

Files end up under ./results/phase1/...
"""

import os
import subprocess
import sys
from pathlib import Path

VOLUME_NAME = "phase1-directions-vol"
LOCAL_OUT = Path("./results")


def main():
    LOCAL_OUT.mkdir(exist_ok=True)

    # Modal volume get with --recursive copies a directory tree.
    # The volume mount root is "/", so we ask for "phase1/".
    cmd = [
        sys.executable, "-m", "modal", "volume", "get",
        VOLUME_NAME,
        "phase1",                # path inside the volume
        str(LOCAL_OUT.resolve()),
        "--force",
    ]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(
            "\nVolume get failed. Falling back to per-file download.\n"
            "If the recursive form is unsupported, list files with:\n"
            f"  python -m modal volume ls {VOLUME_NAME} phase1"
        )
        sys.exit(result.returncode)

    print(f"\nDone. Results synced to {LOCAL_OUT.resolve() / 'phase1'}")


if __name__ == "__main__":
    main()
