"""
One-time diagnostic: confirm HF token reaches the Modal container,
that whoami() succeeds, and that WildGuard is accessible.

Run BEFORE any expensive GPU work:
    python -m modal run modal_diag.py
"""
import modal
from modal_app import app, image, hf_secret


@app.function(image=image, secrets=[hf_secret], timeout=120)
def check_hf():
    import os
    print("=" * 60)
    print("HF environment in the Modal container")
    print("=" * 60)
    for key in ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"]:
        val = os.environ.get(key)
        present = "present" if val else "MISSING"
        snippet = (val[:6] + "...") if val else ""
        print(f"  {key}: {present} {snippet}")

    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
    )

    if not token:
        print("\nNo token found. Check `modal secret list`. Secret name "
              "must be 'huggingface-secret' and the env var inside should be HF_TOKEN.")
        return

    print("\n--- whoami ---")
    try:
        from huggingface_hub import whoami
        info = whoami(token=token)
        print(f"  user:  {info.get('name')}")
        print(f"  email: {info.get('email')}")
        print(f"  type:  {info.get('type')}")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__} — {e}")
        print("  -> token invalid for the account; regenerate at https://huggingface.co/settings/tokens")
        return

    print("\n--- WildGuard access ---")
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info("allenai/wildguard", token=token)
        print(f"  OK — last modified: {info.lastModified}")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__} — {e}")
        print("  -> click 'Agree and access' at https://huggingface.co/allenai/wildguard "
              "while logged into the same HF account.")

    print("\n--- Qwen2.5-14B-Instruct access (sanity) ---")
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info("Qwen/Qwen2.5-14B-Instruct", token=token)
        print(f"  OK — last modified: {info.lastModified}")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__} — {e}")


@app.local_entrypoint()
def main():
    check_hf.remote()