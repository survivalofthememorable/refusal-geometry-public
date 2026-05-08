"""
Spot-check pattern-vs-wildguard disagreements on r-ablation.
Prints the first 5 cases where pattern judge says compliance (>=0.5)
but WildGuard says refusal (<0.5).
"""
import json
import sys

# Make Windows console accept UTF-8 (in case of CJK content).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

with open("results/phase1/06_harmbench/completions_wildguard.json", encoding="utf-8") as f:
    data = json.load(f)

disagree = [
    r for r in data["harmbench"]["ablate_r"]
    if r["compliance_score_pattern"] >= 0.5
    and r["compliance_score_wildguard"] < 0.5
]
print(f"r-ablation disagreement (pattern says compliance, WildGuard says refusal): "
      f"{len(disagree)}/200")
print()

for r in disagree[:5]:
    print("=" * 70)
    print(f"id: {r['id']}")
    print(f"pattern: {r['compliance_score_pattern']}, "
          f"wildguard: {r['compliance_score_wildguard']}")
    print("---")
    completion = r["completion"][:500]
    # Replace any chars Windows console can't handle.
    try:
        print(completion)
    except UnicodeEncodeError:
        print(completion.encode("ascii", "replace").decode())
    print()

# Also do the reverse direction to be thorough — pattern says refusal,
# WildGuard says compliance. These are cases where WildGuard MISSES a
# refusal the pattern judge correctly caught. Should be very few.
reverse = [
    r for r in data["harmbench"]["ablate_r"]
    if r["compliance_score_pattern"] < 0.5
    and r["compliance_score_wildguard"] >= 0.5
]
print()
print("=" * 70)
print(f"REVERSE r-ablation disagreement (pattern=refusal, wildguard=compliance): "
      f"{len(reverse)}/200")
for r in reverse[:3]:
    print("=" * 70)
    print(f"id: {r['id']}")
    completion = r["completion"][:500]
    try:
        print(completion)
    except UnicodeEncodeError:
        print(completion.encode("ascii", "replace").decode())
    print()