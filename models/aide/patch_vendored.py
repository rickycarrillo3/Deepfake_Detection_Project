#!/usr/bin/env python3
"""Apply this repo's patches to the vendored AIDE clone.

The clone at models/aide/AIDE/ is git-ignored, so these edits cannot live in
version control — they have to be reapplied to whatever is on disk. setup.sh
calls this after cloning, but an existing clone (e.g. on a cluster, checked out
before these patches landed) needs it run directly:

    python3 models/aide/patch_vendored.py

Idempotent: already-patched files are reported and left alone. Prints a summary
and exits non-zero if any patch could not be applied, so setup.sh fails loudly
rather than silently producing an unpatched clone.
"""

import sys
from pathlib import Path

AIDE_DIR = Path(__file__).resolve().parent / "AIDE"

# (file, old, new, why). `old` must be absent once `new` is present, so the
# presence of `old` doubles as the "not yet applied" check.
PATCHES = [
    (
        "models/AIDE.py",
        "torch.Tensor(all_hpf_list_5x5)",
        "torch.from_numpy(np.stack(all_hpf_list_5x5)).float()",
        "silence the 'tensor from a list of numpy.ndarrays is extremely slow' "
        "UserWarning (one-time construction cost, not a speedup)",
    ),
    (
        "data/dct.py",
        "_, idx = torch.sort(grade)",
        "_, idx = torch.sort(grade, stable=True)",
        "make patch selection deterministic across devices: torch.sort defaults "
        "to stable=False, and CPU/CUDA break ties differently, so equal-scoring "
        "flat patches (common in image backgrounds, and exactly what the "
        "lowest-rank 'minmin' branch selects) could pick different regions",
    ),
]


def main():
    if not AIDE_DIR.is_dir():
        sys.exit(
            f"No vendored AIDE clone at {AIDE_DIR}.\n"
            "Run models/aide/setup.sh first, or clone it with:\n"
            "  git clone https://github.com/shilinyan99/AIDE.git models/aide/AIDE"
        )

    failed = False
    for rel_path, old, new, why in PATCHES:
        path = AIDE_DIR / rel_path
        if not path.is_file():
            print(f"MISSING  {rel_path} — not found in the clone")
            failed = True
            continue

        src = path.read_text()
        if new in src and old not in src:
            print(f"OK       {rel_path} — already patched")
            continue
        if old not in src:
            print(f"FAILED   {rel_path} — target text not found; upstream may have changed")
            print(f"         looking for: {old!r}")
            failed = True
            continue

        path.write_text(src.replace(old, new))
        print(f"PATCHED  {rel_path} — {why}")

    if failed:
        sys.exit("\nSome patches could not be applied (see above).")
    print("\nAll patches applied.")


if __name__ == "__main__":
    main()
