"""Acceptance test for baselines/: every checkpoint must load strictly.

Run by setup.sh, and worth rerunning after touching networks.py:

    baselines/venv/bin/python baselines/verify_checkpoints.py

Why this exists
---------------
Two of the six architectures (MISLNet, and the prototype head shared by
R50_TF) are transcribed or reconstructed rather than imported from public
upstream code. A wrong transcription does not crash — it produces scores that
look perfectly reasonable and are meaningless. Loading with strict=True turns
that silent failure into a loud one: any missing, unexpected or mis-shaped
tensor aborts.

Each baseline is also run forward once on random input, so a shape error that
only manifests at inference (e.g. the flatten width feeding MISLNet's first FC)
is caught here rather than mid-benchmark.

Baselines whose checkpoint is absent are reported as SKIP, not failed — the
weights are not public and may not all be present.
"""

import os
import sys
import traceback

import torch

BASELINES_DIR = os.path.dirname(os.path.abspath(__file__))
if BASELINES_DIR not in sys.path:
    sys.path.insert(0, BASELINES_DIR)

# name -> (subdirectory, input HxW used for the forward-pass smoke test)
CASES = [
    ("R50_TF (R50-E2P)", "R50_TF", (256, 256)),
    ("R50_nodown (R50-ND)", "R50_nodown", (256, 256)),
    ("NPR", "NPR", (256, 256)),
    ("MISLNet", "MISLNet", (96, 96)),
    ("CLIP-D", "CLIP-D", (224, 224)),
    ("P2G", "P2G", (224, 224)),
]


def _verify_one(subdir, hw):
    """Load one baseline and run it forward once. Runs in its own process."""
    import importlib.util

    model_dir = os.path.join(BASELINES_DIR, subdir)
    sys.path.insert(0, model_dir)

    spec = importlib.util.spec_from_file_location(
        f"verify_{subdir.replace('-', '_')}", os.path.join(model_dir, "model.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model = module.load()
    print(f"strict load OK -> {module.result_stem()}")

    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, *hw).to(next(model.parameters()).device))
    print(f"forward{hw} -> {tuple(out.shape)}")


def main():
    import subprocess

    failures = 0
    skipped = 0

    for label, subdir, hw in CASES:
        model_dir = os.path.join(BASELINES_DIR, subdir)
        weights = os.path.join(model_dir, "best.pt")
        model_py = os.path.join(model_dir, "model.py")

        if not os.path.isfile(weights):
            print(f"[SKIP] {label}: no checkpoint at {weights}")
            skipped += 1
            continue
        if not os.path.isfile(model_py):
            print(f"[SKIP] {label}: not implemented yet ({model_py} missing)")
            skipped += 1
            continue

        # Each baseline is verified in a fresh interpreter. Two of the vendored
        # clones (grip-unina's and NPR's) each ship a top-level package called
        # `networks`, and every model.py permanently prepends its own clone to
        # sys.path — so loading all six in one process would let whichever ran
        # first shadow the others. run.py never hits this (one model per
        # process); a subprocess per case reproduces that same isolation
        # instead of working around it.
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--case", subdir, str(hw[0]), str(hw[1])],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            for line in proc.stdout.strip().splitlines()[-2:]:
                print(f"[OK  ] {label}: {line}" if "strict load" in line else f"       {line}")
        else:
            print(f"[FAIL] {label}:")
            print((proc.stderr or proc.stdout).strip()[-2000:])
            failures += 1

    print()
    print(f"{len(CASES) - failures - skipped} ok, {failures} failed, {skipped} skipped")
    return 1 if failures else 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--case":
        try:
            _verify_one(sys.argv[2], (int(sys.argv[3]), int(sys.argv[4])))
        except Exception:
            traceback.print_exc()
            sys.exit(1)
        sys.exit(0)
    sys.exit(main())
