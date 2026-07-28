#!/usr/bin/env python
"""A/B the device AIDE's DCT/noise patch extraction runs on.

Why this exists
---------------
`models/aide/model.py` runs `DCT_base_Rec_Module` on the CPU by default. That
stage is not incidental: it runs `nn.Unfold` over the image
at its **native resolution** (AIDE's 256x256 resize happens *after* it), so a
1024x1024 input means thousands of 32x32 patch matmuls plus six filter passes,
per image, on one core, inside run.py's timed span.

Moving it to the GPU is a pure placement change, but it is not bit-identical: the
stage ranks patches with `torch.sort` on a summed float score
(`AIDE/data/dct.py`), and CPU/GPU results differ in the last bits. On a tie or
near-tie that selects a *different* patch, which changes the score.

This is not hypothetical — it is why the CPU default stands. A run on a real
shard found **2 predicted-label flips in 20 images**, which is why aide keeps the
slower CPU path rather than the speedup. Use this script to re-measure (at
`--limit 100`, not 20) if you want to revisit that with better evidence.

It scores each image twice with the same weights — once with the DCT on CPU
(the reference / old behaviour), once on the main device — and reports the
latency difference alongside the score agreement.

Usage (from the repo root, with aide's venv):

    models/aide/venv/bin/python ab_dct_device.py --dataset flux_faces --limit 20

On a CPU-only box both arms are the same device, so it reports no speedup; that
is expected and the score-agreement check is still meaningful (it becomes a
determinism check).
"""

import argparse
import importlib.util
import os
import statistics
import sys
import time

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from loaders.dataset import iter_dataset, rel_to_dataset  # noqa: E402
from metrics import THRESHOLD  # noqa: E402

AIDE_DIR = os.path.join(REPO_ROOT, "models", "aide")


def load_aide_module():
    """Import models/aide/model.py the same way run.py does."""
    if AIDE_DIR not in sys.path:
        sys.path.insert(0, AIDE_DIR)
    spec = importlib.util.spec_from_file_location(
        "model_aide_ab", os.path.join(AIDE_DIR, "model.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def time_predict(model, image_path):
    """Score one image, returning (score, seconds).

    Mirrors run.py's timing convention: the predict() call alone, nothing else.
    CUDA is synchronized first so the timer does not close before the async
    work it is measuring has actually finished.
    """
    import torch

    if model.device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    score = model.predict(image_path)
    if model.device.type == "cuda":
        torch.cuda.synchronize()
    return score, time.perf_counter() - start


def summarize(name, times):
    total = sum(times)
    return (
        f"  {name:<18} total {total:7.2f}s | "
        f"mean {statistics.mean(times) * 1000:8.1f}ms | "
        f"median {statistics.median(times) * 1000:8.1f}ms"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="dataset dir under data/")
    parser.add_argument("--limit", type=int, default=20, help="images to compare (default 20)")
    parser.add_argument("--checkpoint", default=None, help="aide checkpoint: genimage|progan|sd14")
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="untimed images before measuring, to absorb CUDA/cudnn init (default 1)",
    )
    args = parser.parse_args()

    import torch

    aide = load_aide_module()

    main_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"main device: {main_device}")
    if main_device.type != "cuda":
        print("NOTE: no CUDA available — both arms run on CPU, so the timing")
        print("      comparison is not meaningful here. Run this on the cluster.")

    # One model, one set of weights: the arms differ only in where self.dct sits,
    # so build it once and move that submodule between them. Loading two full
    # AIDEModel instances would double GPU memory for no reason.
    model = aide.AIDEModel(checkpoint=args.checkpoint).load()

    images = []
    for image_path, _label in iter_dataset(args.dataset):
        images.append(image_path)
        if len(images) >= args.limit + args.warmup:
            break

    if not images:
        raise SystemExit(f"No images found for dataset {args.dataset!r} under data/")

    def set_dct_device(device):
        model.dct_device = torch.device(device)
        model.dct.to(model.dct_device)

    for image_path in images[: args.warmup]:
        for device in ("cpu", main_device):
            set_dct_device(device)
            model.predict(image_path)

    measured = images[args.warmup :]
    print(f"comparing {len(measured)} images from {args.dataset}\n")

    cpu_scores, cpu_times = [], []
    gpu_scores, gpu_times = [], []

    for image_path in measured:
        set_dct_device("cpu")
        score, elapsed = time_predict(model, image_path)
        cpu_scores.append(score)
        cpu_times.append(elapsed)

        set_dct_device(main_device)
        score, elapsed = time_predict(model, image_path)
        gpu_scores.append(score)
        gpu_times.append(elapsed)

    print("Latency (predict() only, run.py's timing convention):")
    print(summarize("dct on cpu", cpu_times))
    print(summarize(f"dct on {main_device.type}", gpu_times))
    speedup = sum(cpu_times) / sum(gpu_times) if sum(gpu_times) else float("nan")
    print(f"  speedup: {speedup:.2f}x\n")

    diffs = [abs(a - b) for a, b in zip(cpu_scores, gpu_scores)]
    flips = [
        (path, a, b)
        for path, a, b in zip(measured, cpu_scores, gpu_scores)
        if (a >= THRESHOLD) != (b >= THRESHOLD)
    ]

    print(f"Score agreement (threshold {THRESHOLD}):")
    print(f"  max  |cpu - {main_device.type}|: {max(diffs):.6f}")
    print(f"  mean |cpu - {main_device.type}|: {statistics.mean(diffs):.6f}")
    print(f"  identical to 1e-6: {sum(d < 1e-6 for d in diffs)}/{len(diffs)}")
    print(f"  predicted-label flips: {len(flips)}/{len(diffs)}")

    for path, a, b in flips[:10]:
        rel = rel_to_dataset(path, args.dataset)
        print(f"    FLIP {rel}: cpu={a:.4f} {main_device.type}={b:.4f}")

    print()
    if not flips and max(diffs) < 1e-4:
        print("VERDICT: safe to keep the DCT on the main device — scores match.")
    elif not flips:
        print("VERDICT: no label flips, but scores drift. Fine for accuracy,")
        print("         but do NOT mix the two regimes within one results CSV.")
    else:
        print("VERDICT: label flips present. The patch ranking is device-sensitive")
        print("         on this data — rerun affected CSVs rather than mixing, or")
        print("         pin --model-arg dct_device=cpu for consistency with old runs.")


if __name__ == "__main__":
    main()
