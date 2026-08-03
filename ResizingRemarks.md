# Resizing / preprocessing remarks

Notes on how each detector preprocesses an image **before** it reaches the
network, written while investigating why `DMimageDetection` is noticeably
slower than the other models despite being a ResNet.

## TL;DR

The architecture is not the problem — the **input size** is. A ResNet is
"lightweight" only because it is usually fed a small fixed input (e.g.
224×224). The slow detectors here feed the network the image at its **native
resolution**, and for a fully-convolutional net the compute scales linearly
with pixel count. A 1024×1024 image is ~21× the pixels of 224×224, so a model
you would expect to be among the fastest becomes among the slowest.

Five of the nine detectors bound their input to 224–384; three run at native
resolution and pay for it in proportion to image size.

## Bounded input — cost is resolution-independent

| Model | Preprocessing before the net | Effective input |
|---|---|---|
| **aide** | `Resize((256,256))` + normalize (`models/aide/model.py:48`) | 256×256 |
| **freqnet** | `Resize((256,256))` + ToTensor (`models/freqnet/model.py:20`) | 256×256 |
| **grag2021** (standalone, GANimageDetection) | `Resize(256, BILINEAR)` + normalize (`models/grag2021/model.py:23`) | short side 256 |
| **co-spy** | `Resize(loadSize)` → `CenterCrop(cropSize)`; sizes 256–384 (semantic/fusion 384, artifact 256→224) | ≤384 |
| **rine** | `CenterCrop(224)` — **no resize**, crops a 224 window (`models/rine/rine/src/utils.py:41`) | 224×224 |

`rine` is a slightly different case: it does not *downscale*, it **crops** a
224×224 patch out of the native image and discards the rest. Cost is fixed and
tiny, but it only ever sees a center patch. The other four in this group
resample the whole image down.

## Native resolution — cost scales with image size (the slow group)

| Model | Preprocessing | Consequence |
|---|---|---|
| **DMimageDetection** (`dmimagedetection_DM` & `dmimagedetection_GAN`) | `ToTensor` + normalize only, **no resize/crop**, fully-conv ResNet-50 (`model.py:101`, upstream `get_list_norm('resnet')`) | ~21× FLOPs on a 1024² image — the model this investigation started from |
| **cnndetection** | `CROP_SIZE = None` → **no crop by default**, native resolution, also a ResNet (`models/cnndetection/model.py:49`) | **same problem** |
| **aeroblade** | reconstruction-distance through Stable Diffusion autoencoders | heaviest by design; has its own bespoke runner |

## Findings

- **DMimageDetection is not the odd one out.** `cnndetection` has the identical
  latency characteristic: `models/cnndetection/model.py:49` sets
  `CROP_SIZE = None`, so it also runs a ResNet over the full native image. The
  difference is purely a config default — cnndetection *exposes* an optional
  `CenterCrop` (`models/cnndetection/model.py:66`) and leaves it off;
  DMimageDetection has no knob at all. Both feel heavy for the same reason:
  native-resolution input, not the architecture.

- **The no-resize behavior is load-bearing, not an oversight.** These forensic
  detectors key on resampling / compression / high-frequency artifacts, and
  downsampling the image destroys exactly those cues. Upstream deliberately runs
  at native resolution. Adding a `Resize` would speed things up but change what
  the detector sees and degrade accuracy — it would no longer be the published
  method, and cross-model correlation numbers would be measuring something
  different.

- **Crop vs resize is a methodological difference, not just a speed one.**
  `rine` and (optionally) `cnndetection` **crop** — preserving native pixel
  statistics on a sub-region. `aide` / `freqnet` / `grag2021` / `co-spy`
  **resize** — resampling the whole frame. Worth keeping in mind when comparing
  scores across the bounded-input models, not only their runtimes.

## Options for the native-resolution ResNets

Roughly in order of how much to trust them:

1. **Run on GPU.** A native-resolution fully-convolutional forward is exactly
   the workload a GPU handles cheaply and a CPU chokes on. If cluster runs are
   landing on CPU, that alone is most of the gap. Confirm
   `torch.cuda.is_available()` is actually true in the cluster environment.
2. **Accept it and just measure it.** Latency comparability across models is a
   stated non-goal (see `CLAUDE.md`). These models are legitimately heavier per
   image; that is a real property worth recording, not a bug to hide.
3. **Cap the long side** (downscale only images beyond, say, ~1024, or
   tile-and-average). A middle ground the upstream authors discuss, but it moves
   off the exact published preprocessing, so it needs an accuracy check on a
   real shard before the scores can be trusted. If added, `cnndetection`
   deserves the same treatment (there it is just setting `CROP_SIZE`) so the two
   stay comparable.
