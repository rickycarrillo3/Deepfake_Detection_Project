# models/freqnet/

Wrapper around [chuangchuangtan/FreqNet-DeepfakeDetection](https://github.com/chuangchuangtan/FreqNet-DeepfakeDetection),
a frequency-domain CNN deepfake detector. Ported from the `FreqNet_setup.ipynb` Colab exploration.

## Setup

```
git clone https://github.com/chuangchuangtan/FreqNet-DeepfakeDetection.git models/freqnet/FreqNet-DeepfakeDetection
python -m venv models/freqnet/venv
source models/freqnet/venv/bin/activate
pip install -r models/freqnet/FreqNet-DeepfakeDetection/requirements.txt
pip install -r models/freqnet/requirements.txt
```

Place the pretrained checkpoint at the root of the vendored clone:

```
models/freqnet/FreqNet-DeepfakeDetection/4-classes-freqnet-v2.pth
```

## Model convention

`freqnet(num_classes=1)` outputs a single logit; `model.py` applies `.sigmoid()` once to get a
fake-probability score directly in `[0, 1]` (higher = more likely fake) — matching
`models/README.md` and `results/README.md`. The notebook's preprocessing (`Resize((256, 256))` +
`ToTensor()`, no normalization) is preserved as-is.

Note: a later cell in the notebook re-applied `sigmoid` to the already-sigmoided score under the
mistaken assumption it was a raw logit (producing a `prob_fake` column). That was a bug in the
notebook's post-hoc analysis, not part of the model's real inference path, so it wasn't ported.

## Usage

```python
from model import FreqNetModel

model = FreqNetModel()
model.load()
score = model.predict("/path/to/image.png")  # float in [0, 1], fake-probability
```

Or run a full shard-based evaluation against `data/<dataset_name>/`, writing to
`results/freqnet/<dataset_name>.csv`:

```
python models/freqnet/run.py --dataset flux_faces
```
