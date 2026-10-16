# models/freqnet/

Wrapper around [chuangchuangtan/FreqNet-DeepfakeDetection](https://github.com/chuangchuangtan/FreqNet-DeepfakeDetection),
a frequency-domain CNN deepfake detector.

## Setup

```bash
cd models/freqnet && bash setup.sh
```

Clones `FreqNet-DeepfakeDetection`, builds `venv/`, and installs the upstream repository's
`requirements.txt`, this directory's, and `torch`/`torchvision`. The pretrained checkpoint
(`4-classes-freqnet-v2.pth`) ships inside the cloned repository — there is no separate download.

Pass `PYTHON=` to select an interpreter on a cluster: `PYTHON=python3.12 bash setup.sh`.

## Score convention

`freqnet(num_classes=1)` outputs a single logit; `model.py` applies `.sigmoid()` once to get a
fake-probability directly in `[0, 1]`, higher meaning more likely fake, as
[`results/README.md`](../../results/README.md) specifies.

Preprocessing is `Resize((256, 256))` + `ToTensor()`, with no normalization — upstream's own
transform, unchanged.

## Running

```bash
models/freqnet/venv/bin/python run.py --model freqnet --dataset flux_test
```

FreqNet was trained on GAN-generated images, so its `result_stem()` is `freqnet_GAN` and the run
writes `results/flux_test_freqnet_GAN_predictions.csv`.

Single image, in Python:

```python
from model import FreqNetModel

model = FreqNetModel()
model.load()
score = model.predict("/path/to/image.png")  # float in [0, 1], fake-probability
```
