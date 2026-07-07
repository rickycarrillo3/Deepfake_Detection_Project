# models/grag2021/

Wrapper around [grip-unina/GANimageDetection](https://github.com/grip-unina/GANimageDetection)
(Grag2021), a ResNet50-nodown GAN-image detector. Ported from the `Grag2021_test.ipynb` Colab
exploration.

## Setup

```
git clone https://github.com/grip-unina/GANimageDetection.git models/grag2021/GANimageDetection
python -m venv models/grag2021/venv
source models/grag2021/venv/bin/activate
pip install -r models/grag2021/requirements.txt
```

Download the pretrained checkpoint into the vendored repo's `weights/` folder:

```
wget -O models/grag2021/GANimageDetection/weights/gandetection_resnet50nodown_stylegan2.pth \
  https://www.grip.unina.it/download/prog/GANdetection/weights/gandetection_resnet50nodown_stylegan2.pth
```

## Model convention

The network outputs a single logit per image (no sigmoid head). Per the notebook's convention,
**negative logit means fake-like**. `model.py` converts this to the project's
fake-probability-score convention via `score = sigmoid(-logit)`, so `score` stays a float in
`[0, 1]` where higher = more likely fake, consistent with `models/README.md` and
`results/README.md`.

Images are resized to 256x256 (bilinear) before normalization with ImageNet mean/std — this
matches the notebook's `FORCE_RESIZE=True` fast path, which trades a small amount of accuracy for
much faster batched inference over the original (which runs at native resolution, one image at a
time).

## Usage

```python
from model import Grag2021Model

model = Grag2021Model()
model.load()
score = model.predict("/path/to/image.png")  # float in [0, 1], fake-probability
```

Or run a full shard-based evaluation against `data/<dataset_name>/`, writing to
`results/grag2021/<dataset_name>.csv`:

```
python models/grag2021/run.py --dataset flux_faces
```
