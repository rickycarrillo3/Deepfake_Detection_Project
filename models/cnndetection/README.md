# models/cnndetection/

Pretrained ResNet-50 detector from
[PeterWang512/CNNDetection](https://github.com/PeterWang512/CNNDetection)
(CVPR 2020 — "CNN-generated images are surprisingly easy to spot...for now").
Uses the blur+JPEG-augmented weights (`blur_jpg_prob0.5.pth`); the checkpoint
outputs a single logit that `sigmoid()` turns into a per-image fake-probability.

```
cnndetection/
├── CNNDetection/          # vendored clone (networks/, demo.py, weights/)
│   └── weights/blur_jpg_prob0.5.pth
├── venv/      # isolated venv
├── setup.sh               # clones the repo, builds the venv, downloads weights
├── requirements.txt
├── model.py               # load()/predict() contract
└── predict_one.py         # single-image CLI
```

See `model.py`'s docstring for architecture/preprocessing details (ResNet-50,
no resize/crop by default, ImageNet normalization, sigmoid → probability) and
how to switch to the `blur_jpg_prob0.1.pth` weight variant.

## Setup

```bash
cd models/cnndetection
bash setup.sh
```

`setup.sh` skips the clone / venv / download steps if they already exist, so
it's safe to re-run.

## Running

Run from the repo root, using this model's venv:

```bash
models/cnndetection/venv/bin/python run.py --model cnndetection --dataset <dataset_name>
```

CNNDetection was trained on GAN-generated images, so its `result_stem()` is `cnndetection_GAN`
and the run writes `results/<dataset_name>_cnndetection_GAN_predictions.csv` — schema in
[`results/README.md`](../../results/README.md).

Because `CROP_SIZE = None` by default, this ResNet-50 runs over the image at **native
resolution**, so its cost scales with pixel count — it and `DMimageDetection` are the slow ones on
large inputs. See [`ResizingRemarks.md`](../../ResizingRemarks.md).

Single image from the terminal:
```bash
cd models/cnndetection
venv/bin/python predict_one.py /path/to/image.png
```
