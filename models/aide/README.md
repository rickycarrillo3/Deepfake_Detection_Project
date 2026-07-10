# models/aide/

Wrapper around [`shilinyan99/AIDE`](https://github.com/shilinyan99/AIDE), ported from the uploaded
`aidecolab.pdf` Colab workflow.

This implementation follows the repository contract:

- `load()` loads the model/checkpoint and prepares inference.
- `predict(image_path) -> score` returns a fake-probability score in `[0, 1]`.
- `run.py` writes `results/aide/<dataset_name>.csv` with exactly:

```csv
filepath,score,label
```

For fake-only datasets, the label should be `1`. The shared loader should already assign `1` to
datasets whose folder name does not contain `ffhq`; you can also force this with `--force-label 1`.

## Setup

From the repository root:

```bash
mkdir -p models/aide
git clone https://github.com/shilinyan99/AIDE.git models/aide/AIDE

python -m venv models/aide/venv
source models/aide/venv/bin/activate

# Install torch/torchvision separately for your CUDA version first.
# Example:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r models/aide/requirements.txt
```

## Checkpoint

By default, `model.py` looks for:

```text
models/aide/AIDE/checkpoints/GenImage_train.pth
```

Create the checkpoint folder:

```bash
mkdir -p models/aide/AIDE/checkpoints
```

Then download/place an AIDE detector checkpoint there. The uploaded Colab used `GenImage_train.pth`
from the AIDE model-zoo folder. If you save it somewhere else, set:

```bash
export AIDE_WEIGHTS_PATH=/absolute/path/to/your_checkpoint.pth
```

or pass it directly:

```bash
python models/aide/run.py --dataset flux_faces --checkpoint /absolute/path/to/your_checkpoint.pth
```

## Model convention

AIDE outputs two logits:

```text
class 0 = real-like
class 1 = fake / AI-generated-like
```

This wrapper returns:

```python
score = softmax([logit_real, logit_fake])[1]
```

so higher `score` means "more fake-like."

## Usage

Quick single-image test:

```python
from model import AIDEModel

model = AIDEModel()
model.load()
score = model.predict("/path/to/image.png")
print(score)
```

Full dataset run:

```bash
python models/aide/run.py --dataset flux_faces --force-label 1
```

Quick smoke test:

```bash
python models/aide/run.py --dataset flux_faces --force-label 1 --limit 10
```

The main output CSV will be:

```text
results/aide/flux_faces.csv
```

with only these columns:

```text
filepath,score,label
```

The runner also writes a timing summary:

```text
results/aide/flux_faces_summary.csv
```

with run metadata such as number of images, total seconds, and average seconds per image.

## Notes

AIDE preprocessing is heavier than many other detectors because each image is converted into five
input tensors:

1. DCT/noise low-rank patch
2. DCT/noise high-rank patch
3. second low-rank patch
4. second high-rank patch
5. resized/normalized image tensor

This matches the uploaded Colab workflow and is why AIDE may run slower than simpler resize-only
models.
