# models/

Holds one subdirectory per pretrained deepfake detector under test.

## Layout

```
models/<model_name>/
├── requirements.txt      # or environment.yml — this model's own dependencies
├── venv/ (or .venv/)     # isolated environment, git-ignored
├── weights/              # model checkpoint(s), git-ignored
└── ...                   # vendored/cloned copy of the model's GitHub repo
```

## Per-model isolation

Each model directory manages its **own environment**. Pretrained detectors pulled from different
GitHub repos commonly pin conflicting torch/tf versions, so there is intentionally no shared
top-level `requirements.txt`.

## Wrapper interface (convention)

Once implemented, each model directory is expected to expose a uniform interface for the
loader/runner to call:

- `load()` — load weights and prepare the model for inference
- `predict(image_path) -> score` — return a float score in `[0, 1]` for a single image

This contract is documented here for consistency; implementation lands with each model.
