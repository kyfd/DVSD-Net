# DVSD-Net

PyTorch implementation for DVSD-Net, an underwater fish counting framework built around
dual-view shared-density supervision.

The project targets dense fish counting in degraded underwater imagery. It generates
paired pseudo-views from each annotated image, learns the full visible density map, and
uses an auxiliary shared-visible branch to encourage cross-view consistent responses.

## Highlights

- Dual-view training with shared-visible fish supervision.
- VGG16-FPN backbone used by the paper model.
- DCFA cross-view feature exchange for the shared branch.
- FAREM foreground-aware Retinex-style enhancement.
- DGRM deformable geometric refinement.
- Minimal training and evaluation code for paper reproduction.

## Repository Layout

```text
DVSD-Net/
  train.py                  # main training entry
  config.py                 # default training and ablation configuration
  datasets/                 # DeepFish paired pseudo-view data loader
  model/                    # DVSD-Net model modules
  misc/                     # losses, Gaussian density generation, utilities
  tools/eval_metrics.py     # count and GAME evaluation
```

Large files such as datasets, checkpoints, generated figures, paper exports, and archives
are intentionally excluded from Git.

## Environment

Python 3.11 is recommended. Install PyTorch according to your CUDA version first, then
install the remaining dependencies.

```bash
conda create -n dvsd-net python=3.11
conda activate dvsd-net

# Example only. Pick the PyTorch/CUDA build that matches your machine.
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

or use the provided Conda environment:

```bash
conda env create -f environment.yml
conda activate dvsd-net
```

## Dataset Format

Download DeepFish from the official project resources:

- Project page: https://alzayats.github.io/DeepFish/
- Code and split reader: https://github.com/alzayats/DeepFish
- Dataset archive: http://data.qld.edu.au/public/Q5842/2020-AlzayatSaleh-00e364223a600e83bd9c3f5bcd91045-DeepFish/
- Paper DOI: https://doi.org/10.1038/s41598-020-71639-x

For counting/localization experiments, use the official DeepFish `loc` split. In
the original DeepFish code this task is named `fish_loc` and reads `train.csv`,
`val.csv`, and `test.csv`; each CSV provides the image `ID`, fish `counts`, and
binary `labels`. Do not create a new random split when reproducing paper numbers.

This repository's loader expects the official LOC split to be arranged as
separate `images/` and `masks/` folders:

```text
DeepFish_LOC/
  train/
    images/
    masks/
  val/
    images/
    masks/
  test/
    images/
    masks/
```

Use the image IDs from the official `train.csv`, `val.csv`, and `test.csv` to
copy or symlink the corresponding `.jpg` images and `.png` point masks into the
three folders above. For example, an entry with `ID=abc123` should become
`images/abc123.jpg` and `masks/abc123.png` in the same split directory.

Mask files are converted to connected-component centroids and used as point-level
supervision. Image and mask stems should match, for example:

```text
images/0001.jpg
masks/0001.png
```

The code also accepts mask names such as `0001_mask.png`.

## Training

Pass dataset paths from the command line, or set `DEEPFISH_TRAIN_ROOT`,
`DEEPFISH_VAL_ROOT`, and `DEEPFISH_TEST_ROOT`.

Minimal single-GPU reproduction command:

```bash
python train.py \
  --profile full \
  --train-root /path/to/DeepFish_LOC/train \
  --val-root /path/to/DeepFish_LOC/val \
  --seed 3035 \
  --epochs 100 \
  --name dvsd_full_loc
```

For multi-GPU training:

```bash
torchrun --master_port 29515 --nproc_per_node=4 train.py \
  --profile full \
  --train-root /path/to/DeepFish_LOC/train \
  --val-root /path/to/DeepFish_LOC/val \
  --seed 3035 \
  --epochs 100 \
  --name dvsd_full_loc
```

Useful ablation profiles:

```bash
python train.py --profile global_only
python train.py --profile dcfa_baseline
python train.py --profile full
```

Individual switches can also be overridden, for example:

```bash
python train.py --use-dcfa 1 --use-dcn 1 --use-enhance 1
```

## Evaluation

Evaluate a checkpoint on the test split:

```bash
python tools/eval_metrics.py \
  --test-root /path/to/DeepFish_LOC/test \
  --model-path exp/DeepFish/<run_name>/output/best_model.pth \
  --profile full \
  --save-json results/dvsd_full_loc_metrics.json
```

The evaluation script reports count and localization-oriented metrics including MAE,
RMSE, and GAME.

For exact table reproduction, report the JSON file produced by the command above
together with the checkpoint path, the commit hash, and whether the run used
`--backbone-pretrained 1` or a `--pretrain-path` checkpoint.

## Reproducibility Notes

- The default `full` profile is the paper-model path: VGG16-FPN, DCFA, DGRM,
  FAREM, and shared-visible supervision enabled, with no inflow/outflow branch.
- This repository intentionally contains no measured result tables, generated paper
  figures, checkpoints, or datasets. Regenerate reported numbers from the released
  checkpoint and the stated test split with `tools/eval_metrics.py`.
- The VGG16-FPN backbone is randomly initialized by default. If a reported experiment
  uses ImageNet initialization, run with `--backbone-pretrained 1` and state that in
  the paper. If it uses a previous counting checkpoint, pass it with `--pretrain-path`
  and disclose the checkpoint source.
- Keep the train/val/test split fixed. Use validation only for model selection and
  report final numbers from the held-out test split.

## Checkpoints and Data

This repository does not include trained weights or datasets. Place checkpoints under
your own `exp/` directory or pass their paths explicitly with `--model-path` or
`--pretrain-path`.

## Citation

The manuscript is under preparation. Please cite the paper when the final citation is
available.
