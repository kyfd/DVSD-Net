# DVSD-Net

PyTorch code for DVSD-Net: underwater fish counting with dual-view shared-density supervision.

Each annotated image is turned into a pair of pseudo-views. The network learns the full visible density map, plus an auxiliary shared-visible branch so the two views stay consistent.

This repo is the training and eval code for the paper. Datasets, checkpoints, and generated figures are **not included**. The diagram below is the `full` profile as implemented in `model/VIC.py`, not a rendered experiment figure.

```text
          paired pseudo-views
                    |
                    v
         UPhy-inspired enhancer
                    |
                    v
              VGG16-FPN
                    |
                    v
         DCFA cross-attention
           (3 stages x 2 blocks)
                    |
                    v
            feature fusion
                    |
                    v
        deformable refine (DGRM)
                    |
           +--------+--------+
           |                 |
           v                 v
   global density      shared-visible
      decoder             decoder
```

What a reviewer should look at after training (not shipped here):

1. Input underwater image
2. Ground-truth density map from point annotations
3. Predicted density map
4. A typical success case and a failure case (heavy occlusion, turbidity, or missed small fish)

Do not treat this README as a results table. Report test MAE / RMSE / GAME from `tools/eval_metrics.py` on the official DeepFish loc split.

## Layout

```text
train.py                training
config.py               defaults and ablation profiles
datasets/               DeepFish paired-view loader
model/                  VGG16-FPN, DCFA, DGRM, FAREM
tools/eval_metrics.py   MAE / RMSE / GAME
```

## Setup

Python 3.11. Install a PyTorch build that matches your CUDA, then:

```bash
pip install -r requirements.txt
```

or:

```bash
conda env create -f environment.yml
conda activate dvsd-net
```

## Data

Use the official DeepFish **loc** split (`fish_loc` in the original code). Do not reshuffle train/val/test.

- Project: https://alzayats.github.io/DeepFish/
- Code: https://github.com/alzayats/DeepFish
- Data: http://data.qld.edu.au/public/Q5842/2020-AlzayatSaleh-00e364223a600e83bd9c3f5bcd91045-DeepFish/
- Paper: https://doi.org/10.1038/s41598-020-71639-x

Arrange the official IDs like this:

```text
DeepFish_LOC/
  train/{images,masks}/
  val/{images,masks}/
  test/{images,masks}/
```

`images/abc123.jpg` pairs with `masks/abc123.png` (or `abc123_mask.png`). Masks are turned into connected-component centroids for point supervision.

## Train

```bash
python train.py \
  --profile full \
  --train-root /path/to/DeepFish_LOC/train \
  --val-root /path/to/DeepFish_LOC/val \
  --seed 3035 \
  --epochs 100 \
  --name dvsd_full_loc
```

Multi-GPU:

```bash
torchrun --master_port 29515 --nproc_per_node=4 train.py \
  --profile full \
  --train-root /path/to/DeepFish_LOC/train \
  --val-root /path/to/DeepFish_LOC/val \
  --seed 3035 \
  --epochs 100 \
  --name dvsd_full_loc
```

Profiles: `global_only`, `dcfa_baseline`, `full`. Paths can also come from `DEEPFISH_TRAIN_ROOT` / `DEEPFISH_VAL_ROOT` / `DEEPFISH_TEST_ROOT`.

`full` is the paper setting (VGG16-FPN + DCFA + DGRM + FAREM + shared-visible). The backbone starts random; pass `--backbone-pretrained 1` or `--pretrain-path` if a reported run used those.

## Eval

```bash
python tools/eval_metrics.py \
  --test-root /path/to/DeepFish_LOC/test \
  --model-path exp/DeepFish/<run_name>/output/best_model.pth \
  --profile full \
  --save-json results/dvsd_full_loc_metrics.json
```

Pick the model on val, report test numbers. Put your own checkpoints under `exp/` or pass `--model-path`.

## Citation

Manuscript in preparation.
