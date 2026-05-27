# Reproducing the paper results

This walks through the full SmolVLA chain, from generalist pretraining and
task-specific top-up to per-group MSE evaluation and the final figure. The
π₀.₅ side reuses the workshop baseline checkpoint and runs a
fine-tuning through OpenPI. The launcher is included for completeness.

All paths below are placeholders. Substitute the ones that fit your machine.

## 1. Generalist pretraining (SmolVLA on AIRoA MoMa)

```bash
python training/train_smolvla_generalist.py \
    --dataset-root /path/to/airoa-moma \
    --output-dir outputs/smolvla_generalist \
    --steps 20000 --batch-size 32
```

Wall-clock on a single RTX 3090 with AMP and batch 32 is roughly 12 hours
for 20k steps. Mid-range checkpoints are saved every 5k steps. We report
results on the 20k checkpoint.

## 2. Task-specific top-up

```bash
python training/train_smolvla_task.py \
    --generalist outputs/smolvla_generalist/checkpoints/020000/pretrained_model \
    --dataset-root /path/to/task6911-v30 \
    --output-dir outputs/smolvla_topup \
    --steps 5000 --batch-size 8
```

This produces the checkpoints used in the per-group MSE sweep. The
checkpoint that wins the offline comparison in our setup is the one around
3k to 4k steps. Later checkpoints overfit the gripper signal and total MSE
starts climbing again.

## 3. Per-group MSE evaluation

For a single checkpoint.

```bash
python eval/per_group_mse.py \
    --checkpoint outputs/smolvla_topup/checkpoints/003000/pretrained_model \
    --dataset-root /path/to/task6911-v30 \
    --n-samples 240
```

To compare several checkpoints in one run, list them in a JSON file.

```bash
cat > checkpoints.json <<'EOF'
{
  "PT 20k":  "outputs/smolvla_generalist/checkpoints/020000/pretrained_model",
  "B8 1k":   "outputs/smolvla_topup/checkpoints/001000/pretrained_model",
  "B8 3k":   "outputs/smolvla_topup/checkpoints/003000/pretrained_model",
  "B8 5k":   "outputs/smolvla_topup/checkpoints/005000/pretrained_model"
}
EOF

python eval/per_group_mse.py \
    --checkpoints-json checkpoints.json \
    --dataset-root /path/to/task6911-v30 \
    --output results.json
```

The script prints a sorted comparison table (Total / Arm / Grip / Head / Base)
and saves the numbers to JSON.

## 4. Figure

The figure from the poster (per-group curves and total MSE bars) is
generated from a hard-coded table of results so it is reproducible without
re-running the sweep.

```bash
python figures/mse_curves.py --out figures/mse_curves.pdf
```

If you have re-run the sweep and want the figure from your own numbers, edit
the `CHECKPOINTS` dict at the top of `figures/mse_curves.py`.

## 5. π₀.₅ fine-tuning

```bash
python training/train_pi05_finetune.py \
    --openpi-root /path/to/openpi \
    --config <pi05_hsr_config_name> \
    --exp-name pi05_finetune \
    --steps <N>
```

The exact config name and the number of training steps used for the
competition submission are covered by the AIRoA NDA and are not
redistributed here. See `docs/CONTEXT.md` for the redistribution
policy and the parts that we do include in the repo (results, robot
scores, LoRA ablation).

OpenPI binds the base checkpoint to the chosen config, not to a CLI flag.
The config we use already wires up the released 80k baseline through its
`weight_loader`. If you want to fine-tune from a different checkpoint,
pass `--weight-loader-dir /path/to/checkpoint` to the launcher and it will
be forwarded as `--weight_loader.checkpoint_dir=...` to OpenPI.

## Changing parameters and seeing what is available

The SmolVLA launchers in `training/` only forward a handful of the flags
that `lerobot-train` accepts. If you want to change something we do not
expose (learning rate, optimizer, scheduler, chunk size, weight decay, log
backend, image-augmentation policy, etc.) you have two options. Either
add the flag to the `cmd` list in the launcher you are using, or run
`lerobot-train` directly with the resolved command that the launcher
prints in `--dry-run` mode.

### SmolVLA (LeRobot)

The canonical list of options is the `TrainPipelineConfig` dataclass at
[`src/lerobot/configs/train.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/configs/train.py).
Every field there can be set on the CLI with the `--field=value` syntax
used by the launchers. The CLI also exposes per-group help.

```bash
lerobot-train --help                # all top-level groups (policy, dataset, optimizer, ...)
lerobot-train --policy.help         # policy-specific options (including SmolVLA flags)
lerobot-train --optimizer.help      # optimizer-specific options
lerobot-train --dataset.help        # dataset / loading options
```

Our launchers also accept `--dry-run`, which prints the resolved
`lerobot-train` command without running it. Useful when you want to copy
the line and tweak a single flag by hand.

```bash
python training/train_smolvla_task.py \
    --generalist .../020000/pretrained_model \
    --dataset-root /path/to/task6911-v30 \
    --output-dir outputs/test \
    --dry-run
```

For broader documentation, the LeRobot
[user docs](https://huggingface.co/docs/lerobot) cover dataset preparation
and the SmolVLA-specific options in depth.

### π₀.₅ (openpi)

The discovery surface is OpenPI's `scripts/train.py`, which uses
[`tyro`](https://github.com/brentyi/tyro) under the hood. The config name
is **positional** (not a flag) and every field of the resolved
`TrainConfig` dataclass can be overridden on the CLI.

```bash
python /path/to/openpi/scripts/train.py --help          # global flags + config list
python /path/to/openpi/scripts/train.py <name> --help   # fields of a specific config
```

The list of registered config names lives in
[`src/openpi/training/config.py`](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py)
(look for `_CONFIGS = [...]`). Each entry is a `TrainConfig` dataclass
with the `weight_loader`, the data mix, the LR schedule and the freezing
strategy already wired up. Override any field at the CLI with
`--field=value`, for example `--num_train_steps=5000` or
`--lr_schedule.peak_lr=1e-4`. To pretty-print a resolved config without
running training, append `--help` to the second invocation above.

### Where to learn what each parameter does

When you find a flag you do not understand, the fastest path is to
open the dataclass at the source link above and look for the field's
docstring or inline comment. They are usually one short line and
informative. If that is not enough, the
[LeRobot user docs](https://huggingface.co/docs/lerobot) and the
[openpi README](https://github.com/Physical-Intelligence/openpi)
cover most options in longer prose. For VLA-specific terms (action
chunk size, freezing the backbone, image augmentation policy,
normalisation stats) `docs/BACKGROUND.md` in this repo gives the short
version and points at the relevant paper section.

## A note on numerical reproducibility

The exact MSE numbers depend on a few things.

- Random seed (we use 1000 for training, 42 for the eval split).
- Number of evaluation samples (240 in the paper).
- Dataset version. AIRoA MoMa and the ICRA dataset have evolved, so the
  generalist results in the paper are on the snapshot at the time of
  submission. The relative ranking between checkpoints, which is the point
  of the paper, is robust to these.
