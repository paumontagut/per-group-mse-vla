# Setup

Two environments are needed, one per model. They share little in practice and
are easiest to keep separate.

## SmolVLA (LeRobot)

We developed against LeRobot v0.5.1, which requires Python 3.12.

```bash
conda create -n lerobot python=3.12
conda activate lerobot

git clone https://github.com/huggingface/lerobot.git
cd lerobot
git checkout v0.5.1   # or any later release that still exposes lerobot-train
pip install -e ".[smolvla]"
```

The `[smolvla]` extra pulls everything the SmolVLA policy needs (transformers,
safetensors, accelerate, num2words, av) with version pins that already work
together, so there is no need to add packages by hand.

A quick sanity check.

```bash
lerobot-info
lerobot-train --help | head -40
python -c "from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy"
```

If the third command exits without an error message the SmolVLA policy
class is importable.

`lerobot-info` prints the detected GPUs, the LeRobot version and the
installed extras. The `--help` of `lerobot-train` lists every available flag
(it is generated from the underlying dataclasses), useful when you want to
change something we do not document explicitly.

## π₀.₅ (OpenPI)

Follow the [openpi](https://github.com/Physical-Intelligence/openpi) setup
instructions. The training launcher (`training/train_pi05_finetune.py`)
needs `--openpi-root` pointing at that checkout.

The π₀.₅ baseline checkpoint distributed by the AIRoA workshop is the
starting point for the fine-tuning.

## GPU

A single 24 GB GPU is enough for everything in this repo. We used an RTX 3090.
Lower VRAM is possible with mixed precision (`--policy.use_amp=true` is on by
default) and smaller batch sizes. See `docs/CONTEXT.md` for the details.

## Datasets

This repo does not redistribute datasets. Pull them yourself from the official
sources.

- **AIRoA MoMa** (generalist pretraining). Public mobile-manipulation dataset
  from the AIRoA project.
- **AIRoA ICRA 2026** (full competition dataset). Released for the ICRA 2026
  workshop "From Data to Decisions". The top-up scripts use the private
  relocation subset that AIRoA distributed to the competing teams.

Both should be in LeRobot v3.0 format before training. If you have a v2.1
snapshot, the LeRobot project ships conversion utilities.
