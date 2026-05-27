---
license: mit
library_name: lerobot
base_model: lerobot/smolvla_base
pipeline_tag: robotics
tags:
- robotics
- vision-language-action
- mobile-manipulation
- toyota-hsr
- smolvla
- icra2026
language:
- en
---

# HSR-SmolVLA (B32, 40k steps)

This is the SmolVLA checkpoint reported as **HSR-SmolVLA (40k)** in the
paper *Per-Group Error, Not Total MSE: Fine-Tuning Vision-Language-Action
Models for 11-DoF Mobile Manipulation*, presented at the ICRA 2026 workshop
"From Data to Decisions" (1st place, 36 teams).

It is fine-tuned in two phases on top of the public
[`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base). A
generalist phase on AIRoA MoMa (Toyota HSR teleoperation episodes)
followed by a task-specific top-up on a private subset of the AIRoA
ICRA 2026 dataset, distributed only to the competing teams during the
workshop. Total parameters, 450 M.

The Toyota HSR has 11 degrees of freedom split across four functionally
distinct joint groups (5 arm, 1 gripper, 2 head, 3 holonomic base). The
companion paper argues that picking the best fine-tuned checkpoint by the
aggregate MSE hides which group is actually responsible for the error, and
introduces **per-group MSE** as a better offline selection signal. This
checkpoint is the one that wins the per-group MSE sweep in our setup.

## How to load

```python
from huggingface_hub import snapshot_download
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

ckpt_dir = snapshot_download("PauMontagut/per-group-mse-smolvla")
policy = SmolVLAPolicy.from_pretrained(ckpt_dir).eval().cuda()
```

The policy expects observations in the Toyota HSR layout (two cameras,
head and hand, plus an 8-D state vector and a natural-language
instruction) and emits action chunks of shape `(T, 11)` in the order
`arm_lift, arm_flex, arm_roll, wrist_flex, wrist_roll, gripper,
head_pan, head_tilt, base_x, base_y, base_theta`.

A WebSocket server skeleton, the I/O contract, the per-group MSE
evaluation script and the raw 60-trial robot scores are all in the
[companion GitHub repository](https://github.com/paumontagut/per-group-mse-vla).
Robot demo videos, the animated per-group analysis and the full story
of the paper are on the
[project page](https://paumontagut.github.io/per-group-mse-vla/).

## Results

Per-group MSE on a held-out `task6911` split, values multiplied by 10³
for readability.

| Group   | MSE (×10⁻³) |
|---------|-------------|
| Arm     | 0.88        |
| Gripper | 3.77        |
| Head    | 0.01        |
| Base    | 3.18        |
| Total   | 1.61        |

Real-robot evaluation on the Toyota HSR, 20 trials over two objects
(ceramic mug and Cheez-It box), 4-point ordinal rubric. Mean score
3.50 / 4. The companion paper compares it against the π₀.₅ baseline
(80k steps, 4.00 / 4) and our own π₀.₅ fine-tune (3.75 / 4). The
π₀.₅ checkpoints are not redistributed because they derive from the
private `baseline_80k` released by the AIRoA workshop under the
competition NDA.

## Training context

The checkpoint was produced on a single RTX 3090 (24 GB), shared with
other lab work, which is why the top-up uses batch 8 in some runs and
batch 32 in others. The release is the batch 32, 40k step version
(referred to as `B32 40k` in `figures/mse_curves.png` of the GitHub
repo), which is the best total-MSE checkpoint on the curve.

The release was sanitised with `scripts/prepare_hf_release.py` to
remove lab-local paths from `train_config.json`. Weights and
normalisation statistics were not modified.

## What this model is, and what it is not

This is **not** the model that won the workshop. The competition
submission was a π₀.₅ fine-tune that stays behind the AIRoA NDA (see
the note in the GitHub repository's README). This SmolVLA checkpoint
is one of the intermediate models we tried during the competition
rounds, a smaller and cheaper option that we trained and evaluated in
parallel with the π₀.₅ track. It is an experimental model that
worked, that runs on a single 24 GB GPU, and that ranks the joint
groups in the same direction as the larger models, which is the
finding the paper is about.

In practice, this means the checkpoint is a useful starting point if
you want to extend the fine-tuning on a related dataset, run the
per-group MSE analysis on a different robot, or just have a working
SmolVLA on the Toyota HSR's 11-DoF layout to build from. It is not
intended as a production policy, and it has not been stressed on
scenes outside the fine-tuning distribution.

## Limitations

- Evaluated only on the Toyota HSR with `task6911` relocation tasks.
  Behaviour on a different robot, a different task family or a
  substantially different scene is not characterised.
- Real-robot results are 20 trials per object pair (60 trials total,
  20 per model in the paper). The arm-MSE versus total-MSE ranking
  claim is robust on this sample but the absolute scores will vary
  with operator, lighting and object pose.
- The model has no built-in stopping condition. It always returns an
  action chunk. Integration code is expected to time the trial and
  decide when to stop.
- Per-group MSE is a better offline signal than total MSE for this
  setup, but it is still offline. Online behaviour can diverge.

## Citation

The arXiv preprint is at
[arxiv.org/abs/2606.00253](https://arxiv.org/abs/2606.00253).

```bibtex
@inproceedings{montagut2026pergroup,
  title  = {Per-Group Error, Not Total {MSE}: Fine-Tuning Vision-Language-Action Models for 11-{DoF} Mobile Manipulation},
  author = {Montagut Bofi, Pau and Garc\'ia Blasco, Mario and Pulli, Tessa and Vincze, Markus},
  booktitle = {ICRA 2026 Workshop on From Data to Decisions},
  year   = {2026},
  eprint = {2606.00253},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url    = {https://arxiv.org/abs/2606.00253}
}
```

If you also use the SmolVLA backbone, please cite Shukor *et al.*,
*SmolVLA, a vision-language-action model for affordable and efficient
robotics*, [arXiv:2506.01844](https://arxiv.org/abs/2506.01844).

## License

MIT. See the
[GitHub repository](https://github.com/paumontagut/per-group-mse-vla)
for the full source, the inference server skeleton, and the
reproducibility instructions, and the
[project page](https://paumontagut.github.io/per-group-mse-vla/) for
the demos and the interactive summary of the results.
