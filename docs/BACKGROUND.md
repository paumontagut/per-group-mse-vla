# Background

This document is for the reader who has not worked with Vision-Language-Action
(VLA) models before. The paper and the rest of the docs assume some of these
concepts. Here we build them up from the ground.

The page walks up from what a VLA is, through how we fine-tune one and
what an action chunk is, into per-group MSE and the gap between offline
and on-robot evaluation. The last section is a glossary of the terms
that appear throughout the repo.

Each section ends with one or two pointers to the canonical paper or
documentation, in case you want to go deeper.

---

## 1. What is a Vision-Language-Action model?

A VLA is a neural network that maps **what the robot sees plus a natural
language instruction** to **the next motions the robot should make**.
Inputs are typically RGB images from the robot's cameras, an internal state
vector with joint positions, and a sentence like *"pick up the mug"*.
The output is one or more actions in joint or end-effector space.

Three things make VLAs different from classical robot policies.

1. **They start from a pretrained vision-language backbone.** The visual
   and linguistic understanding does not come from the robot data, it
   comes from a model trained on billions of internet images and texts.
   The robot data only adapts the last stage that produces motions.
2. **The language input is generic.** You can give them new instructions
   they have never seen as long as the words make sense. A policy
   trained to pick up "the red cup" will often respond to "the
   crimson mug" because the language understanding sits on top of a
   model that knows synonyms.
3. **They are trained on teleoperation, not on demonstrations of a
   single task.** The dataset is one large pool of episodes across many
   tasks. The model learns the joint distribution of "what the world
   looks like, what was said, what the human did" rather than a fixed
   skill.

In this work the VLA backbone is **SmolVLA** (450 million parameters,
based on SmolVLM-2) or **π₀.₅** (3.3 billion parameters, the Physical
Intelligence release).

**Learn more.** For SmolVLA see [the paper on
arXiv](https://arxiv.org/abs/2506.01844). For π₀.₅ see Black *et al.*,
*π₀.₅: a Vision-Language-Action Model with Open-World Generalization*,
[arXiv:2504.16054](https://arxiv.org/abs/2504.16054). For a precursor
that introduced the modern VLA recipe, see
[RT-2](https://arxiv.org/abs/2307.15818).

---

## 2. Fine-tuning, and why two phases

A VLA released by a research group is usually trained on a generic
mixture of robots (Franka arms, mobile manipulators, simulated
environments). If you want it to work on **your** robot, the question is
how much you need to retrain.

We use a two-phase approach.

### 2a. Generalist pretraining

We start from the released SmolVLA checkpoint and continue training it
on **AIRoA MoMa**, a public dataset of Toyota HSR teleoperation across
many tasks. The point of this phase is not to learn any particular
task. It is to teach the model what an HSR observation looks like
(which cameras, which state vector layout) and what its 11-dimensional
action space means. We call the output of this phase the
**generalist checkpoint**.

### 2b. Task-specific top-up

Then we continue training the generalist checkpoint for a few thousand
more steps on a much smaller, task-focused dataset (we used the private
relocation subset of the AIRoA ICRA 2026 dataset distributed to the
competing teams). This phase **specialises** the model to the actual
tasks we are going to evaluate on. We call this the **top-up phase**
and the output a **task checkpoint**.

For π₀.₅ we use a single phase called **expert-only fine-tuning**.
Instead of training everything, we freeze the vision-language backbone
and only update the last stage (the "action expert" head). This is a
common trick to fine-tune large VLAs on a single GPU.

### Why two phases at all

Training only on the small task dataset overfits quickly and forgets
the general-purpose behaviour learned during pretraining. Training only
on the large dataset never specialises. The two-phase recipe is the
robotics equivalent of "pretrain then fine-tune" in NLP, with the same
trade-off between coverage and specialisation.

**Learn more.** The general "pretrain then fine-tune" framing in
robotics is covered in
[Octo](https://arxiv.org/abs/2405.12213) (Octo Team, 2024). For LoRA
and expert-only style adapters in VLAs, see Section 3 of the SmolVLA
paper.

---

## 3. Action chunks

A naive policy outputs one action per step. The robot executes it,
sends a new observation back, the policy outputs the next action, and
so on. This is fine for fast classical controllers but it is a problem
for large VLAs because each forward pass takes around 150 ms and the
robot would spend most of its time waiting for the model.

A **chunked policy** outputs a short trajectory in one shot, for
example 50 future actions. The robot then ingests this chunk, plays a
prefix of it, and only asks for the next chunk when the buffer is
running out. This way the robot is moving continuously while the model
is only invoked every second or so.

The action chunk is the unit the robot consumes and the unit the
offline metric scores. When we compute MSE between the policy and the
ground-truth, we compare the predicted chunk against the human-recorded
chunk of the same length.

**Learn more.** Chunked policies appeared in
[Action Chunking Transformer (ACT)](https://arxiv.org/abs/2304.13705)
(Zhao *et al.*, 2023). Most modern VLAs ship with chunked outputs by
default.

---

## 4. Per-group MSE, the contribution of this paper

The standard way to evaluate a fine-tuned VLA offline is to compute
**Mean Squared Error** between predicted and ground-truth actions on
held-out episodes, average it over all action dimensions, and call that
single number the model's score. Lower is better. You pick the
checkpoint that minimises it and ship it to the robot.

The problem is that the HSR action vector lives in 11 dimensions split
across **four functionally different joint groups**, namely a 5-DoF
arm, a 1-DoF gripper, a 2-DoF head and a 3-DoF holonomic base. These
groups have very different scales and dynamics.

| Group   | Dimensions | Behaviour                          | Typical MSE share |
|---------|------------|------------------------------------|-------------------|
| Arm     | 5          | Continuous, multi-joint manipulator | Medium            |
| Gripper | 1          | Mostly binary (open or close)       | Spiky, high       |
| Head    | 2          | Small range, slow                   | Very low          |
| Base    | 3          | Continuous, holonomic               | High late in training |

Averaging across all 11 dimensions blends signals that should not be
blended. A checkpoint whose total MSE is small can have, for example, a
very well-modelled gripper that pulls the average down and an arm that
has actually degraded. The robot only cares about whether the arm is
right.

The contribution of the paper is to **decompose** the MSE into the four
groups and use the decomposition as the checkpoint-selection signal.
In our 60-trial real-robot evaluation, the **arm** MSE ranked the
models in the same order as the robot did. The **total** MSE did not.

You can see exactly this effect in `docs/img/mse_curves.png` and in
the `Results at a glance` table of the README.

---

## 5. Offline vs online evaluation

"Offline" evaluation means computing a number on a held-out slice of
the training dataset. It is cheap, deterministic and lets you compare
many checkpoints in minutes. The metric here is the (per-group) MSE on
the recorded human actions.

"Online" evaluation means running the policy on the actual robot, in
the actual environment, and scoring whether it completes the task.
Expensive, slow, noisy.

The two can disagree. Reasons include

- The environment at evaluation time is not exactly the one in the
  training dataset (different lighting, slightly different object
  poses, a different operator's calibration).
- The metric is different. Offline MSE rewards matching a specific
  human trajectory step by step. Online success rewards getting to the
  goal, even if the path looks nothing like the human's.
- Small errors compound. The policy sees its own outputs as inputs at
  the next step on the robot, which it never does at offline eval
  time (where every input comes from the recorded human episode).

The paper makes a precise version of the first claim. Offline MSE,
**aggregated**, is a poor predictor of online ranking. Offline MSE
**decomposed per joint group** is a much better predictor. It is not
perfect, but it tracks the robot.

**Learn more.** The general gap between offline and online policy
evaluation is studied in
[Mandlekar *et al.* 2021](https://arxiv.org/abs/2108.03298) and is a
recurring theme in robot learning.

---

## 6. Glossary

- **Action chunk.** A short sequence of future actions predicted in
  one forward pass of the policy, typically 30 to 50 steps long. The
  robot consumes a prefix of the chunk before requesting the next.
- **AIRoA MoMa.** A public dataset of Toyota HSR teleoperation episodes
  collected by the AIRoA project. We use it as the generalist
  pretraining set.
- **AIRoA ICRA 2026.** The large-scale dataset released for the
  workshop competition this paper was submitted to. `task6911` is the
  six-task relocation subset we used for fine-tuning.
- **AMP, mixed precision.** Running the model partly in 16-bit floats
  to roughly halve memory and speed up training. Standard in PyTorch
  via `torch.cuda.amp` and exposed in LeRobot through
  `--policy.use_amp=true`.
- **Backbone.** The pretrained vision-language network that processes
  the image and the instruction. In our fine-tuning runs it is frozen.
- **Checkpoint.** A snapshot of model weights at a given step. Saved
  every few thousand steps during training so we can pick the best
  one offline.
- **Expert-only fine-tuning.** Strategy used for π₀.₅, in which the
  vision-language backbone is frozen and only the action head is
  trained. Reduces trainable parameters to a fraction of the model.
- **Generalist checkpoint.** The output of phase 1 of our SmolVLA
  fine-tuning, on AIRoA MoMa. Used as the starting point for the
  task-specific top-up.
- **Holonomic base.** A mobile base that can translate in any
  direction without first rotating. The HSR's base has three degrees
  of freedom (x, y, theta) that can be commanded independently.
- **LeRobot.** HuggingFace's open-source framework for robot learning.
  Hosts SmolVLA, the dataset format, the trainer and the evaluator.
- **LoRA.** Low-Rank Adaptation. A way to fine-tune a large model
  cheaply by training small low-rank matrices on top of frozen
  weights. We tried it on the π₀.₅ baseline and it underperformed
  expert-only, see `docs/CONTEXT.md`.
- **MSE.** Mean Squared Error. The mean over samples of the squared
  difference between predicted and ground-truth actions.
- **Norm stats.** The means, standard deviations (or quantiles) used
  to standardise observations and actions before they enter the
  policy. Computed once on the training dataset and shipped with the
  checkpoint.
- **openpi.** Physical Intelligence's open-source training stack for
  π-family VLAs. JAX/orbax under the hood.
- **Policy server.** The Python process that loads the policy and
  answers WebSocket requests with action chunks. See `docs/INFERENCE.md`.
- **π₀.₅ ("pi zero point five").** A 3.3 B parameter VLA from
  Physical Intelligence. Released with a pretrained checkpoint we
  use as baseline.
- **SmolVLA.** A small (450 M parameter) VLA from HuggingFace, built
  on the SmolVLM-2 backbone.
- **Top-up.** Our name for phase 2 of SmolVLA fine-tuning, a short
  task-specific run on top of the generalist checkpoint.
- **VLA.** Vision-Language-Action model. The class of models studied
  in this paper.
