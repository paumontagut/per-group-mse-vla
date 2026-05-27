# Context and choices that are not obvious from the code

The paper is short and the scripts are short on purpose. This document
collects the practical decisions behind them. What we tried, what we kept,
what we dropped, and the workarounds that matter when running the code.

## Hardware

A single 24 GB GPU (RTX 3090), within the lab's shared-hardware budget.
The practical consequences are visible throughout the scripts.

- **Batch sizes are 8 to 32**, not the 64+ used in the original SmolVLA paper.
  The top-up runs use batch 8 with mixed precision to fit within a ~4 GB
  footprint. The longer sweeps (B32 in the figure) take the full card.
- **No multi-GPU experiments.** The two phases are sequential. The π₀.₅ side
  is similar, since expert-only fine-tuning is feasible on a single 24 GB
  card because the vision-language backbone is frozen.
- **Dataset size constrains the sweep.** Training the full ICRA 2026
  dataset on a 3090 was not realistic for the time available, so the
  fine-tuning runs on a private subset of the AIRoA ICRA 2026 dataset
  distributed to the competing teams. The generalist phase uses the
  full public AIRoA MoMa set.

## Subset selection

The full AIRoA ICRA 2026 dataset is large and heterogeneous. For the
top-up phase we use `task6911`, the relocation subset, for three reasons.
It is the subset that matches the competition's public evaluation tasks.
It is small enough to converge in a few thousand steps. The loss bottoms
out around 3k to 4k and starts climbing back around 5k, mostly on the
gripper component. And a short top-up on top of a generalist checkpoint
is a cleaner ablation than training from scratch on the small set. The
generalist gives a reasonable initialisation for the four joint groups,
and we can attribute the improvement to the top-up directly.

## Two-phase fine-tuning

Generalist then top-up, rather than a single end-to-end fine-tune, for two
reasons. First, the AIRoA MoMa dataset is much more diverse than the
relocation subset and produces a checkpoint that is robust to camera and
state variation. Second, having an intermediate "generalist" checkpoint lets
us run multiple top-up experiments without paying the long generalist cost
again. The generalist checkpoint at 20k steps is the one used as the
starting point for every top-up reported.

## Why per-group MSE

The four joint groups have very different scales and dynamics.

- The arm is continuous and 5-dimensional. Most of the manipulation signal
  lives here.
- The gripper is essentially binary (open or close). Its squared error is
  dominated by the transition frames.
- The head has a small range and changes slowly. Its MSE is two to three
  orders of magnitude smaller than the rest.
- The base is 3-DoF holonomic and continuous. SmolVLA had no base
  pretraining and this group converges last.

Averaging an MSE over the 11-D vector hides this structure. A checkpoint
whose total MSE looks good can be one where the gripper is well-behaved
but the arm has degraded. On the robot, that checkpoint picks worse. This
is the offline finding that the paper validates with 60 real-robot trials.

## Things we tried and dropped

- **LoRA on the π₀.₅ 80k baseline.** Hypothesis was that a low-rank
  adapter on top of the released baseline would generalize better
  than our π₀.₅ fine-tune for out-of-distribution tasks. We trained
  rank-16 LoRA on the same data and evaluated per task. Our π₀.₅
  fine-tune was 37% better on average across the relocation tasks.
  LoRA only beat it on a single task, and below the chosen 10%
  margin. The fine-tune was kept as the competition submission.
  The LoRA checkpoints survive only as an ablation row.
- **Direct evaluation with `select_action` in the Docker adapter.**
  `select_action` returns a single action from an internal queue and is the
  right API for closed-loop control. The workshop's evaluation framework
  expects a full action chunk instead. The adapter calls
  `predict_action_chunk` and returns the full `(T, 11)` block.
- **Loading the normalizer through the full LeRobot pipeline at inference
  time.** Reconstructing the six-step pipeline by hand was fragile across
  LeRobot versions. We ended up loading the safetensors with the
  normalization statistics directly and applying them in the adapter.

## Workarounds you will run into

- **`--policy.empty_cameras=1`.** SmolVLA expects three cameras. The HSR
  provides two (head and hand), and the third slot is filled with zeros.
- **`--dataset.video_backend=pyav`.** `torchcodec` is faster on short runs
  but leaks file descriptors over long training jobs on large datasets.
- **Corrupt episodes.** In some snapshots of AIRoA MoMa, episode 3997
  has broken metadata. The evaluation script excludes it explicitly.
  Re-run `scripts/find_corrupt_episodes.py` if you suspect new bad
  episodes in a different snapshot.
- **attention_mask dtype.** Some SmolVLA configurations expect a boolean
  attention mask, but the HuggingFace tokenizer returns int64. If you see a
  cryptic shape error from the eager attention path, casting the mask with
  `.bool()` is the fix.
- **`num2words` is a hidden dependency.** The SmolVLM processor uses it but
  some installations don't pull it transitively.

## Things deliberately not in this repo

- **The competition evaluation harness.** That code lives upstream
  (`airoa-org/airoa-evaluation-ICRA`) and is the property of the workshop.
  We forked it for our submission, but this repo does not redistribute it.
- **Our π₀.₅ submission to the competition.** It is a fine-tune of
  `baseline_80k`, which AIRoA distributed only to the competing
  teams under the NDA. Publishing our derivative would effectively
  publish the baseline, so the binary weights stay behind the NDA.
  The training script, the offline results and the 60-trial robot
  scores are all included in the repo. Only the weights themselves
  are redacted.
- **Raw evaluation videos.** The robot videos used in the paper's 60-trial
  evaluation are workshop assets and are not redistributed here.
- **Bucket credentials.** Submission credentials and any cloud paths are
  not in the repository.
