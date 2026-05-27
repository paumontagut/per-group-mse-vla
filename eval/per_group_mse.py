"""
Per-group MSE evaluation for SmolVLA policies on the HSR 11-DoF action space.

The HSR action vector has 11 dimensions split across four functionally distinct
joint groups, namely a 5-DoF arm, a 1-DoF parallel gripper, a 2-DoF head, and
a 3-DoF holonomic base. Aggregating all 11 dimensions into a single MSE hides
which group is actually driving the error. The gripper is binary, the base is
continuous, the head has a small range, and the arm carries most of the
manipulation signal. A checkpoint with low total MSE can still have a degraded
arm and pick worse on the real robot. This script reports MSE per group so the
dominant component is visible.

The script loads a checkpoint with its full preprocessing pipeline (rename,
tokenize, normalize), samples frames from held-out episodes, runs
predict_action_chunk, and reports MSE on the normalized action space. We use
the normalized space because that is what the policy is trained against.
Unnormalizing both sides would change the scale per joint group and obscure
the comparison.

Run it on a single checkpoint

    python eval/per_group_mse.py \\
        --checkpoint outputs/smolvla_topup/checkpoints/003000/pretrained_model \\
        --dataset-root datasets/task6911-v30

or on a sweep of several checkpoints listed in a JSON file

    python eval/per_group_mse.py \\
        --checkpoints-json eval/checkpoints.json \\
        --dataset-root datasets/task6911-v30 \\
        --output results.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# HSR 11D action layout. Order matches both the dataset and the policy output.
# Do not reorder without re-checking the dataset stats.
ACTION_GROUPS = {
    "arm":     slice(0, 5),   # arm_lift, arm_flex, arm_roll, wrist_flex, wrist_roll
    "gripper": slice(5, 6),   # binary open/close
    "head":    slice(6, 8),   # head_pan, head_tilt
    "base":    slice(8, 11),  # base_x, base_y, base_theta (holonomic)
}

# Episodes flagged as corrupt during dataset preparation. Excluded from splits
# to avoid spurious MSE spikes. See scripts/find_corrupt_episodes.py for how
# this set is built.
CORRUPT_EPISODES = {3997}


def split_episodes(episodes_parquet, task_filter=None, test_ratio=0.1, seed=42):
    """Split the episodes of a LeRobot dataset into train/test, deterministically.

    Holding out test episodes by index (not by frame) avoids leakage across
    frames of the same trajectory.
    """
    ep = pd.read_parquet(episodes_parquet)
    if task_filter is not None:
        ep["task_name"] = ep["tasks"].apply(lambda x: x[0] if len(x) > 0 else "unknown")
        ep = ep[ep["task_name"].isin(task_filter)]

    if "task_success" in ep.columns:
        ep = ep[ep["task_success"] == True]

    indices = sorted(ep["episode_index"].astype(int).tolist())
    indices = [i for i in indices if i not in CORRUPT_EPISODES]

    rng = np.random.RandomState(seed)
    shuffled = np.array(indices)
    rng.shuffle(shuffled)
    n_test = max(1, int(len(shuffled) * test_ratio))
    test_eps = sorted(shuffled[:n_test].tolist())
    train_eps = sorted(shuffled[n_test:].tolist())
    return train_eps, test_eps


def evaluate(checkpoint_path, dataset_root, test_episodes, n_samples=240,
             repo_id="local/dataset", device="cuda"):
    """Run a checkpoint over a fixed grid of frames and return per-group MSE."""
    # Imports are local so the module loads without lerobot installed
    # (e.g. when only the helpers above are used from a notebook).
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors

    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=dataset_root,
        episodes=test_episodes,
        video_backend="pyav",  # see docs/CONTEXT.md for why not torchcodec
    )

    policy = SmolVLAPolicy.from_pretrained(str(checkpoint_path))
    policy.eval()
    policy.to(device)

    # The preprocessor handles tokenization, image renames, and normalization
    # using the stats stored inside the checkpoint. Reproducing those steps by
    # hand was fragile. Reusing make_pre_post_processors is the only way to
    # guarantee the input distribution matches what the policy was trained on.
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(checkpoint_path),
        dataset_stats=dataset.meta.stats,
    )

    total_frames = len(dataset)
    n_samples = min(n_samples, total_frames)
    # Linspace sampling rather than random. Gives stable comparisons across
    # checkpoints without having to fix a per-checkpoint seed.
    sample_indices = np.linspace(0, total_frames - 1, n_samples, dtype=int)

    all_mse = []
    per_group = {name: [] for name in ACTION_GROUPS}

    with torch.no_grad():
        for idx in tqdm(sample_indices, desc=Path(checkpoint_path).parent.name):
            frame = dataset[int(idx)]
            gt_action = frame["action"].clone()  # (11,), normalized

            batch = preprocessor(frame)
            pred_chunk = policy.predict_action_chunk(batch)  # (1, T, 11)
            pred_first = pred_chunk[0, 0, :11].cpu()

            error = (pred_first - gt_action[:11]) ** 2
            all_mse.append(error.mean().item())
            for name, slc in ACTION_GROUPS.items():
                per_group[name].append(error[slc].mean().item())

    results = {
        "checkpoint": str(checkpoint_path),
        "n_samples": int(n_samples),
        "n_test_episodes": len(test_episodes),
        "mse_total": float(np.mean(all_mse)),
        "mse_total_std": float(np.std(all_mse)),
    }
    for name in ACTION_GROUPS:
        results[f"mse_{name}"] = float(np.mean(per_group[name]))

    del policy
    if device == "cuda":
        torch.cuda.empty_cache()
    return results


def print_table(results_by_name):
    """Pretty-print a comparison table, sorted by total MSE."""
    print(f"\n{'Model':<24} {'Total':>9} {'Arm':>9} {'Grip':>9} {'Head':>9} {'Base':>9}")
    print("-" * 72)
    ordered = sorted(results_by_name.items(), key=lambda kv: kv[1]["mse_total"])
    for name, r in ordered:
        print(f"{name:<24} {r['mse_total']:>9.4f} {r['mse_arm']:>9.4f} "
              f"{r['mse_gripper']:>9.4f} {r['mse_head']:>9.4f} {r['mse_base']:>9.4f}")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--checkpoint", type=str,
                        help="Path to a single checkpoint (pretrained_model dir)")
    parser.add_argument("--checkpoints-json", type=str,
                        help="JSON file with {name: path} for multiple checkpoints")
    parser.add_argument("--dataset-root", type=str, required=True,
                        help="Root of the LeRobot v3.0 dataset")
    parser.add_argument("--repo-id", type=str, default="local/dataset",
                        help="Placeholder repo_id for the local dataset")
    parser.add_argument("--task-filter", type=str, nargs="*", default=None,
                        help="Restrict to episodes whose first task is in this list")
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-samples", type=int, default=240)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to a JSON file to write results to")
    args = parser.parse_args()

    if not (args.checkpoint or args.checkpoints_json):
        parser.error("Provide either --checkpoint or --checkpoints-json")

    dataset_root = Path(args.dataset_root)
    episodes_parquet = dataset_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    if not episodes_parquet.exists():
        parser.error(f"Episodes parquet not found at {episodes_parquet}. "
                     f"Is the dataset in LeRobot v3.0 format?")

    _, test_eps = split_episodes(
        episodes_parquet,
        task_filter=args.task_filter,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print(f"Held-out episodes: {len(test_eps)}")

    if args.checkpoint:
        checkpoints = {Path(args.checkpoint).parent.name: args.checkpoint}
    else:
        with open(args.checkpoints_json) as f:
            checkpoints = json.load(f)

    results = {}
    for name, path in checkpoints.items():
        if not Path(path).exists():
            print(f"  Skipping {name}: not found at {path}")
            continue
        try:
            results[name] = evaluate(
                checkpoint_path=path,
                dataset_root=dataset_root,
                test_episodes=test_eps,
                n_samples=args.n_samples,
                repo_id=args.repo_id,
                device=args.device,
            )
        except Exception as e:
            print(f"  FAILED on {name}: {e}")
            if args.device == "cuda":
                torch.cuda.empty_cache()

    print_table(results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
