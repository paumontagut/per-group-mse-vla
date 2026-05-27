"""
Generalist pretraining of SmolVLA on AIRoA MoMa.

First of the two phases. The goal is to get SmolVLA out of its 6-7 DoF
pretraining habit and into the HSR action space (11 DoF, four joint groups)
before specializing on the competition tasks. AIRoA MoMa covers ~109k episodes
across a wide variety of mobile manipulation tasks, so the model learns the
HSR's input distribution (head and hand camera, 8-D state) and action layout
without yet overfitting to any specific task.

The output is a checkpoint that is then used as the starting point for the
task-specific top-up (see train_smolvla_task.py).

A typical run looks like

    python training/train_smolvla_generalist.py \\
        --dataset-root datasets/airoa-moma \\
        --output-dir outputs/smolvla_generalist \\
        --steps 20000 --batch-size 32

A few things to know about the flags below. The `--policy.empty_cameras=1`
flag is required because SmolVLA expects 3 cameras and the HSR provides 2
(head and hand), so the third slot is filled with zeros. The `rename_map`
maps the dataset's `observation.image.head` and `.hand` keys to the
`observation.images.camera1` and `camera2` names that SmolVLA expects. And
we use the pyav video backend instead of torchcodec, because torchcodec
leaks file descriptors on long runs over large datasets.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def validate_dataset(root: Path):
    """Light sanity check on the LeRobot v3.0 dataset before launching."""
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        sys.exit(f"ERROR: {info_path} not found. Is the dataset in LeRobot v3.0 format?")

    info = json.loads(info_path.read_text())
    print(f"Dataset: {root}")
    print(f"  Version : {info.get('codebase_version')}")
    print(f"  Episodes: {info.get('total_episodes', 0):,}")
    print(f"  Tasks   : {info.get('total_tasks', 0)}")
    print(f"  FPS     : {info.get('fps', 0)}")

    if info.get("codebase_version") != "v3.0":
        sys.exit(f"ERROR: Expected LeRobot v3.0, got {info.get('codebase_version')}")
    if info.get("total_episodes", 0) == 0:
        sys.exit("ERROR: Dataset reports 0 episodes")


def main():
    parser = argparse.ArgumentParser(description="SmolVLA generalist pretraining on AIRoA MoMa")
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--repo-id", type=str, default="local/airoa-moma",
                        help="Placeholder repo_id for the local dataset")
    parser.add_argument("--policy-base", type=str, default="lerobot/smolvla_base",
                        help="Pretrained policy to start from")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-freq", type=int, default=5000)
    parser.add_argument("--log-freq", type=int, default=200)
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable mixed precision (uses ~2x VRAM)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from a previous checkpoint (pretrained_model dir)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the lerobot-train command and exit")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    validate_dataset(dataset_root)

    cmd = [
        "lerobot-train",
        f"--policy.path={args.resume or args.policy_base}",
        f"--dataset.repo_id={args.repo_id}",
        f"--dataset.root={dataset_root}",
        f"--batch_size={args.batch_size}",
        f"--steps={args.steps}",
        f"--output_dir={args.output_dir}",
        f"--job_name={Path(args.output_dir).name}",
        "--policy.push_to_hub=false",
        "--wandb.enable=false",
        # Map HSR's camera keys to SmolVLA's expected names.
        '--rename_map={"observation.image.head": "observation.images.camera1", '
        '"observation.image.hand": "observation.images.camera2"}',
        "--policy.empty_cameras=1",  # SmolVLA expects 3 cameras, HSR has 2
        "--dataset.video_backend=pyav",  # torchcodec leaks fds on long runs
        f"--seed={args.seed}",
        f"--num_workers={args.num_workers}",
        "--save_checkpoint=true",
        f"--save_freq={args.save_freq}",
        f"--log_freq={args.log_freq}",
    ]
    if not args.no_amp:
        cmd.append("--policy.use_amp=true")

    if args.dry_run:
        print("\nWould run:\n " + " \\\n  ".join(cmd))
        return

    print(f"\nLaunching: {args.steps} steps, batch {args.batch_size}, "
          f"amp={not args.no_amp}")
    print(f"Output: {args.output_dir}")
    sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
