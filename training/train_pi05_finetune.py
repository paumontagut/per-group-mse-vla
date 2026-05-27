"""
π₀.₅ fine-tuning launcher.

π₀.₅ lives in a different ecosystem from SmolVLA. It is JAX/orbax under the
hood and is trained through the OpenPI repository, not LeRobot. This script
is a thin shim around `python scripts/train.py <config> ...` that keeps the
invocation consistent with what we used during the paper. You need a
separate OpenPI checkout (see docs/SETUP.md) and the π₀.₅ baseline
checkpoint that the AIRoA workshop distributed to the participating teams.

The fine-tuning strategy here freezes the vision-language backbone and
trains the action head on the task-specific data, which keeps the
trainable parameters at a fraction of the full 3.3 B and makes the run
feasible on a single 24 GB GPU. A LoRA variant on top of the baseline was
also explored and did not beat this path on in-distribution data, see
docs/CONTEXT.md for the ablation.

How the OpenPI CLI works (used internally by this launcher).

    python scripts/train.py <config-name>
        --exp_name=<run-name>
        --num_train_steps=N
        [--overwrite | --resume]

The base checkpoint to fine-tune from is NOT passed on the command line.
It is bound to the config you select, via the `weight_loader` field of the
TrainConfig dataclass. The config used in the paper ships with OpenPI and
already wires up the correct loader. If you want to point it at a different
baseline checkpoint, you can either edit the config registry in
`src/openpi/training/configs.py` or override the field on the CLI with
`--weight_loader.checkpoint_dir=/path/to/checkpoint`.

A typical run looks like

    python training/train_pi05_finetune.py \\
        --openpi-root /path/to/openpi \\
        --config <pi05_hsr_config_name> \\
        --exp-name pi05_finetune \\
        --steps <N>
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune π₀.₅ via the OpenPI training entry point"
    )
    parser.add_argument("--openpi-root", type=str, required=True,
                        help="Path to the openpi checkout")
    parser.add_argument("--config", type=str, required=True,
                        help="OpenPI training config name (positional to scripts/train.py)")
    parser.add_argument("--exp-name", type=str, required=True,
                        help="Experiment name (forwarded as --exp_name to OpenPI)")
    parser.add_argument("--steps", type=int, required=True,
                        help="Forwarded as --num_train_steps to OpenPI")
    parser.add_argument("--weight-loader-dir", type=str, default=None,
                        help="Optional override of weight_loader.checkpoint_dir, "
                             "if you want to fine-tune from a checkpoint that the "
                             "config does not point at by default")
    parser.add_argument("--overwrite", action="store_true",
                        help="Forwarded as --overwrite to OpenPI")
    parser.add_argument("--resume", action="store_true",
                        help="Forwarded as --resume to OpenPI")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the resolved command without running it")
    args = parser.parse_args()

    openpi_root = Path(args.openpi_root)
    train_py = openpi_root / "scripts" / "train.py"
    if not train_py.exists():
        sys.exit(f"ERROR: {train_py} not found. Is --openpi-root pointing at the openpi checkout?")

    cmd = [
        sys.executable, str(train_py),
        args.config,
        f"--exp_name={args.exp_name}",
        f"--num_train_steps={args.steps}",
    ]
    if args.weight_loader_dir:
        cmd.append(f"--weight_loader.checkpoint_dir={args.weight_loader_dir}")
    if args.overwrite:
        cmd.append("--overwrite")
    if args.resume:
        cmd.append("--resume")

    if args.dry_run:
        print("\nWould run:\n  " + " ".join(cmd))
        return

    print(f"Launching π₀.₅ {args.config} for {args.steps} steps "
          f"(exp_name={args.exp_name})")
    sys.exit(subprocess.run(cmd, cwd=openpi_root).returncode)


if __name__ == "__main__":
    main()
