#!/usr/bin/env python3
"""
Sanitize a SmolVLA checkpoint for public release on the Hugging Face Hub.

LeRobot's `train_config.json` records the exact paths it was launched
with (dataset root, generalist checkpoint path, output dir). Those
paths point at the lab's filesystem and have no business being on the
public Hub. This script copies the checkpoint into a fresh directory
and rewrites the offending fields with placeholders that still document
the intent.

The script is conservative. It does not touch `model.safetensors` or
the preprocessor and postprocessor blobs. It edits `train_config.json`
only, and prints a diff of every key it changes so you can audit it
before pushing.

Run it with

    python scripts/prepare_hf_release.py \\
        --checkpoint /path/to/pretrained_model \\
        --out /tmp/per-group-mse-smolvla

Then upload the result with

    huggingface-cli login
    huggingface-cli upload <your-username>/<repo-name> /tmp/per-group-mse-smolvla

We list the published name and the load snippet in the Models section
of the README once the upload is done.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path


# Keys in train_config.json whose values are lab-local paths. We rewrite
# them to short placeholder strings that document what was there. If
# you find a key here that you DO want to publish (e.g. the upstream
# repo_id of a public dataset), remove it from the dict before running.
REWRITES = {
    "dataset.root":            "<local-path-to-dataset>",
    "dataset.repo_id":         "local/task6911-v30",   # already a placeholder, kept for clarity
    "policy.pretrained_path":  "<local-path-to-generalist-checkpoint>",
    "output_dir":              "<local-output-dir>",
    "job_name":                "per_group_mse_smolvla",
}


def get_at(d, dotted):
    for part in dotted.split("."):
        if not isinstance(d, dict) or part not in d:
            return None
        d = d[part]
    return d


def set_at(d, dotted, value):
    parts = dotted.split(".")
    for part in parts[:-1]:
        d = d[part]
    d[parts[-1]] = value


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the pretrained_model directory")
    parser.add_argument("--out", type=str, required=True,
                        help="Where to write the sanitized copy")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite --out if it exists")
    args = parser.parse_args()

    src = Path(args.checkpoint).resolve()
    dst = Path(args.out).resolve()
    if not src.exists():
        sys.exit(f"ERROR: source {src} does not exist")
    if dst.exists():
        if args.force:
            shutil.rmtree(dst)
        else:
            sys.exit(f"ERROR: {dst} exists, pass --force to overwrite")

    shutil.copytree(src, dst)
    print(f"Copied {src} to {dst}")

    cfg_path = dst / "train_config.json"
    if not cfg_path.exists():
        print(f"WARNING: no train_config.json at {cfg_path}, nothing to sanitize")
        return

    cfg = json.loads(cfg_path.read_text())
    print("\nRewrites in train_config.json")
    print("-" * 72)
    for key, placeholder in REWRITES.items():
        old = get_at(cfg, key)
        if old is None:
            continue
        set_at(cfg, key, placeholder)
        old_short = old if len(str(old)) < 60 else str(old)[:57] + "..."
        print(f"  {key:<28} {old_short!s}\n  {'':<28} -> {placeholder}")
    cfg_path.write_text(json.dumps(cfg, indent=4))

    print("\nFiles in the sanitized release")
    print("-" * 72)
    for p in sorted(dst.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(dst)}  ({p.stat().st_size / (1024*1024):.1f} MB)")

    print("\nNext steps")
    print("-" * 72)
    print("  1. Eyeball train_config.json to make sure nothing else looks lab-internal.")
    print("  2. huggingface-cli login")
    print("  3. huggingface-cli upload <username>/<repo-name>", dst)


if __name__ == "__main__":
    main()
