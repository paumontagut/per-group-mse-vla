#!/usr/bin/env python3
"""
Summarise a LeRobot v3.0 dataset.

I wrote this for myself because every time I came back to a dataset
after a few weeks I had to dig through the parquet files to remember
how many episodes it had, which tasks it covered, and whether I was
looking at the converted v3.0 snapshot or the v2.1 one. It is a five-
minute utility and probably the first thing you want to run after
pulling a dataset that someone else prepared.

It checks the version stamp in `meta/info.json`, counts episodes and
tasks, prints the action/state dimensions and the FPS, lists the task
names with the number of episodes per task, and warns about a few
common gotchas (zero-frame episodes, the corrupt episode 3997 in some
AIRoA MoMa snapshots).

Run it with

    python scripts/dataset_info.py --dataset-root /path/to/dataset
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


CORRUPT_KNOWN = {
    "airoa-moma": {3997},
}


def banner(title: str):
    bar = "-" * max(8, len(title))
    print(f"\n{bar}\n{title}\n{bar}")


def main():
    parser = argparse.ArgumentParser(description="Summarise a LeRobot v3.0 dataset")
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--show-tasks", type=int, default=20,
                        help="How many task names to list before truncating")
    args = parser.parse_args()

    root = Path(args.dataset_root).resolve()
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        sys.exit(f"ERROR: {info_path} not found. Is this a LeRobot v3.0 dataset?")

    info = json.loads(info_path.read_text())
    version = info.get("codebase_version")
    fps = info.get("fps", 0)
    n_episodes = info.get("total_episodes", 0)
    n_tasks = info.get("total_tasks", 0)
    n_frames = info.get("total_frames", 0)

    banner(f"Dataset at {root}")
    print(f"  codebase_version : {version}")
    print(f"  episodes         : {n_episodes:,}")
    print(f"  tasks            : {n_tasks}")
    print(f"  frames           : {n_frames:,}")
    print(f"  fps              : {fps}")

    if version != "v3.0":
        print(f"\n  WARNING: expected v3.0, got {version}. Most scripts here "
              f"assume v3.0. LeRobot ships a converter for older snapshots.")

    # Shape of the action and state vectors, useful sanity check before
    # plugging the dataset into a SmolVLA training run.
    features = info.get("features", {})
    state = features.get("observation.state", {})
    action = features.get("action", {})
    banner("Shapes")
    print(f"  observation.state shape : {state.get('shape', '?')}")
    print(f"  action shape            : {action.get('shape', '?')}")
    img_keys = [k for k in features if k.startswith("observation.image")]
    if img_keys:
        print(f"  image keys              : {img_keys}")

    # Per-task episode count. Most useful number for deciding whether a
    # subset is large enough to fine-tune on.
    parquet = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    if parquet.exists():
        try:
            import pandas as pd
        except ImportError:
            sys.exit("pandas is required for the per-task breakdown")
        ep = pd.read_parquet(parquet)
        ep["task_name"] = ep["tasks"].apply(lambda x: x[0] if len(x) > 0 else "")
        per_task = Counter(ep["task_name"].tolist())
        banner(f"Per-task episode count (top {args.show_tasks})")
        for name, count in per_task.most_common(args.show_tasks):
            print(f"  {count:>6}  {name}")
        if len(per_task) > args.show_tasks:
            print(f"  ... and {len(per_task) - args.show_tasks} more tasks")

        # Sanity checks that have actually caught bugs in our datasets.
        zero_frame = ep[ep.get("length", 0) == 0]
        if len(zero_frame) > 0:
            print(f"\n  WARNING: {len(zero_frame)} episodes report zero frames.")

        if "airoa-moma" in str(root).lower():
            known = CORRUPT_KNOWN["airoa-moma"]
            present = [e for e in known if e in ep["episode_index"].astype(int).tolist()]
            if present:
                print(f"\n  NOTE: episodes known to be corrupt in some snapshots: {present}")
                print(f"        Exclude them with scripts/find_corrupt_episodes.py")
    else:
        print(f"\n  (episodes parquet not found at {parquet}, skipping per-task breakdown)")


if __name__ == "__main__":
    main()
