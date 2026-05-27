#!/usr/bin/env python3
"""
Scan a LeRobot dataset for episodes whose camera videos fail to decode.

LeRobot training crashes hard on a single broken frame, so the simplest
defense is to scan once and pass the clean list with `--dataset.episodes=`.
We hit one corrupt episode in AIRoA MoMa (3997) during the pretraining
sweep, and once we excluded it the training stopped failing intermittently.

The check is intentionally crude. It opens each camera's mp4 with PyAV and
walks every frame. If any frame fails to decode, the episode is reported as
corrupt. This catches both truncated files and partially-encoded frames.

A typical run looks like

    python scripts/find_corrupt_episodes.py \\
        --dataset-root /path/to/dataset \\
        --cameras observation.image.head observation.image.hand \\
        --output clean_episodes.txt
"""

import argparse
import os
import sys
from pathlib import Path


def video_path_for(dataset_root: Path, episode_idx: int, camera_key: str) -> Path:
    """LeRobot v3.0 stores videos under videos/chunk-XXX/<cam>/episode_NNNNNN.mp4."""
    chunk_idx = episode_idx // 1000
    cam_dir = camera_key.replace(".", "/")
    return (dataset_root / "videos" / f"chunk-{chunk_idx:03d}" / cam_dir
            / f"episode_{episode_idx:06d}.mp4")


def decodes_cleanly(video_path: Path):
    """Try to decode every frame. Returns (ok, error_message_or_None)."""
    if not video_path.exists():
        return False, f"file missing: {video_path}"
    try:
        import av
    except ImportError:
        sys.exit("ERROR: PyAV not installed. `pip install av` (or pyav).")

    count = 0
    try:
        container = av.open(str(video_path))
        for _ in container.decode(video=0):
            count += 1
        container.close()
        return True, None
    except Exception as e:
        return False, f"frame ~{count}: {e}"


def list_episodes(dataset_root: Path, task_filter=None, success_only=True):
    """Read episode metadata and return the list of episode indices to scan."""
    import pandas as pd
    parquet = dataset_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    if not parquet.exists():
        sys.exit(f"ERROR: episodes parquet not found at {parquet}")
    ep = pd.read_parquet(parquet)

    if task_filter:
        ep["task_name"] = ep["tasks"].apply(lambda x: x[0] if len(x) > 0 else "")
        ep = ep[ep["task_name"].isin(task_filter)]
    if success_only and "task_success" in ep.columns:
        ep = ep[ep["task_success"] == True]

    return sorted(ep["episode_index"].astype(int).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--cameras", type=str, nargs="+",
                        default=["observation.image.head", "observation.image.hand"],
                        help="Camera keys to scan (full LeRobot keys)")
    parser.add_argument("--task-filter", type=str, nargs="*", default=None,
                        help="Restrict to episodes whose first task is in this list")
    parser.add_argument("--include-failed", action="store_true",
                        help="Include episodes where task_success is False")
    parser.add_argument("--output", type=str, default="clean_episodes.txt",
                        help="Where to write the clean episode list")
    args = parser.parse_args()

    root = Path(args.dataset_root).resolve()
    episodes = list_episodes(root, task_filter=args.task_filter,
                             success_only=not args.include_failed)
    print(f"Scanning {len(episodes)} episodes under {root}...")

    clean, corrupt = [], []
    for i, ep_idx in enumerate(episodes):
        ok = True
        for cam in args.cameras:
            good, err = decodes_cleanly(video_path_for(root, ep_idx, cam))
            if not good:
                ok = False
                print(f"  episode {ep_idx} ({cam}): {err}")
        (clean if ok else corrupt).append(ep_idx)

        if (i + 1) % 50 == 0 or (i + 1) == len(episodes):
            print(f"  ...{i + 1}/{len(episodes)} ({len(corrupt)} corrupt so far)")

    print(f"\nClean:   {len(clean)}")
    print(f"Corrupt: {len(corrupt)}")
    if corrupt:
        print(f"Corrupt indices: {corrupt}")

    clean_str = "[" + ",".join(str(e) for e in clean) + "]"
    Path(args.output).write_text(clean_str)
    print(f"\nClean list written to {args.output}")
    print(f"Pass it to lerobot-train with --dataset.episodes='{clean_str[:60]}...'")


if __name__ == "__main__":
    main()
