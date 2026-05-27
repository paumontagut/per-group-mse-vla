"""
Reproduce the 60-trial robot evaluation results from data/robot_trials.csv.

The script does three things. It prints the mean rubric score for each
model, runs the three pairwise Mann-Whitney U tests, and saves a small
figure with the per-model score distribution. The numbers it prints
should match Table I and the significance statement in the paper to
the third decimal. The CSV is the source of truth, so if you want to
explore a different aggregation (per-object, per-trial-order, etc.)
you can load it with pandas and slice it directly.

Run it with

    python figures/robot_trials.py
    python figures/robot_trials.py --csv data/robot_trials.csv --out figures/robot_trials.pdf
"""

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


PRETTY = {
    "pi05_baseline_80k": "π₀.₅ baseline (80k)",
    "pi05_ours":         "π₀.₅ (ours)",
    "hsr_smolvla_40k":   "HSR-SmolVLA (40k)",
}

# Rendered in the same order as the paper Table I.
ORDER = ["pi05_baseline_80k", "pi05_ours", "hsr_smolvla_40k"]

COLORS = {
    "pi05_baseline_80k":   "#3D7ABF",
    "pi05_ours": "#7B4FAA",
    "hsr_smolvla_40k":     "#D4740E",
}


def summary(df: pd.DataFrame):
    print(f"\n{'Model':<32} {'n':>4} {'mean':>7} {'median':>7}  score histogram (1..4)")
    print("-" * 78)
    for m in ORDER:
        scores = df.loc[df["model"] == m, "score"].to_numpy()
        hist = [int((scores == k).sum()) for k in (1, 2, 3, 4)]
        print(f"{PRETTY[m]:<32} {len(scores):>4} {scores.mean():>7.3f} "
              f"{np.median(scores):>7.1f}  {hist}")


def pairwise(df: pd.DataFrame, alpha: float = 0.05):
    # The paper uses a one-sided test because we had a directional
    # hypothesis going in (the baseline ranks highest, our pi0.5
    # in the middle, HSR-SmolVLA last, by total MSE on the matching
    # offline split). We pass `alternative="greater"` so the test asks
    # "is the first group's distribution stochastically larger than
    # the second's", which is exactly that hypothesis. ORDER below is
    # ranked high to low, so every pair is tested in the expected
    # direction.
    print("\nMann-Whitney U (one-sided, alternative='greater')")
    print("-" * 78)
    results = []
    for a, b in combinations(ORDER, 2):
        xa = df.loc[df["model"] == a, "score"].to_numpy()
        xb = df.loc[df["model"] == b, "score"].to_numpy()
        u, p = mannwhitneyu(xa, xb, alternative="greater")
        marker = "*" if p < alpha else " "
        print(f"  {PRETTY[a]:<28} >  {PRETTY[b]:<28} U={u:>6.1f}  p={p:.4f} {marker}")
        results.append((a, b, u, p))
    print(f"\n* p < {alpha}")
    return results


def figure(df: pd.DataFrame, out_path: Path):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    })
    fig, ax = plt.subplots(figsize=(6.5, 3.2))

    bin_edges = np.arange(0.5, 5.5, 1.0)
    offsets = {-1: -0.22, 0: 0.0, 1: 0.22}
    width = 0.20
    for i, m in enumerate(ORDER):
        scores = df.loc[df["model"] == m, "score"].to_numpy()
        counts, _ = np.histogram(scores, bins=bin_edges)
        x = np.array([1, 2, 3, 4]) + offsets[i - 1]
        ax.bar(x, counts, width=width, color=COLORS[m],
               edgecolor="white", linewidth=0.5, label=PRETTY[m])

    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["1\nno move", "2\nno grasp", "3\ntouch", "4\npick"])
    ax.set_xlabel("Rubric score")
    ax.set_ylabel("Number of trials")
    ax.set_ylim(0, 21)
    ax.set_title("60-trial robot evaluation, score distribution per model",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    png_path = out_path.with_suffix(".png")
    plt.savefig(png_path, bbox_inches="tight", dpi=200)
    print(f"\nSaved {out_path} and {png_path}")


def main():
    parser = argparse.ArgumentParser(description="Per-model summary and Mann-Whitney U on the 60 robot trials")
    parser.add_argument("--csv", type=str,
                        default=str(Path(__file__).resolve().parents[1] / "data" / "robot_trials.csv"))
    parser.add_argument("--out", type=str,
                        default=str(Path(__file__).resolve().parent / "robot_trials.pdf"))
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    expected_cols = {"model", "trial", "score"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise SystemExit(f"CSV is missing columns {missing}")

    summary(df)
    pairwise(df)
    figure(df, Path(args.out))


if __name__ == "__main__":
    main()
