"""
Reproduce the per-joint-group MSE figure from the poster and the paper.

The data are the eval_offline results across the SmolVLA training schedule
(B8 = batch 8 with AMP for the first short top-up sweep, B32 = batch 32 for
the longer runs once we could take the whole GPU). The top panel shows MSE
per joint group on a log scale, the bottom panel shows total MSE so the
"best" checkpoint by total MSE can be located easily for comparison.

We hard-code the numbers here so the figure is reproducible from this file
alone, without re-running the full eval sweep. If you want to regenerate the
numbers from scratch, run eval/per_group_mse.py over the checkpoint set
listed in this dict.

    python figures/mse_curves.py [--out figures/mse_curves.pdf]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# MSE values are raw (not ×10⁻³). Plotting code multiplies by 1000.
# Order corresponds to the effective training schedule, left to right.
CHECKPOINTS = {
    "PT 20k":  {"arm": 0.002668, "gripper": 0.053343, "head": 0.000018, "base": 0.007543, "total": 0.008122},
    "B8 1k":   {"arm": 0.002359, "gripper": 0.035457, "head": 0.000023, "base": 0.011783, "total": 0.007513},
    "B8 3k":   {"arm": 0.002679, "gripper": 0.017073, "head": 0.000009, "base": 0.007686, "total": 0.004868},
    "B8 5k":   {"arm": 0.001661, "gripper": 0.016554, "head": 0.000012, "base": 0.006954, "total": 0.004159},
    "B32 5k":  {"arm": 0.001518, "gripper": 0.011202, "head": 0.000008, "base": 0.007726, "total": 0.003817},
    "B32 10k": {"arm": 0.001359, "gripper": 0.012283, "head": 0.000019, "base": 0.013247, "total": 0.005351},
    "B32 15k": {"arm": 0.001038, "gripper": 0.006650, "head": 0.000008, "base": 0.003916, "total": 0.002146},
    "B32 20k": {"arm": 0.001069, "gripper": 0.004642, "head": 0.000012, "base": 0.005562, "total": 0.002427},
    "B32 25k": {"arm": 0.000905, "gripper": 0.006313, "head": 0.000009, "base": 0.006173, "total": 0.002671},
    "B32 30k": {"arm": 0.000755, "gripper": 0.007221, "head": 0.000007, "base": 0.006455, "total": 0.002762},
    "B32 35k": {"arm": 0.000894, "gripper": 0.007593, "head": 0.000107, "base": 0.004514, "total": 0.002347},
    "B32 40k": {"arm": 0.000883, "gripper": 0.003773, "head": 0.000007, "base": 0.003175, "total": 0.001611},
    "B32 45k": {"arm": 0.001115, "gripper": 0.007790, "head": 0.000013, "base": 0.003714, "total": 0.002230},
    "B32 50k": {"arm": 0.001210, "gripper": 0.006061, "head": 0.000008, "base": 0.004550, "total": 0.002343},
}

GROUPS = ["arm", "gripper", "head", "base"]
COLORS = {"arm": "#2196F3", "gripper": "#E91E63", "head": "#4CAF50", "base": "#FF9800"}
MARKERS = {"arm": "o", "gripper": "s", "head": "^", "base": "D"}
DISPLAY = {
    "arm": "Arm (5 joints)",
    "gripper": "Gripper (1 joint)",
    "head": "Head (2 joints)",
    "base": "Base (3 joints)",
}


def make_figure(out_path: Path):
    labels = list(CHECKPOINTS.keys())
    x = np.arange(len(labels))

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
    })
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6.5),
                                   gridspec_kw={"height_ratios": [3, 2]})

    # Top panel, per-group curves on log scale. Log scale matters here because
    # the head MSE is about 3 orders of magnitude smaller than the gripper MSE.
    # On a linear axis the head curve would collapse onto the x-axis.
    for g in GROUPS:
        vals = [CHECKPOINTS[k][g] * 1000 for k in labels]
        ax1.plot(x, vals, color=COLORS[g], marker=MARKERS[g], markersize=5,
                 linewidth=1.5, label=DISPLAY[g], zorder=3)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax1.set_ylabel("MSE ($\\times 10^{-3}$, log scale)", fontsize=12)
    ax1.set_xlabel("Checkpoint", fontsize=12)
    ax1.set_title("Per-Joint-Group MSE", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10, loc="upper right", framealpha=0.9)
    ax1.grid(True, alpha=0.3, linewidth=0.5)
    ax1.set_yscale("log")
    ax1.set_ylim(bottom=0.005, top=80)

    # Vertical separator between the early batch-8 schedule (first 4 columns)
    # and the longer batch-32 runs.
    ax1.axvline(x=3.5, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
    ax1.text(1.5, 60, "B8", ha="center", fontsize=10, color="gray", fontstyle="italic")
    ax1.text(9, 60, "B32", ha="center", fontsize=10, color="gray", fontstyle="italic")

    # Bottom panel, total MSE bars with the best checkpoint highlighted.
    total_vals = [CHECKPOINTS[k]["total"] * 1000 for k in labels]
    best_idx = int(np.argmin(total_vals))
    bar_colors = ["#78909C"] * len(labels)
    bar_colors[best_idx] = "#26A69A"

    ax2.bar(x, total_vals, color=bar_colors, edgecolor="white", linewidth=0.5, width=0.65)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax2.set_ylabel("Total MSE ($\\times 10^{-3}$)", fontsize=12)
    ax2.set_title("Total MSE", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3, linewidth=0.5, axis="y")

    # Reduction relative to the pretrained checkpoint.
    pt_total = CHECKPOINTS[labels[0]]["total"] * 1000
    reduction = 100 * (pt_total - total_vals[best_idx]) / pt_total

    ax2.annotate(f"Best: {total_vals[best_idx]:.2f}",
                 xy=(best_idx, total_vals[best_idx]),
                 xytext=(best_idx + 1.5, total_vals[best_idx] + 1.5),
                 fontsize=10, fontweight="bold", color="#26A69A",
                 arrowprops=dict(arrowstyle="->", color="#26A69A", lw=1.2))
    ax2.annotate(f"$-${reduction:.1f}%",
                 xy=(best_idx, total_vals[best_idx] + 0.3),
                 ha="center", fontsize=9, color="#26A69A", fontweight="bold")
    ax2.axvline(x=3.5, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    png_path = out_path.with_suffix(".png")
    plt.savefig(png_path, bbox_inches="tight", dpi=200)
    print(f"Saved {out_path} and {png_path}")


def main():
    parser = argparse.ArgumentParser(description="Per-joint-group MSE figure for the paper / poster")
    parser.add_argument("--out", type=str,
                        default=str(Path(__file__).parent / "mse_curves.pdf"))
    args = parser.parse_args()
    make_figure(Path(args.out))


if __name__ == "__main__":
    main()
