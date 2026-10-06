"""
Plot raw ground-truth finger trajectories (train_dg) at full sampling rate.

Examples:
  python -m src.plot_finger_gt --mat_path "...\\sub1_comp.mat" --duration_sec 10
  python -m src.plot_finger_gt --mat_path "...\\sub1_comp.mat" --full
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from .config import ProjectConfig
from .data_loader import load_bci_mat

FINGER_NAMES = ("thumb", "index", "middle", "ring", "little")


def _maybe_decimate(
    t: np.ndarray, seg: np.ndarray, max_points: int
) -> tuple[np.ndarray, np.ndarray, str]:
    n = seg.shape[0]
    if max_points <= 0 or n <= max_points:
        return t, seg, ""
    idx = np.linspace(0, n - 1, num=max_points, dtype=int)
    return t[idx], seg[idx], f" (display: {max_points} pts/channel, evenly sampled from {n})"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat_path", required=True)
    parser.add_argument("--start_sec", type=float, default=0.0, help="Start time in seconds from recording start.")
    parser.add_argument(
        "--duration_sec",
        type=float,
        default=10.0,
        help="How many seconds to plot (ignored if --full).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Plot from --start_sec to the end of train_dg (e.g. full ~400 s at 1 kHz).",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=400_000,
        help="Max points per channel for drawing (decimate if longer). Use 0 to plot every sample (may be slow).",
    )
    parser.add_argument("--sample_rate_hz", type=int, default=1000)
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Output PNG filename under artifacts/ (default: auto by mode).",
    )
    args = parser.parse_args()

    cfg = ProjectConfig()
    os.makedirs(cfg.model_dir, exist_ok=True)

    _, train_y, _ = load_bci_mat(args.mat_path)
    n_total = len(train_y)
    t_total_sec = n_total / args.sample_rate_hz

    start = int(max(0.0, args.start_sec) * args.sample_rate_hz)
    if args.full:
        end = n_total
    else:
        end = int(start + args.duration_sec * args.sample_rate_hz)
        end = min(end, n_total)
    if end <= start:
        raise ValueError("Empty slice: check start_sec / duration_sec / --full.")

    seg = train_y[start:end]
    t = np.arange(seg.shape[0], dtype=np.float64) / args.sample_rate_hz + args.start_sec

    max_pts = 0 if args.max_points == 0 else int(args.max_points)
    t_plot, seg_plot, dec_note = _maybe_decimate(t, seg, max_pts)

    fig_w = 16 if (end - start) / args.sample_rate_hz >= 120.0 else 14
    fig, axes = plt.subplots(5, 1, figsize=(fig_w, 12), sharex=True, constrained_layout=True)
    lw = 0.25 if seg_plot.shape[0] > 50_000 else 0.8
    for k, name in enumerate(FINGER_NAMES):
        axes[k].plot(t_plot, seg_plot[:, k], color="C0", linewidth=lw, rasterized=True)
        axes[k].set_ylabel(f"{name}\n(finger {k})")
        axes[k].grid(True, alpha=0.3)
    axes[-1].set_xlabel("time (s)")
    t0 = float(t_plot[0])
    t1 = float(t_plot[-1])
    fig.suptitle(
        f"Raw train_dg ({args.sample_rate_hz} Hz){dec_note}\n"
        f"window [{t0:.2f}s, {t1:.2f}s], length {(end - start) / args.sample_rate_hz:.2f}s  "
        f"(file total {t_total_sec:.1f}s)",
        fontsize=11,
    )

    if args.out:
        out_name = args.out if args.out.lower().endswith(".png") else f"{args.out}.png"
    else:
        out_name = "finger_gt_full_waveform.png" if args.full else "finger_gt_raw_waveform.png"
    out = os.path.join(cfg.model_dir, out_name)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
