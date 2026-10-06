"""
使用显式流水线（BCIDecodePipeline）在整段数据上模拟在线解码；
可选从 train_manifest.json 读取 channel_target，与多被试 artifacts 对齐。
"""

import argparse
import os
import time
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .config import ProjectConfig
from .data_loader import load_bci_mat, temporal_train_val_split
from .metrics import pearson_r
from .pipeline import BCIDecodePipeline, load_channel_target_from_manifest, run_stream_ticks

FINGER_NAMES = ("thumb", "index", "middle", "ring", "little")


def _finger_plot_ylim(gt: np.ndarray, pred: np.ndarray) -> Tuple[float, float]:
    """
    仅用 ground truth 估计纵轴范围，避免预测中的极端尖峰把 y 轴拉到很大，
    导致真实手指幅度被压成「贴在 0 的直线」。
    预测仍会照常绘制，超出范围的部分在轴边界处被裁剪，可看出异常尖峰。
    """
    g = np.asarray(gt, dtype=np.float64)
    g = g[np.isfinite(g)]
    if g.size == 0:
        p = np.asarray(pred, dtype=np.float64)
        p = p[np.isfinite(p)]
        if p.size == 0:
            return -1.0, 1.0
        lo, hi = np.percentile(p, [2.0, 98.0])
    else:
        lo, hi = np.percentile(g, [1.0, 99.0])
    span = float(max(hi - lo, 1e-6))
    pad = 0.08 * span
    return float(lo - pad), float(hi + pad)


def run_pipeline_simulation(
    mat_path: str,
    model_simple: str,
    model_complex: str,
    model_dir: str,
    manifest_path: str = "",
    n_channels: int = 0,
    window_ms: int = 300,
    step_ms: int = 50,
    max_samples: int = 0,
    mode: str = "blended",
    save_features_npz: str = "",
    latency: bool = False,
    plot_suptitle: str = "Ground truth vs prediction (pipeline)",
    plot_filename: str = "realtime_plot_pipeline.png",
    plot_max_points: int = 1200,
    sim_segment: str = "val",
    val_fraction: float = 0.2,
    include_legacy_fft_bands: bool = False,
    include_physiological_bands: bool = True,
) -> None:
    """
    在单个被试 BCICIV .mat（``train_data`` / ``train_dg``）上跑 BCIDecodePipeline。

    ``sim_segment``:
    - ``val``（默认）: 只取时间轴 **留出段**（与 ``train_nn`` / ``temporal_train_val_split`` 一致），
      避免与训练用的前 (1 - val_fraction) 段重叠。
    - ``full``: 使用整条 ``train_data`` / ``train_dg`` 连续回放。
    """
    cfg = ProjectConfig(
        window_ms=window_ms,
        step_ms=step_ms,
        model_dir=model_dir,
        include_legacy_fft_bands=include_legacy_fft_bands,
        include_physiological_bands=include_physiological_bands,
    )
    full_x, full_y, _ = load_bci_mat(mat_path)
    if sim_segment == "val":
        _, _, train_x, train_y = temporal_train_val_split(full_x, full_y, val_fraction)
        print(
            f"Sim segment: held-out tail only - last {val_fraction:.0%} of train_data/train_dg "
            f"({len(train_x)} samples); matches train_nn val split when val_fraction is the same."
        )
    else:
        train_x, train_y = full_x, full_y
        print(f"Sim segment: full train_data/train_dg ({len(train_x)} samples)")

    ch_from_manifest = load_channel_target_from_manifest(manifest_path) if manifest_path else None
    n_ch = n_channels if n_channels > 0 else ch_from_manifest

    pipeline = BCIDecodePipeline(cfg, model_simple, model_complex, n_channels=n_ch)

    decode_latency_ms: Optional[List[float]] = [] if latency else None
    t0 = time.perf_counter()
    states, preds, gts, feats = run_stream_ticks(
        train_x,
        train_y,
        pipeline,
        mode=mode,
        max_samples=max_samples,
        decode_latency_ms=decode_latency_ms,
    )
    elapsed = time.perf_counter() - t0

    corr = pearson_r(gts.reshape(-1), preds.reshape(-1)) if len(preds) else float("nan")
    print(f"Ticks: {len(states)}, predictions: {len(preds)}, channel_target={n_ch}")
    print(f"Wall time (total): {elapsed:.3f}s")
    if decode_latency_ms is not None and len(decode_latency_ms):
        arr = np.asarray(decode_latency_ms, dtype=np.float64)
        print(f"Decode latency(ms): p50={np.percentile(arr, 50):.3f}, p95={np.percentile(arr, 95):.3f}")
    print(f"Pearson r (overall): {corr:.4f}")
    for k, name in enumerate(FINGER_NAMES):
        if len(preds):
            rk = pearson_r(gts[:, k], preds[:, k])
            print(f"  finger {k} ({name}): Pearson r = {rk:.4f}")

    os.makedirs(cfg.model_dir, exist_ok=True)
    if plot_max_points <= 0:
        show_n = len(preds)
    else:
        show_n = min(plot_max_points, len(preds))
    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True, constrained_layout=True)
    for k, name in enumerate(FINGER_NAMES):
        ax = axes[k]
        if show_n > 0:
            gk = gts[:show_n, k]
            pk = preds[:show_n, k]
            ax.plot(gk, label="ground_truth", color="C0", alpha=0.85)
            ax.plot(pk, label="prediction", color="C1", alpha=0.85)
            y0, y1 = _finger_plot_ylim(gk, pk)
            ax.set_ylim(y0, y1)
        ax.set_ylabel(name)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(f"Pipeline sim: {name} (finger {k})")
    axes[-1].set_xlabel("prediction index")
    title = (
        f"{plot_suptitle}\n(sim: held-out {val_fraction:.0%} time tail; no overlap with train windows)"
        if sim_segment == "val"
        else plot_suptitle
    )
    fig.suptitle(title, fontsize=12)
    fig_path = os.path.join(cfg.model_dir, plot_filename)
    fig.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {fig_path}")

    if save_features_npz and feats:
        np.savez_compressed(save_features_npz, features=np.stack(feats, axis=0))
        print(f"Saved features: {save_features_npz} shape={np.stack(feats, axis=0).shape}")


def main():
    parser = argparse.ArgumentParser(description="Simulate decode pipeline with optional manifest channel alignment.")
    parser.add_argument("--mat_path", required=True, help="BCICIV 单被试 .mat（train_data / train_dg）")
    parser.add_argument("--model_dir", type=str, default="artifacts")
    parser.add_argument(
        "--manifest_path",
        type=str,
        default="",
        help="Optional train_manifest.json (e.g. artifacts_multi/train_manifest.json) for channel_target.",
    )
    parser.add_argument("--n_channels", type=int, default=0, help="Override channel width (0 = use manifest or raw).")
    parser.add_argument("--window_ms", type=int, default=300)
    parser.add_argument("--step_ms", type=int, default=50)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="最多推进的采样点数（0=当前模拟段全长）",
    )
    parser.add_argument(
        "--plot_max_points",
        type=int,
        default=1200,
        help="图中最多绘制的预测点数（0=全部）；与 max_samples 无关，仅影响出图长度",
    )
    parser.add_argument("--mode", choices=("simple", "complex", "blended"), default="blended")
    parser.add_argument("--save_features_npz", type=str, default="", help="Optional path to save stacked feature rows.")
    parser.add_argument(
        "--latency",
        action="store_true",
        help="Print per-decode-step latency p50/p95 (ms) for steps that emit a prediction.",
    )
    parser.add_argument(
        "--sim_segment",
        choices=("full", "val"),
        default="val",
        help="val=仅时间轴留出后段(默认，与 train_nn 一致)；full=整条 train_data。",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.2,
        help="与 train_nn 一致；sim_segment=val 时使用后若干比例作为模拟段。",
    )
    parser.add_argument(
        "--legacy_fft_bands",
        action="store_true",
        help="与训练时一致：额外启用旧版三段 rFFT 带功率（默认仅用 Hz 生理频带）。",
    )
    parser.add_argument(
        "--no_physiological_bands",
        action="store_true",
        help="与训练时一致：关闭 alpha/beta/low-gamma/high-gamma 等 Hz 频带功率。",
    )
    args = parser.parse_args()

    cfg = ProjectConfig(
        window_ms=args.window_ms,
        step_ms=args.step_ms,
        model_dir=args.model_dir,
        include_legacy_fft_bands=args.legacy_fft_bands,
        include_physiological_bands=not args.no_physiological_bands,
    )
    model_simple = os.path.join(cfg.model_dir, "model_simple.joblib")
    model_complex = os.path.join(cfg.model_dir, "model_complex.joblib")
    run_pipeline_simulation(
        mat_path=args.mat_path,
        model_simple=model_simple,
        model_complex=model_complex,
        model_dir=args.model_dir,
        manifest_path=args.manifest_path,
        n_channels=args.n_channels,
        window_ms=args.window_ms,
        step_ms=args.step_ms,
        max_samples=args.max_samples,
        mode=args.mode,
        save_features_npz=args.save_features_npz,
        latency=args.latency,
        plot_max_points=args.plot_max_points,
        sim_segment=args.sim_segment,
        val_fraction=args.val_fraction,
        include_legacy_fft_bands=cfg.include_legacy_fft_bands,
        include_physiological_bands=cfg.include_physiological_bands,
    )


if __name__ == "__main__":
    main()
