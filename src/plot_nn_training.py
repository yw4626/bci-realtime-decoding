"""
NN（MLP）离线训练结果可视化：验证集指标柱状图 + 验证集 GT/预测曲线。
"""

from __future__ import annotations

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .run_pipeline_sim import FINGER_NAMES, _finger_plot_ylim

METRICS_PNG = "training_metrics_mlp.png"
TRACES_PNG = "training_val_curves_mlp.png"


def plot_metrics_bars(result_df: pd.DataFrame, out_path: str, title: str = "MLP offline validation metrics") -> None:
    """按手指绘制 RMSE / MAE / Pearson r 分组柱状图（含 avg 汇总列）。"""
    fam = result_df["family"].iloc[0]
    df = result_df[result_df["family"] == fam]
    ordered: list[pd.Series] = []
    for fi in range(5):
        hit = df[df["finger"] == fi]
        if not hit.empty:
            ordered.append(hit.iloc[0])
    hit_avg = df[df["finger"] == "avg"]
    if not hit_avg.empty:
        ordered.append(hit_avg.iloc[0])
    if not ordered:
        return
    labels: list[str] = []
    for r in ordered:
        fi = r["finger"]
        if fi == "avg" or (isinstance(fi, str) and fi.lower() == "avg"):
            labels.append("avg")
        else:
            labels.append(FINGER_NAMES[int(fi)])
    x = np.arange(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    rmse_v = [float(r["rmse"]) for r in ordered]
    mae_v = [float(r["mae"]) for r in ordered]
    pr_v = [float(r["pearson_r"]) for r in ordered]
    ax.bar(x - w, rmse_v, width=w, label="RMSE", color="C0")
    ax.bar(x, mae_v, width=w, label="MAE", color="C1")
    ax.bar(x + w, pr_v, width=w, label="Pearson r", color="C2")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("value")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_val_traces(
    y_val: np.ndarray,
    pred: np.ndarray,
    out_path: str,
    max_points: int = 4000,
    title: str = "MLP validation: ground truth vs prediction",
) -> None:
    """验证集上前若干时间步的五指 GT / 预测对比（纵轴按 GT 定标，与实时仿真图一致）。"""
    y_val = np.asarray(y_val, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    n = min(len(y_val), len(pred), max_points if max_points > 0 else len(y_val))
    if n <= 0:
        return
    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True, constrained_layout=True)
    for k, name in enumerate(FINGER_NAMES):
        ax = axes[k]
        gk = y_val[:n, k]
        pk = pred[:n, k]
        ax.plot(gk, label="ground_truth", color="C0", alpha=0.85)
        ax.plot(pk, label="prediction", color="C1", alpha=0.85)
        y0, y1 = _finger_plot_ylim(gk, pk)
        ax.set_ylim(y0, y1)
        ax.set_ylabel(name)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(f"Validation: {name} (finger {k})")
    axes[-1].set_xlabel("window index (validation)")
    fig.suptitle(title, fontsize=12)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_nn_training_figures(
    y_val: np.ndarray,
    pred: np.ndarray,
    result_df: pd.DataFrame,
    model_dir: str,
    trace_max_points: int = 4000,
) -> tuple[str, str]:
    """保存指标图与验证曲线图，返回写入的文件路径。"""
    os.makedirs(model_dir, exist_ok=True)
    p_metrics = os.path.join(model_dir, METRICS_PNG)
    p_traces = os.path.join(model_dir, TRACES_PNG)
    plot_metrics_bars(result_df, p_metrics)
    plot_val_traces(y_val, pred, p_traces, max_points=trace_max_points)
    return p_metrics, p_traces


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NN 训练结果图：从 CSV 画指标柱图，或从已有 checkpoint 画验证集 GT/预测曲线。"
    )
    parser.add_argument("--metrics_csv", default="", help="offline_metrics_mlp.csv，画柱状图")
    parser.add_argument("--out", default="", help="柱状图输出路径（默认与 CSV 同目录）")
    parser.add_argument("--model_dir", default="artifacts", help="与 train_nn --model_dir 一致")
    parser.add_argument("--mat_path", default="", help="与训练时相同的 .mat，用于重画验证曲线")
    parser.add_argument(
        "--training_plot_max_points",
        type=int,
        default=4000,
        help="验证曲线最多窗口数（0=全部）",
    )
    args = parser.parse_args()

    if args.metrics_csv:
        df = pd.read_csv(args.metrics_csv)
        out = args.out.strip() or os.path.join(os.path.dirname(os.path.abspath(args.metrics_csv)), METRICS_PNG)
        plot_metrics_bars(df, out, title="MLP offline validation metrics (from CSV)")
        print("Saved:", out)

    if args.mat_path:
        import argparse as ap
        import joblib

        from .train_ridge import add_mat_and_window_args, build_project_config_from_args, load_train_val_windows

        p = ap.ArgumentParser()
        add_mat_and_window_args(p)
        ns = p.parse_args(["--mat_path", args.mat_path, "--model_dir", args.model_dir])
        cfg = build_project_config_from_args(ns)
        tv = load_train_val_windows(cfg, ns.mat_path)
        model_path = os.path.join(cfg.model_dir, "model_mlp.joblib")
        model = joblib.load(model_path)
        pred = model.predict(tv.x_val)
        tmax = args.training_plot_max_points
        if tmax <= 0:
            tmax = len(tv.y_val)
        out_t = os.path.join(cfg.model_dir, TRACES_PNG)
        plot_val_traces(tv.y_val, pred, out_t, max_points=tmax)
        print("Saved:", out_t)

    if not args.metrics_csv and not args.mat_path:
        parser.error("请指定 --metrics_csv 和/或 --mat_path")


if __name__ == "__main__":
    main()
