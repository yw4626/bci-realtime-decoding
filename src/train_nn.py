"""
MLP 神经网络离线训练（每指独立 MLPRegressor）。
单个被试 BCICIV .mat（train_data / train_dg），按 val_fraction 时间切分。

默认特征与 ``train_ridge`` 一致：mean/std/RMS + alpha/beta/low_gamma/high_gamma（Hz）；
可选 ``--legacy_fft_bands`` 追加旧版三段 rFFT 带。
"""

from __future__ import annotations

import argparse
import json
import os
import time

import joblib

from .models_nn import MultiFingerMLP
from .plot_nn_training import save_nn_training_figures
from .train_ridge import (
    add_mat_and_window_args,
    build_project_config_from_args,
    finger_metrics_dataframe,
    load_train_val_windows,
    profile_paths_meta,
)


def _parse_hidden(s: str):
    vals = [int(v.strip()) for v in s.split(",") if v.strip()]
    if not vals:
        raise ValueError("--hidden 不能为空，例如 256,128")
    return tuple(vals)


def main() -> None:
    parser = argparse.ArgumentParser(description="MLP 回归：单 .mat 时间切分。")
    add_mat_and_window_args(parser)
    parser.add_argument("--hidden", type=str, default="256,128", help="隐藏层配置，例如 256,128")
    parser.add_argument("--activation", choices=("relu", "tanh", "logistic"), default="relu")
    parser.add_argument("--alpha", type=float, default=1e-4, help="L2 正则")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--max_iter", type=int, default=200, help="每指网络最大迭代次数")
    parser.add_argument(
        "--no_training_plots",
        action="store_true",
        help="不生成训练结果图（默认写入 training_metrics_mlp.png 与 training_val_curves_mlp.png）",
    )
    parser.add_argument(
        "--training_plot_max_points",
        type=int,
        default=4000,
        help="验证集曲线图最多绘制的窗口数（0=全部）",
    )
    args = parser.parse_args()

    cfg = build_project_config_from_args(args)
    tv = load_train_val_windows(cfg, args.mat_path)
    os.makedirs(cfg.model_dir, exist_ok=True)

    hidden = _parse_hidden(args.hidden)
    model = MultiFingerMLP(
        hidden_layer_sizes=hidden,
        activation=args.activation,
        alpha=args.alpha,
        learning_rate_init=args.lr,
        max_iter=args.max_iter,
        random_state=cfg.random_state,
    )

    t0 = time.perf_counter()
    model.fit(tv.x_tr, tv.y_tr)
    t1 = time.perf_counter()

    pred = model.predict(tv.x_val)
    result_df = finger_metrics_dataframe(tv.y_val, pred, "mlp")
    result_csv = os.path.join(cfg.model_dir, "offline_metrics_mlp.csv")
    result_df.to_csv(result_csv, index=False)

    model_path = os.path.join(cfg.model_dir, "model_mlp.joblib")
    joblib.dump(model, model_path)

    profile = {
        "model": "mlp",
        "data_source": "single_mat_temporal_split",
        "train_windows": int(len(tv.x_tr)),
        "val_windows": int(len(tv.x_val)),
        "max_train_windows": cfg.max_train_windows,
        "max_val_windows": cfg.max_val_windows,
        "train_window_sampling": cfg.train_window_sampling,
        "train_seconds": t1 - t0,
        "hidden_layer_sizes": list(hidden),
        "activation": args.activation,
        "alpha": args.alpha,
        "learning_rate_init": args.lr,
        "max_iter": args.max_iter,
        "val_fraction": cfg.val_fraction,
        "include_legacy_fft_bands": cfg.include_legacy_fft_bands,
        "include_physiological_bands": cfg.include_physiological_bands,
        "sample_rate_hz": cfg.sample_rate_hz,
        **profile_paths_meta(tv),
    }

    profile_path = os.path.join(cfg.model_dir, "train_profile_mlp.json")
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

    print("Saved:", result_csv, model_path)
    if not args.no_training_plots:
        tmax = args.training_plot_max_points
        if tmax <= 0:
            tmax = len(tv.y_val)
        pm, pt = save_nn_training_figures(tv.y_val, pred, result_df, cfg.model_dir, trace_max_points=tmax)
        print("Saved:", pm, pt)
    print(result_df.groupby("family")[["rmse", "mae", "cosine_distance", "pearson_r"]].mean())


if __name__ == "__main__":
    main()
