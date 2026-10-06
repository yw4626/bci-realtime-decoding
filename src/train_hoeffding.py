"""
Hoeffding 树回归（流式 / VFDT 同类）离线训练：逐窗 ``learn_one``，导出 ``model_hoeffding.joblib``。

需先安装：``pip install -r requirements-streaming-tree.txt``
"""

from __future__ import annotations

import argparse
import json
import os
import time

import joblib

from .models_hoeffding import MultiFingerHoeffdingRegressor
from .train_ridge import (
    add_mat_and_window_args,
    build_project_config_from_args,
    finger_metrics_dataframe,
    load_train_val_windows,
    profile_paths_meta,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hoeffding 树回归（River）：单 .mat 时间切分。"
    )
    add_mat_and_window_args(parser)
    parser.add_argument("--grace_period", type=int, default=200, help="分裂前叶内最少样本数")
    parser.add_argument("--delta", type=float, default=1e-6, help="Hoeffding 置信参数（越小越保守）")
    parser.add_argument("--max_size", type=int, default=30, help="每叶最大样本缓冲（River max_size）")
    args = parser.parse_args()

    cfg = build_project_config_from_args(args)

    tv = load_train_val_windows(cfg, args.mat_path)
    os.makedirs(cfg.model_dir, exist_ok=True)

    model = MultiFingerHoeffdingRegressor(
        grace_period=args.grace_period,
        delta=args.delta,
        max_size=args.max_size,
    )
    t0 = time.perf_counter()
    model.fit(tv.x_tr, tv.y_tr)
    t1 = time.perf_counter()

    pred = model.predict(tv.x_val)
    result_df = finger_metrics_dataframe(tv.y_val, pred, "hoeffding_tree")
    result_csv = os.path.join(cfg.model_dir, "offline_metrics_hoeffding.csv")
    result_df.to_csv(result_csv, index=False)

    model_path = os.path.join(cfg.model_dir, "model_hoeffding.joblib")
    joblib.dump(model, model_path)

    profile = {
        "model": "hoeffding_tree",
        "data_source": "single_mat_temporal_split",
        "train_windows": int(len(tv.x_tr)),
        "val_windows": int(len(tv.x_val)),
        "train_seconds": t1 - t0,
        "grace_period": args.grace_period,
        "delta": args.delta,
        "max_size": args.max_size,
        "val_fraction": cfg.val_fraction,
        "include_legacy_fft_bands": cfg.include_legacy_fft_bands,
        "include_physiological_bands": cfg.include_physiological_bands,
        "sample_rate_hz": cfg.sample_rate_hz,
        **profile_paths_meta(tv),
    }

    profile_path = os.path.join(cfg.model_dir, "train_profile_hoeffding.json")
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

    print("Saved:", result_csv, model_path)
    print(result_df.groupby("family")[["rmse", "mae", "cosine_distance", "pearson_r"]].mean())


if __name__ == "__main__":
    main()
