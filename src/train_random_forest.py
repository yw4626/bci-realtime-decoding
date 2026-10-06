"""
Random Forest 离线训练（每指独立 RandomForestRegressor）。
单 .mat 时间切分；数据与滑窗逻辑见 train_ridge。
"""

from __future__ import annotations

import argparse
import json
import os
import time

import joblib

from .models import MultiFingerModel
from .train_ridge import (
    add_mat_and_window_args,
    build_project_config_from_args,
    finger_metrics_dataframe,
    load_train_val_windows,
    profile_paths_meta,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Random Forest：单 .mat 时间划分。"
    )
    add_mat_and_window_args(parser)
    args = parser.parse_args()

    cfg = build_project_config_from_args(args)

    tv = load_train_val_windows(cfg, args.mat_path)
    os.makedirs(cfg.model_dir, exist_ok=True)

    t0 = time.perf_counter()
    model = MultiFingerModel("complex", cfg.random_state, cfg.ridge_alpha).fit(tv.x_tr, tv.y_tr)
    t1 = time.perf_counter()

    pred = model.predict(tv.x_val)
    result_df = finger_metrics_dataframe(tv.y_val, pred, "random_forest")
    result_csv = os.path.join(cfg.model_dir, "offline_metrics_random_forest.csv")
    result_df.to_csv(result_csv, index=False)

    model_path = os.path.join(cfg.model_dir, "model_complex.joblib")
    joblib.dump(model, model_path)

    profile = {
        "model": "random_forest",
        "data_source": "single_mat_temporal_split",
        "train_windows": int(len(tv.x_tr)),
        "val_windows": int(len(tv.x_val)),
        "train_seconds": t1 - t0,
        "val_fraction": cfg.val_fraction,
        "include_legacy_fft_bands": cfg.include_legacy_fft_bands,
        "include_physiological_bands": cfg.include_physiological_bands,
        "sample_rate_hz": cfg.sample_rate_hz,
        **profile_paths_meta(tv),
    }

    profile_path = os.path.join(cfg.model_dir, "train_profile_random_forest.json")
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

    print("Saved:", result_csv, model_path)
    print(result_df.groupby("family")[["rmse", "mae", "cosine_distance", "pearson_r"]].mean())


if __name__ == "__main__":
    main()
