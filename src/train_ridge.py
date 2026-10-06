"""
Ridge 回归离线训练（每指独立 Ridge）。
仅支持单个被试 BCICIV .mat（train_data / train_dg），按 val_fraction 做时间切分。

特征与 ``train_nn`` 相同：默认每通道 mean/std/RMS + Hz 频带功率
（alpha 8–13、beta 13–30、low_gamma 30–70、high_gamma 70–150）；
可选 ``--legacy_fft_bands`` 追加旧版三段 rFFT 带。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from .config import ProjectConfig
from .data_loader import load_bci_mat, temporal_train_val_split
from .features import build_supervised_windows
from .metrics import eval_finger_metrics
from .models import MultiFingerModel


@dataclass
class TrainValWindows:
    x_tr: np.ndarray
    y_tr: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    mat_path: str


def add_mat_and_window_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mat_path",
        required=True,
        help="单个 BCICIV .mat（train_data / train_dg），按 val_fraction 时间切分",
    )
    parser.add_argument("--window_ms", type=int, default=300)
    parser.add_argument("--step_ms", type=int, default=50)
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.2,
        help="时间切分：末尾验证段占比（与 run_pipeline_sim --val_fraction 保持一致）",
    )
    parser.add_argument(
        "--max_train_windows",
        type=int,
        default=0,
        help="训练滑窗上限（0=使用训练段内全部滑窗）；>0 时按 --train_window_sampling 在全时段抽样",
    )
    parser.add_argument(
        "--max_val_windows",
        type=int,
        default=2000,
        help="验证滑窗上限（0=验证段全部滑窗）；>0 时按 --train_window_sampling 抽样",
    )
    parser.add_argument(
        "--train_window_sampling",
        choices=("sequential", "uniform", "stratified"),
        default="stratified",
        help="限额时如何选窗：sequential=只取最前段；uniform=随机；stratified=按时间分桶（推荐）",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="artifacts",
        help="模型与指标输出目录",
    )
    parser.add_argument(
        "--legacy_fft_bands",
        action="store_true",
        help="额外加入旧版按 rFFT bin 索引的三段带功率（默认仅用 Hz：alpha/beta/low_gamma/high_gamma）",
    )
    parser.add_argument(
        "--no_physiological_bands",
        action="store_true",
        help="不使用按 Hz 的 alpha/beta/low-gamma/high-gamma 等频带功率",
    )


def load_train_val_windows(cfg: ProjectConfig, mat_path: str) -> TrainValWindows:
    train_x, train_y, _ = load_bci_mat(mat_path)
    x_tr_raw, y_tr_raw, x_val_raw, y_val_raw = temporal_train_val_split(train_x, train_y, cfg.val_fraction)

    x_tr, y_tr = build_supervised_windows(
        x_tr_raw,
        y_tr_raw,
        cfg.window_size,
        cfg.step_size,
        cfg.max_train_windows,
        sampling=cfg.train_window_sampling,
        random_state=cfg.random_state,
        sample_rate_hz=cfg.sample_rate_hz,
        include_legacy_fft_bands=cfg.include_legacy_fft_bands,
        include_physiological_bands=cfg.include_physiological_bands,
    )
    x_val, y_val = build_supervised_windows(
        x_val_raw,
        y_val_raw,
        cfg.window_size,
        cfg.step_size,
        cfg.max_val_windows,
        sampling=cfg.train_window_sampling,
        random_state=cfg.random_state + 999,
        sample_rate_hz=cfg.sample_rate_hz,
        include_legacy_fft_bands=cfg.include_legacy_fft_bands,
        include_physiological_bands=cfg.include_physiological_bands,
    )

    return TrainValWindows(x_tr=x_tr, y_tr=y_tr, x_val=x_val, y_val=y_val, mat_path=mat_path)


def build_project_config_from_args(args: argparse.Namespace) -> ProjectConfig:
    return ProjectConfig(
        window_ms=args.window_ms,
        step_ms=args.step_ms,
        val_fraction=args.val_fraction,
        max_train_windows=args.max_train_windows,
        max_val_windows=args.max_val_windows,
        train_window_sampling=args.train_window_sampling,
        model_dir=args.model_dir,
        include_legacy_fft_bands=getattr(args, "legacy_fft_bands", False),
        include_physiological_bands=not getattr(args, "no_physiological_bands", False),
    )


def finger_metrics_dataframe(y_val: np.ndarray, pred: np.ndarray, family: str) -> pd.DataFrame:
    rows = []
    for finger in range(y_val.shape[1]):
        m = eval_finger_metrics(y_val[:, finger], pred[:, finger])
        rows.append({"family": family, "finger": finger, **m})
    avg = eval_finger_metrics(y_val.reshape(-1), pred.reshape(-1))
    rows.append({"family": family, "finger": "avg", **avg})
    return pd.DataFrame(rows)


def profile_paths_meta(tv: TrainValWindows) -> dict:
    return {"mat_path": os.path.abspath(tv.mat_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Ridge 回归：单 .mat 时间划分。")
    add_mat_and_window_args(parser)
    parser.add_argument("--ridge_alpha", type=float, default=1.0, help="Ridge 正则强度")
    args = parser.parse_args()

    cfg = build_project_config_from_args(args)
    cfg.ridge_alpha = args.ridge_alpha

    tv = load_train_val_windows(cfg, args.mat_path)
    os.makedirs(cfg.model_dir, exist_ok=True)

    t0 = time.perf_counter()
    model = MultiFingerModel("simple", cfg.random_state, cfg.ridge_alpha).fit(tv.x_tr, tv.y_tr)
    t1 = time.perf_counter()

    pred = model.predict(tv.x_val)
    result_df = finger_metrics_dataframe(tv.y_val, pred, "ridge")
    result_csv = os.path.join(cfg.model_dir, "offline_metrics_ridge.csv")
    result_df.to_csv(result_csv, index=False)

    model_path = os.path.join(cfg.model_dir, "model_simple.joblib")
    joblib.dump(model, model_path)

    profile = {
        "model": "ridge",
        "data_source": "single_mat_temporal_split",
        "train_windows": int(len(tv.x_tr)),
        "val_windows": int(len(tv.x_val)),
        "max_train_windows": cfg.max_train_windows,
        "max_val_windows": cfg.max_val_windows,
        "train_window_sampling": cfg.train_window_sampling,
        "train_seconds": t1 - t0,
        "ridge_alpha": cfg.ridge_alpha,
        "val_fraction": cfg.val_fraction,
        "include_legacy_fft_bands": cfg.include_legacy_fft_bands,
        "include_physiological_bands": cfg.include_physiological_bands,
        "sample_rate_hz": cfg.sample_rate_hz,
        **profile_paths_meta(tv),
    }

    profile_path = os.path.join(cfg.model_dir, "train_profile_ridge.json")
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

    print("Saved:", result_csv, model_path)
    print(result_df.groupby("family")[["rmse", "mae", "cosine_distance", "pearson_r"]].mean())


if __name__ == "__main__":
    main()
