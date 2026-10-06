"""
实时模拟：仅使用 ``python -m src.train_nn`` 产出的 MLP 模型。
将同一 ``model_mlp.joblib`` 载入 simple/complex，使用 ``mode=simple``。
"""

from __future__ import annotations

import argparse
import os

from .run_pipeline_sim import run_pipeline_simulation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Realtime sim using MLP checkpoint (train_nn → model_mlp.joblib)."
    )
    parser.add_argument("--mat_path", required=True)
    parser.add_argument("--model_dir", type=str, default="artifacts")
    parser.add_argument("--model_path", type=str, default="", help="默认 <model_dir>/model_mlp.joblib")
    parser.add_argument("--manifest_path", type=str, default="")
    parser.add_argument("--n_channels", type=int, default=0)
    parser.add_argument("--window_ms", type=int, default=300)
    parser.add_argument("--step_ms", type=int, default=50)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="最多推进的采样点数（0=整段）；在 sim_segment=val 时为留出段全长",
    )
    parser.add_argument(
        "--plot_max_points",
        type=int,
        default=5000,
        help="图中最多绘制的预测点数（0=全部）；与 max_samples 无关，仅影响出图长度",
    )
    parser.add_argument(
        "--sim_segment",
        choices=("full", "val"),
        default="val",
        help="bciciv: val=仅后段留出数据上模拟(默认，与 train_nn 一致)；full=整条 train_data",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.2,
        help="与 train_nn --val_fraction 一致（ProjectConfig 默认 0.2）",
    )
    parser.add_argument("--latency", action="store_true")
    parser.add_argument("--save_features_npz", type=str, default="")
    parser.add_argument(
        "--legacy_fft_bands",
        action="store_true",
        help="须与 train_nn 一致：额外启用旧版三段 rFFT 带（默认仅用 Hz 生理频带）",
    )
    parser.add_argument("--no_physiological_bands", action="store_true")
    args = parser.parse_args()

    path = args.model_path.strip() or os.path.join(args.model_dir, "model_mlp.joblib")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"未找到 {path}，请先运行: python -m src.train_nn ... （或 --model_path 指向 MLP joblib）"
        )

    run_pipeline_simulation(
        mat_path=args.mat_path,
        model_simple=path,
        model_complex=path,
        model_dir=args.model_dir,
        manifest_path=args.manifest_path,
        n_channels=args.n_channels,
        window_ms=args.window_ms,
        step_ms=args.step_ms,
        max_samples=args.max_samples,
        mode="simple",
        save_features_npz=args.save_features_npz,
        latency=args.latency,
        plot_suptitle="Realtime sim — MLP only (train_nn)",
        plot_filename="realtime_plot_mlp_only.png",
        plot_max_points=args.plot_max_points,
        sim_segment=args.sim_segment,
        val_fraction=args.val_fraction,
        include_legacy_fft_bands=args.legacy_fft_bands,
        include_physiological_bands=not args.no_physiological_bands,
    )


if __name__ == "__main__":
    main()
