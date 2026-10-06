"""
实时模拟：仅使用 ``python -m src.train_ridge`` 产出的 Ridge 模型。

``BCIDecodePipeline`` 需要两个 joblib 路径；将同一 ``model_simple.joblib`` 传给
simple/complex 槽位，``mode=simple`` 时只对 simple 模型调用 ``predict``，
与训练时的 ``MultiFingerModel("simple", ...)`` 一致。

示例：
  python -m src.train_ridge --mat_path sub1_comp.mat
  python -m src.run_realtime_sim_ridge_only --mat_path sub1_comp.mat

  python -m src.train_ridge
  python -m src.run_realtime_sim_ridge_only --mat_path sub1_comp.mat
"""

from __future__ import annotations

import argparse
import os

from .run_pipeline_sim import run_pipeline_simulation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Realtime pipeline sim using Ridge-only checkpoint (train_ridge → model_simple.joblib)."
    )
    parser.add_argument("--mat_path", required=True)
    parser.add_argument(
        "--model_dir",
        type=str,
        default="artifacts",
        help="与 train_ridge --model_dir 一致（默认写入 model_simple.joblib）",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="可选：直接指定 Ridge 模型 .joblib；默认使用 <model_dir>/model_simple.joblib",
    )
    parser.add_argument("--manifest_path", type=str, default="")
    parser.add_argument("--n_channels", type=int, default=0)
    parser.add_argument("--window_ms", type=int, default=300)
    parser.add_argument("--step_ms", type=int, default=50)
    parser.add_argument("--max_samples", type=int, default=0, help="0=整段（在 sim_segment=val 时为留出段全长）")
    parser.add_argument(
        "--sim_segment",
        choices=("full", "val"),
        default="val",
        help="val=仅留出后段(默认)；full=整条 train_data",
    )
    parser.add_argument("--val_fraction", type=float, default=0.2, help="与 train_ridge / ProjectConfig 一致")
    parser.add_argument(
        "--plot_max_points",
        type=int,
        default=0,
        help="图中最多绘制的预测点数（0=全部）；与 run_realtime_sim_nn_only 一致",
    )
    parser.add_argument("--latency", action="store_true")
    parser.add_argument("--save_features_npz", type=str, default="")
    parser.add_argument(
        "--legacy_fft_bands",
        action="store_true",
        help="额外启用旧版三段 rFFT 带（默认与 train_ridge 一致：仅 Hz 频带）",
    )
    parser.add_argument("--no_physiological_bands", action="store_true")
    args = parser.parse_args()

    path = args.model_path.strip() or os.path.join(args.model_dir, "model_simple.joblib")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"未找到 {path}，请先运行: python -m src.train_ridge ... "
            f"（或 --model_path 指向 Ridge MultiFingerModel joblib）"
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
        plot_suptitle="Realtime sim — Ridge only (train_ridge)",
        plot_filename="realtime_plot_ridge_only.png",
        plot_max_points=args.plot_max_points,
        sim_segment=args.sim_segment,
        val_fraction=args.val_fraction,
        include_legacy_fft_bands=args.legacy_fft_bands,
        include_physiological_bands=not args.no_physiological_bands,
    )


if __name__ == "__main__":
    main()
