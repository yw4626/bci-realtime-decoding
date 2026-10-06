"""
实时模拟：仅使用 ``python -m src.train_hoeffding`` 产出的 Hoeffding 树模型。

与 RF-only 相同，将同一 ``model_hoeffding.joblib`` 载入 simple/complex 槽位，``mode=complex``
只走第二路预测（两路相同故结果一致）。

需：``pip install -r requirements-streaming-tree.txt``
"""

from __future__ import annotations

import argparse
import os

from .run_pipeline_sim import run_pipeline_simulation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Realtime sim using Hoeffding trees (train_hoeffding → model_hoeffding.joblib)."
    )
    parser.add_argument("--mat_path", required=True)
    parser.add_argument("--model_dir", type=str, default="artifacts")
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="默认 <model_dir>/model_hoeffding.joblib",
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
        help="bciciv: val=仅留出后段(默认)；full=整条 train_data",
    )
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--latency", action="store_true")
    parser.add_argument("--save_features_npz", type=str, default="")
    parser.add_argument("--legacy_fft_bands", action="store_true")
    parser.add_argument("--no_physiological_bands", action="store_true")
    args = parser.parse_args()

    path = args.model_path.strip() or os.path.join(args.model_dir, "model_hoeffding.joblib")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"未找到 {path}，请先: pip install -r requirements-streaming-tree.txt\n"
            f"  然后: python -m src.train_hoeffding ..."
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
        mode="complex",
        save_features_npz=args.save_features_npz,
        latency=args.latency,
        plot_suptitle="Realtime sim — Hoeffding tree (VFDT-class, train_hoeffding)",
        plot_filename="realtime_plot_hoeffding_only.png",
        sim_segment=args.sim_segment,
        val_fraction=args.val_fraction,
        include_legacy_fft_bands=args.legacy_fft_bands,
        include_physiological_bands=not args.no_physiological_bands,
    )


if __name__ == "__main__":
    main()
