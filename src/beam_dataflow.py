"""
Apache Beam 流式作业：将 BCIDecodePipeline 嵌入 Dataflow / DirectRunner。

实现拆至 ``src.beam``；本模块保留 CLI 入口 ``python -m src.beam_dataflow``。

输入（每行 JSON；Pub/Sub 载荷为 UTF-8）。sub1 示例：62 维 ``channels`` 对应单时刻 ECoG：

  {"session_id": "sub1", "channels": [<62 floats>], "sample_seq": 123}

``sample_seq`` / ``seq`` 可选；默认须严格递增，否则消息进死信。发布端反复从 1 重跑时可加
  ``--no_sample_seq_check``（或每条用新 ``session_id`` / 发布端 ``--no_sample_seq``）。

输出（仅 ``stage=decoded`` 写入主 Topic）：五指连续量，例如

  {"session_id": "sub1", "t_index": 6000, "prediction": [t, i, m, r, l], ...}

若项目组织策略禁止 VM 公网 IP（constraints/compute.vmExternalIpAccess），需加
  --no_use_public_ips
并保证 Worker 所用子网已启用 Private Google Access（否则无法访问 GCS/Pub/Sub）。

部署 Dataflow 示例：
  python -m src.beam_dataflow \\
    --runner DataflowRunner \\
    --project YOUR_PROJECT \\
    --region us-central1 \\
    --temp_location gs://YOUR_BUCKET/tmp \\
    --staging_location gs://YOUR_BUCKET/staging \\
    --input_subscription projects/YOUR_PROJECT/subscriptions/bci-samples \\
    --model_simple gs://YOUR_BUCKET/artifacts/model_simple.joblib \\
    --model_complex gs://YOUR_BUCKET/artifacts/model_mlp.joblib \\
    --streaming
"""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
from typing import List, Optional

import apache_beam as beam
from apache_beam.options.pipeline_options import GoogleCloudOptions, PipelineOptions, SetupOptions, StandardOptions

from .beam.builder import build_beam_pipeline


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Beam / Dataflow wrapper for BCIDecodePipeline.")
    parser.add_argument("--input_path", default="", help="本地或 GCS 文本路径，每行一条 JSON（非 streaming 回放）。")
    parser.add_argument("--input_subscription", default="", help="Pub/Sub 订阅 resource 名（流式）。")
    parser.add_argument("--output_path", default="", help="输出前缀（WriteToText），与 Pub/Sub 二选一。")
    parser.add_argument("--output_topic", default="", help="Pub/Sub topic resource 名。")
    parser.add_argument(
        "--deadletter_path",
        default="",
        help="坏消息输出前缀（WriteToText），与 --deadletter_topic 二选一；均未指定时默认写到系统临时目录。",
    )
    parser.add_argument("--deadletter_topic", default="", help="坏消息 Pub/Sub topic。")
    parser.add_argument("--window_ms", type=int, default=300)
    parser.add_argument("--step_ms", type=int, default=50)
    parser.add_argument("--model_simple", required=True, help="simple 分支模型（建议 Ridge: model_simple.joblib）")
    parser.add_argument("--model_complex", required=True, help="complex 分支模型（建议 MLP: model_mlp.joblib）")
    parser.add_argument("--n_channels", type=int, default=0)
    parser.add_argument("--mode", choices=("simple", "complex", "blended"), default="blended")
    parser.add_argument(
        "--blend_alpha",
        type=float,
        default=0.6,
        help="blended 模式下融合权重（与 BCIDecodePipeline / 离线评估一致）。",
    )
    parser.add_argument("--gcs_model_cache_dir", default="", help="worker 上下载 gs:// 模型的目录，默认系统临时目录下子文件夹。")
    parser.add_argument(
        "--legacy_fft_bands",
        action="store_true",
        help="与离线训练一致：额外启用旧版三段 rFFT 带功率（须与 checkpoint 一致）。",
    )
    parser.add_argument(
        "--no_physiological_bands",
        action="store_true",
        help="与离线训练一致：关闭 Hz 生理频带功率特征。",
    )
    parser.add_argument(
        "--no_sample_seq_check",
        action="store_true",
        help="不校验 sample_seq/seq 顺序，也不把序号写入状态；避免发布端从 1 重跑时出现 out_of_order 死信。"
        " 新一段录制仍建议用新 session_id，以免滑窗缓冲与上一段混在一起。",
    )
    known, beam_args = parser.parse_known_args(argv)
    if known.input_subscription and known.input_path:
        parser.error("不要同时指定 --input_subscription 与 --input_path。")
    if not known.input_subscription and not known.input_path:
        parser.error("请指定 --input_subscription（流式）或 --input_path（有界回放）之一。")
    if known.deadletter_path and known.deadletter_topic:
        parser.error("不要同时指定 --deadletter_path 与 --deadletter_topic。")

    deadletter_path = known.deadletter_path
    deadletter_topic = known.deadletter_topic
    if not deadletter_path and not deadletter_topic:
        deadletter_path = os.path.join(tempfile.gettempdir(), "bci_beam_deadletter")
        logging.warning("未指定 --deadletter_path / --deadletter_topic，死信默认写入前缀: %s", deadletter_path)

    options = PipelineOptions(beam_args)
    std = options.view_as(StandardOptions)
    if known.input_subscription and not std.streaming:
        std.streaming = True

    gcs_opts = options.view_as(GoogleCloudOptions)
    if gcs_opts.project is None and os.environ.get("GOOGLE_CLOUD_PROJECT"):
        gcs_opts.project = os.environ.get("GOOGLE_CLOUD_PROJECT")

    setup_opts = options.view_as(SetupOptions)
    if setup_opts.requirements_file is None:
        req = os.path.join(os.path.dirname(os.path.dirname(__file__)), "requirements-dataflow.txt")
        if os.path.isfile(req):
            setup_opts.requirements_file = req

    with beam.Pipeline(options=options) as p:
        logging.info(
            "Beam decode start: mode=%s blend_alpha=%.4f window_ms=%d step_ms=%d strict_sample_seq=%s model_simple=%s model_complex=%s",
            known.mode,
            known.blend_alpha,
            known.window_ms,
            known.step_ms,
            not known.no_sample_seq_check,
            known.model_simple,
            known.model_complex,
        )
        build_beam_pipeline(
            p,
            input_path=known.input_path,
            input_subscription=known.input_subscription,
            output_path=known.output_path,
            output_topic=known.output_topic,
            window_ms=known.window_ms,
            step_ms=known.step_ms,
            model_simple=known.model_simple,
            model_complex=known.model_complex,
            n_channels=known.n_channels,
            mode=known.mode,
            gcs_model_cache_dir=known.gcs_model_cache_dir,
            deadletter_path=deadletter_path,
            deadletter_topic=deadletter_topic,
            blend_alpha=known.blend_alpha,
            include_legacy_fft_bands=known.legacy_fft_bands,
            include_physiological_bands=not known.no_physiological_bands,
            strict_sample_seq=not known.no_sample_seq_check,
        )


if __name__ == "__main__":
    main()
