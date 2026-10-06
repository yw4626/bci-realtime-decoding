"""
将 sub*.mat 中的 ECoG（每行一个时间点、每列一个通道）按顺序发布到 Pub/Sub，
消息格式与 Dataflow ``ParseAndKeyFn`` / ``DecodeSampleFn`` 一致：

  {"session_id": "...", "channels": [62 floats], "sample_seq": n}

``sample_seq`` 单调递增，便于检测乱序；反复重跑同一段时可加 ``--no_sample_seq``，
或每条用新 ``session_id``，或在 Dataflow 侧加 ``--no_sample_seq_check``。

用法示例（需已 ``gcloud auth application-default login`` 且本机安装 google-cloud-pubsub）::

  python -m src.publish_ecog_stream \\
    --project YOUR_PROJECT_ID \\
    --topic bci-ecog-in \\
    --mat_path "./data/sub1_comp.mat" \\
    --segment train \\
    --realtime
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
from typing import Any, Dict, List

import numpy as np
from google.cloud import pubsub_v1
from scipy.io import loadmat


def _topic_path(project: str, topic: str) -> str:
    if topic.startswith("projects/"):
        return topic
    return f"projects/{project}/topics/{topic}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish ECoG rows from .mat to Pub/Sub (BCICIV format).")
    parser.add_argument("--project", required=True, help="GCP 项目 ID（短名）")
    parser.add_argument(
        "--topic",
        required=True,
        help="输入 Topic：短名 bci-ecog-in 或完整 projects/PROJ/topics/bci-ecog-in",
    )
    parser.add_argument("--mat_path", required=True, help="sub1_comp.mat 等")
    parser.add_argument(
        "--segment",
        choices=("train", "test"),
        default="train",
        help="train=train_data（前 2/3）；test=test_data（后 1/3，无标签段仿真）",
    )
    parser.add_argument("--session_id", default="sub1", help="与 Dataflow 中有状态解码的 session 键一致")
    parser.add_argument("--start", type=int, default=0, help="从第几行开始（0-based）")
    parser.add_argument("--max_samples", type=int, default=0, help="最多发送行数，0=发到结尾")
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="按 1000Hz 间隔 sleep（近似实时）；不加则尽快发送",
    )
    parser.add_argument(
        "--no_sample_seq",
        action="store_true",
        help="不传 sample_seq（Dataflow 不做顺序校验）",
    )
    parser.add_argument(
        "--batch_wait_s",
        type=float,
        default=0.0,
        help="每批后额外暂停秒数（限流），可与 --realtime 同用",
    )
    parser.add_argument(
        "--flush_every",
        type=int,
        default=2000,
        metavar="N",
        help="每发出 N 条再在客户端等待一次完成（利用 Publisher 批量 RPC）；设为 1 等价于每条 publish 后立即 result（最慢）。",
    )
    args = parser.parse_args()
    if args.flush_every < 1:
        parser.error("--flush_every 须 >= 1")

    raw = loadmat(args.mat_path)
    key = "train_data" if args.segment == "train" else "test_data"
    if key not in raw:
        raise KeyError(f"{args.mat_path} 中未找到变量 {key}")
    x = np.asarray(raw[key], dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"{key} 应为 2 维 (time, channels)，得到 shape={x.shape}")
    n_ch = x.shape[1]
    print(f"Loaded {key}: shape={x.shape}, n_channels={n_ch}")

    topic_path = _topic_path(args.project, args.topic)
    publisher = pubsub_v1.PublisherClient()
    start = max(0, args.start)
    end = x.shape[0] if args.max_samples <= 0 else min(x.shape[0], start + args.max_samples)

    def _wait_futures(futs: List[concurrent.futures.Future[Any]]) -> None:
        if not futs:
            return
        concurrent.futures.wait(futs, return_when=concurrent.futures.ALL_COMPLETED)
        for f in futs:
            f.result()

    pending: List[concurrent.futures.Future[Any]] = []
    t0 = time.perf_counter()
    try:
        for i in range(start, end):
            row = x[i]
            payload: Dict[str, Any] = {
                "session_id": args.session_id,
                "channels": row.astype(float).tolist(),
            }
            if not args.no_sample_seq:
                payload["sample_seq"] = i - start + 1
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            pending.append(publisher.publish(topic_path, data))
            if len(pending) >= args.flush_every:
                _wait_futures(pending)
                pending.clear()
            if args.realtime:
                time.sleep(0.001)
            if args.batch_wait_s > 0 and (i - start + 1) % 1000 == 0:
                time.sleep(args.batch_wait_s)
        _wait_futures(pending)
    finally:
        publisher.stop()
    elapsed = time.perf_counter() - t0
    n = end - start
    rate = n / elapsed if elapsed > 0 else 0.0
    print(f"Published {n} messages to {topic_path} in {elapsed:.2f}s ({rate:.1f} msg/s, flush_every={args.flush_every})")


if __name__ == "__main__":
    main()
