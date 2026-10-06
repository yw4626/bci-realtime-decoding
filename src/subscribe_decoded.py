"""
订阅 Dataflow 输出 Topic，打印解码结果 JSON 中的 ``prediction``（五指数值）。

示例::

  python -m src.subscribe_decoded \\
    --project YOUR_PROJECT_ID \\
    --subscription bci-decoded-pull \\
    --max_messages 50
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict

from google.cloud import pubsub_v1


def _subscription_path(project: str, subscription: str) -> str:
    if subscription.startswith("projects/"):
        return subscription
    return f"projects/{project}/subscriptions/{subscription}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull decoded finger predictions from Pub/Sub.")
    parser.add_argument("--project", required=True)
    parser.add_argument(
        "--subscription",
        required=True,
        help="订阅短名或完整 resource projects/PROJ/subscriptions/SUB",
    )
    parser.add_argument("--max_messages", type=int, default=20, help="最多拉取条数后退出")
    parser.add_argument(
        "--timeout_s",
        type=float,
        default=60.0,
        help="单条 pull 等待超时（秒）",
    )
    args = parser.parse_args()

    sub_path = _subscription_path(args.project, args.subscription)
    sub_client = pubsub_v1.SubscriberClient()

    received = 0
    while received < args.max_messages:
        resp = sub_client.pull(
            subscription=sub_path,
            max_messages=min(10, args.max_messages - received),
            timeout=args.timeout_s,
        )
        if not resp.received_messages:
            print("No more messages (timeout or empty).")
            break
        ack_ids = []
        for rm in resp.received_messages:
            received += 1
            ack_ids.append(rm.ack_id)
            try:
                text = rm.message.data.decode("utf-8")
                obj: Dict[str, Any] = json.loads(text)
            except Exception as e:
                print("Bad message:", e, rm.message.data[:200])
                continue
            pred = obj.get("prediction")
            sid = obj.get("session_id")
            tid = obj.get("t_index")
            print(f"session={sid} t_index={tid} prediction={pred}")
        if ack_ids:
            sub_client.acknowledge(subscription=sub_path, ack_ids=ack_ids)
    print(f"Done, processed {received} messages.")


if __name__ == "__main__":
    main()
