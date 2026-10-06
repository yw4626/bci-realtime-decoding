from __future__ import annotations

import json
import os
import time
from collections import deque
from typing import Any, Dict, List

import pandas as pd
import streamlit as st
from google.cloud import pubsub_v1


FINGER_NAMES = ["thumb", "index", "middle", "ring", "little"]


def _subscription_path(project: str, subscription: str) -> str:
    if subscription.startswith("projects/"):
        return subscription
    return f"projects/{project}/subscriptions/{subscription}"


def _init_state(max_points: int) -> None:
    if "points" not in st.session_state:
        st.session_state.points = deque(maxlen=max_points)
    if "total_received" not in st.session_state:
        st.session_state.total_received = 0
    if "latest_session" not in st.session_state:
        st.session_state.latest_session = "-"
    if "latest_t_index" not in st.session_state:
        st.session_state.latest_t_index = "-"
    if "running" not in st.session_state:
        st.session_state.running = False


def _resize_buffer(max_points: int) -> None:
    if st.session_state.points.maxlen != max_points:
        st.session_state.points = deque(st.session_state.points, maxlen=max_points)


def _pull_predictions(
    sub_client: pubsub_v1.SubscriberClient,
    sub_path: str,
    pull_batch: int,
    timeout_s: float,
) -> int:
    resp = sub_client.pull(
        subscription=sub_path,
        max_messages=pull_batch,
        timeout=timeout_s,
    )
    if not resp.received_messages:
        return 0

    ack_ids: List[str] = []
    added = 0
    for rm in resp.received_messages:
        ack_ids.append(rm.ack_id)
        try:
            text = rm.message.data.decode("utf-8")
            obj: Dict[str, Any] = json.loads(text)
            pred = obj.get("prediction")
            if not isinstance(pred, list) or len(pred) < 5:
                continue
            point = {
                "event_time": time.time(),
                "session_id": str(obj.get("session_id", "-")),
                "t_index": obj.get("t_index"),
                "thumb": float(pred[0]),
                "index": float(pred[1]),
                "middle": float(pred[2]),
                "ring": float(pred[3]),
                "little": float(pred[4]),
            }
            st.session_state.points.append(point)
            st.session_state.latest_session = point["session_id"]
            st.session_state.latest_t_index = point["t_index"]
            added += 1
        except Exception:
            # Ignore malformed output message in UI path.
            continue

    if ack_ids:
        sub_client.acknowledge(subscription=sub_path, ack_ids=ack_ids)
    st.session_state.total_received += added
    return added


def main() -> None:
    st.set_page_config(page_title="BCI Realtime Decoder", layout="wide")
    st.title("BCI 实时手指解码监控")

    default_project = os.getenv("GOOGLE_CLOUD_PROJECT", "")
    default_subscription = os.getenv("OUTPUT_SUBSCRIPTION", "")

    with st.sidebar:
        st.header("连接配置")
        project = st.text_input("GCP Project", value=default_project)
        subscription = st.text_input("输出订阅名", value=default_subscription)
        pull_batch = st.number_input("每次拉取条数", min_value=1, max_value=200, value=50, step=1)
        timeout_s = st.number_input("Pull 超时(秒)", min_value=1.0, max_value=120.0, value=10.0, step=1.0)
        max_points = st.number_input("图上保留点数", min_value=100, max_value=20000, value=2000, step=100)
        refresh_ms = st.number_input("刷新间隔(ms)", min_value=200, max_value=10000, value=1000, step=100)

        st.caption("需要先执行: gcloud auth application-default login")
        col_a, col_b = st.columns(2)
        if col_a.button("开始", use_container_width=True):
            st.session_state.running = True
        if col_b.button("暂停", use_container_width=True):
            st.session_state.running = False
        if st.button("清空缓存", use_container_width=True):
            st.session_state.points.clear()
            st.session_state.total_received = 0
            st.session_state.latest_session = "-"
            st.session_state.latest_t_index = "-"

    _init_state(int(max_points))
    _resize_buffer(int(max_points))

    if not project.strip() or not subscription.strip():
        st.info("请在侧边栏填写 GCP Project 和输出订阅名，或设置对应环境变量。")
        st.stop()

    try:
        sub_path = _subscription_path(project.strip(), subscription.strip())
        sub_client = pubsub_v1.SubscriberClient()
    except Exception as e:
        st.error(f"初始化 Pub/Sub 客户端失败: {e}")
        st.stop()

    if st.session_state.running:
        added = _pull_predictions(
            sub_client=sub_client,
            sub_path=sub_path,
            pull_batch=int(pull_batch),
            timeout_s=float(timeout_s),
        )
        st.toast(f"本轮新增 {added} 条", icon="✅")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("累计接收(有效 prediction)", st.session_state.total_received)
    c2.metric("缓存点数", len(st.session_state.points))
    c3.metric("最新 session_id", st.session_state.latest_session)
    c4.metric("最新 t_index", st.session_state.latest_t_index)

    if st.session_state.points:
        df = pd.DataFrame(list(st.session_state.points))
        df["event_dt"] = pd.to_datetime(df["event_time"], unit="s")
        chart_df = df[["event_dt"] + FINGER_NAMES].set_index("event_dt")
        st.line_chart(chart_df, height=420)

        with st.expander("最新 20 条原始解码点"):
            st.dataframe(df.tail(20), use_container_width=True, hide_index=True)
    else:
        st.info("暂无数据。请确认 Dataflow 正在向输出主题写入，且订阅配置正确。")

    if st.session_state.running:
        time.sleep(float(refresh_ms) / 1000.0)
        st.rerun()


if __name__ == "__main__":
    main()
