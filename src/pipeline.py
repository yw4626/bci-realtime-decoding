"""
显式解码流水线：与离线训练对齐（窗口长度、步长、ProjectConfig 驱动的 extract_window_features），
用于模拟生产式逐样本处理并可选记录中间特征向量。

说明：模型在“手工特征”上学习；本模块把这些步骤拆成可观测阶段，便于对齐训练/在线行为。
"""

from __future__ import annotations

import json
import os
import struct
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import joblib
import numpy as np
from collections import deque

from .config import ProjectConfig
from .features import extract_window_features
from .models import blend_predictions
from .multi_subject_data import pad_or_trunc_channels


Mode = Literal["simple", "complex", "blended"]


@dataclass
class PipelineState:
    """单次 push 的可观测结果（无预测时除 t_index / stage 外多为 None）。"""

    t_index: int
    stage: str
    n_channels_in: int
    n_channels_aligned: int
    has_prediction: bool
    feature: Optional[np.ndarray] = None
    prediction: Optional[np.ndarray] = None


def load_channel_target_from_manifest(manifest_path: str) -> Optional[int]:
    """从 train_manifest.json 读取 channel_target（多被试 pooled 训练时使用）。"""
    if not manifest_path or not os.path.isfile(manifest_path):
        return None
    with open(manifest_path, encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)
    ch = data.get("channel_target")
    return int(ch) if ch is not None else None


class BCIDecodePipeline:
    """
    阶段：对齐通道 → 滑窗缓冲 →（满足步长）构造窗口 → 特征向量 → 双模型预测/融合。

    与早期单类解码器逻辑对齐，并增加通道对齐与可观测 `PipelineState`。
    """

    def __init__(
        self,
        cfg: ProjectConfig,
        simple_path: str,
        complex_path: str,
        n_channels: Optional[int] = None,
        blend_alpha: float = 0.6,
        simple_model: Any = None,
        complex_model: Any = None,
        retain_features: bool = True,
    ):
        self.cfg = cfg
        self.n_channels = n_channels
        self.blend_alpha = blend_alpha
        self._retain_features = retain_features
        if simple_model is not None and complex_model is not None:
            self.simple = simple_model
            self.complex = complex_model
        elif simple_model is None and complex_model is None:
            self.simple = joblib.load(simple_path)
            self.complex = joblib.load(complex_path)
        else:
            raise ValueError("simple_model 与 complex_model 必须同时提供或同时省略")
        self._buffer: deque = deque(maxlen=cfg.window_size)
        self._sample_counter = 0

    def _align_sample(self, sample: np.ndarray) -> Tuple[np.ndarray, int, int]:
        x = np.asarray(sample, dtype=np.float64).ravel()
        n_in = int(x.shape[0])
        if self.n_channels is None:
            return x, n_in, n_in
        row = pad_or_trunc_channels(x.reshape(1, -1), self.n_channels)
        out = row[0]
        return out, n_in, int(out.shape[0])

    def reset(self) -> None:
        self._buffer.clear()
        self._sample_counter = 0

    def get_stream_state(self) -> Tuple[int, bytes]:
        """供分布式流式引擎持久化：计数器 + 滑窗缓冲（紧凑 bytes，避免每步 tolist 巨型 Python 列表）。"""
        if not self._buffer:
            return self._sample_counter, struct.pack("II", 0, 0)
        arr = np.stack(list(self._buffer), axis=0)
        n_r, n_c = int(arr.shape[0]), int(arr.shape[1])
        payload = arr.astype(np.float64, copy=False).tobytes()
        return self._sample_counter, struct.pack("II", n_r, n_c) + payload

    def set_stream_state(
        self, counter: int, buffer_rows: Union[List[List[float]], np.ndarray, bytes]
    ) -> None:
        """与 `get_stream_state` 配对；支持 list[list]、ndarray 或 get_stream_state 产出的 bytes。"""
        self._sample_counter = int(counter)
        self._buffer.clear()
        if isinstance(buffer_rows, bytes):
            if len(buffer_rows) < 8:
                return
            n_r, n_c = struct.unpack_from("II", buffer_rows, 0)
            if n_r == 0:
                return
            need = 8 + int(n_r) * int(n_c) * 8
            if len(buffer_rows) < need:
                raise ValueError(f"buffer bytes 长度不符: need {need}, got {len(buffer_rows)}")
            arr = np.frombuffer(buffer_rows, dtype=np.float64, offset=8, count=n_r * n_c).reshape(n_r, n_c)
            for i in range(int(n_r)):
                self._buffer.append(np.asarray(arr[i], dtype=np.float64))
            return
        if isinstance(buffer_rows, np.ndarray):
            if buffer_rows.size == 0:
                return
            for i in range(int(buffer_rows.shape[0])):
                self._buffer.append(np.asarray(buffer_rows[i], dtype=np.float64))
            return
        for row in buffer_rows:
            self._buffer.append(np.asarray(row, dtype=np.float64))

    def push(self, sample: np.ndarray, mode: Mode = "blended") -> Optional[PipelineState]:
        self._sample_counter += 1
        aligned, n_in, n_al = self._align_sample(sample)
        self._buffer.append(aligned)

        if len(self._buffer) < self.cfg.window_size:
            return PipelineState(
                t_index=self._sample_counter,
                stage="warmup",
                n_channels_in=n_in,
                n_channels_aligned=n_al,
                has_prediction=False,
            )
        if self._sample_counter % self.cfg.step_size != 0:
            return PipelineState(
                t_index=self._sample_counter,
                stage="skip_step",
                n_channels_in=n_in,
                n_channels_aligned=n_al,
                has_prediction=False,
            )

        window = np.asarray(self._buffer)
        feat_vec = extract_window_features(
            window,
            sample_rate_hz=self.cfg.sample_rate_hz,
            include_legacy_fft_bands=self.cfg.include_legacy_fft_bands,
            include_physiological_bands=self.cfg.include_physiological_bands,
        )
        feat = feat_vec.reshape(1, -1)
        if mode == "simple":
            p_simple = self.simple.predict(feat)
            pred = p_simple[0]
        elif mode == "complex":
            p_complex = self.complex.predict(feat)
            pred = p_complex[0]
        else:
            p_simple = self.simple.predict(feat)
            p_complex = self.complex.predict(feat)
            pred = blend_predictions(p_simple, p_complex, alpha=self.blend_alpha)[0]

        feat_out = feat_vec.copy() if self._retain_features else None
        return PipelineState(
            t_index=self._sample_counter,
            stage="decoded",
            n_channels_in=n_in,
            n_channels_aligned=n_al,
            has_prediction=True,
            feature=feat_out,
            prediction=np.asarray(pred, dtype=np.float64).copy(),
        )


def run_stream_ticks(
    samples: np.ndarray,
    labels: np.ndarray,
    pipeline: BCIDecodePipeline,
    mode: Mode = "blended",
    max_samples: int = 0,
    decode_latency_ms: Optional[List[float]] = None,
) -> Tuple[List[PipelineState], np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    逐样本推进流水线；有预测时收集 pred 与 labels[i]（五指）。

    若传入 `decode_latency_ms`，则对每个产生预测的 `push` 记录一次耗时（毫秒）。
    返回：(每步 state 列表, preds 堆叠, gts 堆叠, 每步特征列表，仅 decoded 步有特征)。
    """
    states: List[PipelineState] = []
    pred_rows: List[np.ndarray] = []
    gt_rows: List[np.ndarray] = []
    feat_rows: List[np.ndarray] = []

    n = len(samples) if max_samples <= 0 else min(len(samples), max_samples)
    for i in range(n):
        if decode_latency_ms is not None:
            t0 = time.perf_counter()
        st = pipeline.push(samples[i], mode=mode)
        states.append(st)
        if st.has_prediction and st.prediction is not None:
            if decode_latency_ms is not None:
                decode_latency_ms.append((time.perf_counter() - t0) * 1000.0)
            pred_rows.append(st.prediction)
            gt_rows.append(labels[i])
            if st.feature is not None:
                feat_rows.append(st.feature)

    preds = np.asarray(pred_rows) if pred_rows else np.zeros((0, 5))
    gts = np.asarray(gt_rows) if gt_rows else np.zeros((0, labels.shape[1]))
    return states, preds, gts, feat_rows
