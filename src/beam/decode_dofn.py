from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, Optional, Tuple

import apache_beam as beam
import joblib
import numpy as np
from apache_beam.metrics import Metrics
from apache_beam.transforms import userstate

from ..config import ProjectConfig
from ..pipeline import BCIDecodePipeline, PipelineState
from .gcs_utils import localize_gcs_uri

logger = logging.getLogger(__name__)

DEADLETTER_TAG = "deadletter"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _model_version_from_uri(uri: str) -> str:
    base = os.path.basename(uri.strip())
    if not base:
        return "unknown"
    return base


def state_to_dict(
    session_id: str,
    st: PipelineState,
    mode: str,
    model_simple_uri: str,
    model_complex_uri: str,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "session_id": session_id,
        "event_time_utc": _utc_now_iso(),
        "t_index": st.t_index,
        "stage": st.stage,
        "n_channels_in": st.n_channels_in,
        "n_channels_aligned": st.n_channels_aligned,
        "has_prediction": st.has_prediction,
        "mode": mode,
        "model_simple_uri": model_simple_uri,
        "model_complex_uri": model_complex_uri,
        "model_simple_version": _model_version_from_uri(model_simple_uri),
        "model_complex_version": _model_version_from_uri(model_complex_uri),
    }
    if st.has_prediction and st.prediction is not None:
        out["prediction"] = st.prediction.astype(float).tolist()
    return out


def make_deadletter(
    session_id: str,
    raw_message: str,
    reason: str,
    detail: str = "",
) -> str:
    payload = {
        "event_time_utc": _utc_now_iso(),
        "session_id": session_id,
        "reason": reason,
        "detail": detail,
        "raw_message": raw_message,
    }
    return json.dumps(payload, ensure_ascii=False)


class DecodeSampleFn(beam.DoFn):
    """按 session_id 分键，用 Beam 托管状态驱动 BCIDecodePipeline。

    输入为 ``(session_id, payload)``（已由 ``ParseAndKeyFn`` 解析好的 dict）；
    主输出仅在 ``stage == 'decoded'`` 时 yield 一条 JSON，warmup/skip 不落主输出。
    """

    STREAM_STATE = userstate.ReadModifyWriteStateSpec("bci_stream", beam.coders.PickleCoder())

    def __init__(
        self,
        window_ms: int,
        step_ms: int,
        model_simple: str,
        model_complex: str,
        n_channels: Optional[int],
        mode: str,
        gcs_model_cache_dir: str,
        blend_alpha: float = 0.6,
        include_legacy_fft_bands: bool = False,
        include_physiological_bands: bool = True,
        strict_sample_seq: bool = True,
    ):
        self._window_ms = window_ms
        self._step_ms = step_ms
        self._model_simple_uri = model_simple
        self._model_complex_uri = model_complex
        self._n_channels = n_channels
        self._mode = mode
        self._gcs_model_cache_dir = gcs_model_cache_dir
        self._blend_alpha = blend_alpha
        self._include_legacy_fft_bands = include_legacy_fft_bands
        self._include_physiological_bands = include_physiological_bands
        self._strict_sample_seq = strict_sample_seq
        self._simple_local: Optional[str] = None
        self._complex_local: Optional[str] = None
        self._simple_model: Any = None
        self._complex_model: Any = None
        self._pipeline: Optional[BCIDecodePipeline] = None

        self._m_total = Metrics.counter("decode", "input_total")
        self._m_bad = Metrics.counter("decode", "input_invalid")
        self._m_out_of_order = Metrics.counter("decode", "input_out_of_order")
        self._m_dup_seq = Metrics.counter("decode", "duplicate_sample_seq_ignored")
        self._m_output_total = Metrics.counter("decode", "output_total")
        self._m_decoded = Metrics.counter("decode", "decoded_events")
        self._m_warmup = Metrics.counter("decode", "stage_warmup")
        self._m_skip = Metrics.counter("decode", "stage_skip_step")

    def setup(self) -> None:
        cache = self._gcs_model_cache_dir or os.path.join(tempfile.gettempdir(), "bci_beam_models")
        self._simple_local = localize_gcs_uri(self._model_simple_uri, cache)
        self._complex_local = localize_gcs_uri(self._model_complex_uri, cache)
        self._simple_model = joblib.load(self._simple_local)
        self._complex_model = joblib.load(self._complex_local)
        cfg = ProjectConfig(
            window_ms=self._window_ms,
            step_ms=self._step_ms,
            model_dir="",
            include_legacy_fft_bands=self._include_legacy_fft_bands,
            include_physiological_bands=self._include_physiological_bands,
        )
        self._pipeline = BCIDecodePipeline(
            cfg,
            "",
            "",
            n_channels=self._n_channels,
            blend_alpha=self._blend_alpha,
            simple_model=self._simple_model,
            complex_model=self._complex_model,
            retain_features=False,
        )
        logger.info(
            "Decode setup complete: mode=%s blend_alpha=%.4f window_ms=%d step_ms=%d simple_uri=%s complex_uri=%s "
            "simple_local=%s complex_local=%s (single BCIDecodePipeline instance)",
            self._mode,
            self._blend_alpha,
            self._window_ms,
            self._step_ms,
            self._model_simple_uri,
            self._model_complex_uri,
            self._simple_local,
            self._complex_local,
        )

    def process(
        self,
        element: Tuple[str, Dict[str, Any]],
        stream_state=beam.DoFn.StateParam(STREAM_STATE),
    ) -> Iterator[str]:
        self._m_total.inc()
        session_id, payload = element
        if self._pipeline is None:
            raise RuntimeError("DecodeSampleFn.setup() 未执行，无法解码")

        if not isinstance(payload, dict):
            self._m_bad.inc()
            raw = json.dumps(payload, ensure_ascii=False)
            yield beam.pvalue.TaggedOutput(
                DEADLETTER_TAG,
                make_deadletter(session_id, raw, "invalid_payload_type", "payload is not object"),
            )
            return

        def _raw_for_dl() -> str:
            return json.dumps(payload, ensure_ascii=False)

        channels = payload.get("channels")
        if channels is None:
            self._m_bad.inc()
            yield beam.pvalue.TaggedOutput(
                DEADLETTER_TAG,
                make_deadletter(session_id, _raw_for_dl(), "missing_channels"),
            )
            return

        try:
            sample = np.asarray(channels, dtype=np.float64)
        except (TypeError, ValueError) as e:
            self._m_bad.inc()
            yield beam.pvalue.TaggedOutput(
                DEADLETTER_TAG,
                make_deadletter(session_id, _raw_for_dl(), "channels_cast_failed", str(e)),
            )
            return

        raw_state = stream_state.read()
        if raw_state is None:
            counter, buffer_part, last_seq = 0, [], None
        elif isinstance(raw_state, tuple) and len(raw_state) == 2:
            counter, buffer_part = raw_state[0], raw_state[1]  # type: ignore[misc]
            last_seq = None
        elif isinstance(raw_state, tuple) and len(raw_state) >= 3:
            counter, buffer_part, last_seq = raw_state[0], raw_state[1], raw_state[2]
        else:
            counter, buffer_part, last_seq = 0, [], None
        if buffer_part is None:
            buffer_part = []

        seq_raw = payload.get("sample_seq", payload.get("seq"))
        seq_int: Optional[int] = None
        if seq_raw is not None:
            try:
                seq_int = int(seq_raw)
            except (TypeError, ValueError) as e:
                self._m_bad.inc()
                yield beam.pvalue.TaggedOutput(
                    DEADLETTER_TAG,
                    make_deadletter(session_id, _raw_for_dl(), "invalid_sample_seq", str(e)),
                )
                return
            if self._strict_sample_seq:
                # At-least-once delivery: same sample_seq can arrive again after state was
                # updated. Rejecting seq_int == last_seq caused spurious out_of_order DLQ.
                if last_seq is not None and seq_int == last_seq:
                    self._m_dup_seq.inc()
                    return
                if last_seq is not None and seq_int < last_seq:
                    self._m_bad.inc()
                    self._m_out_of_order.inc()
                    yield beam.pvalue.TaggedOutput(
                        DEADLETTER_TAG,
                        make_deadletter(
                            session_id,
                            _raw_for_dl(),
                            "out_of_order",
                            f"sample_seq={seq_int} last_seq={last_seq}",
                        ),
                    )
                    return

        pipe = self._pipeline
        pipe.set_stream_state(counter, buffer_part)
        st = pipe.push(sample, mode=self._mode)
        c2, buf2 = pipe.get_stream_state()
        if self._strict_sample_seq:
            new_last = last_seq if seq_int is None else seq_int
        else:
            # Do not persist seq: publisher may restart from 1 while stateful buffer continues.
            new_last = None
        stream_state.write((c2, buf2, new_last))
        if st.stage == "warmup":
            self._m_warmup.inc()
        elif st.stage == "skip_step":
            self._m_skip.inc()
        elif st.stage == "decoded":
            self._m_decoded.inc()
            self._m_output_total.inc()
            yield json.dumps(
                state_to_dict(
                    session_id,
                    st,
                    mode=self._mode,
                    model_simple_uri=self._model_simple_uri,
                    model_complex_uri=self._model_complex_uri,
                ),
                ensure_ascii=False,
            )
