from __future__ import annotations

import json
from typing import Any, Dict, Iterator, Optional, Tuple

import apache_beam as beam
from apache_beam.typehints import KV

from .decode_dofn import DEADLETTER_TAG, DecodeSampleFn, make_deadletter


class ParseAndKeyFn(beam.DoFn):
    """将原始 JSON 解析为 dict，输出 (session_id, payload)，避免下游再次 json.loads。"""

    def process(self, line: str) -> Iterator[Tuple[str, Dict[str, Any]]]:
        line = line.lstrip("\ufeff")
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            yield beam.pvalue.TaggedOutput(
                DEADLETTER_TAG,
                make_deadletter("unknown", line, "invalid_json_before_key", str(e)),
            )
            return
        if not isinstance(obj, dict):
            yield beam.pvalue.TaggedOutput(
                DEADLETTER_TAG,
                make_deadletter("unknown", line, "invalid_payload_type_before_key", "payload is not object"),
            )
            return
        sid = str(obj.get("session_id", "default"))
        yield sid, obj


def build_beam_pipeline(
    p: beam.Pipeline,
    input_path: str,
    input_subscription: str,
    output_path: str,
    output_topic: str,
    window_ms: int,
    step_ms: int,
    model_simple: str,
    model_complex: str,
    n_channels: Optional[int],
    mode: str,
    gcs_model_cache_dir: str,
    deadletter_path: str,
    deadletter_topic: str,
    blend_alpha: float = 0.6,
    include_legacy_fft_bands: bool = True,
    include_physiological_bands: bool = True,
    strict_sample_seq: bool = True,
) -> None:
    decode = DecodeSampleFn(
        window_ms=window_ms,
        step_ms=step_ms,
        model_simple=model_simple,
        model_complex=model_complex,
        n_channels=n_channels if n_channels and n_channels > 0 else None,
        mode=mode,
        gcs_model_cache_dir=gcs_model_cache_dir,
        blend_alpha=blend_alpha,
        include_legacy_fft_bands=include_legacy_fft_bands,
        include_physiological_bands=include_physiological_bands,
        strict_sample_seq=strict_sample_seq,
    )

    if input_subscription:
        lines = p | "ReadPubSub" >> beam.io.ReadFromPubSub(subscription=input_subscription).with_output_types(bytes)

        def _decode_bytes(b: bytes) -> str:
            return b.decode("utf-8")

        text = lines | "BytesToUtf8" >> beam.Map(_decode_bytes)
    else:
        if not input_path:
            raise ValueError("未指定 --input_subscription 时必须提供 --input_path")
        text = p | "ReadText" >> beam.io.ReadFromText(input_path)

    parsed = text | "ParseAndKey" >> beam.ParDo(ParseAndKeyFn()).with_outputs(DEADLETTER_TAG, main="valid")
    keyed = parsed.valid | "TypeHintKeyedKV" >> beam.Map(lambda kv: kv).with_output_types(KV[str, Dict[str, Any]])
    decoded = keyed | "DecodeStateful" >> beam.ParDo(decode).with_outputs(DEADLETTER_TAG, main="decoded")

    decoded_main = decoded.decoded
    deadletters = (parsed[DEADLETTER_TAG], decoded[DEADLETTER_TAG]) | "MergeDeadLetters" >> beam.Flatten()

    if output_topic:
        encoded = decoded_main | "JsonToBytes" >> beam.Map(lambda s: s.encode("utf-8"))
        _ = encoded | "WritePubSub" >> beam.io.WriteToPubSub(topic=output_topic)
    else:
        if not output_path:
            raise ValueError("未指定 --output_topic 时必须提供 --output_path")
        _ = decoded_main | "WriteText" >> beam.io.WriteToText(output_path, file_name_suffix=".jsonl")

    if deadletter_topic:
        dlq_bytes = deadletters | "DeadLetterToBytes" >> beam.Map(lambda s: s.encode("utf-8"))
        _ = dlq_bytes | "WriteDeadLetterPubSub" >> beam.io.WriteToPubSub(topic=deadletter_topic)
    elif deadletter_path:
        _ = deadletters | "WriteDeadLetterText" >> beam.io.WriteToText(deadletter_path, file_name_suffix=".jsonl")
