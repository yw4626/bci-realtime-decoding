#!/usr/bin/env bash
# Submit streaming decode job to Google Cloud Dataflow.
# Prerequisites: gcloud auth, GOOGLE_CLOUD_PROJECT, GCS_BUCKET, models on GCS, Pub/Sub subscription + output topic.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

: "${GOOGLE_CLOUD_PROJECT:?Set GOOGLE_CLOUD_PROJECT}"
: "${GCS_BUCKET:?Set GCS_BUCKET to a bucket for temp/staging (no gs:// prefix)}"
: "${INPUT_SUBSCRIPTION:?Full subscription resource, e.g. projects/PROJ/subscriptions/bci-samples}"
: "${OUTPUT_TOPIC:?Full topic resource, e.g. projects/PROJ/topics/bci-decoded}"
: "${MODEL_SIMPLE:?gs://.../model_simple.joblib (Ridge simple)}"
: "${MODEL_COMPLEX:?gs://.../model_mlp.joblib (MLP complex)}"

REGION="${REGION:-us-central1}"
MODE="${MODE:-blended}"
WINDOW_MS="${WINDOW_MS:-300}"
STEP_MS="${STEP_MS:-50}"
# sub1 固定 62 通道时通常可省略；多被试对齐或变长输入时可设 N_CHANNELS=62
N_CHANNELS="${N_CHANNELS:-0}"
DEADLETTER_TOPIC="${DEADLETTER_TOPIC:-}"
# Org policy constraints/compute.vmExternalIpAccess: set to 1 to disable worker public IPs.
# Subnet needs Private Google Access for workers to reach GCS/Pub/Sub APIs.
NO_USE_PUBLIC_IPS="${NO_USE_PUBLIC_IPS:-0}"

cmd=(
  python -m src.beam_dataflow
  --runner DataflowRunner
  --project "${GOOGLE_CLOUD_PROJECT}"
  --region "${REGION}"
  --temp_location "gs://${GCS_BUCKET}/dataflow/tmp"
  --staging_location "gs://${GCS_BUCKET}/dataflow/staging"
  --setup_file "${ROOT}/setup.py"
  --save_main_session
  --streaming
  --input_subscription "${INPUT_SUBSCRIPTION}"
  --output_topic "${OUTPUT_TOPIC}"
  --model_simple "${MODEL_SIMPLE}"
  --model_complex "${MODEL_COMPLEX}"
  --mode "${MODE}"
  --window_ms "${WINDOW_MS}"
  --step_ms "${STEP_MS}"
)

if [[ -n "${N_CHANNELS}" && "${N_CHANNELS}" != "0" ]]; then
  cmd+=(--n_channels "${N_CHANNELS}")
fi

if [[ -n "${DEADLETTER_TOPIC}" ]]; then
  cmd+=(--deadletter_topic "${DEADLETTER_TOPIC}")
fi

if [[ "${NO_USE_PUBLIC_IPS}" == "1" ]]; then
  cmd+=(--no_use_public_ips)
fi

"${cmd[@]}" "$@"
