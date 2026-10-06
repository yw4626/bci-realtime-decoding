#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

: "${GOOGLE_CLOUD_PROJECT:?Set GOOGLE_CLOUD_PROJECT}"
: "${REGION:?Set REGION (e.g. us-central1)}"
: "${GCS_BUCKET:?Set GCS_BUCKET (without gs://)}"
: "${INPUT_SUBSCRIPTION:?projects/PROJECT/subscriptions/SUB_NAME}"
: "${OUTPUT_TOPIC:?projects/PROJECT/topics/TOPIC_NAME}"
: "${MODEL_SIMPLE:?gs://.../model_simple.joblib (Ridge)}"
: "${MODEL_COMPLEX:?gs://.../model_mlp.joblib (MLP)}"

JOB_NAME="${JOB_NAME:-bci-decode-$(date +%Y%m%d-%H%M%S)}"
TEMPLATE_OBJECT="${TEMPLATE_OBJECT:-templates/bci-beam-decode-flex.json}"
TEMPLATE_GCS_PATH="gs://${GCS_BUCKET}/${TEMPLATE_OBJECT}"
DEADLETTER_TOPIC="${DEADLETTER_TOPIC:-}"
TEMP_LOCATION="${TEMP_LOCATION:-gs://${GCS_BUCKET}/dataflow/tmp}"
STAGING_LOCATION="${STAGING_LOCATION:-gs://${GCS_BUCKET}/dataflow/staging}"
MODE="${MODE:-blended}"
WINDOW_MS="${WINDOW_MS:-300}"
STEP_MS="${STEP_MS:-50}"
N_CHANNELS="${N_CHANNELS:-0}"
GCS_MODEL_CACHE_DIR="${GCS_MODEL_CACHE_DIR:-/tmp/bci_beam_models}"

PARAMS="input_subscription=${INPUT_SUBSCRIPTION},output_topic=${OUTPUT_TOPIC},model_simple=${MODEL_SIMPLE},model_complex=${MODEL_COMPLEX},mode=${MODE},window_ms=${WINDOW_MS},step_ms=${STEP_MS},n_channels=${N_CHANNELS},gcs_model_cache_dir=${GCS_MODEL_CACHE_DIR},temp_location=${TEMP_LOCATION},staging_location=${STAGING_LOCATION}"
if [[ -n "${DEADLETTER_TOPIC}" ]]; then
  PARAMS="${PARAMS},deadletter_topic=${DEADLETTER_TOPIC}"
fi

gcloud dataflow flex-template run "${JOB_NAME}" \
  --project "${GOOGLE_CLOUD_PROJECT}" \
  --region "${REGION}" \
  --template-file-gcs-location "${TEMPLATE_GCS_PATH}" \
  --parameters "${PARAMS}" \
  --additional-user-labels "model_simple=ridge,model_complex=mlp" \
  "$@"
