#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

: "${GOOGLE_CLOUD_PROJECT:?Set GOOGLE_CLOUD_PROJECT}"
: "${REGION:?Set REGION (e.g. us-central1)}"
: "${AR_REPOSITORY:?Set AR_REPOSITORY (Artifact Registry repo name)}"
: "${GCS_BUCKET:?Set GCS_BUCKET (without gs://)}"

IMAGE_NAME="${IMAGE_NAME:-bci-beam-decode}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
TEMPLATE_OBJECT="${TEMPLATE_OBJECT:-templates/bci-beam-decode-flex.json}"

IMAGE_URI="${REGION}-docker.pkg.dev/${GOOGLE_CLOUD_PROJECT}/${AR_REPOSITORY}/${IMAGE_NAME}:${IMAGE_TAG}"
TEMPLATE_GCS_PATH="gs://${GCS_BUCKET}/${TEMPLATE_OBJECT}"

echo "[1/3] Build image: ${IMAGE_URI}"
gcloud builds submit --project "${GOOGLE_CLOUD_PROJECT}" \
  --config "${ROOT}/cloudbuild.dataflow.yaml" \
  --substitutions="_IMAGE_URI=${IMAGE_URI}" \
  "${ROOT}"

echo "[2/3] Build flex template: ${TEMPLATE_GCS_PATH}"
gcloud dataflow flex-template build "${TEMPLATE_GCS_PATH}" \
  --project "${GOOGLE_CLOUD_PROJECT}" \
  --region "${REGION}" \
  --image "${IMAGE_URI}" \
  --sdk-language "PYTHON" \
  --metadata-file "${ROOT}/dataflow_flex_template.json"

echo "[3/3] Done"
echo "IMAGE_URI=${IMAGE_URI}"
echo "TEMPLATE_GCS_PATH=${TEMPLATE_GCS_PATH}"
