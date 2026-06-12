#!/usr/bin/env bash
# Build the mini-swe-agent sidecar tool image.
#
# Usage:
#   bash examples/swe_agent_blackbox/build_tool.sh --registry <your-registry>
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE_NAME="${TOOL_IMAGE:-mini-swe-agent-tool}"
IMAGE_TAG="${TOOL_TAG:-latest}"

# Parse args
REGISTRY=""
PIP_INDEX_URL=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --registry) REGISTRY="$2"; shift 2 ;;
        --pip-index) PIP_INDEX_URL="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

BUILD_ARGS=()
if [[ -n "${PIP_INDEX_URL}" ]]; then
    BUILD_ARGS+=(--build-arg PIP_INDEX_URL="${PIP_INDEX_URL}")
fi

echo "==> Building tool image: ${IMAGE_NAME}:${IMAGE_TAG}"
docker build \
    -f "${SCRIPT_DIR}/Dockerfile.mini-swe-agent-tool" \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    "${BUILD_ARGS[@]}" \
    "${SCRIPT_DIR}/"

if [[ -n "${REGISTRY}" ]]; then
    FULL_TAG="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
    echo "==> Tagging and pushing: ${FULL_TAG}"
    docker tag "${IMAGE_NAME}:${IMAGE_TAG}" "${FULL_TAG}"
    docker push "${FULL_TAG}"
    echo "    Pushed."
fi

echo ""
echo "Tool image ready: ${IMAGE_NAME}:${IMAGE_TAG}"
if [[ -n "${REGISTRY}" ]]; then
    echo "  Remote sandbox: ${FULL_TAG}"
fi
