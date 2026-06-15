#!/usr/bin/env bash
# Build the Claude Code sidecar tool image.
#
# Usage:
#   bash build_tool.sh
#   bash build_tool.sh --claude-code-version latest
#   bash build_tool.sh --claude-code-version 2.1.118
#   bash build_tool.sh --npm-registry https://registry.npmmirror.com
#   bash build_tool.sh --registry swr.cn-east-3.myhuaweicloud.com/openyuanrong
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE_NAME="${TOOL_IMAGE:-claude-code-tool}"
IMAGE_TAG="${TOOL_TAG:-latest}"
CLAUDE_CODE_VERSION="${CLAUDE_CODE_VERSION:-latest}"
NPM_REGISTRY="${NPM_REGISTRY:-}"
REGISTRY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --registry) REGISTRY="$2"; shift 2 ;;
        --npm-registry) NPM_REGISTRY="$2"; shift 2 ;;
        --claude-code-version) CLAUDE_CODE_VERSION="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

BUILD_ARGS=(
    --build-arg "CLAUDE_CODE_VERSION=${CLAUDE_CODE_VERSION}"
)
if [[ -n "${NPM_REGISTRY}" ]]; then
    BUILD_ARGS+=(--build-arg "NPM_REGISTRY=${NPM_REGISTRY}")
fi

echo "==> Building Claude Code tool image: ${IMAGE_NAME}:${IMAGE_TAG}"
docker build \
    -f "${SCRIPT_DIR}/Dockerfile.claude-code-tool" \
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
