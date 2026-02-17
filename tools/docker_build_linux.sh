#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE_TAG="llamadart-native-linux-builder:ubuntu24.04"
DOCKER_PLATFORM="${DOCKER_PLATFORM:-linux/amd64}"
ARCH="x64"
JOBS=""
REBUILD_IMAGE=0
CLEAN=0

usage() {
  cat <<'EOF'
Usage: tools/docker_build_linux.sh [options]

Build Linux outputs inside a cached Docker image.

Options:
  --arch <x64|arm64|all>   Target architecture (default: x64)
  --jobs <N>               Pass through to tools/build.py --jobs
  --clean                  Pass through to tools/build.py --clean
  --image <tag>            Docker image tag override
  --platform <platform>    Docker platform override (default: linux/amd64)
  --rebuild-image          Force docker image rebuild
  -h, --help               Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arch)
      ARCH="$2"
      shift 2
      ;;
    --jobs)
      JOBS="$2"
      shift 2
      ;;
    --clean)
      CLEAN=1
      shift
      ;;
    --image)
      IMAGE_TAG="$2"
      shift 2
      ;;
    --platform)
      DOCKER_PLATFORM="$2"
      shift 2
      ;;
    --rebuild-image)
      REBUILD_IMAGE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "${ARCH}" != "x64" && "${ARCH}" != "arm64" && "${ARCH}" != "all" ]]; then
  echo "Invalid --arch value: ${ARCH}" >&2
  exit 1
fi

if [[ -n "${JOBS}" && ! "${JOBS}" =~ ^[0-9]+$ ]]; then
  echo "--jobs must be an integer, got: ${JOBS}" >&2
  exit 1
fi

build_image() {
  echo "Building image ${IMAGE_TAG} (${DOCKER_PLATFORM}) ..."
  docker build \
    --platform "${DOCKER_PLATFORM}" \
    -f "${REPO_ROOT}/tools/docker/linux-builder.Dockerfile" \
    -t "${IMAGE_TAG}" \
    "${REPO_ROOT}/tools/docker"
}

if [[ "${REBUILD_IMAGE}" -eq 1 ]]; then
  build_image
elif ! docker image inspect "${IMAGE_TAG}" >/dev/null 2>&1; then
  build_image
else
  echo "Using cached image ${IMAGE_TAG}"
fi

build_one() {
  local target_arch="$1"
  local clean_flag=""
  local jobs_flag=""

  if [[ "${CLEAN}" -eq 1 ]]; then
    clean_flag="--clean"
  fi
  if [[ -n "${JOBS}" ]]; then
    jobs_flag="--jobs ${JOBS}"
  fi

  echo "Building linux/${target_arch} in Docker ..."
  docker run --rm \
    --platform "${DOCKER_PLATFORM}" \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -e CCACHE_DIR=/work/.ccache \
    -v "${REPO_ROOT}:/work" \
    -w /work \
    "${IMAGE_TAG}" \
    bash -lc "set -euo pipefail; git submodule update --init --recursive; python3 tools/build.py linux --arch ${target_arch} ${clean_flag} ${jobs_flag}"

  echo "Output libraries:"
  find "${REPO_ROOT}/bin/linux/${target_arch}" -maxdepth 1 -type f | sort
}

cd "${REPO_ROOT}"

if [[ "${ARCH}" == "all" ]]; then
  build_one x64
  build_one arm64
else
  build_one "${ARCH}"
fi
