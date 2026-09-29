#!/usr/bin/env bash
# Build the pinned ChromatinHD wheel published as this repository's release asset.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_ROOT="${TMPDIR:?TMPDIR must identify allocation-local storage}/chromatinhd-wheel-build"
SDIST_URL="https://files.pythonhosted.org/packages/2d/75/c2440103fa39561397a928e6e2e7db0f3dc48e10ecd1101d3c5a1195c293/chromatinhd-0.4.3.tar.gz"
SDIST_SHA256="6976968bf5291025bff686b12fd356f1c64564453cca089aa9133e4e9be33ac6"

rm -rf "$BUILD_ROOT"
mkdir -p "$BUILD_ROOT" "$ROOT/dist"

if [[ "$(gcc -dumpfullversion)" != 14.3.0 ]]; then
  echo "GCC 14.3.0 is required" >&2
  exit 1
fi

curl --fail --location --max-time 300 --output "$BUILD_ROOT/chromatinhd-0.4.3.tar.gz" "$SDIST_URL"
echo "$SDIST_SHA256  $BUILD_ROOT/chromatinhd-0.4.3.tar.gz" | sha256sum --check

UV_CACHE_DIR="$BUILD_ROOT/uv-cache" timeout 1800 uv build --wheel --python 3.13 \
  --build-constraints "$ROOT/scripts/build-constraints.txt" \
  --out-dir "$ROOT/dist" "$BUILD_ROOT/chromatinhd-0.4.3.tar.gz"
sha256sum "$ROOT"/dist/chromatinhd-0.4.3-*.whl
