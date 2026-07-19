#!/usr/bin/env bash
# Build seir_core.{dylib,so,dll} from seir_core.c.
#
# Usage:
#   bash simulation/c/build.sh
#   → produces simulation/c/seir_core.{dylib|so}
#
# Python loader (simulation/sim/stepper.py HAS_C_BACKEND branch) auto-detects.

set -euo pipefail
cd "$(dirname "$0")"

OS=$(uname -s)
case "$OS" in
    Darwin) EXT=dylib ;;
    Linux)  EXT=so ;;
    MINGW*|MSYS*|CYGWIN*) EXT=dll ;;
    *) echo "unknown OS: $OS"; exit 1 ;;
esac

OUT="seir_core.${EXT}"
SRC="seir_core.c"

# Pick compiler
if command -v cc >/dev/null 2>&1; then CC=cc
elif command -v gcc >/dev/null 2>&1; then CC=gcc
elif command -v clang >/dev/null 2>&1; then CC=clang
else echo "no C compiler found"; exit 1
fi

# B-P3 (M7): -ffast-math REMOVED — it disabled IEEE NaN/Inf semantics (defeating
# the downstream finite-value gate: the kernel could emit plausible-but-wrong
# finite garbage instead of a detectable NaN) and permitted FP reassociation →
# non-reproducible results. -march=native is now OPT-IN (MPH_NATIVE_ARCH=1) so the
# default build is portable + bit-reproducible across machines.
FLAGS="-O3 -fPIC -fvisibility=hidden -Wall -Wextra"
if [ "${MPH_NATIVE_ARCH:-0}" = "1" ] && \
   ${CC} -march=native -E -x c /dev/null -o /dev/null >/dev/null 2>&1; then
    FLAGS="${FLAGS} -march=native"
    echo "  [build] MPH_NATIVE_ARCH=1 → -march=native (non-portable, non-reproducible binary)"
fi

echo "Building ${OUT} with ${CC} ${FLAGS}"
${CC} ${FLAGS} -shared "${SRC}" -o "${OUT}" -lm
echo "  size: $(stat -f '%z bytes' "${OUT}" 2>/dev/null || stat -c '%s bytes' "${OUT}") "
echo "  ✓ ${OUT} ready — ctypes will find it at simulation/c/${OUT}"
