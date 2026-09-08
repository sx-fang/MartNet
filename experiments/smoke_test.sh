#!/bin/bash
# Quick end-to-end check that the code runs in your environment: one tiny
# Sec. 4.1-style linear-parabolic solve (d = 4, I = 20, M = 512; about a
# minute on one GPU). Not a numerical-accuracy test -- it only verifies the
# training loop, CSV logging, and exit code.
# Usage:   bash experiments/smoke_test.sh     (from the repository root)
# Override the interpreter with PYTHON=... if needed.
# Success: exit 0 and outputs/*_smoke.csv with at least 20 iteration rows.
set -u
PY="${PYTHON:-python3}"
ITERS=20

"$PY" code/SOCMartNet-v3-refactored/run.py --example linear --dim 4 --preset sisc \
    --dtype float32 --max-iter "$ITERS" --num-paths 512 --seed 0 --tag smoke

f=$(ls outputs/*smoke*.csv 2>/dev/null | head -1)
if [ -z "$f" ]; then
  echo "SMOKE FAIL: no outputs/*smoke*.csv produced"; exit 1
fi
n=$(awk 'END{print NR}' "$f")
if [ "$n" -lt "$ITERS" ]; then
  echo "SMOKE FAIL: only $n rows in $f"; exit 1
fi
echo "SMOKE OK: $f ($n rows)"
