#!/bin/bash
# submit_job.sh <tag> <run.py args...>
# Frozen-copy submission: freeze the code into a job-private copy, submit
# the job running FROM that copy, and let the archive epilogue MOVE the
# same copy into runs/<jobid>/code/ -- so the archived code is identical
# by construction to the executed code, immune to any later edit of the
# live code dir.
# MUST be run from the workdir root (this repository's root: the dir
# containing code/SOCMartNet-v3-refactored/ and slurm/).
set -euo pipefail
TAG="$1"; shift
[ -d code/SOCMartNet-v3-refactored ] \
  || { echo "ERROR: run from the workdir root (code/SOCMartNet-v3-refactored not found)"; exit 1; }

TS=$(date +%Y%m%d%H%M%S)
FROZEN="frozen/${TAG}_${TS}_$$"
mkdir -p "$FROZEN"
# mirror the repo-root layout, minus runtime products (outputs/logs/frozen;
# frozen itself excluded to avoid self-recursion) and the report assets
# (results/figures/katex -- reference data, not needed by the running job)
rsync -a --exclude outputs --exclude 'outputs_*' --exclude logs \
      --exclude frozen --exclude __pycache__ --exclude .git \
      --exclude results --exclude figures --exclude katex ./ "$FROZEN"/
# version anchor: full-tree md5 manifest, paths relative to the frozen root
# so `md5sum -c MANIFEST.md5` works verbatim inside runs/<jobid>/code/
(cd "$FROZEN" && find . -type f -name '*.py' | sort | xargs md5sum > MANIFEST.md5)
# sweep residues of submissions whose jobs never ran (cancelled while
# queued); a live queued/running job's copy is younger than 7 days
find frozen -mindepth 1 -maxdepth 1 -type d -mtime +7 -exec rm -rf {} + 2>/dev/null || true

# Slurm opens logs/%j_%x.{out,err} when the job starts -- before the batch
# script's own mkdir runs -- so the log dir must exist from submission time
mkdir -p logs outputs
sbatch --export=ALL,ENGINE_DIR="$FROZEN" -J "$TAG" slurm/run_one.slurm "$@"
echo "frozen copy: $FROZEN  (epilogue moves it to runs/<jobid>/code/)"
