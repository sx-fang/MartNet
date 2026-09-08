#!/bin/bash
#SBATCH -A YOUR_SLURM_ACCOUNT
#SBATCH -p YOUR_PARTITION
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH -t 04:00:00
#SBATCH -o logs/%j_%x.out
#SBATCH -e logs/%j_%x.err
# Generic single-GPU runner for the SOC-MartNet full-repro code
# (code/SOCMartNet-v3-refactored).  SITE TEMPLATE: edit the #SBATCH header
# (account, partition, walltime) and the PY variable below to match your
# cluster; the SBATCH_ACCOUNT / SBATCH_PARTITION / SBATCH_TIMELIMIT
# environment variables override the #SBATCH lines, and PYTHON overrides PY.
# Usage (from the workdir root that contains code/ and slurm/):
#   slurm/submit_job.sh <jobname> <run.py args...>   (frozen copy, preferred)
#   sbatch -J <jobname> slurm/run_one.slurm <run.py args...>   (legacy, live tree)
# The job name (-J) doubles as the log-file tag (%x).
#
# Optional archiving: set ARCHIVE_SCRIPT to a helper invoked as
# <jobid> <repo-tag> [note] that snapshots logs/outputs/code into
# runs/<jobid>/ (with a frozen copy it is MOVED there, so the archived
# code equals the executed code -- see submit_job.sh).  With ARCHIVE_SCRIPT
# unset the run simply leaves logs/ and outputs/ in the workdir.

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs outputs
export PYTHONUNBUFFERED=1
PY="${PYTHON:-python3}"

# Frozen code copy: submit_job.sh exports ENGINE_DIR = job-private frozen
# repo-root copy; unset falls back to the live tree (legacy direct sbatch).
ENG="${ENGINE_DIR:-.}"
# record the exact invocation and the code version anchor in the log
echo "JOBID=$SLURM_JOB_ID TAG=$SLURM_JOB_NAME"
echo "ARGS: $*"
echo "ENGINE_DIR=$ENG"
[ -f "$ENG/MANIFEST.md5" ] && cat "$ENG/MANIFEST.md5"

$PY "$ENG/code/SOCMartNet-v3-refactored/run.py" "$@" --out outputs

# optional archive epilogue
if [ -n "${ARCHIVE_SCRIPT:-}" ]; then
  if [ -n "${ENGINE_DIR:-}" ]; then
    ARCHIVE_CODE_DIR="$ENG" bash "$ARCHIVE_SCRIPT" \
      $SLURM_JOB_ID "${REPO_TAG:-socmartnet}" auto
  else
    bash "$ARCHIVE_SCRIPT" $SLURM_JOB_ID "${REPO_TAG:-socmartnet}" auto
  fi
else
  echo "ARCHIVE_SCRIPT unset -- skipping archiving (logs/ and outputs/ remain in the workdir)"
fi
