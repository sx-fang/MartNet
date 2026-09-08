# Shared harness for the per-example reproduction scripts (sourced, not run).
# Expected runtime layout: the repository root IS the workdir, with
#   code/SOCMartNet-v3-refactored/  and  slurm/  at its root (submit_job.sh
#   freezes a job-private copy of the tree with a full-tree MANIFEST.md5,
#   so the executed code is archived verbatim with each run).
set -u
WORKDIR="${WORKDIR:-$PWD}"          # <-- adapt if running from elsewhere
SUB="$WORKDIR/slurm/submit_job.sh"
export SBATCH_PARTITION="${SBATCH_PARTITION:-batch}"
export SBATCH_TIMELIMIT="${SBATCH_TIMELIMIT:-1-00:00:00}"
cd "$WORKDIR"

submit() { # tag gpus args...
  local tag=$1 g=$2; shift 2
  SBATCH_GPUS=$g bash "$SUB" "$tag" "$@"
}
