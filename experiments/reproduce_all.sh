#!/bin/bash
# Reproduce every numerical result of the SOC-MartNet (SISC accepted) full
# reproduction, using the frozen code in ../code via the frozen-copy
# submitter (../slurm/submit_job.sh + run_one.slurm: each submission
# freezes a job-private copy of the code with a full-tree MANIFEST.md5,
# so the executed code is archived verbatim with the run). All argument
# strings are the AUTHORITATIVE ARGS recorded in the producing jobs' logs
# (see experiments.csv for jobid provenance; results/ for the reference
# numbers).
#
# This driver runs the per-example scripts in order. Each sec*.sh /
# table*.sh script is self-contained (they source common.sh for the
# harness) and can be run standalone to reproduce just that example:
#
#   table1_equaltime.sh       Table 1 (Sec. 4.8 paper Table 1, equal-time final
#                             caliber, 50 jobs)
#   sec41_linear_parabolic.sh Sec. 4.1 paper Fig. 1 (15 jobs)
#   sec42_semilinear.sh       Sec. 4.2 paper Fig. 2 (24 jobs)
#   sec43_time_convergence.sh Sec. 4.3 paper Fig. 3 (35 jobs)
#   sec44_fig4_hjb1.sh        Sec. 4.4 paper Fig. 4, HJB-1 (12 jobs)
#   sec44_fig5_hjb23.sh       Sec. 4.4 Fig. 5, HJB-2/-3 high-dim (12 jobs)
#   sec45_domain_validity.sh  Sec. 4.5 paper Fig. 6 (5 jobs)
#   sec46_shifted_target.sh   Sec. 4.6 paper Fig. 7, 3 configurations
#                             (25 jobs)
#   sec47_epsilon.sh          Sec. 4.7 paper Fig. 8 (20 jobs)
#   table2_runtime.sh         Sec. 4.8 Table 2 paper Table 2 (20 jobs)
#
# Execution environment: any SLURM cluster with suitable GPUs -- the
#   single-GPU families run on any GPU with >= 24 GB; the d=10^4 cells need
#   8 x 80 GB GPUs (DDP). Multi-GPU cells set SBATCH_GPUS; run.py --gpus -1
# uses all visible GPUs; MASTER_PORT is derived deterministically per job,
# so co-scheduled multi-GPU jobs on one node do not collide.
#
# Usage: run from the repository root -- the repo doubles as the workdir
#   (code/SOCMartNet-v3-refactored/ and slurm/ at its root; submit_job.sh
#   expects exactly this layout). Set WORKDIR in common.sh only if running
#   from a copied tree elsewhere.
#   >=5 seeds per cell; tags match the archived runs. SOC examples
#   throughout use the implementation flags --fd-residual --controlled-pool.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for s in table1_equaltime sec41_linear_parabolic sec42_semilinear \
         sec43_time_convergence sec44_fig4_hjb1 sec44_fig5_hjb23 \
         sec45_domain_validity sec46_shifted_target sec47_epsilon \
         table2_runtime; do
  echo "=== $s ==="
  bash "$HERE/$s.sh"
done

echo "ALL_SUBMITTED (check CSV row counts: I1000->1002, I2000->2002, I3000->3002, I4000->4002, I6000->6002, I8000->8002 rows)"
