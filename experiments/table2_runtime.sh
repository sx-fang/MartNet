#!/bin/bash
# Sec. 4.8 Table 2 (paper Table 2): DDP wall-clock runtime, HJB-3 (delta0=0.1),
# I=6000, w=d+10; d in {100,500,800,1000,2000} x GPUs {1,2,4,8}; 20 jobs,
# single run per cell; uniform batch 128 on all cells (the report's
# reproduction caliber).
# RT = the CSV's cumulative training wall-clock column (rt).
#
# Reference: results/t2b_fix_agg.csv (per-cell RT with RE columns; per-run
# final rows in results/t2b_raw_final.txt). Multi-GPU jobs pin MASTER_PORT
# deterministically per job id, so cells can share a node.
# Success check: I=6000 -> 6002 CSV rows; read rt from the final row.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

for d in 100 500 800 1000 2000; do for g in 1 2 4 8; do
  submit "t2b128_g${g}_d${d}" $g \
    --example hjblq --delta0 0.1 --dim $d --preset sisc --dtype float32 \
    --max-iter 6000 --num-paths 10000 --renew-frac 0.2 --fd-residual \
    --lam-bar 1000 --controlled-pool --batch-size 128 --seed 0 \
    --tag "t2b128_g${g}_d${d}"
done; done
echo "TABLE2_SUBMITTED (20 jobs, uniform batch 128; reference: results/t2b_fix_agg.csv)"
