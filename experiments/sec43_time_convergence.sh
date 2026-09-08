#!/bin/bash
# Sec. 4.3 time-discretization convergence rate (paper Fig. 3): LinearSinAC
# (eq. (4.2) with Allen-Cahn source), d=100, N = 3/4/5 (I=3000) and
# N = 6/8/12/18 (I=6000) -- the paper's split protocol; 5 seeds (35 jobs).
#
# Reference (results/s413_5seed_agg.csv): all seven paper anchors hit,
# ratios 1.005-1.36 (N=18: 1.005); our log-log slope -1.087 vs the paper's
# reference slope O(N^-1.01).
# Success check: I=3000 -> 3002 CSV rows; I=6000 -> 6002.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

for N in 3 4 5; do for s in 0 1 2 3 4; do
  submit "s43fix_n${N}_d100_s${s}" 1 \
    --example linsin --dim 100 --preset sisc --dtype float32 --max-iter 3000 \
    --num-dt $N --num-paths 10000 --renew-frac 0.2 \
    --seed $s --tag "s43fix_n${N}_d100_s${s}"
done; done
for N in 6 8 12 18; do for s in 0 1 2 3 4; do
  submit "s43fix_n${N}_d100_s${s}" 1 \
    --example linsin --dim 100 --preset sisc --dtype float32 --max-iter 6000 \
    --num-dt $N --num-paths 10000 --renew-frac 0.2 \
    --seed $s --tag "s43fix_n${N}_d100_s${s}"
done; done
echo "SEC43_SUBMITTED (35 jobs; reference: results/s413_5seed_agg.csv)"
