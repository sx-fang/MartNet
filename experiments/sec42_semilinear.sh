#!/bin/bash
# Sec. 4.2 semilinear parabolic equation with oscillatory terminal
# (paper Fig. 2): d=100, T = 1/0.1/0.05/0.01, 5 seeds (20 jobs), plus the
# d=10 exact-reference curves (4 one-iteration jobs producing the top row
# of the paper figure).
#
# Reference (results/s413_5seed_agg.csv):
#   T=1     RE 1.46e-2 +- 0.27e-2 ; T=0.1  RE 3.24e-3 +- 0.59e-3 ;
#   T=0.05  RE 5.64e-3 +- 0.69e-3 ; T=0.01 RE 2.03e-2 +- 0.51e-3
# Success check: numerical runs I=2000 -> 2002 CSV rows; reference runs
# produce curve CSVs only (max-iter 1).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

### numerical row (d=100)
for T in 1.0 0.1 0.05 0.01; do for s in 0 1 2 3 4; do
  submit "s42cb_t${T}_d100_s${s}" 1 \
    --example semilinear --dim 100 --te $T --preset sisc --dtype float32 \
    --max-iter 2000 --num-dt 100 --num-paths 10000 --renew-frac 0.2 \
    --seed $s --tag "s42cb_t${T}_d100_s${s}"
done; done

### exact-reference row (d=10, curves only)
for T in 1.0 0.1 0.05 0.01; do
  submit "s42h_true_d10_t${T}" 1 \
    --example semilinear --dim 10 --te $T --preset sisc --dtype float32 \
    --max-iter 1 --num-dt 100 --num-paths 10000 --renew-frac 0.2 \
    --seed 0 --tag "s42h_true_d10_t${T}"
done
echo "SEC42_SUBMITTED (24 jobs; reference: results/s413_5seed_agg.csv)"
