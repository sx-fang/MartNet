#!/bin/bash
# Sec. 4.1 linear parabolic equation (paper Fig. 1): manufactured sin
# solution, d = 100/1000 (I=2000, M=1e4) and d = 2000 (I=4000, M=6000),
# 5 seeds each (15 jobs, single GPU).
#
# Reference (results/s413_5seed_agg.csv):
#   d100  RE 6.16e-3 +- 1.24e-3 ; d1000  6.54e-3 +- 0.68e-3 ;
#   d2000 RE 4.54e-3 +- 0.57e-3   (paper section is figure-only, no anchors)
# Success check: I=2000 -> 2002 CSV rows; I=4000 -> 4002.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

for d in 100 1000; do for s in 0 1 2 3 4; do
  submit "s41cb_d${d}_s${s}" 1 \
    --example linear --dim $d --preset sisc --dtype float32 --max-iter 2000 \
    --num-dt 100 --num-paths 10000 --renew-frac 0.2 \
    --seed $s --tag "s41cb_d${d}_s${s}"
done; done
for s in 0 1 2 3 4; do
  submit "s41cb_d2000_s${s}" 1 \
    --example linear --dim 2000 --preset sisc --dtype float32 --max-iter 4000 \
    --num-dt 100 --num-paths 6000 --renew-frac 0.2 \
    --seed $s --tag "s41cb_d2000_s${s}"
done
echo "SEC41_SUBMITTED (15 jobs; reference: results/s413_5seed_agg.csv)"
