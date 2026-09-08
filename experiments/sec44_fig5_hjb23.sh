#!/bin/bash
# Sec. 4.4 HJB-2 / HJB-3 high-dimensional (paper Fig. 5):
#   d=2000  single-run r0 curves per the paper (seed 0, 8 GPUs, I=8000,
#           lr x4 -- HJB-2 delta0=0.2, HJB-3 delta0=0.1)
#   d=10000 5 seeds, 8 GPUs, I=6000, width d+10, batch 64 (authors' d=10000
#           INI tier), lr x4
#
# Reference (results/s44_d1e4_5seed_agg.csv, figures/fig5_hjb_highdim.csv):
#   HJB-2 d=2000  RE 0.0184 (archived 5-run anchor 0.01281 +- 0.00098, 1.43x)
#   HJB-3 d=2000  RE 0.0215 (anchor 0.03289 +- 0.00311, 0.65x)
#   HJB-2 d=1e4   RE 0.01387 +- 0.00086 (anchor 0.00728 +- 0.00082, 1.91x)
#   HJB-3 d=1e4   RE 0.02838 +- 0.00359 (anchor 0.02086 +- 0.00102, 1.36x)
# Success check: I=8000 -> 8002 CSV rows; I=6000 -> 6002.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

### d=2000 (single run per equation)
for eq in 2 3; do
  if [ $eq = 2 ]; then DD=0.2; else DD=0.1; fi
  submit "s44g_hjb${eq}_lr4" 8 \
    --example hjblq --dim 2000 --delta0 $DD --preset sisc --dtype float32 \
    --max-iter 8000 --num-paths 10000 --renew-frac 0.2 --fd-residual \
    --lam-bar 1000 --controlled-pool \
    --lr-uv 2.7438303115639573e-05 --lr-rho 2.7438303115639574e-04 \
    --seed 0 --tag "s44g_hjb${eq}_lr4"
done

### d=10000 (5 seeds per equation)
for s in 0 1 2 3 4; do for eq in 2 3; do
  if [ $eq = 2 ]; then DD=0.2; else DD=0.1; fi
  submit "s44i_hjb${eq}_d1e4b64lr4_s${s}" 8 \
    --example hjblq --dim 10000 --delta0 $DD --preset sisc --dtype float32 \
    --max-iter 6000 --num-paths 10000 --renew-frac 0.2 --fd-residual \
    --lam-bar 1000 --controlled-pool --width 10010 \
    --batch-size 64 --lr-uv 7.5714866e-06 --lr-rho 7.5714866e-05 \
    --seed $s --tag "s44i_hjb${eq}_d1e4b64lr4_s${s}"
done; done
echo "SEC44_FIG5_SUBMITTED (12 jobs; reference: results/s44_d1e4_5seed_agg.csv)"
