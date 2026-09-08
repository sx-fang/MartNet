#!/bin/bash
# Sec. 4.7 epsilon-perturbation suite (paper Fig. 8): HJB-2 d=1000 with
# eps_purb in {1, 1/2, 1/4, 1/8} added to H, I=4000, 5 seeds (20 jobs).
#
# Reference (results/s413_5seed_agg.csv): RE vs the eps=0 reference
# solution decreases monotonically -- 1.743 / 0.823 / 0.370 / 0.203
# (5-seed means; the paper's claim of monotone approach reproduced).
# Success check: I=4000 -> 4002 CSV rows.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

for EPS in 1.0 0.5 0.25 0.125; do for s in 0 1 2 3 4; do
  submit "s47f_eps${EPS}_d1000_s${s}" 1 \
    --example hjblq --dim 1000 --eps-purb $EPS --preset sisc --dtype float32 \
    --max-iter 4000 --num-paths 10000 --renew-frac 0.2 --fd-residual \
    --lam-bar 1000 --controlled-pool --seed $s \
    --tag "s47f_eps${EPS}_d1000_s${s}"
done; done
echo "SEC47_SUBMITTED (20 jobs; reference: results/s413_5seed_agg.csv)"
