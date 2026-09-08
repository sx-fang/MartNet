#!/bin/bash
# Sec. 4.5 validity on the space-time region (paper Fig. 6): HJB-2,
# d=1000, I=8000, lr x4, lam-bar 10, with --eval-ptx (v(r, s1+r1) curves at
# r=0.125/0.25) and --eval-path-re (RE(t) along sample paths);
# 5 seeds (5 jobs, single GPU).
#
# Reference (figures/fig6_domain_validity.csv):
#   terminal RE 0.02062 +- 0.00172 (archived anchor 1.41e-2, ratio 1.46);
#   ptx |dv| = 0.0172 (r=0.125) / 0.0308 (r=0.25); path RE grows mildly in t.
# Success check: I=8000 -> 8002 CSV rows (+ *_curve_* / *_ptx / *_pathre CSVs).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

for s in 0 1 2 3 4; do
  submit "s45fcb_d1000_s${s}" 1 \
    --example hjblq --dim 1000 --preset sisc --dtype float32 --max-iter 8000 \
    --num-paths 10000 --renew-frac 0.2 --fd-residual --lam-bar 10 \
    --controlled-pool --eval-ptx --eval-path-re \
    --lr-uv 4.7772864e-5 --lr-rho 4.7772864e-4 --seed $s \
    --tag "s45fcb_d1000_s${s}"
done
echo "SEC45_SUBMITTED (5 jobs; reference: figures/fig6_domain_validity.csv)"
