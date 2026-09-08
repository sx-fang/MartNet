#!/bin/bash
# Sec. 4.6 shifted-target SOCP (paper Fig. 7): ShiftTargetHJB with
# g = 10 ln(0.5(1+|x-3*1_d|^2)), d = 100/500, 5 seeds. Three configurations:
#
#   (a) paper-text configuration I=1000 (reference batch; the paper figure's
#       3000-iteration axis makes this batch not directly comparable);
#   (b) authors' archived-INI configuration I=3000, lr_rho = 10 x lr_u
#       (producing INIs SOCMN154/156) -- d500 headline: 3.23e-3 +- 0.51e-3,
#       J(u) gap to the theoretical bound 0.29%; d100 gives
#       1.16e-2 +- 0.18e-2 in this configuration;
#   (c) d100 with strengthened adversarial training (I=3000, lr_uv x4
#       = 1.2e-3, lr_rho = 40 x lr_uv = 4.8e-2, renew 0.1) -- report
#       headline: RE 1.45e-3 +- 0.44e-3, J(u) gap 0.11%.
#
# Reference: results/s413_5seed_agg.csv + figures/fig7_shifted_target_summary.csv.
# Success check: I=1000 -> 1002 CSV rows; I=3000 -> 3002.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

### (a) paper-text configuration (I1000, reference)
for d in 100 500; do for s in 0 1 2 3 4; do
  submit "s46f_st_d${d}_s${s}" 1 \
    --example shifttarget --dim $d --preset sisc --dtype float32 \
    --max-iter 1000 --num-paths 10000 --renew-frac 0.2 --fd-residual \
    --lam-bar 1000 --controlled-pool --seed $s \
    --tag "s46f_st_d${d}_s${s}"
done; done

### (b) authors' archived-INI configuration (I3000, lr_rho = 10 x lr_u)
for spec in "100 0.003" "500 1.3416407864998738e-3"; do
  set -- $spec; d=$1; rho=$2
  for s in 0 1 2 3 4; do
    submit "s46i_st_d${d}_s${s}" 1 \
      --example shifttarget --dim $d --preset sisc --dtype float32 \
      --max-iter 3000 --num-paths 10000 --renew-frac 0.2 --fd-residual \
      --lam-bar 1000 --controlled-pool --lr-rho "$rho" \
      --seed $s --tag "s46i_st_d${d}_s${s}"
  done
done

### (c) d100 strengthened adversarial training (report headline)
### (archived tags: seed0 = s46j, seeds 1-4 = s46k)
submit "s46j_rho40x_s0" 1 \
  --example shifttarget --dim 100 --preset sisc --dtype float32 \
  --max-iter 3000 --num-paths 10000 --renew-frac 0.1 --fd-residual \
  --lam-bar 1000 --controlled-pool --lr-uv 1.2e-3 --lr-rho 4.8e-2 \
  --seed 0 --tag "s46j_rho40x_s0"
for s in 1 2 3 4; do
  submit "s46k_rho40x_s${s}" 1 \
    --example shifttarget --dim 100 --preset sisc --dtype float32 \
    --max-iter 3000 --num-paths 10000 --renew-frac 0.1 --fd-residual \
    --lam-bar 1000 --controlled-pool --lr-uv 1.2e-3 --lr-rho 4.8e-2 \
    --seed $s --tag "s46k_rho40x_s${s}"
done
echo "SEC46_SUBMITTED (25 jobs; reference: results/s413_5seed_agg.csv)"
