#!/bin/bash
# Sec. 4.4 HJB-1 (paper Fig. 4): NonDegHJB with oscillatory terminal,
# d = 100/1000/2000 x T = 1/0.5/0.1/0.01 (12 cells, single-run per cell
# following the archived repeat convention; single GPU per job).
# lam-bar = 100 for d>=1000 with T=0.01 (paper body convention), else 1000.
#
# Reference RE (figures/fig4_hjb1_tscan.csv):
#   d100:  6.15e-3 / 6.44e-3 / 1.18e-2 / 3.16e-2
#   d1000: 5.48e-3 / 6.98e-3 / 1.94e-2 / 1.14e-1
#   d2000: 2.90e-3 / 2.61e-3 / 1.96e-2 / 9.24e-2   ("smaller T = harder")
# Success check: I=1000 -> 1002 CSV rows.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

for d in 100 1000 2000; do for T in 1.0 0.5 0.1 0.01; do
  LB=1000; if [ "$d" -ge 1000 ] && [ "$T" = "0.01" ]; then LB=100; fi
  submit "s42g_hjb1_d${d}_t${T}_s0" 1 \
    --example hjb --dim $d --te $T --terminal oscillatory --delc 0.1 \
    --preset sisc --dtype float32 --max-iter 1000 --num-paths 10000 \
    --renew-frac 0.2 --fd-residual --lam-bar $LB \
    --controlled-pool --seed 0 --tag "s42g_hjb1_d${d}_t${T}_s0"
done; done
echo "SEC44_FIG4_SUBMITTED (12 jobs; reference: figures/fig4_hjb1_tscan.csv)"
