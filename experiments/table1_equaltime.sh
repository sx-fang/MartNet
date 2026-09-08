#!/bin/bash
# Table 1 (Sec 4.8, paper Table 1) EQUAL-TIME reproduction -- the report's final
# caliber for Table 1: MartNet runs within the paper's printed DeepBSDE
# running-time budget of each cell (I = quota / measured per-iteration
# cost: 3200-4900 iterations, budget utilization 85-111%), then the errors
# are compared. 10 cells (d = 100/300/500/800/1000 x {w=256, w=d+10}) x
# 5 seeds = 50 single-GPU jobs.
#
# Common configuration (paper text): HJB-2, point evaluation x0 = 0,
# 4 hidden layers, W per cell, rho width r = 600, uniform batch 128,
# lr_u = 3e-3 d^-0.5, lambda_bar 1e3, M = 1e4, renew 0.2, fp32.
# Per-cell deltas are I (time quota) and lr_rho: the paper-text flat 1e-2
# in the five low-dimensional cells, the authors' relative convention
# (10 x lr_u; lr_u at w256 d1000) in the five high-dimensional cells
# (the flat 1e-2 diverges there). Values verbatim from the producing
# jobs' logs.
#
# Tags match the archived runs (per-wave prefixes t1e2/t1e3/t1e5/t1e6;
# the u1r10/u1r1 suffixes encode lr_rho = 10 x lr_u / = lr_u).
#
# Reference: results/t1e_final_table.csv (10/10 cells <= 5x DeepBSDE
# printed, geomean 0.89x; rebuild it from the raw extracts with
# plots/build_t1e_final_table.py --write; default mode is check-only).
# Success check: I3000 -> 3002 CSV rows; I3200 -> 3202; I3300 -> 3302;
# I3500 -> 3502; I4100 -> 4102; I4600 -> 4602; I4800 -> 4802; I4900 -> 4902.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
export SBATCH_TIMELIMIT=04:00:00

submit_cell() { # tag_base W d I lrrho seed
  local base=$1 W=$2 d=$3 I=$4 lrrho=$5 s=$6
  [ "$W" = "auto" ] && W=$((d+10))
  local tag="${base}_s${s}"
  local lruv
  lruv=$(awk -v d=$d 'BEGIN{printf "%.10e", 3e-3*(d^-0.5)}')
  [ "$lrrho" = "auto10" ] && lrrho=$(awk -v a="$lruv" 'BEGIN{printf "%.10e", a*10}')
  [ "$lrrho" = "auto1" ] && lrrho=$lruv
  submit "$tag" 1 \
    --example hjblq --dim $d --preset sisc --dtype float32 \
    --x0-mode point --num-hidden 4 --width $W --r-dim 600 \
    --batch-size 128 --max-iter $I --num-paths 10000 \
    --renew-frac 0.2 --fd-residual --lam-bar 1000 --controlled-pool \
    --lr-uv "$lruv" --lr-rho "$lrrho" --seed $s --tag "$tag"
}

for s in 0 1 2 3 4; do
  #                tag_base                W    d    I     lr_rho
  submit_cell      t1e2_w256_d100          256  100  3200  1e-2    $s
  submit_cell      t1e2_w256_d300          256  300  3300  1e-2    $s
  submit_cell      t1e2_w256_d500          256  500  4600  1e-2    $s
  submit_cell      t1e3_w256_d800_u1r10    256  800  4800  auto10 $s
  submit_cell      t1e3_w256_d1000_u1r1    256  1000 4900  auto1  $s
  submit_cell      t1e6_wd10_d100          auto 100  3500  1e-2    $s
  submit_cell      t1e2_wd10_d300          auto 300  3500  1e-2    $s
  submit_cell      t1e5_wd10_d500_u1r10    auto 500  3500  auto10 $s
  submit_cell      t1e3_wd10_d800_u1r10    auto 800  3000  auto10 $s
  submit_cell      t1e3_wd10_d1000_u1r10   auto 1000 4100  auto10 $s
done
echo "TABLE1_EQUALTIME_SUBMITTED (50 jobs; reference: results/t1e_final_table.csv)"
