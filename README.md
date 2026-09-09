# SOC-MartNet for Parabolic Equations and HJB Equations

**Update (v1.1, 2026-09-08):** This release supersedes the initial codebase and provides a complete numerical reproduction package covering the SOC-MartNet results in Sections 4.1–4.8 of the paper.

**Why this release:** After the paper's results were produced, and before its release as the initial codebase, the code evolved through several versions to improve its efficiency and applicability to a wider range of problems. As a result, the initial codebase is not fully consistent with the paper's description and instead more closely aligns with our companion paper's code (https://github.com/sx-fang/DRDM). This release resolves that discrepancy. The implementation was refactored from the original v3-era codebase, available in the preprint's TeX source at https://arxiv.org/src/2405.03169v3, to more faithfully reflect the paper's description.

The package has been validated to reproduce the original results on SLURM clusters with 8 $\times$ A100 GPUs and on RTX 4090 GPUs for smaller-scale experiments. The reproduction report is available in `REPORT.html` (open it in a browser).

## Summary

This repository contains the source code for the numerical experiments presented in:
**SOC-MartNet: A Martingale Neural Network for the Hamilton-Jacobi-Bellman Equation without
Explicit $\inf_{u \in U} H$ in Stochastic Optimal Controls**.

```bibtex
@article {Cai2025SOCMartNet,
    AUTHOR = {Cai, Wei and Fang, Shuixin and Zhou, Tao},
     TITLE = {S{OC}-{M}art{N}et: a martingale neural network for the
              {H}amilton-{J}acobi-{B}ellman equation without explicit
              {$\inf_{u \in U}H$} in stochastic optimal controls},
   JOURNAL = {SIAM J. Sci. Comput.},
  FJOURNAL = {SIAM Journal on Scientific Computing},
    VOLUME = {47},
      YEAR = {2025},
    NUMBER = {4},
     PAGES = {C795--C819},
      ISSN = {1064-8275,1095-7197},
       DOI = {10.1137/24M1681033},
}
```
The paper's preprint is available on arXiv: [https://arxiv.org/abs/2405.03169](https://arxiv.org/abs/2405.03169).

## Contents

| Path | Contents |
|---|---|
| `REPORT.html` | Summary report: all result tables, figures, parameter settings, reproduction guide |
| `code/SOCMartNet-v3-refactored/` | Reproduction code: entry `run.py` + the `socmartnet` package + `tests/validation_anchors/` (bit-level validation data) |
| `slurm/` | Frozen-copy submission channel: `submit_job.sh` + `run_one.slurm` (a site template; see the reproduction guide) |
| `experiments/` | Exact producing command lines per experiment family (`experiments.csv`) + single-case reproduction scripts (headers state coverage / reference values / expected CSV line count) + the driver `reproduce_all.sh` + `smoke_test.sh` |
| `results/` | Aggregate CSVs + the raw final-row extracts (`t2b_raw_final.txt`, `t1e*_raw_final.txt`) + `final_rows.csv` (per-run final-row evidence) |
| `figures/` | Figure PNGs + per-figure summary-data CSVs |
| `plots/` | Plotting/aggregation scripts: `plot_*.py` and `build_final_rows.py` / `build_s44_d1e4_agg.py` expect the archival `runs/<jobid>/outputs` layout; `build_t1e_final_table.py` / `build_t2b_agg.py` re-aggregate the raw extracts in `results/` (default: check against the shipped files without writing; `--write` regenerates) |

## Reproduction guide

The environment is pinned in `pyproject.toml`; install it with

```bash
pip install .
```

On hosts whose GPU driver predates CUDA 13 (e.g. the CUDA 12.8 driver of the A100
cluster the results were produced on), the PyPI default `torch` wheel fails at CUDA
init ("NVIDIA driver ... too old"); install the cu128 build the results were produced
with first -- `pip install .` then keeps it instead of replacing it:

```bash
pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

A quick end-to-end check that the code runs in your environment (one tiny linear-parabolic
solve, about a minute on one GPU; prints `SMOKE OK` on success; `PYTHON=...` overrides the
interpreter):

```bash
bash experiments/smoke_test.sh
```

The full reproduction runs on a SLURM cluster with A100-class GPUs:

```bash
# 1) This repository doubles as the workdir: clone (or copy the tree) onto your cluster
#    and work at its root -- submit_job.sh and the scripts expect exactly this layout.
#    Adapt the site template: edit slurm/run_one.slurm's #SBATCH account/partition header
#    and PY variable (or export SBATCH_ACCOUNT / SBATCH_PARTITION / PYTHON)
# 2) Submit: run the single-case scripts on demand (e.g. bash experiments/sec41_linear_parabolic.sh;
#    the full list is in the header of experiments/reproduce_all.sh), or bash experiments/reproduce_all.sh for everything
# 3) Success check: CSV line count (1000 iterations → 1002 lines, etc.); sacct COMPLETED alone is not sufficient
# 4) Compare: aggregate CSVs under results/ and final_rows.csv
```

## AI assistance

During the preparation of the reproduction code and report, the AI assistants GLM 5.3 and
Kimi K3 (256K) participated in refactoring the v3-era codebase, running the reproduction
experiments, and writing the reproduction report.
