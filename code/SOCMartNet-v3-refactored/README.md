# SOCMartNet v3 (SOC-MartNet baseline, refactored)

A refactor of the arXiv:2405.03169 **v3** codebase (the lineage that
produced the paper's numbers): modules
irrelevant to the paper's numerical experiments are removed, variable symbols are aligned with
the paper, and a single CLI covers all §4 examples. **Numerical behavior is item-by-item
identical to the v3 code** (refactor only, no numerical change — seeds, RNG call order, DDP
bias correction, schedulers, and the λ update position are all preserved as in the original).

## Removed modules (relative to the v3 codebase)

| Blueprint item | Disposition |
|---|---|
| `prabequations.py` (PINN/FD legacy, scipy ODE) | removed (not referenced by testsocmart) |
| `VovNet` / `StackNet` / `ortho_linear` (unused network variants) | removed, unified into `DNNtx` |
| `SDEsolver`, `loss_zc`, `loss_term`, `ortho_loss`, `tensor_sample`, `pretrain_test` | removed (dead code) |
| `NonDegHJBv2`, `Counter2/2_1/expneqx2/expcos/sin`, `Counter3`, `SDEGen` | removed (no corresponding example in the v3 paper) |
| `plot*.py`, `srun_socp8.sh` | removed, replaced by CSV + optional plotting |
| `testsocmart.py` (hardcoded loop lists) | replaced by the `run.py` CLI |
| `fbsnn.py` (FBSNN/deep-BSDE-style benchmark trainer) | removed (2026-09-07 cleanup: never invoked by any formal run of this reproduction; the §4.8 deep BSDE comparison uses the paper's printed values) |

Preserved: the SOC-MartNet solver (Alg 3.1 SOC mode + Alg 3.2 PDE mode), the Cole–Hopf
blockwise MC reference solution, and the D₀ segment grid.

## Paper §4 examples → command mapping

| Paper example | Command | Dimensions / times |
|---|---|---|
| §4.1 linear parabolic (manufactured sin solution) | `python run.py --example linear` | d=50,100,500,1000, T=1 |
| §4.2.1 semilinear smooth g | `--example semilinear --terminal smooth` | d=50,100,500,1000, T=1 |
| §4.2.2 semilinear oscillatory g | `--example semilinear --terminal oscillatory` | d=100, T=1,0.1,0.05,0.01 |
| §4.3.1 HJB smooth g (control mode) | `--example hjb --terminal smooth` | d=50..2000, T=1 |
| §4.3.2 HJB oscillatory g | `--example hjb --terminal oscillatory` | d=50,100,1000,2000 × T=1,0.5,0.1,0.01; `--lam-bar 100` for d≥1000 with T=0.01 |

Common defaults (paper §4 opening): T=1, N=100, M=1e5 (X₀ on the D₀ segment grid), RMSProp,
batch 200→400→800→1600 (tiered at I/4), lr δ₁=δ₂=δ₀·1e-3 and δ₃=1e-2 decayed by 0.01^{i/I}
(approximated as StepLR(step=I/10, γ=0.01^{1/9})), λ₀=10, δ₄=10, λ̄=1e3, J=2K=2,
u/v networks 4 hidden layers × (2d+20) ReLU, ρ=sin(W₁t+W₂x+b) ∈ R^{2d+300}, fp64, DDP.
Two presets: `--preset v3paper` (the paper-text settings above); `--preset cube` (the actual
configuration behind the authors' archived CSVs: 3 hidden layers, r=2d+500, lr=1e-3, λ̄=1e4,
I=1000 (d≤100) / 2000 (d>100)).

## Symbol correspondence (paper ↔ code)

`u_alpha`/`v_theta`/`rho_eta` ↔ u_α/v_θ/ρ_η; `lam0/delta4/lam_bar` ↔ λ₀/δ₄/λ̄;
`J/K` ↔ J/K (step_uv/step_rho); `mart_loss` ↔ |G|² (DDP-corrected version);
`ham_mean` ↔ (1/|A|)ΣH (the paper's first term carries an extra Δt factor, absorbed by λ and
the learning rates — see the note above); `dM` ↔ ΔM/Δt (code convention; the paper's G carries
an extra Δt factor); `segment_grid` ↔ D₀⊂S₁∪S₂.

## Refactoring correctness criterion (reproduction test) — **passed ✅**

Example: §4.3.1, HJB smooth g, d=100, T=1, SOCMartNet (Alg. 3.1), cube preset, seed 0.
Criterion baseline: the authors' reference CSV
`tests/validation_anchors/reference_data/SOCMartNet_Non-deg_HJB_d100.csv`
(I=1000, final RE=1.6792e-2, mart loss 1.12e-3, rt≈31s @ 8×V100 DDP).
Two-arm comparison: A = the v3 original code (unmodified numerical path);
B = this refactored code (`run.py`). The anchor data ships with the
package: `tests/validation_anchors/`
(`outputs_orig/` = arm A, `outputs_refac/` = arm B, `reference_data/` = the
authors' reference).

**Results (artifacts: `tests/validation_anchors/outputs_orig/` = arm A,
`tests/validation_anchors/outputs_refac/` = arm B):**

- **A vs B (same seed, same machine, same configuration): the 1001-row training history × all
  columns (hami / mart loss / RE / epoch) has a row-wise maximum difference of 0.000e+00 —
  bit-level identity**: the refactor has zero numerical drift from the v3 code; the
  criterion passes decisively.
- B vs the authors' reference (different hardware / RNG): final RE 1.6489e-2 vs 1.6792e-2
  (ratio 0.982); min RE 4.83e-3 vs 5.59e-3; checkpoint REs (it=0/100/200/…/1000) all at the
  1e-2 level; mart loss 6.3e-5 vs 1.12e-3 (DDP correction-factor difference, ws=1 vs ws=8, see
  above); rt 72.6 s (1×A100) vs 31.4 s (8×V100).

## Known convention differences (inherited from the blueprint, left unchanged)

- The paper's L carries a Δt factor in the H term and in G; the code uses dM=ΔM/Δt and
  ham_mean without Δt (the scaling is absorbed by λ / the learning rates).
- The λ update sits inside the J loop in the code (the paper's Alg 3.1 places it in the
  (λ,η) ascent block).
- §4.3.2 paper: λ̄=100 for d≥1000 with T=0.01; the authors' d1000 directory uses λ̄=1e2 for
  all T, the d2000 directory uses λ̄=1e4 — specify explicitly via the CLI `--lam-bar`.
- The paper's Eq. (4.3) writes the oscillatory terminal with δ=π/10; the code uses delc/π
  (delc=0.1). The reference solution and training use the same g — internally consistent,
  not affecting the reproduction criterion.
- The v3 paper has no Table 1 (it first appears in the SISC final version §4.8); HJB-2/3
  (b=1, δ₀ scaling) are outside the v3 code's coverage.

## Accepted-version (SISC) extensions

All examples of the accepted paper (SOCMartNet_accepted.tex) §4.1–4.8 are ported
**additively** on top of the v3 refactor base; the existing v3paper/cube numerical paths are
unchanged (bit-identical when all new switches are off).

### Accepted-version §4 examples → command mapping

| Paper example | Command | Notes |
|---|---|---|
| §4.1 linear parabolic Counter | `--example linear` | v3 class reused (≡ the accepted-version Counter) |
| §4.2 semilinear oscillatory | `--example semilinear --terminal oscillatory` | v3 class reused |
| §4.3 time-convergence eq_linsin | `--example linsin` | new class `LinearSinAC`: μ=sin(2x), σ=1+0.5 sin(5t+x), manufactured solution v=1+mean sin(t+x); `--allen-cahn` keeps the explicit v−v³ |
| §4.4 HJB-1 | `--example hjb --terminal oscillatory` | ≡ NonDegHJBv3 (b=0, δ₀=1, delc=0.1, terminal 1+g) |
| §4.4 HJB-2 / HJB-3 | `--example hjblq` (default δ₀=0.2 = HJB-2) / `--delta0 0.1` (HJB-3) | new class `HJBLQ`: b=1, delc=0.3, terminal without +1 |
| §4.5 space-time domain validity | `--example hjblq --eval-ptx --eval-path-re` | r=0.125/0.25 curves + RE(t) along 8 sampled paths |
| §4.6 shifted terminal | `--example shifttarget` | new class `ShiftTargetHJB`: b=0, δ₀=0.1, g=10·ln(0.5(1+\|x−3·1_d\|²)), J(u) logged per iteration (256-path MC, `--cost-paths`) |
| §4.7 ε perturbation | `--example hjblq --eps-purb {1,0.5,0.25,0.125}` | H plus ε·sin(1_dᵀk); reference solution at ε=0 (paper convention) |
| §4.8 Table 1/2 | `--example hjblq --x0-mode point` | single point x=0; Table 1 uses HJB-2, Table 2 uses `--delta0 0.1 --max-iter 6000` |

### `--preset sisc` (accepted-version §4 opening + the authors' archived configs)

| Item | sisc | v3paper | cube |
|---|---|---|---|
| Hidden layers × width | 6 × (d+10) | 4 × 2(d+10) | 3 × 2(d+10) |
| ρ dimension r | 600 | 2d+300 | 2d+500 |
| batch | 256 (d≤1000) / 128 (d>1000), fixed | 200/400/800/1600 tiered | same as v3paper |
| lr (u,v) | 3d^{-0.5}·1e-3 (d>1000: 3d^{-0.8}·1e-3) | same | 1e-3 |
| lr decay | ≤100-step staircase quantization of 0.01^{i/I} (authors' archived configs: StepLR(100, 0.01^{100/I})) | StepLR(I/10, 0.01^{1/9}) | same as v3paper |
| I | 1000 (Table-1 convention; Table 2 uses `--max-iter 6000`) | 1000/2000 | 1000/2000 |
| λ̄ | 1e3 | 1e3 | 1e4 |

The sisc preset enables extended logging by default (the CSV appends `rel_linf`,
`mean_vtrue_t0`, and, for SOC examples, a `cost` column) plus end-state region-curve CSVs
(`*_curve_{e1,diag,manifold}.csv`). `--dtype float32` returns to the archived convention
(the fp64 default is the v3 validation convention; the paper does not state it).

### Evaluation utilities

- `socmartnet/evaluate.py`: `curve_values` (S1/S2/S3 curves), `ptx_values` (§4.5
  v(r, s𝟏+r𝟏)), `path_re` (§4.5 path RE(t), by default 4 paths each from 0_d and l(0.75)).
  Multi-seed aggregation is handled by the top-level `plots/build_*.py`.

### Accepted-version new conventions (recorded only, not modified)

- **HJB-1/2/3 mapping**: HJB-1 ≡ v3 NonDegHJBv3 (terminal 1+g, Cole–Hopf reference 1−ln…);
  HJB-2/3 use the accepted-version HJBLQ lineage (terminal g without +1, reference −ln…). The paper's +1 in
  J(u)=1+E[…] is likewise omitted in the accepted-version archived comput_cost.
- **ε₁ notation**: the paper's PDE writes ε₁Δv, its SDE √(2ε₁)dB, c₁=ε₁^{−2}; the
  self-consistent reading is the code convention (σ=δ₀√2, PDE diffusion δ₀², c₁=δ₀^{−2},
  Cole–Hopf θ=1). HJB-2/3's δ₀=0.2/0.1 is exactly the paper's ε₁.
- **ε₀**: the paper writes 0.1π/0.3π; the code convention is delc/π (delc=0.1/0.3). This
  reproduction follows the code convention.
- **§4.6 center of g**: the paper text has target=3·1_d; the authors' archived runs (SOCMN154) actually
  uses a g without the shift (and the SOCMN147 archive uses the quadratic mean((x−3)²) with
  coefficient 1). **This reproduction follows the paper** — default target=𝟑, log form,
  C_g=10; the archived convention is recorded only and can be reproduced with
  `--target-shift 0` as a cross-check.
- **§4.3 Allen–Cahn**: the paper has f=v−v³+f̄; the archived SOCMN179's LinearSin driver
  contains no explicit v−v³ (absorbed into f̄, with update_f=False precomputed offline, so it
  structurally cannot depend on network values). **This reproduction's judgment: the SOCMN179
  version may not be the code that actually produced the paper's time-convergence figure
  (Fig. 3) — the true implementation may be lost — so the paper convention is followed**: the
  explicit v−v³ is kept by default (`LinearSinAC(allen_cahn=True)`, loss f = −(v−v³) − f̄ with
  f̄ the manufactured compensator); `--no-allen-cahn` returns to the archived-compensator
  convention as a cross-check. The two forms have identical residuals at the exact solution
  (the v−v³ term cancels against the corresponding term in f̄).
- **Batch sampling and the dM scheme**: the paper text describes permutation blocks within an
  epoch; the code keeps the v3 validation convention (per-iteration randperm subsets + an
  explicit H via autograd on v_x). The authors' accepted-version platform's SocMartNet class
  uses another
  convention: a finite-difference approximation of dM = Δv/Δt + (μ_sys−μ_pil)·v_x
  (x_forw = x + Δμ·Δt²) plus path renewal with epochsize=10000, rate_newpath=0.2. The
  accepted-version FD
  scheme is implemented as an optional solver path (`--fd-residual`).
