#!/usr/bin/env python3
"""fig7_shifted_target (Sec 4.6, 6 panels): per dim d in {100,500}
  (a/d) s -> v(0, s*1_d) diag (seed0 curve, like the paper's r0_diag),
  (b/e) RE vs iteration (5 seeds, mean + 2*SD band),
  (c/f) J(u_theta) vs iteration (5 seeds, mean + 2*SD band) with the
        theoretical lower bound J(u*) = v(0, 0) (seed0 diag v_true at s=0).
Data: d100 = 510052-510056, d500 = 509996-510000 (5 seeds each, point
mode; the seed0 run of each dim also carries the curve_diag CSV).
Outputs: figs/fig7_shifted_target.png + figs/fig7_shifted_target_summary.csv
"""
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DIMS = {100: [510052, 510053, 510054, 510055, 510056],
        500: [509996, 509997, 509998, 509999, 510000]}
# d100: strengthened configuration (I=3000, lr_uv x4 = 1.2e-3, lr_rho =
# 40 x lr_uv = 4.8e-2, renew 0.1), RE 1.45e-3 +- 0.0004.
# d500: authors' archived-INI configuration (I=3000, lr_rho = 10 x lr_u,
# renew 0.2), RE 3.23e-3.

fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.4))
summary = []
col0_extents = []
for row, d in enumerate(DIMS):
    # (a) diag from seed0
    f0 = glob.glob(f'runs/{DIMS[d][0]}/outputs/*curve_diag.csv')[0]
    df0 = pd.read_csv(f0)
    vstar = float(np.interp(0.0, df0.s, df0.v_true))
    ax = axes[row, 0]
    ax.plot(df0.s, df0.v_true, '-', color='#1f77b4', lw=1.6, label='exact')
    ax.plot(df0.s, df0.v_pred, 'o', ms=2.5, color='#d62728',
            label=r'SOC-MartNet', markevery=3)
    ax.set_xlabel(r'$s$')
    ax.set_ylabel(r'$v(0, s\,\mathbf{1}_d)$')
    ax.set_title(f'$d={d}$', fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(ls=':', alpha=0.5)
    col0_extents.append((min(df0.v_true.min(), df0.v_pred.min()),
                         max(df0.v_true.max(), df0.v_pred.max())))

    # histories: 5 seeds
    its, res, js = None, [], []
    for j in DIMS[d]:
        fh = glob.glob(f'runs/{j}/outputs/*.csv')
        fh = [x for x in fh if 'curve' not in x][0]
        h = pd.read_csv(fh)
        its = h['iter step'].to_numpy()
        res.append(h['error'].to_numpy())
        js.append(h['cost'].to_numpy())
    res, js = np.array(res), np.array(js)
    re_m, re_sd = res.mean(0), res.std(0)
    j_m, j_sd = js.mean(0), js.std(0)
    ax = axes[row, 1]
    ax.plot(its, re_m, color='#d62728', lw=1.4)
    # log axis: one-sided upper band mean..mean+2SD
    ax.fill_between(its, re_m, re_m + 2 * re_sd,
                    color='#d62728', alpha=0.25)
    ax.set_yscale('log')
    ax.set_xlabel('iteration')
    ax.set_ylabel('RE')
    ax.set_title(f'$d={d}$', fontsize=10)
    ax.grid(ls=':', alpha=0.5, which='both')

    ax = axes[row, 2]
    ax.plot(its, j_m, color='#1f77b4', lw=1.4, label=r'$J(u_\theta)$')
    ax.fill_between(its, j_m - 2 * j_sd, j_m + 2 * j_sd,
                    color='#1f77b4', alpha=0.25)
    ax.axhline(vstar, color='orange', ls='--', lw=1.5,
               label=r'$J(u^*)=v(0,\mathbf{0})$')
    ax.set_xlabel('iteration')
    ax.set_ylabel(r'$J(u_\theta)$')
    ax.set_title(f'$d={d}$', fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(ls=':', alpha=0.5)
    summary.append(dict(d=d, RE_final=re_m[-1], RE_sd=res[:, -1].std(),
                        J_final=j_m[-1], Jstar=vstar,
                        gap=(j_m[-1] - vstar) / abs(vstar)))

# the first-column (diag curve) panels of both rows share one vertical
# axis range so the d=100 / d=500 solution amplitudes compare directly
lo = min(e[0] for e in col0_extents)
hi = max(e[1] for e in col0_extents)
pad = 0.05 * (hi - lo)
for row in range(len(DIMS)):
    axes[row, 0].set_ylim(lo - pad, hi + pad)

fig.tight_layout()
fig.savefig('tmp_convfig/figs/fig7_shifted_target.png', dpi=200)
pd.DataFrame(summary).to_csv('tmp_convfig/figs/fig7_shifted_target_summary.csv',
                             index=False)
print(pd.DataFrame(summary).to_string(index=False))
