#!/usr/bin/env python3
"""fig6_domain_validity (Sec 4.5, 3 panels): HJB-2 d=1000, 5 seeds.
  (a) s -> v(r, s*1_d + r*1_d), r = 0.125   (b) same, r = 0.25
  (c) RE vs t along sample paths (pathre: t, re_pool, re_path_mean/std).
Data: per-seed jobids hardcoded below; reads the runs/<jobid>/outputs layout.
Outputs: figs/fig6_domain_validity.png + figs/fig6_domain_validity.csv
"""
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

JOBS = [5431552, 491371, 491372, 491373, 491374]

def read(j, suffix):
    f = glob.glob(f'runs/{j}/outputs/*{suffix}.csv')
    return pd.read_csv(f[0]) if f else None

fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
outcsv = {}
for k, r in enumerate(('0.125', '0.25')):
    preds, s, vt = [], None, None
    for j in JOBS:
        df = read(j, f'ptx{r}')
        if df is None:
            continue
        s, vt = df.s.to_numpy(), df.v_true.to_numpy()
        preds.append(df.v_pred.to_numpy())
    mean = np.mean(preds, axis=0)
    sd = np.std(preds, axis=0)
    ax = axes[k]
    ax.plot(s, vt, '-', color='#1f77b4', lw=1.6, label='exact')
    ax.plot(s, mean, 'o', ms=2.5, color='#d62728', markevery=3,
            label='SOC-MartNet (5-run mean)')
    ax.fill_between(s, mean - 2 * sd, mean + 2 * sd, color='#d62728',
                    alpha=0.2)
    ax.set_xlabel(r'$s$')
    ax.set_ylabel(rf'$v({r},\, s\mathbf{{1}}_d + {r}\mathbf{{1}}_d)$')
    ax.set_title(rf'$r = {r}$', fontsize=10)
    ax.legend(frameon=False, fontsize=7.5)
    ax.grid(ls=':', alpha=0.5)
    outcsv[f's_ptx{r}'] = s
    outcsv[f'vtrue_ptx{r}'] = vt
    outcsv[f'vpred_ptx{r}'] = mean

ts, means, sds = None, [], []
for j in JOBS:
    df = read(j, 'pathre')
    if df is None:
        continue
    ts = df.t.to_numpy()
    means.append(df.re_path_mean.to_numpy())
    sds.append(df.re_path_std.to_numpy())
mean, sd = np.mean(means, axis=0), np.mean(sds, axis=0)
ax = axes[2]
ax.plot(ts, mean, '-', color='#d62728', lw=1.5,
        label='RE on paths (5-run mean)')
ax.fill_between(ts, mean - 2 * sd, mean + 2 * sd, color='#d62728', alpha=0.2)
ax.set_xlabel(r'$t$')
ax.set_ylabel('RE')
ax.set_title('RE vs $t$', fontsize=10)
ax.legend(frameon=False, fontsize=7.5)
ax.grid(ls=':', alpha=0.5)
outcsv['t'] = pd.Series(ts)
outcsv['re_path_mean5'] = pd.Series(mean)
outcsv['re_path_sd2x'] = pd.Series(2 * sd)

fig.tight_layout()
fig.savefig('tmp_convfig/figs/fig6_domain_validity.png', dpi=200)
pd.DataFrame({k: pd.Series(v) for k, v in outcsv.items()}).to_csv(
    'tmp_convfig/figs/fig6_domain_validity.csv', index=False)
print('ptx mean |v_p - v_t| (5-run): r=.125 %.4f  r=.25 %.4f; pathre final %.4f'
      % (np.abs(outcsv['vpred_ptx0.125'] - outcsv['vtrue_ptx0.125']).mean(),
         np.abs(outcsv['vpred_ptx0.25'] - outcsv['vtrue_ptx0.25']).mean(),
         mean[-1]))
