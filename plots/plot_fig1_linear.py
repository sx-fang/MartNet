#!/usr/bin/env python3
"""fig1_linear (Sec 4.1, 12 panels): rows d=100/1000/2000, cols
  s->v(0,s e_1) (r0), s->v(0,s 1_d) (r0), Mart. loss vs iter (5-run
  mean+2SD), RE vs iter (5-run mean+2SD).
Data: s41cb jobs (jobids in results/s413_5seed_agg.csv).
Outputs: figs/fig1_linear.png + figs/fig1_linear.csv
"""
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AGG = pd.read_csv('tmp_convfig/s413_cb_5seed_agg.csv')

def jobs_of(cell):
    r = AGG[AGG.cell == cell].iloc[0]
    ids = [int(r.seed0_jobid), int(r.s1_jobid), int(r.s2_jobid),
           int(r.s3_jobid), int(r.s4_jobid)]
    return [j for j in ids if not np.isnan(j)]

CELLS = {'d100': 's41_d100', 'd1000': 's41_d1000', 'd2000': 's41_d2000'}
fig, axes = plt.subplots(3, 4, figsize=(12.6, 8.2))
rows = []
for i, (tag, cell) in enumerate(CELLS.items()):
    jobs = jobs_of(cell)
    # (a) diag from seed0 -- collect both curve panels first, then share ylim
    curves = {}
    for k, cv in enumerate(('curve_e1', 'curve_diag')):
        f = glob.glob(f'runs/{jobs[0]}/outputs/*{cv}.csv')[0]
        curves[k] = pd.read_csv(f)
        ax = axes[i, k]
        ax.plot(curves[k].s, curves[k].v_true, '-', color='#1f77b4', lw=1.5,
                label='exact')
        ax.plot(curves[k].s, curves[k].v_pred, 'o', ms=2.5,
                color='#d62728', markevery=3, label='SOC-MartNet')
        ax.set_title(rf'$d={tag[1:]}$' +
                     (r', $s\,\mathbf{e}_1$' if k == 0 else
                      r', $s\,\mathbf{1}_d$'), fontsize=9)
        if i == 0 and k == 0:
            ax.legend(frameon=False, fontsize=7.5)
        ax.grid(ls=':', alpha=0.5)
        if i == 2:
            ax.set_xlabel('$s$')
    # shared y-range across the two curve panels of this row (same-row
    # cols 1-2 share one vertical axis range)
    lo = min(curves[0].v_true.min(), curves[0].v_pred.min(),
             curves[1].v_true.min(), curves[1].v_pred.min())
    hi = max(curves[0].v_true.max(), curves[0].v_pred.max(),
             curves[1].v_true.max(), curves[1].v_pred.max())
    pad = 0.05 * (hi - lo)
    axes[i, 0].set_ylim(lo - pad, hi + pad)
    axes[i, 1].set_ylim(lo - pad, hi + pad)
    # histories 5 runs
    its = None
    loss, err = [], []
    for j in jobs:
        fh = [x for x in glob.glob(f'runs/{j}/outputs/*.csv')
              if 'curve' not in x][0]
        h = pd.read_csv(fh)
        its = h['iter step'].to_numpy()
        loss.append(h['mart loss'].to_numpy())
        err.append(h['error'].to_numpy())
    loss, err = np.array(loss), np.array(err)
    for k, (arr, name) in enumerate(((loss, 'Mart. loss'), (err, 'RE'))):
        m, sd = arr.mean(0), arr.std(0)
        ax = axes[i, 2 + k]
        ax.plot(its, m, color='#2ca02c' if k == 0 else '#d62728', lw=1.3)
        # log axis: one-sided upper band mean..mean+2SD (mean-2SD can go
        # negative)
        ax.fill_between(its, m, m + 2 * sd, alpha=0.25,
                        color='#2ca02c' if k == 0 else '#d62728')
        ax.set_yscale('log')
        ax.set_title(rf'$d={tag[1:]}$, {name} vs Iter.', fontsize=9)
        ax.grid(ls=':', alpha=0.5, which='both')
        if i == 2:
            ax.set_xlabel('iteration')
    r = AGG[AGG.cell == cell].iloc[0]
    rows.append(dict(d=int(tag[1:]), RE_mean=r['mean'], RE_sd=r.sd,
                     jobs=';'.join(map(str, jobs))))

fig.tight_layout()
fig.savefig('tmp_convfig/figs/fig1_linear.png', dpi=200)
pd.DataFrame(rows).to_csv('tmp_convfig/figs/fig1_linear.csv', index=False)
print(pd.DataFrame(rows).to_string(index=False))
