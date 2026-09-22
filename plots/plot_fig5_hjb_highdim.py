#!/usr/bin/env python3
"""Paper Fig. 5 (Sec 4.4, full 3x4 layout mirroring the accepted paper):
  row 1 (d=2000, 5-seed mean curves with mean+-2SD bands):
    (a) HJB-2 diag  (b) HJB-2 manifold  (c) HJB-3 diag  (d) HJB-3 manifold
  row 2 (HJB-2, d=1e4): (e) manifold r0 curve, (f) Mart. loss vs iter,
    (g) Hamiltonian vs iter -- LINEAR y-axis fixed to [-0.5, 0.5],
    (h) RE vs iter.
  row 3 (HJB-3, d=1e4): (i) manifold r0 curve, (j)-(l) same histories.
  History panels: mean across 5 seeds; log-axis panels (Mart loss, RE)
  use the one-sided mean..mean+2SD band (paper caption: "mean + 2 x SD");
  linear-axis panels (Hamiltonian, d=2000 value curves) use two-sided
  mean+-2SD bands.
Data: d2000 = 5 seeds each -- HJB-2 496285 (s0) + 534414/534417/534418/534419
      (s1-4); HJB-3 496291 (s0) + 534420/534421/534422/534423 (s1-4)
      (lam1000+lr4x+I8000);
      HJB-2 d1e4 = 496434 (seed0, the selected batch-64 scan arm)
                   + 500699/701/703/705 (seeds 1-4);
      HJB-3 d1e4 = 496436 (seed0, the selected batch-64 scan arm)
                   + 500700/702/704/706 (seeds 1-4).
Outputs: figs/fig5_hjb_highdim.png + figs/fig5_hjb_highdim.csv
"""
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

D2000 = [('HJB-2', 'diag', [496285, 534414, 534417, 534418, 534419]),
         ('HJB-2', 'manifold', [496285, 534414, 534417, 534418, 534419]),
         ('HJB-3', 'diag', [496291, 534420, 534421, 534422, 534423]),
         ('HJB-3', 'manifold', [496291, 534420, 534421, 534422, 534423])]
D1E4 = {'HJB-2': [496434, 500699, 500701, 500703, 500705],
        'HJB-3': [496436, 500700, 500702, 500704, 500706]}
HAM_LIM = (-0.5, 0.5)

fig, axes = plt.subplots(3, 4, figsize=(13.4, 9.6))
rows = []

# ---- row 1: d2000 curves, 5-seed mean +- 2SD -------------------------------
for k, (eq, cv, jobs) in enumerate(D2000):
    ax = axes[0, k]
    vp, res = [], []
    for j in jobs:
        f = glob.glob(f'runs/{j}/outputs/*curve_{cv}.csv')[0]
        vp.append(pd.read_csv(f).v_pred.to_numpy())
        fh = [x for x in glob.glob(f'runs/{j}/outputs/*.csv')
              if 'curve' not in x][0]
        res.append(pd.read_csv(fh).error.iloc[-1])
    df0 = pd.read_csv(glob.glob(f'runs/{jobs[0]}/outputs/*curve_{cv}.csv')[0])
    m, sd = np.mean(vp, 0), np.std(vp, 0)
    ax.plot(df0.s, df0.v_true, '-', color='#1f77b4', lw=1.5, label='exact')
    ax.plot(df0.s, m, 'o', ms=2.2, color='#d62728', markevery=3,
            label='SOC-MartNet')
    ax.fill_between(df0.s, m - 2 * sd, m + 2 * sd, color='#d62728',
                    alpha=0.25, label=r'mean$\pm$2SD (5 runs)')
    ax.set_title(rf'{eq}, $d=2000$, {cv}' + '\n' +
                 rf'RE$ = {np.mean(res):.2e} \pm {np.std(res, ddof=1):.1e}$',
                 fontsize=9)
    ax.grid(ls=':', alpha=0.5)
    ax.set_xlabel('$s$')
    if k == 0:
        ax.legend(fontsize=7, loc='best')
    rows.append(dict(eq=eq, d=2000, curve=cv,
                     jobid=';'.join(map(str, jobs)),
                     RE=float(np.mean(res)),
                     RE_sd=float(np.std(res, ddof=1))))

# ---- rows 2-3: d1e4 per equation -------------------------------------------
def d1e4_block(row, eq):
    jobs = D1E4[eq]
    # manifold curve from seed0
    ax = axes[row, 0]
    f = glob.glob(f'runs/{jobs[0]}/outputs/*curve_manifold.csv')[0]
    df = pd.read_csv(f)
    fh = [x for x in glob.glob(f'runs/{jobs[0]}/outputs/*.csv')
          if 'curve' not in x][0]
    re0 = pd.read_csv(fh).error.iloc[-1]
    ax.plot(df.s, df.v_true, '-', color='#1f77b4', lw=1.5, label='exact')
    ax.plot(df.s, df.v_pred, 'o', ms=2.2, color='#d62728', markevery=3)
    ax.set_title(rf'{eq}, $d=10^4$, manifold' + '\n' +
                 rf'RE$={re0:.2e}$ (r0)', fontsize=9)
    ax.grid(ls=':', alpha=0.5)
    ax.set_xlabel('$s$')
    rows.append(dict(eq=eq, d=10000, curve='manifold', jobid=jobs[0],
                     RE=float(re0)))

    # histories: mean across seeds
    its, mart, ham, err = None, [], [], []
    for j in jobs:
        fh = [x for x in glob.glob(f'runs/{j}/outputs/*.csv')
              if 'curve' not in x][0]
        h = pd.read_csv(fh)
        its = h['iter step'].to_numpy()
        mart.append(h['mart loss'].to_numpy())
        ham.append(h['hami'].to_numpy())
        err.append(h['error'].to_numpy())
    # (f) Mart. loss: log axis, one-sided mean..mean+2SD
    m, sd = np.mean(mart, 0), np.std(mart, 0)
    ax = axes[row, 1]
    ax.plot(its, m, color='#2ca02c', lw=1.3)
    ax.fill_between(its, m, m + 2 * sd, color='#2ca02c', alpha=0.25)
    ax.set_yscale('log')
    ax.set_title(rf'Mart. loss vs Iter., {eq} $d=10^4$ (5 runs)', fontsize=9)
    ax.grid(ls=':', alpha=0.5, which='both')
    ax.set_xlabel('iteration')
    # (g) Hamiltonian: LINEAR axis fixed to [-0.5, 0.5]
    m, sd = np.mean(ham, 0), np.std(ham, 0)
    ax = axes[row, 2]
    ax.plot(its, m, color='#9467bd', lw=1.3)
    ax.fill_between(its, np.clip(m - 2 * sd, *HAM_LIM),
                    np.clip(m + 2 * sd, *HAM_LIM),
                    color='#9467bd', alpha=0.25)
    ax.set_ylim(*HAM_LIM)
    ax.axhline(0., color='0.6', lw=0.6)
    ax.set_title(rf'Hamilt. vs Iter., {eq} $d=10^4$' + '\n' +
                 r'(linear axis $[-0.5, 0.5]$)', fontsize=9)
    ax.grid(ls=':', alpha=0.5)
    ax.set_xlabel('iteration')
    # (h) RE: log axis, one-sided mean..mean+2SD
    m, sd = np.mean(err, 0), np.std(err, 0)
    ax = axes[row, 3]
    ax.plot(its, m, color='#d62728', lw=1.3)
    ax.fill_between(its, m, m + 2 * sd, color='#d62728', alpha=0.25)
    ax.set_yscale('log')
    ax.set_title(rf'RE vs Iter., {eq} $d=10^4$ (5 runs)', fontsize=9)
    ax.grid(ls=':', alpha=0.5, which='both')
    ax.set_xlabel('iteration')
    rows.append(dict(eq=eq, d=10000, curve='hist', jobid=';'.join(
        map(str, jobs)), RE=float(np.mean([e[-1] for e in err]))))

d1e4_block(1, 'HJB-2')
d1e4_block(2, 'HJB-3')

import os
OUT_DIR = os.environ.get('OUT_DIR', 'tmp_convfig/figs')
os.makedirs(OUT_DIR, exist_ok=True)
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/fig5_hjb_highdim.png', dpi=200)
pd.DataFrame(rows).to_csv(f'{OUT_DIR}/fig5_hjb_highdim.csv', index=False)
print(pd.DataFrame(rows).to_string(index=False))
