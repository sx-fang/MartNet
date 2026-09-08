#!/usr/bin/env python3
"""fig4_hjb1_tscan (Sec 4.4, HJB-1 = NonDegHJB f_cost): 12 diag panels
  rows d = 100 / 1000 / 2000, cols T = 1 / 0.5 / 0.1 / 0.01.
Annotate each panel with the run's final RE.
Data: s42g jobs 495819-26 (d100/d1000), 495827 + 496077-79 (d2000).
Outputs: figs/fig4_hjb1_tscan.png + figs/fig4_hjb1_tscan.csv
"""
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CELLS = {(100, '1.0'): 495819, (100, '0.5'): 495820, (100, '0.1'): 495821,
         (100, '0.01'): 495822, (1000, '1.0'): 495823, (1000, '0.5'): 495824,
         (1000, '0.1'): 495825, (1000, '0.01'): 495826,
         (2000, '1.0'): 495827, (2000, '0.5'): 496077,
         (2000, '0.1'): 496078, (2000, '0.01'): 496079}
DIMS = [100, 1000, 2000]
TS = ['1.0', '0.5', '0.1', '0.01']

fig, axes = plt.subplots(3, 4, figsize=(13.2, 8.6), sharex=True)
rows = []
g_lo, g_hi = float('inf'), float('-inf')
for i, d in enumerate(DIMS):
    for k, t in enumerate(TS):
        j = CELLS[(d, t)]
        fc = glob.glob(f'runs/{j}/outputs/*curve_diag.csv')
        ax = axes[i, k]
        if not fc:
            ax.set_visible(True)
            ax.text(.5, .5, 'MISSING', ha='center', va='center')
            ax.set_title(f'$d={d}$, $T={t}$', fontsize=9)
            continue
        df = pd.read_csv(fc[0])
        fh = [x for x in glob.glob(f'runs/{j}/outputs/*.csv')
              if 'curve' not in x]
        re_final = pd.read_csv(fh[0]).error.iloc[-1]
        ax.plot(df.s, df.v_true, '-', color='#1f77b4', lw=1.5, label='exact')
        ax.plot(df.s, df.v_pred, 'o', ms=2.2, color='#d62728', markevery=3,
                label='SOC-MartNet')
        ax.set_title(f'$d={d}$, $T={t}$   RE$={re_final:.2e}$', fontsize=9)
        if i == 0 and k == 0:
            ax.legend(frameon=False, fontsize=8)
        ax.grid(ls=':', alpha=0.5)
        if i == 2:
            ax.set_xlabel('$s$')
        if k == 0:
            ax.set_ylabel(r'$v(0, s\,\mathbf{1}_d)$')
        g_lo = min(g_lo, df.v_true.min(), df.v_pred.min())
        g_hi = max(g_hi, df.v_true.max(), df.v_pred.max())
        rows.append(dict(d=d, T=float(t), jobid=j, RE_final=float(re_final)))

# all twelve panels share one vertical axis range
pad = 0.05 * (g_hi - g_lo)
for i in range(len(DIMS)):
    for k in range(len(TS)):
        axes[i, k].set_ylim(g_lo - pad, g_hi + pad)

fig.tight_layout()
fig.savefig('tmp_convfig/figs/fig4_hjb1_tscan.png', dpi=200)
pd.DataFrame(rows).to_csv('tmp_convfig/figs/fig4_hjb1_tscan.csv', index=False)
print(pd.DataFrame(rows).to_string(index=False))
