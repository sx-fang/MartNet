#!/usr/bin/env python3
"""Sec 4.2 semilinear fig (paper layout, 2x4):
  top row: d=10 EXACT solution, s -> v(t, s 1_d) at t=0 (Cole-Hopf
    1e6-sample reference, s42h_true_d10 jobs) and t=T (terminal 1+g,
    closed form: g = (1/d) sum_i [sin(x_i - pi/2) + sin(1/(0.1/pi + x_i^2))],
    problems.py NonDegHJB 'oscillatory' + paper v(T)=1+g).
  bottom row: d=100 numerical t=0 diag, 5-run mean +- 2SD (s42cb jobs).
Outputs: figs/fig2_semilinear.png + figs/fig2_semilinear.csv
"""
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AGG = pd.read_csv('tmp_convfig/s413_cb_5seed_agg.csv')
TS = ['1.0', '0.1', '0.05', '0.01']
DELPI = 0.1 / np.pi          # code configuration: delc/pi with delc=0.1

def g_term(x):
    d = x.shape[-1]
    return (np.sin(x - np.pi / 2) + np.sin(1.0 / (DELPI + x ** 2))).mean(-1)

def jobs_of(cell):
    r = AGG[AGG.cell == cell].iloc[0]
    return [int(r[c]) for c in ('seed0_jobid', 's1_jobid', 's2_jobid',
                                's3_jobid', 's4_jobid')]

fig, axes = plt.subplots(2, 4, figsize=(12.6, 5.6))
top_done = True
bottom_extents = []
for k, t in enumerate(TS):
    ax = axes[0, k]
    # jobids 496287..496290 follow TS order
    jid = 496287 + k
    fs = glob.glob(f'runs/{jid}/outputs/*curve_diag.csv')
    if not fs:
        top_done = False
        ax.text(.5, .5, 'reference eval pending', ha='center', va='center',
                transform=ax.transAxes, fontsize=8)
    else:
        df = pd.read_csv(fs[0])
        ax.plot(df.s, df.v_true, '-', color='#1f77b4', lw=1.6,
                label=r'exact $t=0$')
        s = df.s.to_numpy()
        xdiag = np.stack([s] * 10, axis=1)      # d=10, s*1_d
        ax.plot(s, 1 + g_term(xdiag), '--', color='#ff7f0e', lw=1.6,
                label=r'exact $t=T$ (1+g)')
        ax.legend(frameon=False, fontsize=7.5)
    ax.set_title(rf'$T={t}$, $d=10$ (exact)', fontsize=9)
    ax.grid(ls=':', alpha=0.5)

    jobs = jobs_of(f's42_t{t}')
    preds, s, vt = [], None, None
    for j in jobs:
        fs = glob.glob(f'runs/{j}/outputs/*curve_diag.csv')
        if not fs:
            continue
        df = pd.read_csv(fs[0])
        s, vt = df.s.to_numpy(), df.v_true.to_numpy()
        preds.append(df.v_pred.to_numpy())
    ax = axes[1, k]
    if preds:
        mean, sd = np.mean(preds, 0), np.std(preds, 0)
        ax.plot(s, vt, '-', color='#1f77b4', lw=1.5, label='exact')
        ax.plot(s, mean, 'o', ms=2.2, color='#d62728', markevery=3,
                label='SOC-MartNet (5-run)')
        ax.fill_between(s, mean - 2 * sd, mean + 2 * sd, color='#d62728',
                        alpha=0.2)
        if k == 0:
            ax.legend(frameon=False, fontsize=7.5)
        # collect extents for the shared bottom-row y-range
        row_lo = min(vt.min(), (mean - 2 * sd).min())
        row_hi = max(vt.max(), (mean + 2 * sd).max())
        row_extent = (row_lo, row_hi)
    else:
        row_extent = None
    bottom_extents.append(row_extent)
    ax.set_title(rf'$T={t}$, $d=100$ (numerical)', fontsize=9)
    ax.grid(ls=':', alpha=0.5)
    ax.set_xlabel('$s$')
    if k == 0:
        ax.set_ylabel(r'$v(0, s\,\mathbf{1}_d)$')

# all bottom-row (d=100 numerical) panels share one vertical axis range so
# solution amplitudes compare directly across T
ext = [e for e in bottom_extents if e is not None]
if len(ext) == len(TS):
    lo = min(e[0] for e in ext)
    hi = max(e[1] for e in ext)
    pad = 0.05 * (hi - lo)
    for k in range(len(TS)):
        axes[1, k].set_ylim(lo - pad, hi + pad)

fig.tight_layout()
fig.savefig('tmp_convfig/figs/fig2_semilinear.png', dpi=200)
pd.DataFrame({'T': [float(t) for t in TS],
              'RE_mean': [AGG[AGG.cell == f's42_t{t}'].iloc[0]['mean']
                          for t in TS],
              'RE_sd': [AGG[AGG.cell == f's42_t{t}'].iloc[0].sd
                        for t in TS],
              'top_row_done': top_done}).to_csv(
    'tmp_convfig/figs/fig2_semilinear.csv', index=False)
print('done, top row complete:', top_done)
