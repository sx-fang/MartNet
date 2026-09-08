#!/usr/bin/env python3
"""fig8_epsilon (Sec 4.7): s -> v(0, s*1_d), d=1000, eps in {1,1/2,1/4,1/8}
(bottom up) + top reference curve = HJB-2 exact (eps=0, in-code v_true).
Data: 5-seed s47f curve_diag CSVs (jobs 491409-12 seeds0, 495979-94 s1-4).
Outputs: figs/fig8_epsilon.png + figs/fig8_epsilon.csv
"""
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EPS = ['1.0', '0.5', '0.25', '0.125']
JOBS = {'1.0': 491409, '0.5': 491410, '0.25': 491411, '0.125': 491412}
S1 = {'1.0': 495979, '0.5': 495983, '0.25': 495987, '0.125': 495991}

fig, ax = plt.subplots(figsize=(5.6, 4.2))
out = None
# colorblind-friendly: CVD-safe viridis (brighter spread) + redundant
# encoding (distinct markers + linestyles + direct end-of-curve labels), so
# no information rides on hue alone
colors = plt.cm.viridis(np.linspace(0.12, 0.88, len(EPS)))
markers = ['o', 's', '^', 'D']
styles = ['-', '--', '-.', ':']
for c, mk, st, eps in zip(colors, markers, styles, EPS):
    jobs = [JOBS[eps] + i for i in range(5)] if eps == '1.0' else \
           [JOBS[eps]] + [S1[eps] + i for i in range(4)]
    preds, s = [], None
    for j in jobs:
        f = glob.glob(f'runs/{j}/outputs/*curve_diag.csv')
        if not f:
            continue
        df = pd.read_csv(f[0])
        s = df.s.to_numpy()
        preds.append(df.v_pred.to_numpy())
    mean = np.mean(preds, axis=0)
    lbl = r'$\varepsilon=%s$' % eps.rstrip('0').rstrip('.') if '.' in eps \
        else r'$\varepsilon=%s$' % eps
    ax.plot(s, mean, color=c, lw=1.7, ls=st, marker=mk, ms=3.5,
            markevery=8, label=lbl)
    ax.annotate(r'$\varepsilon=%g$' % float(eps),
                xy=(s[-1], mean[-1]), xytext=(4, 0),
                textcoords='offset points', color=c, fontsize=8,
                va='center')
    if out is None:
        ref = df.v_true.to_numpy()
        out = pd.DataFrame({'s': s, 'v_true_ref': ref})
    out[f'v_pred_eps{eps}'] = mean

ax.plot(s, ref, color='#E69F00', lw=2.4, ls=(0, (6, 2)),
        label=r'reference ($\varepsilon=0$)')
ax.annotate(r'ref. $\varepsilon=0$', xy=(s[-1], ref[-1]), xytext=(4, 0),
            textcoords='offset points', color='#E69F00', fontsize=8,
            va='center')
ax.set_xlabel(r'$s$')
ax.set_ylabel(r'$v(0, s\,\mathbf{1}_d)$')
ax.set_xlim(s[0], s[-1] + 0.22)     # room for the direct labels
ax.legend(frameon=False, fontsize=8.5, loc='upper left', ncol=2)
ax.grid(ls=':', alpha=0.5)
fig.tight_layout()
fig.savefig('tmp_convfig/figs/fig8_epsilon.png', dpi=200)
out.to_csv('tmp_convfig/figs/fig8_epsilon.csv', index=False)
print('eps curves:', {e: round(float(out[f"v_pred_eps{e}"].mean()), 4) for e in EPS},
      'ref mean:', round(float(ref.mean()), 4))
