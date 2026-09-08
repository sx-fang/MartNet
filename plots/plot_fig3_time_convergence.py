#!/usr/bin/env python3
"""fig3_time_convergence (Sec 4.3 time-discretization convergence rate): RE vs N, log-log.

Paper configuration: eq_linsin + Allen-Cahn source, d=100,
N in {3,4,5,6,8,12,18} (I=3000 for N<=5, I=6000 for N>=6), reference slope
O(N^{-1.01}).
Ours: 5-seed final-iteration RE (tmp_convfig/s413_cb_5seed_agg.csv).
Paper anchors: accepted tex tikz coordinates (same CSV, paper_anchor col).
Outputs: figs/fig3_time_convergence.png + figs/fig3_time_convergence.csv
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

os.makedirs('tmp_convfig/figs', exist_ok=True)
agg = pd.read_csv('tmp_convfig/s413_cb_5seed_agg.csv')
s43 = agg[agg.cell.str.startswith('s43_n')].copy()
s43['N'] = s43.cell.str.slice(5).astype(int)
s43 = s43.sort_values('N')
s43['paper_anchor'] = pd.to_numeric(s43['paper_anchor'], errors='coerce')

N = s43.N.to_numpy()
ours_mean = s43['mean'].to_numpy()
ours_sd = s43.sd.to_numpy()
# per-seed scatter values
seeds = s43[['seed0', 's1', 's2', 's3', 's4']].to_numpy()

# reference slope O(N^{-1.01}) (paper's reported rate) anchored at our first
# anchor point
slope = -1.01
ref = ours_mean[0] * (N / N[0]) ** slope

fig, ax = plt.subplots(figsize=(5.2, 4.0))
ax.plot(N, ref, '-', color='0.6', lw=1.0, zorder=1,
        label=r'reference slope $\mathcal{O}(N^{%.2f})$' % slope)
ax.errorbar(N, ours_mean, yerr=ours_sd, fmt='o-', color='#d62728', ms=5,
            lw=1.4, capsize=3, zorder=3,
            label='ours (5 seeds, final iter)')
ax.scatter(np.repeat(N, 5), seeds.ravel(), s=8, color='#d62728', alpha=0.45,
           zorder=4, linewidths=0)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'number of time steps $N$')
ax.set_ylabel(r'relative error $\mathrm{RE}_1$')
ax.set_xticks(N)
ax.set_xticklabels([str(n) for n in N])
ax.minorticks_off()
ax.grid(True, which='major', ls=':', alpha=0.5)
ax.legend(frameon=False, fontsize=8.5)
fig.tight_layout()
fig.savefig('tmp_convfig/figs/fig3_time_convergence.png', dpi=200)

out = pd.DataFrame({'N': N, 'ours_mean': ours_mean, 'ours_sd': ours_sd})
out.to_csv('tmp_convfig/figs/fig3_time_convergence.csv', index=False)
print(out.to_string(index=False))
# regression slope of our points (log-log)
p = np.polyfit(np.log(N), np.log(ours_mean), 1)
print('our log-log slope = %.3f' % p[0])
