#!/usr/bin/env python3
"""Sec 4.4 d=2000 5-seed aggregation (final configuration: lr4x + I=8000).

seed0 = the historical single runs (496285 hjb2 / 496291 hjb3, tag
s44g_hjb{eq}_lr4); seeds 1-4 = tags s44g_hjb{eq}_lr4_s{k}. Archived anchors:
SOCMN144/143 d2000 summary final rel_l1err (5-run mean).
Run from the repo root (needs the runs/ mirror); writes
results/s44_d2000_5seed_agg.csv.
"""
import csv
import glob
import os

JOBS = {
    'hjb2_d2000': [496285, 534414, 534417, 534418, 534419],
    'hjb3_d2000': [496291, 534420, 534421, 534422, 534423],
}
ARCH = {'hjb2_d2000': 0.012809664439327841,
        'hjb3_d2000': 0.032892168136345595}


def final_error(jobid):
    fs = [x for x in glob.glob(f'runs/{jobid}/outputs/*.csv')
          if 'curve' not in os.path.basename(x)]
    with open(fs[0]) as fh:
        last = fh.read().strip().splitlines()[-1].split(',')
    hdr = open(fs[0]).readline().split(',')
    return float(dict(zip(hdr, last))['error'])


rows = [('cell', 'config', 'arch_anchor', 's0', 's1', 's2', 's3', 's4',
         'mean', 'sd', 'ratio_vs_arch', 'jobids')]
for cell, jobs in JOBS.items():
    vals = [final_error(j) for j in jobs]
    mean = sum(vals) / len(vals)
    sd = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
    rows.append((cell, 'lr4x_I8000', f'{ARCH[cell]:.5f}') +
                tuple(f'{v:.5f}' for v in vals) +
                (f'{mean:.5f}', f'{sd:.5f}', f'{mean / ARCH[cell]:.2f}',
                 ';'.join(map(str, jobs))))

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                   'results', 's44_d2000_5seed_agg.csv')
with open(out, 'w', newline='\n') as fh:
    csv.writer(fh).writerows(rows)
for r in rows:
    print(*r, sep=' | ')
print('wrote', os.path.normpath(out))