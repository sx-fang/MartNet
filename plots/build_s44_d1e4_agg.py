#!/usr/bin/env python3
"""Sec 4.4 d1e4 5-seed formal aggregation (final configuration: batch64 +
lr4x).

seed0 = the selected batch-64 validation arm (496434 hjb2 / 496436 hjb3);
seeds 1-4 = s44i formal jobs. Archived anchors: SOCMN144/143 d10000 summary
final rel_l1err (5-run mean).
Output: tmp_convfig/s44_d1e4_5seed_agg.csv
"""
import glob
import pandas as pd

JOBS = {
    'hjb2_d1e4': [496434, 500699, 500701, 500703, 500705],
    'hjb3_d1e4': [496436, 500700, 500702, 500704, 500706],
}
ARCH = {'hjb2_d1e4': 0.007278192695230246,
        'hjb3_d1e4': 0.020857696235179902}

rows = []
for cell, jobs in JOBS.items():
    vals = []
    for j in jobs:
        f = [x for x in glob.glob(f'runs/{j}/outputs/*.csv')
             if 'curve' not in x][0]
        vals.append(float(pd.read_csv(f).error.iloc[-1]))
    mean = sum(vals) / len(vals)
    sd = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
    rows.append([cell, 'b64_lr4x_I6000', f'{ARCH[cell]:.5f}'] +
                [f'{v:.5f}' for v in vals] +
                [f'{mean:.5f}', f'{sd:.5f}', f'{mean / ARCH[cell]:.2f}',
                 ';'.join(map(str, jobs))])
df = pd.DataFrame(rows, columns=[
    'cell', 'config', 'arch_anchor', 's0', 's1', 's2', 's3', 's4',
    'mean', 'sd', 'ratio_vs_arch', 'jobids'])
df.to_csv('tmp_convfig/s44_d1e4_5seed_agg.csv', index=False)
print(df.to_string(index=False))
