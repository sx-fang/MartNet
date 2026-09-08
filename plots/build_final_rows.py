#!/usr/bin/env python3
"""Rebuild results/final_rows.csv -- the per-run evidence index (one final
CSV row per archived run) -- from the runs/ archive mirror.

Needs the runs/ mirror produced by the archive epilogue + rsync backfill
(all runs, converged and diverged alike; one directory per jobid with
outputs/ holding the training CSVs). Only main history CSVs are indexed;
curve/ptx/pathre side files and non-CSV artifacts are skipped.

Run from the package root:   python3 plots/build_final_rows.py
(RUNS_DIR overrides the mirror location, default ../runs; default mode
prints to stdout and does NOT touch the shipped file -- the shipped
results/final_rows.csv indexes the curated subset of runs behind this
package's reported results, so a --write against the full mirror yields
a superset; pass --write to regenerate results/final_rows.csv)
"""
import csv
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.normpath(os.path.join(HERE, os.pardir))
RUNS = os.environ.get('RUNS_DIR', os.path.join(PKG, os.pardir, 'runs'))
OUT = os.path.join(PKG, 'results', 'final_rows.csv')

rows = []
for d in sorted(os.listdir(RUNS)):
    od = os.path.join(RUNS, d, 'outputs')
    if not os.path.isdir(od):
        continue
    for f in sorted(glob.glob(os.path.join(od, '*.csv'))):
        b = os.path.basename(f)[:-4]
        if any(k in b for k in ('_curve_', '_ptx', '_pathre')):
            continue
        try:
            with open(f, encoding='utf-8') as fh:
                lines = fh.read().strip().splitlines()
            if len(lines) < 3:
                continue
            hdr = lines[0].split(',')
            last = lines[-1].split(',')
            rec = dict(zip(hdr, last))
            rows.append([d, b, len(lines) - 1,
                         rec.get('iter step', rec.get('it', '')),
                         rec.get('rt', ''), rec.get('error', ''),
                         rec.get('rel_l1err', '')])
        except Exception as e:
            rows.append([d, b, 'ERR', str(e)[:40], '', '', ''])

if '--write' in sys.argv:
    w = csv.writer(open(OUT, 'w', newline='', encoding='utf-8'))
    dest = OUT
else:  # default dry mode: print to stdout, do not write
    w = csv.writer(sys.stdout)
    dest = '<stdout>'
w.writerow(['jobid', 'stem', 'rows', 'final_iter', 'final_rt',
            'final_error', 'final_rel_l1err_archived'])
w.writerows(rows)
print('runs summarized: %d -> %s' % (len(rows), dest),
      file=sys.stderr)
