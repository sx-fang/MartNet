#!/usr/bin/env python3
"""Regenerate results/t1e_final_table.csv -- the Table 1 (Sec 4.8, paper Table 1)
EQUAL-TIME final-caliber table -- from the six raw final-row extract files
in results/.

Each extract line is   <csv stem>|<final iter>,<final rt>,<final error>
(final row of an archived run CSV; the stem carries tag + seed). Which
(family wave, arm) each cell's five seeds come from is encoded below; tags
match the archived runs:

  flat lr_rho = 1e-2 cells (paper-text value):   families t1e2 / t1e6
  relative lr_rho cells (authors' convention):   families t1e3 / t1e5,
    arm suffix u1r10 (lr_rho = 10 x lr_u) or u1r1 (lr_rho = lr_u)

Seed provenance per cell: seed 0 from the wave-1 arm scan, seeds 1-4 from
the completion waves (t1e4 lines reuse the t1e2_* tags, t1e3w2 lines reuse
the t1e3_* tags). Rows follow the paper Table 1 order: the W_h = 256 block
(d ascending) first, then the W_h = d+10 block.

Run from the package root:   python3 plots/build_t1e_final_table.py
(default: compare against the shipped results/t1e_final_table.csv without
writing; --write regenerates it)
"""
import os
import re
import sys
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.normpath(os.path.join(HERE, os.pardir, 'results'))

SOURCES = ['t1e2_raw_final.txt', 't1e3_raw_final.txt', 't1e3w2_raw_final.txt',
           't1e4_raw_final.txt', 't1e5_raw_final.txt', 't1e6_raw_final.txt']

# cell, families, wtag, d, I, lr_rho caliber, lr_rho value, arm, DeepBSDE printed, quota (s)
CELLS = [
    ('w256_d100',  {'t1e2'}, 'w256', 100,  3200, '1e-2 (paper)', '1.0e-2',    '',      3.23e-3,  73),
    ('w256_d300',  {'t1e2'}, 'w256', 300,  3300, '1e-2 (paper)', '1.0e-2',    '',      1.18e-3,  90),
    ('w256_d500',  {'t1e2'}, 'w256', 500,  4600, '1e-2 (paper)', '1.0e-2',    '',      1.05e-3, 116),
    ('w256_d800',  {'t1e3'}, 'w256', 800,  4800, '10 x lr_u',    '1.0607e-03', 'u1r10', 2.54e-3, 145),
    ('w256_d1000', {'t1e3'}, 'w256', 1000, 4900, '1 x lr_u',     '9.4868e-05', 'u1r1',  1.86e-3, 170),
    ('wd10_d100',  {'t1e6'}, 'wd10', 100,  3500, '1e-2 (paper)', '1.0e-2',    '',      2.86e-3,  53),
    ('wd10_d300',  {'t1e2'}, 'wd10', 300,  3500, '1e-2 (paper)', '1.0e-2',    '',      3.27e-4, 103),
    ('wd10_d500',  {'t1e5'}, 'wd10', 500,  3500, '10 x lr_u',    '1.3416e-03', 'u1r10', 6.41e-4, 184),
    ('wd10_d800',  {'t1e3'}, 'wd10', 800,  3000, '10 x lr_u',    '1.0607e-03', 'u1r10', 1.28e-3, 386),
    ('wd10_d1000', {'t1e3'}, 'wd10', 1000, 4100, '10 x lr_u',    '9.4868e-04', 'u1r10', 3.77e-3, 615),
]

HEADER = ('cell,I,lr_rho_caliber,lr_rho_value,re_mean,re_sd,re_s0,re_s1,re_s2,'
          're_s3,re_s4,dbsde_printed,ratio_vs_dbsde,quota_s,rt_mean,rt_s0,rt_s1,'
          'rt_s2,rt_s3,rt_s4,util')


def main():
    write = '--write' in sys.argv
    runs = {}  # (cell, seed) -> (iter, rt, error)
    for fname in SOURCES:
        with open(os.path.join(RESULTS, fname)) as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                stem, payload = ln.rsplit('|', 1)
                it_s, rt_s, err_s = payload.split(',')
                it, rt, err = float(it_s), float(rt_s), float(err_s)
                for cell, fams, wtag, d, I, cal, val, arm, ref, quota in CELLS:
                    suffix = '_' + arm if arm else ''
                    for fam in fams:
                        if re.search(r'_%s_%s_d%d%s_s(\d)\.csv$'
                                     % (fam, wtag, d, suffix), stem):
                            if int(it) != I:
                                continue  # superseded runs at other quotas
                            runs[(cell, int(stem[-5]))] = (it, rt, err)

    out = [HEADER]
    for cell, fams, wtag, d, I, cal, val, arm, ref, quota in CELLS:
        seeds = []
        for s in range(5):
            key = (cell, s)
            if key not in runs:
                sys.exit('missing run for %s seed %d' % key)
            seeds.append(runs[key])
        errs = [e for (_, _, e) in seeds]
        rts = [r for (_, r, _) in seeds]
        re_mean = statistics.mean(errs)
        re_sd = statistics.stdev(errs)  # sample SD, ddof = 1
        rt_mean = statistics.mean(rts)
        row = [cell, str(I), cal, val,
               '%.4e' % re_mean, '%.4e' % re_sd] \
              + ['%.4e' % e for e in errs] \
              + ['%.2e' % ref, '%.2f' % (re_mean / ref), str(quota),
                 '%.1f' % rt_mean] \
              + ['%.1f' % r for r in rts] \
              + ['%d%%' % round(rt_mean / quota * 100)]
        out.append(','.join(row))

    text = '\n'.join(out) + '\n'
    target = os.path.join(RESULTS, 't1e_final_table.csv')
    if write:
        with open(target, 'w', newline='\n') as fh:
            fh.write(text)
        print('wrote %s (%d data rows)' % (target, len(out) - 1))
    else:
        shipped = open(target).read()
        if shipped == text:
            print('CHECK OK: regenerated table is byte-identical to %s' % target)
        else:
            sys.exit('CHECK FAILED: regenerated table differs from %s' % target)


if __name__ == '__main__':
    main()
