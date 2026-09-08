#!/usr/bin/env python3
"""Table 2 (Sec 4.8 DDP RT) aggregation from the per-run final-row extract.

Input:  results/t2b_raw_final.txt -- 40 lines `t2b{b}_g{g}_d{d}|<final CSV
        row>` (uniform-batch arms b128/b256, 20 cells each; final-row fields:
        iter, epoch, rt, hami, mart, error, rel_linf, mean_vtrue_t0, u_lag).
Output: results/t2b_fix_agg.csv -- per cell (dim x gpus): RT and final-iter
        RE per arm, the paper's printed RT, the b128/printed ratio, the
        batch cost (256/128), and DDP speedups per arm.
Owner ruling 2026-09-04: b128 arm = Table 2 reproduction caliber (the b256
arm is kept as a batch ablation in this data file).

Run from the package root:   python3 plots/build_t2b_agg.py
(default: compare against the shipped results/t2b_fix_agg.csv without
writing; --write regenerates it)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.normpath(os.path.join(HERE, os.pardir, 'results'))

# paper Table 2 printed RTs (seconds), keyed by (gpus, dim)
PRINTED = {
    (1, 100): 153, (1, 500): 775, (1, 800): 1350, (1, 1000): 1909, (1, 2000): 5032,
    (2, 100): 151, (2, 500): 430, (2, 800): 721, (2, 1000): 1001, (2, 2000): 2582,
    (4, 100): 142, (4, 500): 233, (4, 800): 393, (4, 1000): 536, (4, 2000): 1387,
    (8, 100): 148, (8, 500): 153, (8, 800): 231, (8, 1000): 302, (8, 2000): 773,
}

DIMS = (100, 500, 800, 1000, 2000)
GPUS = (1, 2, 4, 8)


def main():
    rt, re_ = {}, {}
    with open(os.path.join(RESULTS, 't2b_raw_final.txt')) as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            stem, payload = ln.rsplit('|', 1)
            t = stem.split('_')              # t2b{128|256}, g{g}, d{d}
            key = ('b' + t[0][3:], int(t[1][1:]), int(t[2][1:]))
            cols = payload.split(',')
            rt[key], re_[key] = float(cols[2]), float(cols[5])
    need = {(a, g, d) for a in ('b128', 'b256') for (g, d) in PRINTED}
    assert need <= set(rt), 'missing runs: %s' % sorted(need - set(rt))

    out = ['dim,gpus,rt_b256,rt_b128,re_b256,re_b128,rt_paper_printed,'
           'ratio_b128_printed,batch_cost_256_128,ddp_speed_b256,'
           'ddp_speed_b128']
    for d in DIMS:
        for g in GPUS:
            r256, r128 = rt[('b256', g, d)], rt[('b128', g, d)]
            p = PRINTED[(g, d)]
            out.append('%d,%d,%.4f,%.4f,%.6f,%.6f,%d,%.4f,%.4f,%.4f,%.4f'
                       % (d, g, r256, r128, re_[('b256', g, d)],
                          re_[('b128', g, d)], p, r128 / p,
                          r256 / r128,
                          rt[('b256', 1, d)] / r256, rt[('b128', 1, d)] / r128))
    text = '\n'.join(out) + '\n'
    target = os.path.join(RESULTS, 't2b_fix_agg.csv')
    if '--write' in sys.argv:
        with open(target, 'w', newline='', encoding='utf-8') as fh:
            fh.write(text)
        print('wrote %s (%d cells)' % (target, len(out) - 1))
    else:
        with open(target, encoding='utf-8') as fh:
            shipped = fh.read()
        if shipped == text:
            print('CHECK OK: regenerated table is byte-identical to %s'
                  % target)
        else:
            sys.exit('CHECK FAILED: regenerated table differs from %s'
                     % target)


if __name__ == '__main__':
    main()
