"""Command-line interface of the SOC-MartNet reproduction code (split out
of run.py for module reuse). See run.py's module docstring for the example
catalogue."""

import argparse

from .factory import PDE_EXAMPLES, SOC_EXAMPLES

_DOC = 'Run one numerical experiment of the SOC-MartNet paper (Sec. 4).'


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=_DOC,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--example', choices=SOC_EXAMPLES + PDE_EXAMPLES,
                   required=True)
    p.add_argument('--method', choices=['socmartnet', 'prabmartnet'],
                   default=None,
                   help='default: socmartnet (Alg. 3.1) for SOC examples, '
                        'prabmartnet (Alg. 3.2) otherwise')
    p.add_argument('--terminal', choices=['smooth', 'oscillatory'],
                   default='smooth', help='terminal g for semilinear/hjb')
    p.add_argument('--dim', type=int, required=True)
    p.add_argument('--te', type=float, default=1.0)
    p.add_argument('--preset', choices=['v3paper', 'cube', 'sisc'],
                   default='v3paper')
    p.add_argument('--max-iter', type=int, default=None,
                   help='default: 1000 (d<=500, v3paper) / 2000 (d>500); '
                        'cube preset: 2000 iff d>100; sisc preset: 1000 '
                        '(paper Table 1 value; Table 2 uses 6000)')
    p.add_argument('--width', type=int, default=None,
                   help='override hidden width of u/v nets (Table 1 W=256 '
                        'row: --width 256)')
    p.add_argument('--num-hidden', type=int, default=None,
                   help='override number of hidden layers of u/v nets')
    p.add_argument('--r-dim', type=int, default=None,
                   help='override adversarial net output dimension r')
    p.add_argument('--lr-uv', type=float, default=None,
                   help='override u/v learning rate (archived-configuration '
                        'probes)')
    p.add_argument('--lr-rho', type=float, default=None,
                   help='override adversarial-net learning rate (archived '
                        'configurations use 10 x lr_uv; paper delta3 = 1e-2)')
    p.add_argument('--batch-size', type=int, default=None,
                   help='override minibatch size with a single constant '
                        'segment (sisc preset: 256 if d<1000 else 128); '
                        'replaces the preset batsize schedule')
    p.add_argument('--num-paths', type=int, default=10**5,
                   help='training path-pool size M; the accepted-version '
                        'path-renewal configuration uses the epochsize as '
                        'the pool, e.g. '
                        '--num-paths 10000 --renew-frac 0.2')
    p.add_argument('--renew-frac', type=float, default=0.,
                   help='fraction of the path pool regenerated at each '
                        'epoch boundary (the authors\' accepted-version '
                        'rate_newpath=0.2); '
                        '0 disables')
    p.add_argument('--fd-residual', action='store_true',
                   help='use the accepted-version finite-difference '
                        'martingale residual (lambda-free loss=mart+ctr, '
                        'u-gradient bug fixed) instead of the autograd-H '
                        'trapezoid + lambda augmentation')
    p.add_argument('--num-dt', type=int, default=100)
    p.add_argument('--lam-bar', type=float, default=None,
                   help='default 1e3 (v3paper/sisc) / 1e4 (cube); paper '
                        'Sec. 4.4 uses 100 for d>=1000, T=0.01')
    p.add_argument('--controlled-pool', action=argparse.BooleanOptionalAction,
                   default=False,
                   help='(train_fd only): renew the path pool with '
                        'CONTROLLED paths simulated online under the current '
                        'u_alpha, storing generation-time controls u_old; '
                        'the FD residual corrects only the lag '
                        '2(u-u_old).grad v (no lag bias, no double count). '
                        'Requires --renew-frac > 0; initial pool stays '
                        'pilot (u_old=0). Default off recovers the pilot '
                        'pool bitwise')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--gpus', type=int, default=-1,
                   help='number of GPUs to use; -1 = all visible')
    p.add_argument('--dtype', choices=['float64', 'float32'],
                   default='float64',
                   help='default float64 (validated v3 configuration); '
                        'float32 recovers the accepted-version archived runs')
    p.add_argument('--x0-scale', type=float, default=1.0,
                   help='half-length of the D_0 segments (s in [-scale, '
                        'scale])')
    p.add_argument('--unit-ball', action='store_true',
                   help='normalize the diagonal segment to the unit sphere '
                        '(linear/semilinear/hjb only)')
    p.add_argument('--x0-mode', choices=['region', 'point'], default='region',
                   help='point: D_0 = {0} (Sec. 4.8, hjblq/shifttarget only)')
    p.add_argument('--region', choices=['s1s2', 's2s3', 's2'], default=None,
                   help='D_0 set for hjblq/shifttarget; default s2s3 for '
                        'hjblq (Sec. 4.4), s2 for shifttarget (Sec. 4.6)')
    # HJBLQ family parameters (Sec. 4.4-4.8)
    p.add_argument('--bval', type=float, default=None,
                   help='drift b of hjblq; default 1.0')
    p.add_argument('--delta0', type=float, default=None,
                   help='diffusion scale delta_0 of hjblq (sigma = delta0 '
                        'sqrt(2)); default 0.2 (HJB-2); use 0.1 for HJB-3')
    p.add_argument('--delc', type=float, default=None,
                   help='epsilon_0 configuration: code uses delc/pi in '
                        'sin(1/(eps0 + x^2)); default 0.3 for hjblq, 0.1 '
                        'for semilinear/hjb')
    p.add_argument('--eps-purb', type=float, default=None,
                   help='Sec. 4.7 perturbation: add eps*sin(1_d^T k) to the '
                        'Hamiltonian (eps = 1, 1/2, 1/4, 1/8)')
    p.add_argument('--target-shift', type=float, default=3.0,
                   help='shifttarget: target point s*1_d coordinate (paper '
                        'text: 3; the paper configuration is authoritative, '
                        'the accepted-version archived runs used 0)')
    p.add_argument('--allen-cahn', action=argparse.BooleanOptionalAction,
                   default=True,
                   help='linsin: keep the v - v^3 Allen-Cahn source explicit '
                        '(paper configuration, default on); '
                        '--no-allen-cahn recovers the archived '
                        'compensator-only form (SOCMN179, likely not the '
                        'code that produced the paper\'s time-convergence '
                        'figure, Fig. 3)')
    # logging / evaluation artifacts
    p.add_argument('--extended-log', action='store_true',
                   help='append rel_linf / mean_vtrue_t0 (/ cost) columns to '
                        'the history CSV; default on for --preset sisc')
    p.add_argument('--cost-track', action=argparse.BooleanOptionalAction,
                   default=None,
                   help='log J(u) by Monte-Carlo each iteration (default on '
                        'for shifttarget; consumes RNG draws)')
    p.add_argument('--cost-paths', type=int, default=256,
                   help='number of MC paths for the J(u) estimate (Sec. 4.6)')
    p.add_argument('--cost-num-dt', type=int, default=100)
    p.add_argument('--save-curves', action=argparse.BooleanOptionalAction,
                   default=None,
                   help='write v(0, .) curve CSVs on the D_0 generators at '
                        'the end of training (default on for sisc region '
                        'runs, off otherwise)')
    p.add_argument('--curve-points', type=int, default=100)
    p.add_argument('--eval-ptx', action='store_true',
                   help='Sec. 4.5: write s -> v(r, s 1_d + r 1_d) CSVs for '
                        'r = 0.125, 0.25')
    p.add_argument('--eval-path-re', action='store_true',
                   help='Sec. 4.5: write RE(t) along 8 fresh pilot paths')
    p.add_argument('--out', type=str, default='outputs')
    p.add_argument('--tag', type=str, default='')
    return p.parse_args(argv)
