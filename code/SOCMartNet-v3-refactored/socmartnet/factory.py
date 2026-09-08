"""Problem factory: maps the --example CLI choice to a problem instance
(split out of run.py for module reuse)."""

import torch

from . import (HJBLQ, LinearParabolicSin, LinearSinAC, NonDegHJB,
               ShiftTargetHJB)

SOC_EXAMPLES = ('hjb', 'hjblq', 'shifttarget')
PDE_EXAMPLES = ('linear', 'semilinear', 'linsin')


def build_problem(args):
    t0 = torch.tensor(0.)
    te = torch.tensor(args.te)
    if args.example == 'linear':
        if args.x0_mode == 'point':
            raise ValueError('--x0-mode point is only supported for '
                             'hjblq/shifttarget')
        return LinearParabolicSin(args.dim, t0=t0, te=te,
                                  x0_scale=torch.tensor(args.x0_scale),
                                  unit_ball=args.unit_ball)
    if args.example in ('semilinear', 'hjb'):
        if args.x0_mode == 'point':
            raise ValueError('--x0-mode point is only supported for '
                             'hjblq/shifttarget')
        delc = args.delc if args.delc is not None else 0.1
        return NonDegHJB(args.dim, t0=t0, te=te, terminal=args.terminal,
                         delc=delc, x0_scale=torch.tensor(args.x0_scale),
                         unit_ball=args.unit_ball)
    common = dict(t0=t0, te=te, x0_scale=args.x0_scale,
                  x0_mode=args.x0_mode)
    if args.example == 'hjblq':
        return HJBLQ(args.dim, b_val=args.bval if args.bval is not None else 1.,
                     delta0=args.delta0 if args.delta0 is not None else 0.2,
                     delc=args.delc if args.delc is not None else 0.3,
                     eps_purb=args.eps_purb,
                     region=args.region or 's2s3', **common)
    if args.example == 'shifttarget':
        return ShiftTargetHJB(args.dim, target_shift=args.target_shift,
                              region=args.region or 's2', **common)
    # linsin
    if args.x0_mode == 'point':
        raise ValueError('--x0-mode point is only supported for '
                         'hjblq/shifttarget')
    return LinearSinAC(args.dim, t0=t0, te=te, allen_cahn=args.allen_cahn,
                       x0_scale=args.x0_scale)
