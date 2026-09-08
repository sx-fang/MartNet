"""Hyperparameter presets (paper Sec. 4 global settings + archived INI
fallbacks), split out of run.py for module reuse.

Presets:
  v3paper   v3-paper-text settings (4 hidden layers of 2d+20, r = 2d+300,
            lr0 = 3 d^{-0.5} 1e-3, lam_bar = 1e3)
  cube      settings behind the authors' v3-era archived CSVs (3 hidden
            layers of 2(d+10), r = 2d+500, lr0 = 1e-3, lam_bar = 1e4);
            used by the reproduction test
  sisc      accepted-paper Sec.-4 intro settings: 6 hidden layers of d+10
            ReLU, r = 600, fixed batch 256 (d<1000) / 128 (1000<=d<1e4) /
            64 (d>=1e4, archived INI tier), lr_uv = 3 d^{-0.5|-0.8} 1e-3,
            staircase lr decay 0.01^{i/I} quantized in <=100-iter plateaus
            (authors' StepLR(100, 0.01^{100/I})), I = 1000, lambda0 = 10,
            delta4 = 10, lam_bar = 1e3, J = 2K = 2.
"""

_CLI_OVERRIDES = (('width', 'width'), ('num_hidden', 'num_hidden'),
                  ('r_dim', 'r'), ('lr_uv', 'lr_uv'), ('lr_rho', 'lr_rho'))


def hyperparams(args):
    """Hyperparameter dict; the v3paper/cube branches are value-identical to
    the validated baseline."""
    d = args.dim
    if args.preset == 'cube':
        # config behind the authors' v3-era archived CSVs
        hp = dict(num_hidden=3, width=2 * (d + 10), r=2 * d + 500,
                  lr_uv=1e-3, lr_rho=1e-2,
                  max_iter=2000 if d > 100 else 1000, lam_bar=1e4,
                  batsize=[200, 400, 800, 1600], sch_mode='v3')
    elif args.preset == 'sisc':
        # accepted-paper Sec.-4 intro + the authors' accepted-version
        # archived configs: d >= 1000 uses the producing configuration of
        # the archived HJB2b INIs (SOCMN144 et al.): lr exponent 0.8,
        # batch 128, rho lr 10 x value lr (documented paper-vs-code
        # discrepancies; the archive fallback was adopted after
        # paper-configuration runs failed at d=1000 with RE ~1).
        # lr exponent: paper 0.5 for d < 1000, archived/taskmaker 0.8 for
        # d >= 1000 (at d=1000 the two disagree; the archived value wins
        # since the paper value failed with RE ~1).
        # batch tiers follow the archived INIs: 256 (d<1000, SOCMN126-129),
        # 128 (1000<=d<1e4, SOCMN129/144/150-153), 64 (d>=1e4, SOCMN143/144
        # d10000 INIs -- the missing 64 tier had cost a 2x rt gap and a
        # residual RE gap at d=1e4).
        expo = -0.5 if d < 1000 else -0.8
        lr_uv = 3e-3 * d**expo
        hp = dict(num_hidden=6, width=d + 10, r=600,
                  lr_uv=lr_uv, lr_rho=1e-2 if d < 1000 else 10 * lr_uv,
                  max_iter=1000, lam_bar=1e3,
                  batsize=[256 if d < 1000 else 128 if d < 10000 else 64],
                  sch_mode='quantized')
    else:  # v3paper
        hp = dict(num_hidden=4, width=2 * (d + 10), r=2 * d + 300,
                  lr_uv=3e-3 * d**(-0.5 if d <= 1000 else -0.8),
                  lr_rho=1e-2,
                  max_iter=1000 if d <= 500 else 2000, lam_bar=1e3,
                  batsize=[200, 400, 800, 1600], sch_mode='v3')
    if args.max_iter is not None:
        hp['max_iter'] = args.max_iter
    if args.lam_bar is not None:
        hp['lam_bar'] = args.lam_bar
    for cli_key, hp_key in _CLI_OVERRIDES:
        if getattr(args, cli_key) is not None:
            hp[hp_key] = getattr(args, cli_key)
    if args.batch_size is not None:
        hp['batsize'] = [args.batch_size]
    return hp
