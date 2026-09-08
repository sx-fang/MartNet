"""Run one numerical experiment of the SOC-MartNet paper (accepted version,
Section 4; arXiv:2405.03169).

Examples (accepted-paper Sec. 4):
  --example linear       Sec. 4.1  linear parabolic, manufactured sin solution
  --example semilinear   Sec. 4.2  semilinear parabolic (PDE mode, Alg. 3.2)
  --example linsin       Sec. 4.3  time-convergence test, eq. (4.2) with
                         variable-coefficient L and Allen-Cahn-type source
                         (PDE mode; the v - v^3 source is kept explicit by
                         default per the paper text)
  --example hjb          Sec. 4.4  HJB-1: NonDegHJB in SOC mode (Alg. 3.1)
                         with --terminal oscillatory (delc = 0.1)
  --example hjblq        Secs. 4.4-4.8  HJB-2 (default: --bval 1 --delta0 0.2
                         --delc 0.3) / HJB-3 (--delta0 0.1); Sec. 4.7 adds
                         --eps-purb; Sec. 4.8 (Tables 1-2) uses --x0-mode point
  --example shifttarget  Sec. 4.6  shifted-target SOCP (b=0, delta0=0.1,
                         g = 10 ln(0.5(1+|x-3*1_d|^2)), D_0 = S_2)

Presets (see socmartnet/presets.py and README for the full table):
  --preset v3paper / cube / sisc (accepted-paper configuration; see the
  presets module docstring for the tier/decay/batch details).

Default execution mirrors the authors' v3 code: float64 (--dtype float32
recovers the accepted-version archived runs' precision), RMSProp, offline pilot paths
(M = 1e5, N = 100), J = 2K = 2.  The v3paper/cube presets additionally use
the batch schedule 200/400/800/1600 with milestones at I/4, I/2, 3I/4 and
StepLR(step=I/10, gamma=0.01^{1/9}); those two presets are bit-identical to
the validated baseline as long as the new logging/eval switches stay off.
Single GPU runs without torch.distributed; multiple GPUs use mp.spawn + DDP
with the authors' all-reduce bias correction.

Module layout: the CLI lives in socmartnet/cli.py, the
preset table in socmartnet/presets.py, the problem factory in
socmartnet/factory.py, and the training orchestration in
socmartnet/train.py; this file is the thin entry point.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from socmartnet.train import main

if __name__ == '__main__':
    main()
