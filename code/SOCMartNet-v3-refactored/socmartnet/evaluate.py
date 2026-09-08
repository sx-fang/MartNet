"""End-of-training evaluation utilities backing the accepted paper's figures.

Every figure panel of Section 4 is backed by a CSV produced from these
functions (run.py writes them).  All evaluations run under torch.no_grad.
When called after DDP training on a single rank, pass the unwrapped network
(v_theta.module) -- DDP forwards involve collectives and would hang.
"""

import torch

from .problems import diag_points, e1_points, manifold_points

CURVE_GENS = {'e1': e1_points, 'diag': diag_points,
              'manifold': manifold_points}


def curve_values(problem, v_theta, curve, radius=1., num_points=100,
                 t_val=None):
    """(s, v_true, v_pred) on a named curve at fixed time (default t0).

    curve: 'e1' (S_1), 'diag' (S_2) or 'manifold' (S_3).
    """
    s_coord, x = CURVE_GENS[curve](problem.dim_x, num_points, radius)
    if t_val is None:
        t_val = float(problem.t0)
    t = torch.full([num_points, 1], float(t_val))
    with torch.no_grad():
        v_pred = v_theta(t, x).squeeze(-1)
    v_true = problem.v_exact(t, x).squeeze(-1)
    return s_coord, v_true, v_pred


def ptx_values(problem, v_theta, r_frac, radius=1., num_points=100):
    """Sec. 4.5, Fig. 8ab: s -> v(r, s 1_d + r 1_d) at fixed time r."""
    s_coord, x = diag_points(problem.dim_x, num_points, radius)
    x = x + r_frac
    t = torch.full([num_points, 1], float(r_frac))
    with torch.no_grad():
        v_pred = v_theta(t, x).squeeze(-1)
    v_true = problem.v_exact(t, x).squeeze(-1)
    return s_coord, v_true, v_pred


def manifold_point(dim_x, s0):
    """Single point l(s0) of the S_3 space curve (Eq. (4.11))."""
    e_vec = torch.arange(1, dim_x + 1)
    return s0 * torch.sign(torch.sin(e_vec)) \
        + torch.cos(e_vec + s0 * torch.pi)


def path_re(problem, v_theta, num_dt=100, paths_per_start=4, starts=None):
    """Sec. 4.5, Fig. 8c: relative error of v_theta along fresh pilot paths.

    starts: list of [dim_x] start points; default [0_d, l(0.75)] -- the
    paper's X_0 = 0 for S2 and X_0 = l(0.75) for S3, with 4 paths each.
    Returns (t_grid, re_pool, re_path_mean, re_path_std) where re_pool is the
    mean|err|/mean|v| ratio pooled over all paths at each time slice, and
    re_path_mean/std are the mean and sample std over paths of the per-path
    ratio.
    """
    if starts is None:
        starts = [torch.zeros(problem.dim_x),
                  manifold_point(problem.dim_x, 0.75)]
    dt = float(problem.te - problem.t0) / num_dt

    trajs = []
    for start in starts:
        xt = start.unsqueeze(0).repeat(paths_per_start, 1)
        traj = [xt]
        for n in range(num_dt):
            tn = torch.full([paths_per_start, 1],
                            float(problem.t0) + dt * n)
            dw = torch.normal(torch.zeros_like(xt), std=dt**0.5)
            xt = xt + problem.mu(tn, xt) * dt + problem.sigma(tn, xt) * dw
            traj.append(xt)
        trajs.append(torch.stack(traj))          # [N+1, P, d]
    xt_all = torch.cat(trajs, dim=1)             # [N+1, 2P, d]

    num_times = num_dt + 1
    num_paths = xt_all.shape[1]
    t_grid = torch.linspace(float(problem.t0), float(problem.te), num_times)
    t_flat = t_grid.repeat_interleave(num_paths).unsqueeze(-1)
    x_flat = xt_all.reshape(-1, problem.dim_x)
    with torch.no_grad():
        v_pred = v_theta(t_flat, x_flat).squeeze(-1)
    v_true = problem.v_exact(t_flat, x_flat).squeeze(-1)

    err = (v_pred - v_true).abs().reshape(num_times, num_paths)
    vabs = v_true.abs().reshape(num_times, num_paths)
    re_pool = err.mean(-1) / vabs.mean(-1)
    re_per_path = err / vabs
    re_path_mean = re_per_path.mean(-1)
    re_path_std = re_per_path.std(-1) if num_paths > 1 \
        else torch.zeros(num_times)
    return t_grid, re_pool, re_path_mean, re_path_std
