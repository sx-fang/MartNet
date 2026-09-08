"""Training orchestration: DDP worker, method dispatch, and artifact
writers (split out of run.py for module reuse)."""

import csv
import random
from pathlib import Path

import numpy as np
import torch

from . import (SOCMartNet, control_net, relative_l1, test_net,
               value_net)
from . import evaluate
from .cli import parse_args
from .factory import SOC_EXAMPLES, build_problem
from .presets import hyperparams


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def worker(rank, world_size, args):
    print(f"Running process on rank {rank}.")
    setup_seed(args.seed + rank)

    if world_size > 1:
        import os
        import zlib
        import torch.distributed as dist
        os.environ['MASTER_ADDR'] = 'localhost'
        if 'MASTER_PORT' not in os.environ:
            # Deterministic per-job port: all ranks derive the same value from
            # shared environment without communicating (chicken-and-egg with
            # init_process_group). Distinct jobs co-located on one node get
            # distinct ports -- the authors' platform INI used
            # master_port=random; the old fixed 12355 made two co-located
            # multi-GPU jobs die on EADDRINUSE.
            port_seed = os.environ.get('SLURM_JOB_ID', '0') + '_' + str(args.seed)
            os.environ['MASTER_PORT'] = str(
                20000 + zlib.crc32(port_seed.encode()) % 40000)
        try:
            dist.init_process_group("nccl", rank=rank, world_size=world_size)
        except RuntimeError:
            dist.init_process_group("gloo", rank=rank, world_size=world_size)

    torch.set_default_dtype(torch.float64 if args.dtype == 'float64'
                            else torch.float32)
    torch.set_default_device(f'cuda:{rank}')

    hp = hyperparams(args)
    max_iter, lam_bar = hp['max_iter'], hp['lam_bar']
    problem = build_problem(args)
    N = args.num_dt
    Dt = (torch.tensor(args.te) - problem.t0) / N
    batsize_gpu = [max(int(s / world_size), 1) for s in hp['batsize']]
    if len(hp['batsize']) > 1:
        milestones = [int(max_iter / len(hp['batsize']) * (i + 1))
                      for i in range(len(hp['batsize']) - 1)]
    else:
        milestones = []

    # RNG call order mirrors the authors' testsocmart.py:
    # grid (deterministic) -> reference MC -> network init -> paths -> training
    x0 = problem.x0_points(int(args.num_paths / world_size))
    x_test = problem.test_points(10**3)
    v_true = problem.v_exact(problem.t0.unsqueeze(-1), x_test)

    u_alpha = control_net(args.dim, problem.dim_u, hp['width'],
                          hp['num_hidden'])
    v_theta = value_net(args.dim, hp['width'], hp['num_hidden'])
    rho_eta = test_net(args.dim, hp['r'])

    if world_size > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        u_alpha = DDP(u_alpha.to(rank), device_ids=[rank])
        v_theta = DDP(v_theta.to(rank), device_ids=[rank])
        rho_eta = DDP(rho_eta.to(rank), device_ids=[rank])
    else:
        u_alpha, v_theta, rho_eta = (u_alpha.to(rank), v_theta.to(rank),
                                     rho_eta.to(rank))

    soc_mode = args.example in SOC_EXAMPLES
    optim_u = torch.optim.RMSprop(u_alpha.parameters(), lr=hp['lr_uv'])
    optim_v = torch.optim.RMSprop(v_theta.parameters(), lr=hp['lr_uv'])
    optim_rho = torch.optim.RMSprop(rho_eta.parameters(), lr=hp['lr_rho'])

    if hp['sch_mode'] == 'quantized':
        # staircase 0.01^{i/I} in <=100-iter plateaus (authors' accepted-version configs)
        sch_step = min(100, max(int(max_iter / 10), 1))
        sch_gam = 0.01**(sch_step / max_iter)
    else:  # v3paper/cube: validated baseline schedule
        sch_step = max(int(max_iter / 10), 1)
        sch_gam = 0.01**(1 / 9)
    sch_u = torch.optim.lr_scheduler.StepLR(optim_u, sch_step,
                                            sch_gam) if soc_mode else None
    sch_v = torch.optim.lr_scheduler.StepLR(optim_v, sch_step, sch_gam)
    sch_rho = torch.optim.lr_scheduler.StepLR(optim_rho, sch_step, sch_gam)

    def err_func():
        return relative_l1(v_theta(problem.t0.unsqueeze(-1), x_test), v_true)

    extended = args.extended_log or args.preset == 'sisc'
    cost_track = args.cost_track
    if cost_track is None:
        cost_track = args.example == 'shifttarget'
    if cost_track and not hasattr(problem, 'comput_cost'):
        raise ValueError(f'--cost-track needs a comput_cost method; '
                         f'{args.example} does not provide one')

    aux_func = None
    if extended or cost_track:
        vtrue_linf = v_true.abs().max()

        def aux_func():
            out = {}
            with torch.no_grad():
                v_pred = v_theta(problem.t0.unsqueeze(-1), x_test)
                out['rel_linf'] = (v_pred - v_true).abs().max() / vtrue_linf
                out['mean_vtrue_t0'] = v_true.mean()
                if cost_track and soc_mode:
                    x0c = torch.zeros([args.cost_paths, args.dim])
                    cost = problem.comput_cost(u_alpha, x0c,
                                               num_dt=args.cost_num_dt)
                    if world_size > 1:
                        dist.all_reduce(cost, op=dist.ReduceOp.AVG)
                    out['cost'] = cost
                ul = getattr(solver, 'u_lag_last', None)
                if ul is not None:
                    out['u_lag'] = ul
            return out

    method = args.method
    if method is None:
        method = 'socmartnet' if soc_mode else 'prabmartnet'
    if method == 'socmartnet' and not soc_mode:
        raise ValueError('socmartnet (Alg. 3.1) applies to the SOC examples '
                         '(hjb/hjblq/shifttarget); linear/semilinear/linsin '
                         'use prabmartnet (Alg. 3.2)')
    if method == 'prabmartnet' and soc_mode and args.example != 'hjb':
        raise ValueError('prabmartnet (Alg. 3.2, PDE mode) among the SOC '
                         'examples applies only to hjb (the HJB-1 <-> '
                         'semilinear reduction); hjblq/shifttarget provide '
                         'no PDE driver f')
    method_label = {'socmartnet': 'SOCMartNet',
                    'prabmartnet': 'PrabMartNet'}[method]
    sav_name = (f"{args.out}/{method_label}_{problem.name}_d{args.dim}"
                f"_te{args.te}" + (f"_{args.tag}" if args.tag else ""))

    if method == 'prabmartnet':
        solver = SOCMartNet(Dt, problem.mu, problem.sigma, problem.H,
                            problem.v_term, problem.dim_w, t0=problem.t0,
                            f_fun=problem.f,
                            H_depends_on_vx=problem.H_depends_on_vx)
        # forward renew_frac -- the authors' accepted-version PDE INIs set
        # rate_newpath = 0.2 (Count/LinSinCR), matching the SOC
        # configuration.
        solver.train((u_alpha, v_theta, rho_eta),
                     (optim_u, optim_v, optim_rho), (sch_u, sch_v, sch_rho),
                     max_iter, x0, batsize_gpu, rank=rank,
                     lam0=1., delta4=1., lam_bar=1.,  # lambda pinned at 1
                     batsize_milestone=milestones, N=N, err_func=err_func,
                     log_gap=1, J=2, K=1, aux_func=aux_func,
                     renew_frac=args.renew_frac)
    else:
        solver = SOCMartNet(Dt, problem.mu, problem.sigma, problem.H,
                            problem.v_term, problem.dim_w, t0=problem.t0,
                            H_depends_on_vx=problem.H_depends_on_vx,
                            f_cost_fun=getattr(problem, 'f_cost', None))
        if args.fd_residual:
            if not soc_mode or getattr(problem, 'f_cost', None) is None:
                raise ValueError('--fd-residual needs an SOC example with a '
                                 'f_cost method (e.g. hjblq/shifttarget)')
            # The accepted-version configuration pins lambda at 1; --lam-bar > 1 upweights
            # the martingale term
            lam_bar_fd = lam_bar if args.lam_bar is not None else 1.
            solver.train_fd((u_alpha, v_theta, rho_eta),
                            (optim_u, optim_v, optim_rho),
                            (sch_u, sch_v, sch_rho),
                            max_iter, x0, batsize_gpu, rank=rank,
                            batsize_milestone=milestones, N=N,
                            err_func=err_func, log_gap=1, J=2, K=1,
                            aux_func=aux_func, renew_frac=args.renew_frac,
                            lam0=min(10., lam_bar_fd), delta4=10.,
                            lam_bar=lam_bar_fd,
                            ctr_pool=args.controlled_pool)
        else:
            solver.train((u_alpha, v_theta, rho_eta),
                         (optim_u, optim_v, optim_rho),
                         (sch_u, sch_v, sch_rho),
                         max_iter, x0, batsize_gpu, rank=rank,
                         lam0=10., delta4=10., lam_bar=lam_bar,
                         batsize_milestone=milestones, N=N, err_func=err_func,
                         log_gap=1, J=2, K=1, aux_func=aux_func,
                         renew_frac=args.renew_frac)

    if rank == 0:
        Path(args.out).mkdir(exist_ok=True)
        cols = [solver.it_hist, solver.epoch_hist, solver.rt_hist,
                solver.ham_hist, solver.lossmart_hist, solver.error_hist]
        header = ['iter step', 'epoch', 'rt', 'hami', 'mart loss', 'error']
        for key, val in getattr(solver, 'aux_hist', {}).items():
            header.append(key)
            cols.append(val)
        log_tensor = torch.stack(cols, dim=1)
        with open(f'{sav_name}.csv', 'w', encoding='UTF8',
                  newline='') as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            writer.writerows(log_tensor.cpu().numpy())
        v_save = v_theta.module if world_size > 1 else v_theta
        torch.save(v_save.state_dict(), f'{sav_name}_vnn.pkl')
        print(f'saved: {sav_name}.csv')

        save_curves = args.save_curves
        if save_curves is None:
            save_curves = (args.preset == 'sisc'
                           and args.x0_mode == 'region')
        if save_curves and args.x0_mode == 'region':
            region = args.region
            if region is None:
                region = {'hjblq': 's2s3', 'shifttarget': 's2'}.get(
                    args.example, 's1s2')
            curves = {'s1s2': ['e1', 'diag'],
                      's2s3': ['diag', 'manifold'],
                      's2': ['diag']}[region]
            for cv in curves:
                s, v_t, v_p = evaluate.curve_values(
                    problem, v_save, cv, radius=args.x0_scale,
                    num_points=args.curve_points)
                with open(f'{sav_name}_curve_{cv}.csv', 'w',
                          encoding='UTF8', newline='') as fh:
                    writer = csv.writer(fh)
                    writer.writerow(['s', 'v_true', 'v_pred'])
                    writer.writerows(torch.stack(
                        [s, v_t, v_p], dim=1).cpu().numpy())
                print(f'saved: {sav_name}_curve_{cv}.csv')

        if args.eval_ptx:
            for r_frac in (0.125, 0.25):
                s, v_t, v_p = evaluate.ptx_values(
                    problem, v_save, r_frac, radius=args.x0_scale,
                    num_points=args.curve_points)
                with open(f'{sav_name}_ptx{r_frac}.csv', 'w',
                          encoding='UTF8', newline='') as fh:
                    writer = csv.writer(fh)
                    writer.writerow(['s', 'v_true', 'v_pred'])
                    writer.writerows(torch.stack(
                        [s, v_t, v_p], dim=1).cpu().numpy())
                print(f'saved: {sav_name}_ptx{r_frac}.csv')

        if args.eval_path_re:
            t_grid, re_pool, re_mean, re_std = evaluate.path_re(
                problem, v_save, num_dt=N)
            with open(f'{sav_name}_pathre.csv', 'w', encoding='UTF8',
                      newline='') as fh:
                writer = csv.writer(fh)
                writer.writerow(['t', 're_pool', 're_path_mean',
                                 're_path_std'])
                writer.writerows(torch.stack(
                    [t_grid, re_pool, re_mean, re_std], dim=1).cpu().numpy())
            print(f'saved: {sav_name}_pathre.csv')

    if world_size > 1:
        dist.destroy_process_group()


def main(argv=None):
    args = parse_args(argv)
    n_gpus = torch.cuda.device_count()
    if args.gpus > 0:
        n_gpus = min(args.gpus, n_gpus)
    n_gpus = max(n_gpus, 1)
    print(f'Number of used GPUs: {n_gpus}')
    if n_gpus > 1:
        import torch.multiprocessing as mp
        mp.spawn(worker, args=(n_gpus, args), nprocs=n_gpus, join=True)
    else:
        worker(0, 1, args)
