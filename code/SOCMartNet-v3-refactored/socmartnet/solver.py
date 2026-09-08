"""SOC-MartNet solver (arXiv:2405.03169v3), Algorithms 3.1 (SOC mode) and
3.2 (parabolic/PDE mode).

Paper notation used throughout:
  pi_N, Dt      -- uniform time partition {t_n} and step Δt (Eq. (3.28))
  u_alpha, v_theta, rho_eta -- the three networks (Eqs. (3.24)-(3.25))
  H             -- Hamiltonian along the paths (Eq. (3.3)); in PDE mode the
                   parabolic nonlinearity f plays its role (Sec. 3.4)
  dM            -- trapezoid martingale increment residual
                   dM_{n+1} = (v_{n+1} - v_n)/Dt + (H_n + H_{n+1})/2
                   (Eq. (3.29) divided through by Dt -- the code scaling;
                   the paper's G carries an extra Dt factor, absorbed by lambda)
  G             -- projected residual (1/|A|) sum rho_eta * dM  in R^r
                   (Eq. (3.30)); mart_loss = |G|^2 with the authors' DDP
                   minibatch-bias correction (all-reduce on detached G)
  L             -- mean(H) + lam * |G|^2   (Eq. (3.31); the paper's first term
                   (1/|A|) sum H Dt differs by the constant factor Dt)
  lam           -- multiplier, updated as
                   lam <- min{lam_bar, lam + delta4 * |G|^2}   (Alg. 3.1, L.11)
  J, K          -- inner descent/ascent steps per iteration (J = 2K = 2)

State paths X follow the *pilot* SDE (mu, sigma), fixed and independent of the
control, and are generated offline once before training (Remark 3.6).
"""

import time

import torch
import torch.distributed as dist


def _all_reduce_mean(t):
    """Mean over ranks; no-op when DDP is not initialized.

    Unlike the authors' accepted-version code (SUM with no world-size compensation, which
    scales the gradient by ws in multi-GPU), this uses ReduceOp.AVG so the
    detached factor is the true cross-rank average.  In single-GPU mode
    (ws=1) it reduces to identity, preserving the archived numbers.
    """
    if dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.AVG)
    return t


def bat_vgrad(v_net, t, x):
    """(grad_x v, v) with the graph kept; works on DDP-wrapped nets too."""
    is_req = x.requires_grad
    x.requires_grad = True
    val = v_net(t, x)
    dx_val = torch.autograd.grad(val, x,
                                 grad_outputs=torch.ones_like(val),
                                 create_graph=True)[0]
    x.requires_grad = is_req
    return dx_val.unsqueeze(-2), val


class SOCMartNet:

    def __init__(self, Dt, mu, sigma, H_fun, g_term, dim_w,
                 f_fun=None, H_depends_on_vx=True, H_depends_on_vxx=False,
                 t0=torch.tensor(0.), f_cost_fun=None):
        if H_depends_on_vxx:
            raise RuntimeError('unsupported H_depends_on_vxx')
        self.Dt = Dt
        self.t0 = t0
        self.mu = mu                  # pilot drift  b_func
        self.sigma = sigma            # pilot diffusion diagonal  sgmd_func
        self.H_fun = H_fun            # Hamiltonian  h_func
        self.f_fun = f_fun            # parabolic nonlinearity (PDE mode)
        self.g_term = g_term          # terminal function  v_term
        self.dim_w = dim_w
        self.H_depends_on_vx = H_depends_on_vx
        # FD residual: running cost f(u) for the control term
        self.f_cost_fun = f_cost_fun
        # mean |u_cur - u_old| of the last delta_m_fd call when the
        # controlled pool is in use; logged via aux_func as 'u_lag'
        self.u_lag_last = None

    @property
    def soc_mode(self):
        """True: Algorithm 3.1 (control u_alpha trained). False: Alg. 3.2."""
        return self.f_fun is None

    # ---- sample paths of the pilot SDE by the Euler scheme (Eq. (3.28)) ----
    def simu_paths(self, x0, N):
        num_path = x0.shape[0]
        xt = [x0]
        t_part = [self.t0]
        tn = self.t0
        for _ in range(N):
            drift = self.mu(tn, xt[-1]) * self.Dt
            dB = torch.normal(mean=torch.zeros([num_path, self.dim_w]),
                              std=torch.sqrt(self.Dt))
            xt.append(xt[-1] + drift + self.sigma(tn, xt[-1]) * dB)
            tn = tn + self.Dt
            t_part.append(tn)
        return torch.stack(t_part), torch.stack(xt)

    # ---- sample paths of the CONTROLLED SDE (drift mu + 2*u_alpha)
    # with u_alpha evaluated online along the path under no_grad, and record
    # the generation-time control u_old per node.  The FD residual on these
    # paths corrects only the lag 2*(u_cur - u_old).grad v instead of the
    # full 2u.grad v (no lag bias, no double count --
    # u_old cancels identically in the residual drift).  RNG consumption has
    # the same count/shape/order as simu_paths, so the noise stream is
    # preserved.  Returns (t_part, xt, ut_old) with ut_old [N, path, x]. ----
    def simu_paths_ctr(self, x0, N, u_alpha):
        num_path = x0.shape[0]
        xt = [x0]
        ut = []
        t_part = [self.t0]
        tn = self.t0
        with torch.no_grad():
            for _ in range(N):
                un = u_alpha(tn.reshape(1), xt[-1])
                drift = (self.mu(tn, xt[-1]) + 2 * un) * self.Dt
                dB = torch.normal(mean=torch.zeros([num_path,
                                                    self.dim_w]),
                                  std=torch.sqrt(self.Dt))
                xt.append(xt[-1] + drift + self.sigma(tn, xt[-1]) * dB)
                ut.append(un)
                tn = tn + self.Dt
                t_part.append(tn)
        return torch.stack(t_part), torch.stack(xt), torch.stack(ut)

    # ---- regenerate num_newpath pilot paths (Sec. 4 authors'
    # accepted-version
    # configuration: epochsize pool + rate_newpath renewal).  New start
    # points are resampled from x0_all and the paths follow the SAME pilot
    # SDE, so the
    # martingale residual keeps its form; only the empirical path measure is
    # refreshed each epoch.  Returns xt with num_newpath slots renewed. ----
    def renew_paths(self, x0_all, N, num_newpath, xt):
        x0_new = x0_all[torch.randperm(x0_all.shape[0])[:num_newpath]]
        _, xt_new = self.simu_paths(x0_new, N)
        return torch.cat([xt[:, num_newpath:], xt_new], dim=1)

    # ---- controlled renewal -- same FIFO discipline as renew_paths
    # (drop the oldest num_newpath slots, append fresh ones) and identical
    # RNG consumption, but the fresh paths follow the controlled SDE under
    # the CURRENT u_alpha and their generation-time controls replace the
    # matching slots of the companion u_old pool. ----
    def renew_paths_ctr(self, x0_all, N, num_newpath, xt, u_old, u_alpha):
        x0_new = x0_all[torch.randperm(x0_all.shape[0])[:num_newpath]]
        _, xt_new, ut_new = self.simu_paths_ctr(x0_new, N, u_alpha)
        return (torch.cat([xt[:, num_newpath:], xt_new], dim=1),
                torch.cat([u_old[:, num_newpath:], ut_new], dim=1))

    # ---- martingale loss |G|^2: detached cross-rank average in one factor
    # (DDP minibatch-bias correction; ReduceOp.AVG, unlike the accepted-version
    # SUM without world-size compensation) ----
    def loss_mart(self, v_path, H_path, rho_path, vte):
        # terminal replacement: v_theta(T, x) = g(x)  (Eq. (3.24))
        v_path = torch.cat([v_path, vte.unsqueeze(0)], dim=0)
        H_path = torch.cat([H_path, H_path[-1].unsqueeze(0)], dim=0)

        dv = (v_path[1:] - v_path[:-1]) / self.Dt
        avg_H = 0.5 * (H_path[1:] + H_path[:-1])
        dM = dv + avg_H                                    # trapezoid residual
        return self.loss_mart_dm(dM, rho_path)

    def loss_mart_dm(self, dM, rho_path):
        """|G|^2 projection applied to a precomputed per-node residual
        dM [N, batch, 1]:  2 * mean(G^2)  with G = (rho_path * dM) averaged
        over the batch axis; the detached cross-rank mean in one factor is
        the authors' DDP minibatch-bias correction."""
        G = (rho_path * dM).mean([0, 1])
        G_det = _all_reduce_mean(G.detach())
        return 2 * (G_det * G).mean()

    # ---- FD residual (the accepted-version SocMartNet.delta_m, with
    # the u-gradient bug FIXED).  The controlled-drift contribution
    # (mu_sys - mu_pil).grad v = 2 u.grad v is approximated by a forward
    # finite difference of step h = Dt^2 evaluated at the CURRENT u_alpha on
    # every call, so no path-drift lag is involved.
    #
    # Reuse: v_bat, u_bat, f_bat, x_forw and the shifted forward
    # v_theta(t, x_forw) are each computed ONCE per call and shared by the
    # martingale and control branches; only the detach pattern differs.
    def delta_m_fd(self, t_bat, xt_bat, vte_bat, u_alpha, v_theta,
                   uold_bat=None):
        h = self.Dt.pow(2)
        u_bat = u_alpha(t_bat, xt_bat)
        v_bat = v_theta(t_bat, xt_bat)
        f_bat = self.f_cost_fun(u_bat)
        if uold_bat is None:
            x_forw = xt_bat + 2 * u_bat * h
        else:
            # a controlled-pool path already embeds the
            # drift 2*u_old (generation-time snapshot), so the FD term
            # corrects only the lag 2*(u_cur - u_old).grad v.  u_old is a
            # stored constant (no grad); the u-gradient flows through
            # u_bat exactly as in the pilot-pool case.  For the initial
            # pilot pool u_old = 0, recovering the full-FD form bitwise.
            x_forw = xt_bat + 2 * (u_bat - uold_bat) * h
            self.u_lag_last = (u_bat - uold_bat).abs().mean().detach()

        # frozen-parameter forward: gradient flows through the input x_forw
        # (hence to u_alpha) but NOT to v_theta's weights -- the HEAD-style
        # fix of the authors' no_grad bug, which had killed this u path.
        for p in v_theta.parameters():
            p.requires_grad_(False)
        vfw_u = v_theta(t_bat, x_forw)
        for p in v_theta.parameters():
            p.requires_grad_(True)

        # shifted forward for the martingale branch: x_forw detached so no
        # u-gradient, but v_theta keeps its gradient (accepted-version vgrad branch)
        vfw_v = v_theta(t_bat, x_forw.detach())

        # time increment with terminal replacement v_theta(T, x_N) = g(x_N)
        v_path = torch.cat([v_bat, vte_bat.unsqueeze(0)], dim=0)
        dv = (v_path[1:] - v_path[:-1]) / self.Dt

        # controlled-drift FD term (pointwise per node, matching the
        # accepted-version delta_m); trapezoid average is applied to f only.
        d_fd_u = (vfw_u - v_bat.detach()) / h
        d_fd_v = (vfw_v - v_bat) / h
        f_bat = torch.cat([f_bat, f_bat[-1].unsqueeze(0)], dim=0)
        f_avg = 0.5 * (f_bat[1:] + f_bat[:-1])

        # martingale branch: trains v_theta only (u / f detached)
        dM_vgrad = dv + d_fd_v + f_avg.detach()
        # control branch: trains u_alpha only (dv v-part detached; the
        # u-gradient is kept through x_forw in d_fd_u and through f_avg)
        dM_ugrad = dv.detach() + d_fd_u + f_avg
        return dM_vgrad, dM_ugrad

    def train_fd(self, nets, optims, schs, max_iter, x0, batsize_arr,
                 rank='None', batsize_milestone=None, N=10,
                 err_func=lambda: torch.nan, log_gap=10, J=2, K=1,
                 max_epoch=float('inf'), aux_func=None, renew_frac=0.,
                 lam0=1., delta4=0., lam_bar=1., ctr_pool=False):
        """FD-residual training loop (the accepted-version configuration).

        loss_desc = lam * mart_loss + ctr_loss, loss_asc = -mart_loss, with
        lam <- min(lam_bar, lam + delta4 * mart_loss) (Alg. 3.1 Line 11).
        The accepted-version configuration pins lam at 1 (lam0=1, delta4=0,
        lam_bar=1);
        the lam_bar=10 variant upweights the martingale term.
        The martingale residual is delta_m_fd; the u-gradient bug is fixed.
        ctr_pool: renew the pool with CONTROLLED paths simulated
        under the current u_alpha, storing the generation-time controls in
        a companion u_old pool; delta_m_fd then corrects only the lag
        2*(u_cur - u_old).grad v.  The initial pool stays pilot with
        u_old = 0 (full-FD form), so ctr_pool differs from pilot renewal
        only after the first epoch boundary.  Requires renew_frac > 0.
        """
        u_alpha, v_theta, rho_eta = nets
        unn_optim, vnn_optim, rhonn_optim = optims
        unn_sch, vnn_sch, rhonn_sch = schs

        rt0 = time.time()
        u_alpha.train()
        v_theta.train()
        rho_eta.train()

        tot_size = x0.shape[0]
        t_part, xt = self.simu_paths(x0, N)[:2]
        t = t_part.expand(1, tot_size, -1).transpose(0, 2).contiguous()
        vte = self.g_term(xt[-1])

        renew_on = renew_frac > 0. and self.soc_mode
        num_newpath = max(1, int(tot_size * renew_frac)) if renew_on else 0
        n_epoch_done = 0
        # companion pool of generation-time controls; the initial
        # pilot pool carries u_old = 0 (its embedded control drift is zero)
        ctr_on = ctr_pool and renew_on
        u_old = xt.new_zeros((N, tot_size, xt.shape[-1])) if ctr_on else None

        it_hist, epoch_hist, rt_hist = [], [], []
        ham_hist, lossmart_hist, error_hist = [], [], []
        aux_hist = {}

        lam = lam0
        it = 0
        epoch = 0.
        bat_stage = 0
        bat_size = batsize_arr[bat_stage]
        if batsize_milestone is None:
            batsize_milestone = []
        batsize_milestone = batsize_milestone + [max_iter + 1]

        while True:
            if (it > max_iter) or (epoch > max_epoch):
                break
            if it >= batsize_milestone[bat_stage]:
                bat_stage += 1
                bat_size = batsize_arr[bat_stage]

            if renew_on and int(epoch) > n_epoch_done:
                if ctr_on:
                    xt, u_old = self.renew_paths_ctr(x0, N, num_newpath, xt,
                                                     u_old, u_alpha)
                else:
                    xt = self.renew_paths(x0, N, num_newpath, xt)
                vte = self.g_term(xt[-1])
                n_epoch_done = int(epoch)

            bat_idx = torch.randperm(tot_size)[:bat_size]
            t_bat = t[:-1, bat_idx]
            xt_bat = xt[:-1, bat_idx]
            vte_bat = vte[bat_idx]
            uold_bat = u_old[:, bat_idx] if ctr_on else None
            rho_bat = rho_eta(t_bat, xt_bat).detach()

            dM_v_last = None
            for _ in range(J):
                dM_v, dM_u = self.delta_m_fd(t_bat, xt_bat, vte_bat,
                                             u_alpha, v_theta,
                                             uold_bat=uold_bat)
                mart_loss = self.loss_mart_dm(dM_v, rho_bat)
                lam = min(lam_bar, lam + delta4 * mart_loss.detach())
                ctr_loss = dM_u.mean()
                loss_tot = lam * mart_loss + ctr_loss
                loss_tot.backward()
                unn_optim.step()
                vnn_optim.step()
                unn_optim.zero_grad()
                vnn_optim.zero_grad()
                dM_v_last = dM_v

            ham_mean = ctr_loss.detach()  # logged as the control term
            for _ in range(K):  # ascent on eta; reuse the last descent dM_v
                rho_bat = rho_eta(t_bat, xt_bat)
                mart_asc = self.loss_mart_dm(dM_v_last.detach(), rho_bat)
                loss_test = -mart_asc
                loss_test.backward()
                rhonn_optim.step()
                rhonn_optim.zero_grad()

            if it % log_gap == 0:
                error = err_func().detach()
                lr = vnn_optim.param_groups[0]['lr']
                rt = time.time() - rt0
                it_hist.append(it)
                epoch_hist.append(epoch)
                rt_hist.append(rt)
                ham_hist.append(ham_mean)
                lossmart_hist.append(mart_loss.detach())
                error_hist.append(error.detach())
                if aux_func is not None:
                    for key, val in aux_func().items():
                        aux_hist.setdefault(key, []).append(
                            float(val.detach()))
                print(
                    f"rank: {rank}\niter step: [{it}/{max_iter}], rt: {rt:.2f}, epoch: {epoch:.2f},\nbat_size: {bat_size}, lr: {lr:.5}, lambda: {lam:.5},\nloss_mart: {mart_loss.item():.5}, ctr: {ctr_loss.item():.5},\nerror: {error:.5}\n"
                )

            it += 1
            epoch += bat_size / tot_size
            if (unn_sch is not None) and self.soc_mode:
                unn_sch.step()
            if vnn_sch is not None:
                vnn_sch.step()
            if rhonn_sch is not None:
                rhonn_sch.step()

        self.it_hist = torch.tensor(it_hist)
        self.rt_hist = torch.tensor(rt_hist)
        self.epoch_hist = torch.tensor(epoch_hist)
        self.ham_hist = torch.stack(ham_hist, dim=0)
        self.lossmart_hist = torch.stack(lossmart_hist, dim=0)
        self.error_hist = torch.stack(error_hist, dim=0)
        self.aux_hist = {key: torch.tensor(val)
                         for key, val in aux_hist.items()}

        u_alpha.eval()
        v_theta.eval()
        rho_eta.eval()

    # ---- batched evaluation of v_theta and H on the training batch ----
    def eval_vH(self, t_bat, x_bat, u_alpha, v_theta):
        if self.H_depends_on_vx:
            vx_bat, v_bat = bat_vgrad(v_theta, t_bat, x_bat)
        else:
            v_bat = v_theta(t_bat, x_bat)
            vx_bat = None

        if self.soc_mode:
            u_bat = u_alpha(t_bat, x_bat)
            H_bat = self.H_fun(t_bat, x_bat, u_bat, v_bat, vx_bat, None)
        else:
            H_bat = self.f_fun(t_bat, x_bat, v_bat, vx_bat, None)
        return v_bat, H_bat

    # ---- Algorithm 3.1 / 3.2 ----
    def train(self, nets, optims, schs, max_iter, x0, batsize_arr,
              rank='None', lam0=1., batsize_milestone=None, N=10,
              err_func=lambda: torch.nan, log_gap=10, J=2, K=1,
              delta4=10, lam_bar=1000., max_epoch=float('inf'),
              aux_func=None, renew_frac=0.):
        # aux_func: optional logging-only hook evaluated at every log point
        # (after the optimizer steps, typically under no_grad); must return a
        # dict of scalar tensors.  It does not affect the training trajectory
        # beyond its own RNG consumption; default None reproduces the
        # validated baseline exactly.
        # renew_frac: fraction of the path pool regenerated at the end
        # of each epoch (the authors' accepted-version rate_newpath,
        # e.g. 0.2); 0 disables.

        u_alpha, v_theta, rho_eta = nets
        unn_optim, vnn_optim, rhonn_optim = optims
        unn_sch, vnn_sch, rhonn_sch = schs

        rt0 = time.time()
        u_alpha.train()
        v_theta.train()
        rho_eta.train()

        tot_size = x0.shape[0]

        t_part, xt = self.simu_paths(x0, N)[:2]
        t = t_part.expand(1, tot_size, -1).transpose(0, 2).contiguous()
        vte = self.g_term(xt[-1])

        # path-renewal config (the authors' accepted-version rate_newpath
        # configuration).
        # the soc_mode gate is dropped -- the authors' accepted-version PDE-mode
        # INIs (CodeAndFig_Revision Count/LinSinCR, e.g. 0_Counter_d100_
        # w110_h6.ini) set rate_newpath = 0.2, so PDE runs also renew the
        # pool at epoch boundaries.
        # Legacy PDE calls never forwarded renew_frac, so no validated run
        # shifts.
        renew_on = renew_frac > 0.
        num_newpath = max(1, int(tot_size * renew_frac)) if renew_on else 0
        n_epoch_done = 0

        it_hist, epoch_hist, rt_hist = [], [], []
        ham_hist, lossmart_hist, error_hist = [], [], []
        aux_hist = {}

        lam = lam0
        it = 0
        epoch = 0.
        bat_stage = 0
        bat_size = batsize_arr[bat_stage]
        if batsize_milestone is None:
            batsize_milestone = []
        batsize_milestone = batsize_milestone + [max_iter + 1]

        while True:
            if (it > max_iter) or (epoch > max_epoch):
                break
            if it >= batsize_milestone[bat_stage]:
                bat_stage += 1
                bat_size = batsize_arr[bat_stage]

            # renew renew_frac of the path pool at each epoch boundary
            if renew_on and int(epoch) > n_epoch_done:
                xt = self.renew_paths(x0, N, num_newpath, xt)
                vte = self.g_term(xt[-1])
                n_epoch_done = int(epoch)

            # index subset A_i = {0..N-1} x M_i (Eq. (3.32) style)
            bat_idx = torch.randperm(tot_size)[:bat_size]
            t_bat = t[:-1, bat_idx]
            xt_bat = xt[:-1, bat_idx]
            vte_bat = vte[bat_idx]
            rho_bat = rho_eta(t_bat, xt_bat).detach()

            for _ in range(J):  # descent on (alpha, theta)
                v_bat, H_bat = self.eval_vH(t_bat, xt_bat, u_alpha, v_theta)
                mart_loss = self.loss_mart(v_bat, H_bat, rho_bat, vte_bat)

                # multiplier update (Alg. 3.1 Line 11; applied inside the
                # J-loop, as in the authors' code)
                lam = min(lam_bar, lam + delta4 * mart_loss.detach())

                ham_mean = H_bat.mean()
                if self.soc_mode:
                    loss_tot = ham_mean + lam * mart_loss
                    loss_tot.backward()
                    unn_optim.step()
                    vnn_optim.step()
                    unn_optim.zero_grad()
                    vnn_optim.zero_grad()
                else:
                    loss_tot = lam * mart_loss
                    loss_tot.backward()
                    vnn_optim.step()
                    vnn_optim.zero_grad()

            v_bat, H_bat = self.eval_vH(t_bat, xt_bat, u_alpha, v_theta)
            for _ in range(K):  # ascent on eta
                rho_bat = rho_eta(t_bat, xt_bat)
                mart_asc = self.loss_mart(v_bat.detach(), H_bat.detach(),
                                          rho_bat, vte_bat)
                loss_test = -mart_asc
                loss_test.backward()
                rhonn_optim.step()
                rhonn_optim.zero_grad()

            if it % log_gap == 0:
                error = err_func().detach()
                lr = vnn_optim.param_groups[0]['lr']
                rt = time.time() - rt0
                it_hist.append(it)
                epoch_hist.append(epoch)
                rt_hist.append(rt)
                ham_hist.append(ham_mean.detach())
                lossmart_hist.append(mart_loss.detach())
                error_hist.append(error.detach())
                if aux_func is not None:
                    for key, val in aux_func().items():
                        aux_hist.setdefault(key, []).append(
                            float(val.detach()))
                print(
                    f"rank: {rank}\niter step: [{it}/{max_iter}], rt: {rt:.2f}, epoch: {epoch:.2f},\nbat_size: {bat_size}, lr: {lr:.5}, lambda: {lam:.5},\nloss_mart: {mart_loss.item():.5}, ham: {ham_mean.item():.5},\nerror: {error:.5}\n"
                )

            it += 1
            epoch += bat_size / tot_size
            if (unn_sch is not None) and self.soc_mode:
                unn_sch.step()
            if vnn_sch is not None:
                vnn_sch.step()
            if rhonn_sch is not None:
                rhonn_sch.step()

        self.it_hist = torch.tensor(it_hist)
        self.rt_hist = torch.tensor(rt_hist)
        self.epoch_hist = torch.tensor(epoch_hist)
        self.ham_hist = torch.stack(ham_hist, dim=0)
        self.lossmart_hist = torch.stack(lossmart_hist, dim=0)
        self.error_hist = torch.stack(error_hist, dim=0)
        self.aux_hist = {key: torch.tensor(val)
                         for key, val in aux_hist.items()}

        u_alpha.eval()
        v_theta.eval()
        rho_eta.eval()
        return u_alpha, v_theta, rho_eta
