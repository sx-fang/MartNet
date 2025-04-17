# Define metaclass for PDEs and related functions
import abc
from abc import abstractmethod
from itertools import cycle
from pathlib import Path
from typing import Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist


def tensor2ndarray(tensor):
    return tensor.detach().cpu().numpy()


def split_number(num, num_parts):
    part_size = num // num_parts
    parts = [part_size] * num_parts
    for i in range(num % num_parts):
        parts[i] += 1
    return parts


def get_interval(s_range: Union[float, Sequence[float]]):
    if isinstance(s_range, Sequence) is True:
        left_end, right_end = s_range
    else:
        left_end = -s_range
        right_end = s_range
    return left_end, right_end


def e1_curve(
    dim_x: int,
    num_points: int,
    s_range: Union[float, Sequence[float]],
):
    left_end, right_end = get_interval(s_range)
    s_coord = torch.linspace(left_end, right_end, num_points)
    e_1stcoord = torch.zeros([dim_x])
    e_1stcoord[0] = 1.
    xe1 = torch.outer(s_coord, e_1stcoord)
    return s_coord, xe1


def diag_curve(
    dim_x: int,
    num_points: int,
    s_range: Union[float, Sequence[float]],
):
    left_end, right_end = get_interval(s_range)
    s_coord = torch.linspace(left_end, right_end, num_points)
    e_diag = torch.ones((dim_x, ))
    s_coord = torch.linspace(left_end, right_end, num_points)
    xdiag = torch.outer(s_coord, e_diag)
    return s_coord, xdiag


def manifold_curve(
    dim_x: int,
    num_points: int,
    s_range: Union[float, Sequence[float]],
):
    left_end, right_end = get_interval(s_range)
    s_coord = torch.linspace(left_end, right_end, num_points)
    e_vec = torch.arange(1, dim_x + 1)
    x_diag = torch.outer(s_coord, torch.sign(torch.sin(e_vec)))
    x = x_diag + torch.cos(e_vec + s_coord.unsqueeze(-1) * torch.pi)
    return s_coord, x


def origin_point(dim_x, num_points):
    s_coord = torch.zeros([num_points])
    x = torch.zeros([num_points, dim_x])
    return s_coord, x


def plot_error_path(t_grid, errors, labels, sav_name):
    cmap = plt.get_cmap('tab10')(np.arange(len(labels)))
    markers = cycle(('s', 'D', '^', 'v', 'x', '+'))
    t_grid = tensor2ndarray(t_grid)
    errors = [tensor2ndarray(err) for err in errors]
    for err, lab, cm in zip(errors, labels, cmap):
        plt.plot(t_grid, err, color=cm, label=lab, marker=next(markers))
    plt.legend()
    plt.xlabel('$t$')
    plt.ylabel('Relative error')
    ymax = max([err.max() for err in errors])
    plt.ylim(top=min(2 * ymax, 1.))
    Path(sav_name).parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(f'{sav_name}repath.pdf')
    plt.close()


def plot_vtxpath(t_grid, vt_true, vt_appr, sav_name):

    c_list = plt.get_cmap('tab10')(np.arange(2))
    t_grid = tensor2ndarray(t_grid)
    vt_true = tensor2ndarray(vt_true)
    vt_appr = tensor2ndarray(vt_appr)
    for i in range(vt_true.shape[1]):
        if i == 0:
            labels = ("Predicted $v(t, X_t)$", "Exact $v(t, X_t)$",
                      "Exact $v(0, X_0)$", "Exact $v(T, X_T)$")
        else:
            labels = (None, None, None, None)
        plt.plot(t_grid, vt_appr[:, i], color=c_list[1], label=labels[0])
        plt.plot(t_grid, vt_true[:, i], color=c_list[0], label=labels[1])
        plt.scatter(t_grid[0],
                    vt_true[0, i],
                    color='black',
                    label=labels[0],
                    marker='s')
        plt.scatter(t_grid[-1],
                    vt_true[-1, i],
                    color='black',
                    label=labels[0],
                    marker='o')
    plt.legend()
    plt.xlabel('$t$')
    Path(sav_name).parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(f'{sav_name}vpath.pdf')
    plt.close()


def plot_on_curve(s, v_true, v_approx, sav_prefix, labels=None, ylim=None):
    if labels is None:
        labels = ['True', 'Predicted']
    plt.plot(s, v_true, label=labels[0], color='blue')
    plt.scatter(s, v_approx, label=labels[1], color='red')
    if ylim is not None:
        plt.ylim(*ylim)
    plt.legend()
    Path(sav_prefix).parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(sav_prefix)
    plt.close()


def modify_ylim(y_min, y_max, rate_edge=0.2, include_zero=True, min_high=1.):
    if y_max - y_min < min_high:
        shift = min_high - (y_max - y_min)
        y_min = y_min - shift / 2
        y_max = y_max + shift / 2

    edge = rate_edge * (y_max - y_min)
    y_min = y_min - edge
    y_max = y_max + edge

    if include_zero is True:
        y_min = min(0, y_min)
        y_max = max(0, y_max)
    return y_min, y_max


class PDE(metaclass=abc.ABCMeta):

    name = 'Default_PDE_Name'
    musys_depends_on_v = True
    sgmsys_depends_on_v = True
    f_depends_on_v = True

    v_shellfunc = None

    def __init__(self, dim_x, t0=0., te=1., use_dist=False) -> None:
        self.dim_x = dim_x
        self.dim_w = dim_x

        if isinstance(t0, float) or isinstance(t0, int):
            t0 = torch.tensor(t0)
        if isinstance(te, float) or isinstance(te, int):
            te = torch.tensor(te)
        self.t0 = t0
        self.te = te

        self.use_dist = use_dist
        if use_dist is True:
            self.world_size = dist.get_world_size()
        else:
            self.world_size = 1

    @abstractmethod
    def mu_pil(self, _t, _x):
        pass

    @abstractmethod
    def sgm_pil(self, _t, _x, _dw):
        pass

    @abstractmethod
    def mu_sys(self, _t, _x, _k):
        pass

    @abstractmethod
    def sgm_sys(self, _t, _x, _k, _dw):
        pass

    @abstractmethod
    def f(self, _t, _x, _v):
        pass

    @abstractmethod
    def x0_points(self, num_points):
        pass

    def produce_logfunc(self, _v_approx, _num_testpoint=1000):
        pass

    def produce_results(self, _v_approx, _sav_prefix):
        pass

    def gen_pilpath(self, x0: torch.Tensor, num_dt: int):
        dt = (self.te - self.t0) / num_dt
        dwt = torch.normal(mean=0.,
                           std=torch.sqrt(dt),
                           size=(num_dt, x0.shape[-2], self.dim_w),
                           device=x0.device)

        t = [torch.full((1, 1), self.t0)]
        xt = [x0]
        for n in range(num_dt):
            tn = t[n]
            xtn = xt[n]
            mu_tn = self.mu_pil(tn, xtn)
            sgm_dwtn = self.sgm_pil(tn, xtn, dwt[n])
            t.append(tn + dt)
            xt.append(xtn + mu_tn * dt + sgm_dwtn)
        t = torch.stack(t, dim=0)
        xt = torch.stack(xt, dim=0)
        # xt: [time, path, dim of x]
        return t, xt

    def gen_syspath(self, x0: torch.Tensor, num_dt: int, v_func):
        dt = (self.te - self.t0) / num_dt
        dwt = torch.normal(mean=0.,
                           std=torch.sqrt(dt),
                           size=(num_dt, x0.shape[-2], self.dim_w),
                           device=x0.device)

        t = [torch.full((1, 1), self.t0)]
        xt = [x0]
        vt = [v_func(t[0], x0)]
        for n in range(num_dt):
            tn = t[n]
            xtn = xt[n]
            vtn = vt[n]
            mu_tn = self.mu_sys(tn, xtn, vtn)
            sgm_dwtn = self.sgm_sys(tn, xtn, vtn, dwt[n])
            t.append(tn + dt)
            xt.append(xtn + mu_tn * dt + sgm_dwtn)
            vt.append(v_func(t[-1], xt[-1]))
        t = torch.stack(t, dim=0)
        xt = torch.stack(xt, dim=0)
        vt = torch.stack(vt, dim=0)
        # xt: [time, path, dim of x]
        return t, xt, vt


class PDEwithVtrue(PDE):
    name = 'Default_TVP_Name'

    x0pil_range = 1.
    x0_for_train = {'S2': diag_curve, 'S3': manifold_curve}
    record_linf_error = True

    @abstractmethod
    def v(self, _t, _x):
        pass

    def x0_points(self, num_points):
        curve_funcs = self.x0_for_train.values()
        num_points = split_number(num_points, len(curve_funcs))
        x0_list = [
            c_func(self.dim_x, nump, self.x0pil_range)[1]
            for nump, c_func in zip(num_points, curve_funcs)
        ]
        x0 = torch.concatenate(x0_list, dim=0)
        return x0

    def produce_logfunc(self, v_approx, num_testpoint=1000):
        t0 = self.t0.unsqueeze(-1)
        x_test = self.x0_points(num_testpoint)
        vtrue_ontest = self.v(t0, x_test)
        vtrue_ontest_l1 = torch.abs(vtrue_ontest).mean()
        vtrue_ontest_linf = torch.abs(vtrue_ontest).max()

        def log_func(_it: int, _t: torch.Tensor, _xt: torch.Tensor):
            err = v_approx(t0, x_test) - vtrue_ontest
            abs_err = torch.abs(err)
            l1err = abs_err.mean() / vtrue_ontest_l1
            log = {'rel_l1err': l1err.item()}
            if self.record_linf_error is True:
                linf_err = abs_err.max() / vtrue_ontest_linf
                log['rel_linferr'] = linf_err.item()
            return log

        return log_func

    def __x0_for_res(self, num_points=50):
        args = (self.dim_x, num_points, self.x0pil_range)
        curve_name = list(self.x0_for_train.keys())
        s, xs = zip(*[x0f(*args) for x0f in self.x0_for_train.values()])
        return curve_name, s, xs

    def __save_res_oncurve(self, sav_prefix, curve_name, s, vtrue_x0s,
                           vapprox_x0s):
        res_header = []
        res_col = []
        for (cname, si, vtrue, vappr) in \
                zip(curve_name, s, vtrue_x0s, vapprox_x0s):
            res_header.extend(
                [f'coord_{cname}', f'vtrue_{cname}', f'vappr_{cname}'])
            res_col.extend(
                [tensor2ndarray(si),
                 vtrue.squeeze(-1),
                 vappr.squeeze(-1)])
        res_arr = np.stack(res_col, axis=1)
        np.savetxt(sav_prefix + 'res_on_line.csv',
                   res_arr,
                   delimiter=',',
                   header=','.join(res_header))

    def __res_on_curve(self, sav_prefix, v_approx, y_min=None, y_max=None):
        curve_name, s, x0s = self.__x0_for_res()
        t0_uns = self.t0.unsqueeze(-1)
        vtrue_x0s = [tensor2ndarray(self.v(t0_uns, x0)) for x0 in x0s]
        vapprox_x0s = [tensor2ndarray(v_approx(t0_uns, x0)) for x0 in x0s]

        if (y_min is None) or (y_max is None):
            v_max = np.nanmax([vtrue_x0s, vapprox_x0s])
            v_min = np.nanmin([vtrue_x0s, vapprox_x0s])
            v_min, v_max = modify_ylim(v_min, v_max)
            y_min = v_min if y_min is None else y_min
            y_max = v_max if y_max is None else y_max
        for i in range(len(s)):
            plot_on_curve(s[i].cpu().numpy(),
                          vtrue_x0s[i],
                          vapprox_x0s[i],
                          sav_prefix + curve_name[i] + '.pdf',
                          ylim=(y_min, y_max))

        self.__save_res_oncurve(sav_prefix, curve_name, s, vtrue_x0s,
                                vapprox_x0s)

    def __res_on_path(
        self,
        sav_prefix,
        v_approx,
        num_path=8,
        num_dt=50,
    ):
        res_header = []
        res_col = []
        relerrs = []
        curve_name, _, xs = self.__x0_for_res(num_points=num_path)
        for x0name, x0 in zip(curve_name, xs):
            with torch.no_grad():
                t, xt = self.gen_pilpath(x0, num_dt)
                vt_appr = v_approx(t, xt)
                vt_true = self.v(t, xt)
            # vt_true1 = self.v(t[:-1], xt[:-1])
            # vt_true2 = self.v_term(xt[-1:])
            # vt_true = torch.cat([vt_true1, vt_true2], dim=0)
            t_grid = t.squeeze()

            # plot_vtxpath(t_grid, vt_true, vt_appr, f'{sav_prefix}{x0name}_')
            err = (vt_appr - vt_true).abs().mean([1])
            relerr = err / vt_true.abs().mean()

            idx_path = range(vt_true.shape[1])
            res_header.extend([f'vtrue_{x0name}_path{i}' for i in idx_path])
            res_header.extend([f'vappr_{x0name}_path{i}' for i in idx_path])
            res_header.extend([f'relerr_{x0name}_path{i}' for i in idx_path])
            res_col.extend([vt_true.squeeze(-1), vt_appr.squeeze(-1), relerr])
            relerrs.append(relerr)
        res_header.insert(0, 't')
        res_col.insert(0, t.squeeze(-1))
        res_col = tensor2ndarray(torch.concat(res_col, dim=-1))
        np.savetxt(sav_prefix + 'res_on_path.csv',
                   res_col,
                   delimiter=',',
                   header=','.join(res_header))
        plot_error_path(t_grid, relerrs, list(curve_name), f'{sav_prefix}')

    def produce_results(self, v_approx, sav_prefix):
        self.__res_on_curve(sav_prefix, v_approx)
        self.__res_on_path(sav_prefix, v_approx)


class HJB(PDEwithVtrue):
    name = 'Default_HJB_equation'
    compute_cost_gap = None
    compute_cost_maxit = torch.inf

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dim_u = None

    def comput_cost(self, x0, u_func, num_dt=100):
        t, xt, ut = self.gen_syspath(x0, num_dt, u_func)
        f_val = self.f(t[:-1], xt[:-1], ut[:-1])
        run_cost = f_val.mean([0, 1]) * (self.te - self.t0)
        term_cost = self.v_term(xt[-1]).mean(0)
        cost = run_cost + term_cost
        if self.use_dist is True:
            dist.all_reduce(cost)
            cost = cost / self.world_size
        return cost

    def compute_cost_onx0points(self,
                                x0,
                                u_approx,
                                num_dt=100,
                                num_cost_path=64):
        # x0: [points, x]
        cost_list = []
        for x0i in x0:
            x0i_mc = torch.ones([num_cost_path, 1]) * x0i
            cost_x0i = self.comput_cost(x0i_mc, u_approx, num_dt=num_dt)
            cost_list.append(cost_x0i)
        cost_onpoints = torch.stack(cost_list, dim=0)
        return cost_onpoints

    def x0_for_cost(self, num_points):
        return self.x0_points(num_points)

    def produce_logfunc(self,
                        v_approx,
                        u_approx,
                        num_testpoint=1000,
                        num_cost_path=64):
        t0 = self.t0.unsqueeze(-1)
        x0_cost = self.x0_for_cost(num_cost_path)
        x0_value = self.x0_points(num_testpoint)

        vtrue_onx0 = self.v(t0, x0_value)
        vtrue_onx0cost = self.v(t0, x0_cost).mean()

        vtrue_ontest_l1 = torch.abs(vtrue_onx0).mean()
        vtrue_ontest_linf = torch.abs(vtrue_onx0).max()

        def log_func(it: int, _t: torch.Tensor, _xt: torch.Tensor):
            with torch.no_grad():
                err = v_approx(t0, x0_value) - vtrue_onx0
                abs_err = torch.abs(err)
                l1err = abs_err.mean() / vtrue_ontest_l1
            log = {'rel_l1err': l1err.item()}

            if self.record_linf_error is True:
                linf_err = abs_err.max() / vtrue_ontest_linf
                log['rel_linferr'] = linf_err.item()

            if self.compute_cost_gap is not None:
                if (it % self.compute_cost_gap == 0) and \
                        (it < self.compute_cost_maxit):
                    with torch.no_grad():
                        cost = self.comput_cost(x0_cost, u_approx).item()
                        cost_v0 = cost / vtrue_onx0cost.item()
                    log['cost'] = cost
                    log['vtrue_on_x0cost'] = vtrue_onx0cost.item()
                    log['cost/v0'] = cost_v0
            return log

        return log_func


class EVP(PDEwithVtrue):

    name = 'EigenValueMeta'
    x0_for_train = {'S1': e1_curve}
    # x0_for_train = {'S2': diag_curve}
    x0pil_range = 3.

    musys_depends_on_v = False
    sgmsys_depends_on_v = False
    f_depends_on_v = True

    rate_newlamb = 0.1
    fourier_frequency = None

    def __init__(self, dim_x, **kwargs) -> None:
        super().__init__(dim_x, **kwargs)
        self.te = 10
        self.dim_w = dim_x

        self.relu6 = torch.nn.ReLU6()
        self.softplus = torch.nn.Softplus()

        self.lamb_init = 0.
        self.lamb_val = torch.tensor(self.lamb_init)
        self.lamb_from_v = self.lamb_val

    @abstractmethod
    def set_vnn_forward(self, _vnn):
        pass

    def register_vnn(self, vnn: torch.nn.Module):
        self.vnn = vnn

    def get_lamb(self):
        new_lamb = self.vnn.module.lamb
        r = self.rate_newlamb
        self.lamb_val = (1 - r) * self.lamb_val.detach().item() + r * new_lamb
        return self.lamb_val

    def lamb_v(self, t, vpath):
        v_mean = vpath.mean(1, keepdim=True)
        lamb_batch = torch.log(
            (v_mean[1:] / v_mean[:1]).abs()) / (t[1:] - t[:1])
        lamb_val = lamb_batch.mean()

        r = self.rate_newlamb
        self.lamb_from_v = (
            1 - r) * self.lamb_from_v.detach().item() + r * lamb_val

        lamb_var = (lamb_batch - lamb_val).pow(2).mean()
        return lamb_val, lamb_var

    def f(self, t, _x, v):
        # lamb_val = self.get_lamb()
        with torch.no_grad():
            lamb_val = self.lamb_v(t, v)[0]
        f_val = -lamb_val * v
        return f_val

    def additional_loss(self, t_path, x_path, v_approx):
        # x0 = self.x0_points(100)
        x0 = torch.zeros((1, self.dim_x), device=self.t0.device)
        vappr_val = v_approx(None, x0)
        vtrue_val = self.v(None, x0)
        loss_val = (vtrue_val - vappr_val).mean().pow(2)

        vappr_path = v_approx(t_path, x_path)
        xnorm_mask = (x_path.norm(dim=-1, keepdim=True) > 3.)
        loss_val = loss_val + (xnorm_mask * vappr_path.pow(2)).mean()

        return loss_val

    def x0_points(self, num_points):
        curve_funcs = self.x0_for_train.values()
        num_points = split_number(num_points, len(curve_funcs))
        x0_list = [
            c_func(self.dim_x, nump, self.x0pil_range)[1]
            for (nump, c_func) in zip(num_points, curve_funcs)
        ]
        x0 = torch.concatenate(x0_list, dim=0)
        return x0

    def produce_logfunc(self, v_approx, num_testpoint=1000):
        t0 = self.t0.unsqueeze(-1)
        x_test = self.x0_points(num_testpoint)
        vtrue_ontest = self.v(t0, x_test)
        vtrue_ontest_l1 = torch.abs(vtrue_ontest).mean()
        vtrue_ontest_linf = torch.abs(vtrue_ontest).max()

        def log_func(_it: int, t: torch.Tensor, xt: torch.Tensor):
            v0_approx = v_approx(t0, x_test)
            err = v0_approx - vtrue_ontest
            abs_err = torch.abs(err)
            l1err = abs_err.mean() / vtrue_ontest_l1
            with torch.no_grad():
                vappr_path = v_approx(t, xt)
                lamb_from_v = self.lamb_v(t, vappr_path)[0]

            log = {
                'ev': self.lamb_val.detach().item(),
                'ev_from_v': lamb_from_v.item(),
                'ev_error': (self.true_eigenval - lamb_from_v).abs().item(),
                'rel_l1err': l1err.item(),
            }
            if self.record_linf_error is True:
                linf_err = abs_err.max() / vtrue_ontest_linf
                log['rel_linferr'] = linf_err.item()
            return log

        return log_func
