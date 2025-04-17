# Core components of MartNet
import abc
from abc import abstractmethod
import time
from collections.abc import Callable
from typing import Optional, Sequence, Union, Tuple
import math

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

from exmeta import PDE


def update_paths(path, new_path):
    if new_path is None:
        updated_path = path
    elif path is None:
        updated_path = new_path
    elif new_path.shape[1] >= path.shape[1]:
        updated_path = new_path
    else:
        old_path = path[:, new_path.shape[1]:]
        updated_path = torch.cat((old_path, new_path), dim=1)
    return updated_path


def append_hist(hist: dict, hist_keys, hist_vals):
    if len(hist) == 0:
        hist.update({k: [v] for k, v in zip(hist_keys, hist_vals)})
    else:
        for k, v in zip(hist_keys, hist_vals):
            hist[k].append(v)


class FourierFeatures(nn.Module):

    def __init__(self, min_frequency: int = 1, max_frequency: int = 10):
        super().__init__()
        self.freqs = nn.Parameter(
            1.0 * torch.arange(min_frequency, max_frequency + 1))
        self.num_freqs = self.freqs.shape[0]

    def forward(self, x):
        x_proj = torch.einsum('...i,j->...ij', x, self.freqs)
        sin_val = torch.sin(x_proj).flatten(start_dim=-2, end_dim=-1)
        cos_val = torch.cos(x_proj).flatten(start_dim=-2, end_dim=-1)
        return torch.cat([sin_val, cos_val], dim=-1)


class DNNx(nn.Module):

    def __init__(self,
                 xdims: Sequence,
                 shell_func=None,
                 act_func=nn.ReLU,
                 multi_scale=False,
                 scale_factor=10.,
                 fourier_frequency=None):
        # xdims: Sequence of ints, or [int, int, ..., int, [int, int]].
        # shell_func: (t, x, layer_output) -> dnn_output

        super().__init__()
        dim_x = xdims[0]
        dim_out = xdims[-1]
        self.dim_x = dim_x
        self.dim_out = dim_out
        self.num_hidden = len(xdims) - 2

        if fourier_frequency is not None:
            min_freq, max_freq = fourier_frequency
            ff_layer = FourierFeatures(min_frequency=min_freq,
                                       max_frequency=max_freq)
            outdim_fflayer = 2 * ff_layer.num_freqs * dim_x
            self.xlayer = nn.Sequential(
                ff_layer,
                nn.Linear(outdim_fflayer, xdims[1]),
            )
        else:
            self.xlayer = nn.Linear(dim_x, xdims[1])

        layers = []
        if self.num_hidden > 0:
            layers.append(act_func())
            for i in range(1, self.num_hidden):
                layers.append(nn.Linear(xdims[i], xdims[i + 1]))
                layers.append(act_func())
            if isinstance(dim_out, int):  # if the NN is vector-valued
                layers.append(nn.Linear(xdims[-2], dim_out))
            else:  # if the NN is tensor-valued
                dim_flatout = torch.prod(torch.tensor(dim_out))
                layers.append(nn.Linear(xdims[-2], dim_flatout))
                layers.append(nn.Unflatten(-1, dim_out))
        else:
            layers.append(nn.Identity())

        self.layers = nn.Sequential(*layers)
        if multi_scale is True:
            w = 1. + torch.arange(0, dim_out) / dim_out * scale_factor
            self.scale_layer = lambda z: w * z
        else:
            self.scale_layer = lambda z: z

        if shell_func is None:
            self.shell_func = self.__default_shellfunc
        else:
            self.shell_func = shell_func

    def __default_shellfunc(self, _x, y):
        return y

    def call(self, _t, x):
        ltx = self.scale_layer(self.xlayer(x))
        y = self.shell_func(x, self.layers(ltx))
        return y

    def forward(self, _t, x):
        return self.call(_t, x)

    @property
    def module(self):
        return self


class EigenFuncValue(DNNx):

    def __init__(self,
                 xdims,
                 shell_func=None,
                 act_func=nn.ReLU,
                 multi_scale=False,
                 scale_factor=10.,
                 init_lamb=1.0,
                 fourier_frequency=None):
        super().__init__(xdims,
                         shell_func=shell_func,
                         act_func=act_func,
                         multi_scale=multi_scale,
                         scale_factor=scale_factor,
                         fourier_frequency=fourier_frequency)
        self.lamb = Parameter(torch.tensor(init_lamb))

    def eigenfunc_parameters(self):
        params = [p for n, p in self.named_parameters() if n != "lamb"]
        return params

    def eigenval_parameters(self):
        return [self.lamb]


class DNNtx(DNNx):

    def __init__(self,
                 xdims: list,
                 shell_func=None,
                 act_func=nn.ReLU,
                 multi_scale=False,
                 scale_factor=10.):
        super().__init__(xdims,
                         shell_func=shell_func,
                         act_func=act_func,
                         multi_scale=multi_scale,
                         scale_factor=scale_factor)
        self.tlayer = nn.Linear(1, xdims[1])

    def call(self, t, x):
        tx = self.tlayer(t) + self.xlayer(x)
        ltx = self.layers(tx)
        y = self.shell_func(x, self.scale_layer(ltx))
        return y


class PathSampler(object):

    def __init__(self,
                 pde: PDE,
                 size_per_epoch: int,
                 num_dt: int = 100,
                 ctr_func=None,
                 rank=None,
                 rate_newpath=0.2):
        self.pde = pde
        self.rank = rank
        self.num_dt = num_dt

        self.size_per_epoch = size_per_epoch
        self.epoch = 0.
        self.epoch_finished = 0.
        self.path_idx = torch.randperm(self.size_per_epoch)

        if rate_newpath <= 0.:
            num_newpath = 0
        elif rate_newpath >= 1.0:
            num_newpath = self.size_per_epoch
        else:
            num_newpath = int(self.size_per_epoch * rate_newpath)
            num_newpath = max(1, num_newpath)
        self.num_newpath = num_newpath

        self.dt = (self.pde.te - self.pde.t0) / num_dt
        t_path = torch.linspace(self.pde.t0, self.pde.te, self.num_dt + 1)
        t_path = t_path.unsqueeze(-1).unsqueeze(-1)
        self.t_path = t_path
        self.ctr_func = ctr_func

    def gen_xtpath(self, num_path, ctr_func=None):
        x0 = self.pde.x0_points(num_path)
        x0 = x0[torch.randperm(x0.shape[0])]
        if ctr_func is None:
            _, xt = self.pde.gen_pilpath(x0, self.num_dt)
        else:
            _, xt, _ = self.pde.gen_syspath(x0, self.num_dt, ctr_func)
        # xt: [t0 to tN, path, dim of x]
        return xt

    def gen_sgmsys_offline(self, t_path, xt_path: torch.Tensor):
        dwt_shape = (xt_path.shape[0], xt_path.shape[1], self.pde.dim_w)
        dwt = torch.normal(mean=0.,
                           std=torch.sqrt(self.dt),
                           size=dwt_shape,
                           device=self.rank)
        sgmt_path = self.pde.sgm_sys(t_path, xt_path, None, dwt)
        return sgmt_path

    def gen_musys_offline(self, t_path, xt_path):
        mut_path = self.pde.mu_sys(t_path, xt_path, None) * self.dt
        return mut_path

    def gen_ft_offline(self, t_path, xt_path):
        return self.pde.f(t_path, xt_path, None)

    def gen_xtpil_offlines(self, num_path, ctr_func=None):
        with torch.no_grad():
            xt_pil = self.gen_xtpath(num_path, ctr_func=ctr_func)

        path_print = ['Xt_pilot']
        xtsys_offline = xt_pil[:-1]
        if self.pde.sgmsys_depends_on_v is False:
            sgmt_path = self.gen_sgmsys_offline(self.t_path[:-1], xt_pil[:-1])
            xtsys_offline = xtsys_offline + sgmt_path
            path_print.append('mu_offline')
        if self.pde.musys_depends_on_v is False:
            mut_path = self.gen_musys_offline(self.t_path[:-1], xt_pil[:-1])
            xtsys_offline = xtsys_offline + mut_path
            path_print.append('sigma_offline')

        if self.pde.f_depends_on_v is False:
            ft_offline = self.gen_ft_offline(self.t_path, xt_pil)
        else:
            ft_offline = None
        print(
            f'Rank {self.rank}: new paths are generated for {path_print}, \nnum_newpath={self.num_newpath}/{self.size_per_epoch}'
        )
        return xt_pil, xtsys_offline, ft_offline

    def get_pathbat(self, bat_size):

        if self.epoch == 0.:
            offlines = self.gen_xtpil_offlines(self.size_per_epoch,
                                               ctr_func=None)
            self.xt_pil = offlines[0]
            self.xtsys_offline = offlines[1]
            self.ft_offline = offlines[2]
        elif (self.epoch - self.epoch_finished) >= 1.:
            # If an entire epoch has be finished, then update the paths
            offlines = self.gen_xtpil_offlines(self.num_newpath,
                                               ctr_func=self.ctr_func)
            self.xt_pil = update_paths(self.xt_pil, offlines[0])
            self.xtsys_offline = update_paths(self.xtsys_offline, offlines[1])
            self.ft_offline = update_paths(self.ft_offline, offlines[2])
            self.epoch_finished += 1.

        bat_idx = self.path_idx[:bat_size]
        xt_bat = self.xt_pil[:, bat_idx]
        if self.xtsys_offline is not None:
            xtsysoff_bat = self.xtsys_offline[:, bat_idx]
        else:
            xtsysoff_bat = None
        if self.ft_offline is not None:
            ftoff_bat = self.ft_offline[:, bat_idx]
        else:
            ftoff_bat = None
        self.epoch += min(bat_size / self.size_per_epoch, 1.)
        self.path_idx = self.path_idx.roll(-bat_size)

        # xt_bat, ftoff_bat: [t_0 to t_N, batch, dim of x]
        # xtsysoff_bat: [t_0 to t_{N-1}, batch, dim of self]
        return xt_bat, xtsysoff_bat, ftoff_bat


class MartNet(metaclass=abc.ABCMeta):
    """
    MartNet is an abstract base class for solving partial differential equations (PDEs) using deep neural networks (DNNs).
    Attributes:
        name (str): The name of the class.
        problem (PDE): The PDE problem to be solved.
        nets (Sequence[DNNtx]): A sequence of neural networks used in the solution.
        num_dt (int): The number of time steps.
        use_dist (bool): Flag indicating whether to use distributed training.
        rank (int): The rank of the current process in distributed training.
        world_size (int): The total number of processes in distributed training.
        dt (float): The time step size.
        t_path (torch.Tensor): The time path tensor.
        unn (Optional[DNNtx]): The neural network for 'u' term.
        vnn (DNNtx): The neural network for 'v' term.
        rhonn (DNNtx): The neural network for 'rho' term.
        has_vterm (bool): Flag indicating whether the problem has a 'v' term.
        _rho_val (Optional[torch.Tensor]): Cached value of rho.
        _deltam (Optional[torch.Tensor]): Cached value of delta m.
        mart_loss (Optional[torch.Tensor]): Cached value of martingale loss.
        ctr_loss (Optional[torch.Tensor]): Cached value of control loss.
    Methods:
        __init__(self, problem: PDE, nets: Sequence[DNNtx], num_dt: int = 100, use_dist: bool = False, rank: int = 0) -> None:
            Initializes the MartNet class with the given parameters.
        init_train(self) -> None:
            Initializes the training process by setting the neural networks to training mode.
        finalize_train(self) -> None:
            Finalizes the training process by setting the neural networks to evaluation mode.
        loss_mart(self, deltam: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
            Computes the martingale loss given delta m and rho tensors.
        delta_m(self, xt_pil: torch.Tensor, xtsys_offline: Optional[torch.Tensor] = None, f_offline: Optional[torch.Tensor] = None, compute_ugrad: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            Abstract method to compute delta m. Must be implemented by subclasses.
        loss_ctr(self, deltam: torch.Tensor) -> torch.Tensor:
            Abstract method to compute control loss. Must be implemented by subclasses.
        loss_desc(self, xt_pil: torch.Tensor, xtsys_offline: Optional[torch.Tensor] = None, ft_offline: Optional[torch.Tensor] = None) -> torch.Tensor:
            Computes the total loss for the descent step, including martingale and control losses.
        loss_asc(self, xt_pil: torch.Tensor, xtsys_offline: Optional[torch.Tensor] = None, ft_offline: Optional[torch.Tensor] = None) -> torch.Tensor:
            Computes the total loss for the ascent step, including martingale loss.
    """
    name = 'MartNet'

    def __init__(self,
                 problem: PDE,
                 nets: Sequence[DNNtx],
                 num_dt: int = 100,
                 use_dist: bool = False,
                 rank: int = 0) -> None:
        self.problem = problem
        if use_dist is True:
            self.world_size = dist.get_world_size()
        else:
            self.world_size = 1
        self.use_dist = use_dist
        self.rank = rank

        self.num_dt = num_dt
        self.dt = (self.problem.te - self.problem.t0) / num_dt
        t_path = torch.linspace(self.problem.t0, self.problem.te,
                                self.num_dt + 1)
        self.t_path = t_path.unsqueeze(-1).unsqueeze(-1)

        self.nets = nets
        if len(nets) == 3:
            self.unn, self.vnn, self.rhonn = nets
        elif len(nets) == 2:
            self.vnn, self.rhonn = nets
            self.unn = None
        else:
            raise ValueError('The number of nets should be 2 or 3')

        self.has_vterm = hasattr(self.problem, 'v_term')

    def init_train(self):
        self._rho_val = None
        self._deltam = None
        for net in self.nets:
            net.train()

    def finalize_train(self):
        for net in self.nets:
            net.eval()

    def loss_mart(self, deltam, rho):
        rd = (rho * deltam).mean([0, 1])
        rd_mdt = rd.detach()
        if self.use_dist is True:
            dist.all_reduce(rd_mdt)
            rd_mdt = rd_mdt / self.world_size
        # Correct the bias caused by DDP
        loss_val = (rd_mdt * rd).mean() / 2
        return loss_val

    @abstractmethod
    def delta_m(
        self,
        xt_pil: torch.Tensor,
        xtsys_offline: Optional[torch.Tensor] = None,
        f_offline: Optional[torch.Tensor] = None,
        compute_ugrad: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # xt_pil, xtsys_offline, f_offline: [time, path, dim of x]
        pass

    @abstractmethod
    def loss_ctr(self, deltam: torch.Tensor) -> torch.Tensor:
        # deltam: [time, path, dim of self]
        pass

    def loss_desc(self, xt_pil, xtsys_offline=None, ft_offline=None):
        self._deltam = None
        if self._rho_val is None:
            with torch.no_grad():
                self._rho_val = self.rhonn(self.t_path[:-1], xt_pil[:-1])

        deltam_vgrad, deltam_ugrad = self.delta_m(
            xt_pil,
            xtsys_offline=xtsys_offline,
            f_offline=ft_offline,
            compute_ugrad=True,
        )
        mart_loss = self.loss_mart(deltam_vgrad, self._rho_val)
        ctr_loss = self.loss_ctr(deltam_ugrad)
        loss_tot = mart_loss + ctr_loss
        self.mart_loss = mart_loss.detach()
        self.ctr_loss = ctr_loss.detach()

        if hasattr(self.problem, 'additional_loss'):
            loss_tot = loss_tot + self.problem.additional_loss(
                self.t_path,
                xt_pil,
                self.vnn,
            )
        return loss_tot

    def loss_asc(self, xt_pil, xtsys_offline=None, ft_offline=None):
        self._rho_val = None
        if self._deltam is None:
            with torch.no_grad():
                self._deltam, _ = self.delta_m(xt_pil,
                                               xtsys_offline=xtsys_offline,
                                               f_offline=ft_offline,
                                               compute_ugrad=False)
        rho_bat = self.rhonn(self.t_path[:-1], xt_pil[:-1])
        loss_test = -self.loss_mart(self._deltam, rho_bat)
        return loss_test

    # def loss_mart(self, deltam: torch.Tensor,
    #               rho: torch.Tensor) -> torch.Tensor:
    #     # deltam, rho: [time, path, dim of self]
    #     # the number of path must be even
    #     rdm = (rho * deltam).unflatten(1, (-1, 2)).mean([0, 1])
    #     rdm_det = rdm.detach()
    #     if self.use_dist is True:
    #         dist.all_reduce(rdm_det)
    #         rdm_det = rdm_det / self.world_size

    #     # Correct the bias caused by DDP and mini-batch sampling
    #     loss_val = (rdm_det[0] * rdm[1] + rdm_det[1] * rdm[0]).mean()
    #     return loss_val / 4


class DfSocMartNet(MartNet):
    name = 'Df-SocMartNet'

    def _xt_next(self, t: torch.Tensor, x: torch.Tensor, ctr: torch.Tensor):
        x_next = x
        if self.problem.musys_depends_on_v is True:
            x_next = x_next + self.problem.mu_sys(t, x, ctr) * self.dt
        if self.problem.sgmsys_depends_on_v is True:
            dwt_shape = x.shape[:-1] + (self.problem.dim_w, )
            dwt = torch.normal(mean=0.,
                               std=torch.sqrt(self.dt),
                               size=dwt_shape,
                               device=self.rank)
            x_next = x_next + self.problem.sgm_sys(t, x, ctr, dwt)
        return x_next

    def _v_next(self, t_next, x_next):
        if self.has_vterm is True:
            v_next0 = self.vnn(t_next[:-1], x_next[:-1])
            v_te = self.problem.v_term(x_next[-1:])
            v_next = torch.cat((v_next0, v_te), dim=0)
        else:
            v_next = self.vnn(t_next, x_next)
        return v_next

    def delta_m(self,
                xt_pil,
                xtsys_offline=None,
                f_offline=None,
                compute_ugrad=True):
        t = self.t_path[:-1]
        xt = xt_pil[:-1]
        ut = self.unn(t, xt)
        vt = self.vnn(t, xt)

        if xtsys_offline is None:
            xtsys_offline = xt_pil
        if self.problem.f_depends_on_v is True:
            f_path = self.problem.f(t, xt, ut)
        else:
            f_path = f_offline[:-1]

        x_next = self._xt_next(t, xtsys_offline, ut)
        t_next = self.t_path[1:]
        vnext_ud = self._v_next(t_next, x_next.detach())
        deltam_vgrad = (vnext_ud - vt) / self.dt + f_path.detach()

        if compute_ugrad is True:
            with torch.no_grad():
                vnext_vd = self._v_next(t_next, x_next)
            deltam_ugrad = (vnext_vd - vt.detach()) / self.dt + f_path
        else:
            deltam_ugrad = None
        return deltam_vgrad, deltam_ugrad

    def loss_ctr(self, deltam):
        loss = deltam.mean()
        return loss


class SocMartNet(MartNet):
    name = 'SocMartNet'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.problem.sgmsys_depends_on_v is False, \
            'Currently, SocMartNet is only for HJB equation with sgmsys_depends_on_v == False'

    def delta_m(self,
                xt_pil,
                xtsys_offline=None,
                f_offline=None,
                compute_ugrad=True):
        # xtsys_offline is not used in this class

        ut = self.unn(self.t_path, xt_pil)
        if self.problem.f_depends_on_v is True:
            f_path = self.problem.f(self.t_path, xt_pil, ut)
        else:
            f_path = f_offline
        fpath_mean = (f_path[1:] + f_path[:-1]) / 2

        t = self.t_path[:-1]
        xt = xt_pil[:-1]
        if self.has_vterm is True:
            vt = self.vnn(t, xt)
            v_te = self.problem.v_term(xt_pil[-1:])
            v_path = torch.cat((vt, v_te), dim=0)
        else:
            v_path = self.vnn(self.t_path, xt_pil)
            vt = v_path[:-1]
        deltav = (v_path[1:] - vt) / self.dt

        # The following approximates (mu_sys - mu_pil)^{\top} v_x using finite differences instead of automatic differentiation.
        # This approach is adopted because torch.autograd.grad() is incompatible with torch.nn.parallel.DistributedDataParallel (PyTorch 2.6).
        # See, https://pytorch.org/docs/2.6/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel
        sz_diff = self.dt**2 # The size of the finite difference
        musys_val = self.problem.mu_sys(t, xt, ut[:-1])
        mupil_val = self.problem.mu_pil(t, xt)
        delta_mu = musys_val - mupil_val
        x_forw = xt + delta_mu * sz_diff
        vforw_vgrad = self.vnn(t, x_forw.detach())
        deltamu_vdx = (vforw_vgrad - vt) / sz_diff

        deltam_vgrad = deltav + deltamu_vdx + fpath_mean.detach()
        if compute_ugrad is True:
            with torch.no_grad():
                vforw_ugrad = self.vnn(t, x_forw)
            deltam_ugrad = (vforw_ugrad - vt.detach()) / sz_diff + fpath_mean
        else:
            deltam_ugrad = None
        return deltam_vgrad, deltam_ugrad

    def loss_ctr(self, deltam):
        loss = deltam.mean()
        return loss


class QuasiMartNet(SocMartNet):
    name = 'QuasiMartNet'

    def loss_ctr(self, _):
        return torch.tensor(0.)

    def delta_m(self,
                xt_pil,
                f_offline=None,
                xtsys_offline=None,
                compute_ugrad=False):
        # xtsys_offline and compute_ugrad are not used in this class

        ut = self.vnn(self.t_path, xt_pil)
        if self.problem.f_depends_on_v is True:
            f_path = self.problem.f(self.t_path, xt_pil, ut)
        else:
            f_path = f_offline
        fpath_mean = (f_path[1:] + f_path[:-1]) / 2

        t = self.t_path[:-1]
        xt = xt_pil[:-1]
        if self.has_vterm is True:
            vt = self.vnn(t, xt)
            v_te = self.problem.v_term(xt_pil[-1:])
            v_path = torch.cat((vt, v_te), dim=0)
        else:
            v_path = self.vnn(self.t_path, xt_pil)
            vt = v_path[:-1]
        deltav = (v_path[1:] - vt) / self.dt

        # Approximate (mu_sys - mu_pil)^{\top} v_x by finite difference
        sz_diff = self.dt**2
        musys_val = self.problem.mu_sys(t, xt, ut[:-1])
        mupil_val = self.problem.mu_pil(t, xt)
        delta_mu = musys_val - mupil_val
        x_forw = xt + delta_mu * sz_diff
        vforw_vgrad = self.vnn(t, x_forw)
        deltamu_vdx = (vforw_vgrad - vt) / sz_diff

        deltam_vgrad = deltav + deltamu_vdx + fpath_mean
        return deltam_vgrad, None


class DfQuasiMartNet(DfSocMartNet):
    name = 'Df-QuasiMartNet'

    def loss_ctr(self, _):
        return torch.tensor(0.)

    def delta_m(self,
                xt_pil,
                xtsys_offline=None,
                f_offline=None,
                compute_ugrad=None):
        # compute_ugrad is not used in this class

        t = self.t_path[:-1]
        xt = xt_pil[:-1]
        vt = self.vnn(t, xt)
        if self.problem.f_depends_on_v is True:
            f_path = self.problem.f(t, xt, vt)
        else:
            f_path = f_offline[:-1]

        if xtsys_offline is None:
            xtsys_offline = xt_pil
        x_next = self._xt_next(t, xtsys_offline, vt)
        t_next = self.t_path[1:]
        vnext_val = self._v_next(t_next, x_next)
        deltam_val = (vnext_val - vt) / self.dt + f_path
        return deltam_val, None


class DfEvMartNet(DfSocMartNet):
    name = 'Df-EvMartNet'

    def init_train(self):
        super().init_train()
        self.problem.register_vnn(self.vnn)

    def loss_ctr(self, _):
        return self.problem.lamb_val.detach()

    def delta_m(self,
                xt_pil,
                xtsys_offline=None,
                f_offline=None,
                compute_ugrad=None):
        # f_offline and compute_ugrad are not used in this class

        t = self.t_path[:-1]
        xt = xt_pil[:-1]
        vt = self.vnn(t, xt)

        if xtsys_offline is None:
            xtsys_offline = xt_pil
        x_next = self._xt_next(t, xtsys_offline, vt)
        t_next = self.t_path[1:]
        # with torch.no_grad():
        vnext = self._v_next(t_next, x_next)
        lv = (vnext - vt) / self.dt
        if hasattr(self.problem, 'update_eigenvalue'):
            self.problem.update_eigenvalue(xt, lv, vt)
        delta_mart = lv + self.problem.f(t, xt, vt)
        if hasattr(self.problem, 'shell_deltamart'):
            delta_mart = self.problem.shell_deltamart(t, xt, vt, delta_mart)

        # delta_mart = vt.detach() * delta_mart / vt.detach().mean()
        # print(f"lambda = {lamb.item()}")
        return delta_mart, None

    # def loss_mart(self, deltam, rho):
    #     rd = (rho * deltam).mean([0, 1])
    #     rd_mdt = rd.detach()
    #     if self.use_dist is True:
    #         dist.all_reduce(rd_mdt)
    #         rd_mdt = rd_mdt / self.world_size
    #     # Correct the bias caused by DDP
    #     loss_val = (rd_mdt * rd).mean() / 2
    #     return loss_val

    def loss_mart(self, deltam: torch.Tensor,
                  rho: torch.Tensor) -> torch.Tensor:
        # deltam, rho: [time, path, dim of self]
        # the number of path must be even
        rdm = (rho * deltam).unflatten(1, (-1, 2)).mean([0, 1])
        rdm_det = rdm.detach()
        if self.use_dist is True:
            dist.all_reduce(rdm_det)
            rdm_det = rdm_det / self.world_size

        # Correct the bias caused by DDP and mini-batch sampling
        loss_val = (rdm_det[0] * rdm[1] + rdm_det[1] * rdm[0]).mean()
        return loss_val / 4


def train_martnet(
    martnet: MartNet,
    pathsamp_func: Callable[[int], tuple],
    optim_desc: Union[Optimizer, Sequence[Optimizer]],
    optim_asc: Optimizer,
    batsize: int = 64,
    max_iter: int = 1000,
    step_desc: int = 2,
    step_asc: int = 1,
    schs: Optional[Sequence[LRScheduler]] = None,
    log_func: Optional[Callable[[int], dict]] = None,
    ip_time_gap: float = 0.,
) -> dict:

    schs = () if schs is None else schs
    if isinstance(optim_desc, Optimizer):
        optim_desc = (optim_desc, )
    log_func = (lambda _it, _t, _x: {}) if log_func is None else log_func
    log_keys = ['it', 'rt', 'mart_loss', 'g1_loss']
    hist_dict = {}

    # Ensure that the batch size is even
    batsize = math.ceil(batsize / 2) * 2

    rt0 = time.time()
    martnet.init_train()
    time_print = rt0
    for it in range(max_iter + 1):
        xt_pil, xtsys_offline, ft_offline = pathsamp_func(batsize)
        for _ in range(step_desc):
            loss_desc = martnet.loss_desc(
                xt_pil,
                xtsys_offline=xtsys_offline,
                ft_offline=ft_offline,
            )
            loss_desc.backward()
            for opt in optim_desc:
                opt.step()
                opt.zero_grad()

        for _ in range(step_asc):
            loss_asc = martnet.loss_asc(
                xt_pil,
                xtsys_offline=xtsys_offline,
                ft_offline=ft_offline,
            )
            loss_asc.backward()
            optim_asc.step()
            optim_asc.zero_grad()
        for sch in schs:
            sch.step()
        rt = time.time() - rt0

        with torch.no_grad():
            newlog_dict = log_func(it, martnet.t_path, xt_pil)
        mart_loss = martnet.mart_loss.abs().item()
        ctr_loss = martnet.ctr_loss.item()

        log_vals = [it, rt, mart_loss, ctr_loss]
        new_histkeys = log_keys + list(newlog_dict.keys())
        new_histvals = log_vals + list(newlog_dict.values())
        append_hist(hist_dict, new_histkeys, new_histvals)

        if (time.time() - time_print >= ip_time_gap) or (it in (0, max_iter)):
            lr = optim_desc[0].param_groups[0]['lr']
            pr_str = f"rank: {martnet.rank}\niter step: [{it}/{max_iter}], rt: {rt:.2f}, lr: {lr:.5}, \nmart_loss: {mart_loss:.5}, ctr_loss: {ctr_loss:.5}\n"
            pr_str += "\n".join([f'{k}: {v}' for k, v in newlog_dict.items()])
            pr_str += '\n'
            print(pr_str)
            time_print = time.time()

    martnet.finalize_train()
    return hist_dict
