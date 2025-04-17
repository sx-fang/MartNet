# Some examples of PDE and SOCP
import gc
import types
import math
import psutil
import torch
from exmeta import PDE, HJB, PDEwithVtrue, origin_point, diag_curve, e1_curve, EVP


def free_cache(device: torch.device):
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'xpu':
        torch.xpu.empty_cache()
    elif device.type == 'cpu':
        gc.collect()
    else:
        raise ValueError('The device type is not supported.')


def get_safe_chunksize(num_item_persize: int,
                       dtype: torch.dtype,
                       device: torch.device,
                       use_percent=0.5):
    if device.type == 'cuda':
        avail_mem = torch.cuda.mem_get_info(device)[0]
    elif device.type == 'xpu':
        avail_mem = torch.xpu.mem_get_info(device)[0]
    elif device.type == 'cpu':
        avail_mem = psutil.virtual_memory().available
    else:
        raise ValueError('The device type is not supported.')

    safe_chunksize = avail_mem * use_percent // (num_item_persize *
                                                 dtype.itemsize)
    safe_chunksize = int(safe_chunksize)
    if safe_chunksize == 0:
        raise MemoryError(
            f'The memory of {device} is not enough to get safe_chunksize >= 1.'
        )
    return safe_chunksize


class HJB0(HJB):
    name = 'HJB_Example0'
    musys_depends_on_v = True
    sgmsys_depends_on_v = False
    f_depends_on_v = True
    nsamp_mc = 10**6

    def __init__(self, dim_x, **kwargs) -> None:
        super().__init__(dim_x, **kwargs)
        self.dim_w = dim_x
        self.dim_u = dim_x

        self.b_val = torch.tensor(1.)
        self.delta0 = torch.tensor(1 / 4)

    def mu_sys(self, _t, x, k):
        drift = self.b_val
        return drift + 2 * k

    def sgm_sys(self, t, _x, _k, dw):
        c_sgm = self.delta0 * torch.tensor(2.).pow(0.5)
        return c_sgm * dw

    def mu_pil(self, _t, x):
        return torch.full_like(x, self.b_val)

    def sgm_pil(self, t, x, dw):
        return self.sgm_sys(t, x, None, dw)

    def g(self, x):
        x = x - self.b_val
        return torch.log(0.5 * (1 + (x**2).sum(-1, keepdim=True)))

    def v_term(self, x):
        return self.g(x)

    def f(self, _t, _x, k):
        return self.delta0.pow(-2) * k.pow(2).sum(-1, keepdims=True)

    def v(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # x: [time step, batch, x], or [batch, x]
        # the shape of t should admit the validity of t + x
        assert (t.ndim == 1) or (t.ndim == x.ndim)
        assert x.ndim >= 2

        c_sgm = self.delta0 * torch.tensor(2.).pow(0.5)
        std_dw = c_sgm * (self.te - t).pow(0.5)
        x_bdt = (x + self.b_val * (self.te - t)).unsqueeze(-2)

        nsamp_mc = self.nsamp_mc
        multiplier = self.dim_w * (1 + std_dw.numel() + x_bdt.numel())
        chunksize = min(
            get_safe_chunksize(multiplier, x.dtype, x.device),
            nsamp_mc,
        )
        cum_size = 0
        cum_mean = 0.
        print(f"Monte-Carlo for Ref. solution on {x.device}...")
        while cum_size < nsamp_mc:
            try:
                chunksize = min(chunksize, nsamp_mc - cum_size)
                rand_mc = torch.normal(
                    mean=0.,
                    std=1.,
                    size=(chunksize, self.dim_w),
                    device=x.device,
                )
                # dw_tte: [axis of t[..., 0], MC samples, dim_w]
                dw_tte = torch.einsum('...j, ij -> ...ij', std_dw, rand_mc)
                del rand_mc
                new_mean = torch.exp(-self.g(x_bdt + dw_tte)).mean(-2)
                del dw_tte
                cum_size += chunksize
                new_rate = chunksize / cum_size
                cum_mean = (1 - new_rate) * cum_mean + new_rate * new_mean
                del new_mean
            except RuntimeError as err:
                if 'out of memory' in str(err):
                    free_cache(x.device)
                    chunksize = int(chunksize // 2)
                    print(
                        f"Restricted by memory of {x.device}, reduce chunksize to {chunksize}"
                    )
                    if chunksize == 0:
                        print('Memory Error: Cannot reduce chunksize=0')
                        raise err
                else:
                    raise err

        return -torch.log(cum_mean)


class HJB0a(HJB0):
    name = 'HJB_Example0a'

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.delta0 = torch.tensor(0.5)


class HJB0b(HJB0):
    name = 'HJB_Example0b'

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.delta0 = torch.tensor(0.1)


class HJB0c(HJB0):
    name = 'HJB_Example0c'

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.delta0 = torch.tensor(0.05)


class HJB1(HJB0):
    '''
    This example is modified from Section 3.1 of
        Bachouch, A., Huré, C., Langrené, N. et al. Deep Neural Networks Algorithms for Stochastic Control Problems on Finite Horizon: Numerical Applications. Methodol Comput Appl Probab 24, 143-178 (2022).
    and Section 4.3 of
        Weinan E, Han J, Jentzen A (2017) Deep learning-based numerical methods for high-dimensional parabolic partial differential equations and backward stochastic differential equations.
        In: Communications in mathematics and statistics 5, vol 5, pp 349-380.
    '''
    name = 'HJB_Example1'

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.b_val = torch.tensor(0.)
        self.delta0 = torch.tensor(1.)


class HJB1ShiftTarget(HJB0):
    """
    This example is inspired by section 4.2.3 of 
    @article {MR4793480,
    AUTHOR = {Li, Xingjian and Verma, Deepanshu and Ruthotto, Lars},
     TITLE = {A neural network approach for stochastic optimal control},
   JOURNAL = {SIAM J. Sci. Comput.},
  FJOURNAL = {SIAM Journal on Scientific Computing},
    VOLUME = {46},
      YEAR = {2024},
    NUMBER = {5},
     PAGES = {C535--C556},
      ISSN = {1064-8275,1095-7197},
   MRCLASS = {65M70 (35F21 49K45 49L20 68T07 93-08 93E20)},
  MRNUMBER = {4793480},
       DOI = {10.1137/23M155832X},
       URL = {https://doi.org/10.1137/23M155832X},
}
    """
    name = 'HJB_ShiftTarget'
    x0_train = {'diag': diag_curve}
    coeff_g = 10.
    compute_cost_gap = 1
    compute_cost_maxit = 300

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._target = torch.full((self.dim_x, ), 3.)
        self.b_val = torch.tensor(0.)
        self.delta0 = torch.tensor(0.1)

    def x0_for_cost(self, num_points):
        return torch.zeros((num_points, self.dim_x))

    def g(self, x):
        x = x - self.b_val - self._target
        g_inner = torch.log(0.5 * (1 + (x**2).sum(-1, keepdim=True)))
        return self.coeff_g * g_inner


class HJB2(HJB0):
    name = 'HJB_Example2'

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.b_val = torch.tensor(1.)
        self.delta0 = torch.tensor(0.1)
        self.delcpi = 0.3 / torch.pi

    def g(self, x):
        x = x - self.b_val
        return (torch.sin(x - torch.pi / 2) +
                torch.sin(1 / (self.delcpi + x.pow(2)))).mean(-1, keepdim=True)


class HJB2a(HJB2):
    name = 'HJB_Example2a'

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.delta0 = torch.tensor(0.5)


class HJB2b(HJB2):
    name = 'HJB_Example2b'

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.delta0 = torch.tensor(0.2)


class HJB2c(HJB2):
    name = 'HJB_Example2c'

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.delta0 = torch.tensor(0.1)


class HJB1Onep(HJB1):
    name = 'HJB_Example1Onep'
    x0_train = {'x0': origin_point}


class HJB0b1p(HJB0b):
    name = 'HJB_Example0b1p'
    x0_train = {'x0': origin_point}


class HJB0c1p(HJB0c):
    name = 'HJB_Example0c1p'
    x0_train = {'x0': origin_point}


class HJB2b1p(HJB2b):
    name = 'HJB_Example2b1p'
    x0_train = {'x0': origin_point}


class HJB2c1p(HJB2c):
    name = 'HJB_Example2c1p'
    x0_train = {'x0': origin_point}


class HJB2bPba(HJB2b):
    c_purb = 2.0
    name = 'HJB_Example2bPba'
    x0_train = {'diag': diag_curve}

    def f(self, _t, _x, k):
        f_val = self.delta0.pow(-2) * k.pow(2).sum(-1, keepdims=True)
        purb = self.c_purb * torch.sin(k.sum(-1, keepdims=True))
        return f_val + purb


class HJB2bPbb(HJB2bPba):
    c_purb = 1.0
    name = 'HJB_Example2bPbb'


class HJB2bPbc(HJB2bPba):
    c_purb = 1 / 2
    name = 'HJB_Example2bPbc'


class HJB2bPbd(HJB2bPba):
    c_purb = 1 / 4
    name = 'HJB_Example2bPbd'


class HJB2bPbe(HJB2bPba):
    c_purb = 1 / 8
    name = 'HJB_Example2bPbe'


class HJB2bPbf(HJB2bPba):
    c_purb = 1 / 18
    name = 'HJB_Example2bPbf'


class HJB2bPbg(HJB2bPba):
    c_purb = 1 / 32
    name = 'HJB_Example2bPbg'


class Counter(PDEwithVtrue):
    name = 'Counter_Example'
    x0_train = {'1stcoord': e1_curve, 'diag': diag_curve}

    musys_depends_on_v = False
    sgmsys_depends_on_v = False
    f_depends_on_v = False

    def __init__(self, dim_x, **kwargs) -> None:
        super().__init__(dim_x, **kwargs)
        self.dim_w = dim_x

    def mu_pil(self, _t, x):
        return torch.zeros_like(x)

    def sgm_pil(self, _t, x, dw):
        return torch.ones_like(x) * dw

    def mu_sys(self, _t, x, _v):
        return torch.zeros_like(x)

    def sgm_sys(self, _t, x, _v, dw):
        return torch.ones_like(x) * dw

    def v_term(self, x):
        return 1. + torch.sin(self.te + x).mean(-1, keepdim=True)

    def f(self, t, x, _v):
        dt_u = torch.cos(t + x).mean(-1, keepdim=True)
        dxx_u = -torch.sin(t + x).mean(-1, keepdim=True)
        return -dt_u - 0.5 * dxx_u

    def v(self, t, x):
        return 1. + torch.sin(t + x).mean(-1, keepdim=True)


class AllenCahnSin(PDEwithVtrue):
    name = 'AllenCahn_Sinx'
    x0_train = {'diag': diag_curve}

    musys_depends_on_v = False
    sgmsys_depends_on_v = False
    f_depends_on_v = True
    c_in_sgm = 5.

    def __init__(self, dim_x, **kwargs) -> None:
        super().__init__(dim_x, **kwargs)
        self.dim_w = dim_x

    def mu_pil(self, _t, x):
        mu = torch.sin(2 * x)
        return mu

    def sgm_pil(self, t, x, dw):
        sgm = 1. + 0.5 * torch.sin(self.c_in_sgm * t + x)
        return sgm * dw

    def mu_sys(self, _t, x, _v):
        mu = torch.sin(2 * x)
        return mu

    def sgm_sys(self, t, x, _v, dw):
        sgm = 1. + 0.5 * torch.sin(self.c_in_sgm * t + x)
        return sgm * dw

    def v_term(self, x):
        return 1. + torch.sin(self.te + x).mean(-1, keepdim=True)

    def f(self, t, x, v):
        sintx = torch.sin(t + x)
        costx = torch.cos(t + x)
        dt_v = costx.mean(-1, keepdim=True)

        mu = torch.sin(2 * x)
        mu_vx = (mu * costx).mean(-1, keepdim=True)

        sgm_val = 1. + 0.5 * torch.sin(self.c_in_sgm * t + x)
        tr_sgm_vxx = -(sgm_val.pow(2) * sintx).mean(-1, keepdim=True)

        v_v3 = v - v.pow(3)
        v_exact = 1 + sintx.mean(-1, keepdim=True)
        v_v3_exact = v_exact - v_exact.pow(3)
        return -dt_v - mu_vx - 0.5 * tr_sgm_vxx - v_v3_exact + v_v3

    def v(self, t, x):
        return 1. + torch.sin(t + x).mean(-1, keepdim=True)


class LinearSin(PDEwithVtrue):
    name = 'Linear_Sinx'
    x0_train = {'diag': diag_curve}

    musys_depends_on_v = False
    sgmsys_depends_on_v = False
    f_depends_on_v = False
    c_in_sgm = 5.

    def __init__(self, dim_x, **kwargs) -> None:
        super().__init__(dim_x, **kwargs)
        self.dim_w = dim_x

    def mu_pil(self, _t, x):
        mu = torch.sin(2 * x)
        return mu

    def sgm_pil(self, t, x, dw):
        sgm = 1. + 0.5 * torch.sin(self.c_in_sgm * t + x)
        return sgm * dw

    def mu_sys(self, _t, x, _v):
        mu = torch.sin(2 * x)
        return mu

    def sgm_sys(self, t, x, _v, dw):
        sgm = 1. + 0.5 * torch.sin(self.c_in_sgm * t + x)
        return sgm * dw

    def v_term(self, x):
        return 1. + torch.sin(self.te + x).mean(-1, keepdim=True)

    def f(self, t, x, _v):
        sintx = torch.sin(t + x)
        costx = torch.cos(t + x)
        dt_v = costx.mean(-1, keepdim=True)

        mu = torch.sin(2 * x)
        mu_vx = (mu * costx).mean(-1, keepdim=True)

        sgm_val = 1. + 0.5 * torch.sin(self.c_in_sgm * t + x)
        tr_sgm_vxx = -(sgm_val.pow(2) * sintx).mean(-1, keepdim=True)
        return -dt_v - mu_vx - 0.5 * tr_sgm_vxx

    def v(self, t, x):
        return 1. + torch.sin(t + x).mean(-1, keepdim=True)


class AllenCahn(PDE):
    '''
    This example is taken from Eq. [15] in
        J. Han, A. Jentzen, W. E, Solving high-dimensional partial differential equations using deep learning, Proc. Natl. Acad. Sci. U.S.A. 115 (34) 8505-8510, (2018).
    '''
    name = 'Allen-Cahn-equation'
    musys_depends_on_v = False
    sgmsys_depends_on_v = False
    f_depends_on_v = True

    def __init__(self, dim_x, t0=0., te=0.3, use_dist=False) -> None:
        super().__init__(dim_x, t0=t0, te=te, use_dist=use_dist)
        self.dim_w = dim_x

        if dim_x != 100:
            raise ValueError("Only dim_x = 100 is available for this example.")
        if t0 != 0.:
            raise ValueError("Only t_0 = 0 is available for this example.")

    def mu_pil(self, _t, x):
        return torch.zeros_like(x)

    def sgm_pil(self, _t, x, dw):
        return torch.full_like(x, 2**(0.5)) * dw

    def mu_sys(self, t, x, _v):
        return self.mu_pil(t, x)

    def sgm_sys(self, t, x, _v, dw):
        return self.sgm_pil(t, x, dw)

    def v_term(self, x):
        y = 1 / (2 + 0.4 * x.pow(2).sum(-1, keepdims=True))
        return y

    def f(self, _t, _x, v):
        return v - v.pow(3)

    def x0_points(self, num_points):
        x0 = torch.zeros((num_points, self.dim_x))
        return x0

    def produce_logfunc(self, v_approx, num_testpoint=None):
        v_val = torch.tensor((0.0528, ))
        x_test = torch.zeros((self.dim_x, ))
        v_val_l1 = torch.abs(v_val).mean()

        def log_func(_it: int, _t: torch.Tensor, _xt: torch.Tensor):
            vpred = v_approx(self.t0.unsqueeze(-1), x_test)
            err = vpred - v_val
            l1err = torch.abs(err).mean() / v_val_l1
            log = {
                'rel_l1err': l1err.item(),
                'vpred_x0': vpred.item(),
                'vtrue_x0': v_val.item(),
            }
            return log

        return log_func

    def produce_results(self, *args, **kwargs):
        pass


class BSE(PDE):
    '''
    This example is taken from Eq. [11] in
        J. Han, A. Jentzen, W. E, Solving high-dimensional partial differential equations using deep learning, Proc. Natl. Acad. Sci. U.S.A. 115 (34) 8505-8510, (2018).
    '''
    name = 'Black-Scholes-equation'
    musys_depends_on_v = False
    sgmsys_depends_on_v = False
    f_depends_on_v = True

    def __init__(self, dim_x, t0=0., **kwargs) -> None:
        super().__init__(dim_x, t0=t0, **kwargs)
        self.dim_w = dim_x

        if t0 != 0.:
            raise ValueError("Only t_0 = 0 is available for this example.")

        self.bar_mu = 0.02
        self.bar_sgm = 0.2
        self.delta = 2 / 3
        self.r = 0.02

        if dim_x == 100:
            self.vh = 50.
            self.vl = 70.
            self.v0 = torch.tensor((60.781, ))
        elif dim_x == 1:
            self.vh = 50.
            self.vl = 120.
            self.v0 = torch.tensor((97.705, ))
        else:
            assert dim_x in (1, 100)

        self.relu6 = torch.nn.ReLU6()

    def mu_pil(self, _t, x):
        return self.bar_mu * x

    def sgm_pil(self, _t, x, dw):
        return self.bar_sgm * x * dw

    def mu_sys(self, t, x, _v):
        return self.mu_pil(t, x)

    def sgm_sys(self, t, x, _v, dw):
        return self.sgm_pil(t, x, dw)

    def v_term(self, x):
        y = x.min(-1, keepdims=True)[0]
        return y

    def q_func(self, v):
        relu6_val = self.relu6((v - self.vh) / (self.vl - self.vh) * 6.)
        q = 0.2 + (0.02 - 0.2) * relu6_val / 6.
        return q

    def f(self, _t, _x, v):
        f_val = -(1 - self.delta) * self.q_func(v) * v - self.r * v
        return f_val

    def x0_points(self, num_points):
        x0 = torch.full((num_points, self.dim_x), 100.)
        return x0

    def produce_logfunc(self, v_approx, num_testpoint=None):
        v_val = self.v0
        x_test = torch.full((self.dim_x, ), 100.)
        v_val_l1 = torch.abs(v_val).mean()

        def log_func(_it: int, _t: torch.Tensor, _xt: torch.Tensor):
            vpred = v_approx(self.t0.unsqueeze(-1), x_test)
            err = vpred - v_val
            l1err = torch.abs(err).mean() / v_val_l1
            log = {
                'rel_l1err': l1err.item(),
                'vpred_x0': vpred.item(),
                'vtrue_x0': v_val.item(),
            }
            return log

        return log_func

    def produce_results(self, *args, **kwargs):
        pass


class AllenCahnSDGD(PDEwithVtrue):
    '''
    This example is modified from 
    title = {Tackling the curse of dimensionality with physics-informed neural networks},
    journal = {Neural Networks},
    author = {Zheyuan Hu and Khemraj Shukla and George Em Karniadakis and Kenji Kawaguchi},
    volume = {176},
    pages = {106369},
    year = {2024},
    doi = {https://doi.org/10.1016/j.neunet.2024.106369}.
    '''
    name = 'Allen-Cahn-SDGD'
    musys_depends_on_v = False
    sgmsys_depends_on_v = False
    f_depends_on_v = True
    x0pil_scale = 1.5
    x0_for_train = {'S2': diag_curve}

    def __init__(self, dim_x, **kwargs) -> None:
        super().__init__(dim_x, **kwargs)
        self.dim_w = dim_x

        xi = torch.linspace(-torch.pi / 2, torch.pi / 2, dim_x)
        self.ci = (1.5 + torch.sin(xi)) / dim_x

    @staticmethod
    def x_shift(t, x):
        return t + x - torch.tensor(0.5)

    def mu_pil(self, _t, x):
        return torch.zeros_like(x)

    def sgm_pil(self, _t, x, dw):
        return torch.full_like(x, 2**0.5) * dw

    def mu_sys(self, _t, x, _v):
        return torch.zeros_like(x)

    def sgm_sys(self, _t, x, _v, dw):
        return torch.full_like(x, 2**0.5) * dw

    def v(self, t, x):
        x = self.x_shift(t, x)
        cosx = torch.cos(x)
        cosx_r1 = torch.roll(cosx, -1, dims=-1)
        x_r1 = torch.roll(x, -1, dims=-1)
        s = torch.sin(x + cosx_r1 + x_r1 * cosx)
        return (self.ci * s).sum(-1, keepdims=True)

    def v_term(self, x):
        return self.v(self.te, x)

    def d_v(self, t, x):
        x = self.x_shift(t, x)
        sinx = torch.sin(x)
        cosx = torch.cos(x)
        sinx_r1 = torch.roll(sinx, -1, dims=-1)
        cosx_r1 = torch.roll(cosx, -1, dims=-1)
        x_r1 = torch.roll(x, -1, dims=-1)
        x_inner = x + cosx_r1 + x_r1 * cosx

        sin_xinn = torch.sin(x_inner)
        cos_xinn = torch.cos(x_inner)

        dx_xinn1 = 1. - x_r1 * sinx
        dx_xinn2 = cosx - sinx_r1
        dx_xinn = dx_xinn1 + dx_xinn2
        sumdx_v = (self.ci * cos_xinn * dx_xinn).sum(-1, keepdims=True)

        dxx_xinn1 = -dx_xinn1.pow(2) - dx_xinn2.pow(2)
        dxx_xinn2 = -x_r1 * cosx - cosx_r1

        sxinn_cxinn = sin_xinn * dxx_xinn1 + cos_xinn * dxx_xinn2
        sumdxx_v = (self.ci * sxinn_cxinn).sum(-1, keepdims=True)

        d_v = sumdx_v + sumdxx_v
        v = (self.ci * sin_xinn).sum(-1, keepdims=True)
        return d_v, v

    def f(self, t, x, v):
        dv_val, v_true = self.d_v(t, x)
        return v - v.pow(3) - dv_val - v_true + v_true.pow(3)


class AllenCahnSDGDXrad0d5(AllenCahnSDGD):
    x0_scale = 0.5
    name = f'Allen-Cahn-SDGD-Xrad{x0_scale}'
    x0pil_scale = x0_scale


class AllenCahnSDGDXrad0(AllenCahnSDGDXrad0d5):
    x0_scale = 0.
    name = f'Allen-Cahn-SDGD-Xrad{x0_scale}'

    @staticmethod
    def x_shift(t, x):
        return t + x - torch.tensor(1.0)


class AllenCahnSDGDXrad1d0(AllenCahnSDGDXrad0d5):
    x0_scale = 1.0
    name = f'Allen-Cahn-SDGD-Xrad{x0_scale}'


class AllenCahnSDGDXrad1d5(AllenCahnSDGDXrad0d5):
    x0_scale = 1.5
    name = f'Allen-Cahn-SDGD-Xrad{x0_scale}'


class AllenCahnSDGDXrad2d0(AllenCahnSDGDXrad0d5):
    x0_scale = 2.0
    name = f'Allen-Cahn-SDGD-Xrad{x0_scale}'


class AllenCahnSDGDXrad2d5(AllenCahnSDGDXrad0d5):
    x0_scale = 2.5
    name = f'Allen-Cahn-SDGD-Xrad{x0_scale}'


class AllenCahnSDGDVlap(PDEwithVtrue):
    '''
    This example is modified from 
    title = {Tackling the curse of dimensionality with physics-informed neural networks},
    journal = {Neural Networks},
    author = {Zheyuan Hu and Khemraj Shukla and George Em Karniadakis and Kenji Kawaguchi},
    volume = {176},
    pages = {106369},
    year = {2024},
    doi = {https://doi.org/10.1016/j.neunet.2024.106369}.
    '''
    name = 'Allen-Cahn-SDGD-vLaplace'
    musys_depends_on_v = False
    sgmsys_depends_on_v = True
    f_depends_on_v = True
    x0pil_scale = 1.5

    def __init__(self, dim_x, **kwargs) -> None:
        super().__init__(dim_x, **kwargs)
        self.dim_w = dim_x

        xi = torch.linspace(-torch.pi / 2, torch.pi / 2, dim_x)
        self.ci = (1.5 + torch.sin(xi)) / dim_x

    @staticmethod
    def x_shift(t, x):
        return t + x - torch.tensor(0.5)

    def mu_pil(self, _t, x):
        return torch.zeros_like(x)

    def sgm_pil(self, _t, x, dw):
        return 2**0.5 * dw

    def mu_sys(self, _t, x, _v):
        return torch.zeros_like(x)

    def sgm_sys(self, _t, x, v, dw):
        return v * dw

    def v(self, t, x):
        x = self.x_shift(t, x)
        cosx = torch.cos(x)
        cosx_r1 = torch.roll(cosx, -1, dims=-1)
        x_r1 = torch.roll(x, -1, dims=-1)
        s = torch.sin(x + cosx_r1 + x_r1 * cosx)
        return (self.ci * s).sum(-1, keepdims=True)

    def v_term(self, x):
        return self.v(self.te, x)

    def d_v(self, t, x):
        x = self.x_shift(t, x)
        sinx = torch.sin(x)
        cosx = torch.cos(x)
        sinx_r1 = torch.roll(sinx, -1, dims=-1)
        cosx_r1 = torch.roll(cosx, -1, dims=-1)
        x_r1 = torch.roll(x, -1, dims=-1)
        x_inner = x + cosx_r1 + x_r1 * cosx

        sin_xinn = torch.sin(x_inner)
        cos_xinn = torch.cos(x_inner)

        dx_xinn1 = 1. - x_r1 * sinx
        dx_xinn2 = cosx - sinx_r1
        dx_xinn = dx_xinn1 + dx_xinn2
        sumdx_v = (self.ci * cos_xinn * dx_xinn).sum(-1, keepdims=True)

        dxx_xinn1 = -dx_xinn1.pow(2) - dx_xinn2.pow(2)
        dxx_xinn2 = -x_r1 * cosx - cosx_r1

        sxinn_cxinn = sin_xinn * dxx_xinn1 + cos_xinn * dxx_xinn2
        sumdxx_v = (self.ci * sxinn_cxinn).sum(-1, keepdims=True)

        v = (self.ci * sin_xinn).sum(-1, keepdims=True)
        d_v = sumdx_v + 0.5 * v.pow(2) * sumdxx_v
        return d_v, v

    def f(self, t, x, v):
        dv_val, v_true = self.d_v(t, x)
        return v - v.pow(3) - dv_val - v_true + v_true.pow(3)


class QuasiSDGD(PDEwithVtrue):
    '''
    This example is modified from 
    title = {Tackling the curse of dimensionality with physics-informed neural networks},
    journal = {Neural Networks},
    author = {Zheyuan Hu and Khemraj Shukla and George Em Karniadakis and Kenji Kawaguchi},
    volume = {176},
    pages = {106369},
    year = {2024},
    doi = {https://doi.org/10.1016/j.neunet.2024.106369}.
    '''
    name = 'Quasi-SDGD'
    musys_depends_on_v = True
    sgmsys_depends_on_v = True
    f_depends_on_v = True
    x0pil_scale = 1.5

    def __init__(self, dim_x, **kwargs) -> None:
        super().__init__(dim_x, **kwargs)
        self.dim_w = dim_x

        xi = torch.linspace(-torch.pi / 2, torch.pi / 2, dim_x)
        # The definition of self.ci differs from the example used in SDGD, 
        # as our test typically involves solving v(0, x) for x located on the diagonal of a cube, rather than within a ball.
        self.ci = (1.5 + torch.sin(xi)) / dim_x

    @staticmethod
    def x_shift(t, x):
        return t + x - torch.tensor(0.5)

    def mu_pil(self, _t, x):
        return torch.zeros_like(x)

    def sgm_pil(self, _t, x, dw):
        return torch.full_like(x, 2**0.5) * dw

    def mu_sys(self, _t, x, v):
        return -1. + 0.5 * v * torch.ones_like(x)

    def sgm_sys(self, _t, x, v, dw):
        return v * torch.ones_like(x) * dw

    def v(self, t, x):
        x = self.x_shift(t, x)
        cosx = torch.cos(x)
        cosx_r1 = torch.roll(cosx, -1, dims=-1)
        x_r1 = torch.roll(x, -1, dims=-1)
        s = torch.sin(x + cosx_r1 + x_r1 * cosx)
        return (self.ci * s).sum(-1, keepdims=True)

    def v_term(self, x):
        return self.v(self.te, x)

    def d_v(self, t, x):
        x = self.x_shift(t, x)
        sinx = torch.sin(x)
        cosx = torch.cos(x)
        sinx_r1 = torch.roll(sinx, -1, dims=-1)
        cosx_r1 = torch.roll(cosx, -1, dims=-1)
        x_r1 = torch.roll(x, -1, dims=-1)
        x_inner = x + cosx_r1 + x_r1 * cosx

        sin_xinn = torch.sin(x_inner)
        cos_xinn = torch.cos(x_inner)

        dx_xinn1 = 1. - x_r1 * sinx
        dx_xinn2 = cosx - sinx_r1
        dx_xinn = dx_xinn1 + dx_xinn2
        sumdx_v = (self.ci * cos_xinn * dx_xinn).sum(-1, keepdims=True)

        dxx_xinn1 = -dx_xinn1.pow(2) - dx_xinn2.pow(2)
        dxx_xinn2 = -x_r1 * cosx - cosx_r1
        sxinn_cxinn = sin_xinn * dxx_xinn1 + cos_xinn * dxx_xinn2
        sumdxx_v = (self.ci * sxinn_cxinn).sum(-1, keepdims=True)

        v = (self.ci * sin_xinn).sum(-1, keepdims=True)
        d_v = 0.5 * (v * sumdx_v + v.pow(2) * sumdxx_v)
        return d_v, v

    def f(self, t, x, v):
        dv_val, v_true = self.d_v(t, x)
        f_val = v.pow(2) - v_true.pow(2) - dv_val
        return f_val


class QuasiSDGDFullHess(PDEwithVtrue):
    name = 'Quasi-SDGD-Full-Hess'
    musys_depends_on_v = True
    sgmsys_depends_on_v = True
    f_depends_on_v = True

    x0pil_scale = 1.5

    def __init__(self, dim_x, **kwargs) -> None:
        super().__init__(dim_x, **kwargs)
        self.dim_w = dim_x
        xi = torch.linspace(-torch.pi / 2, torch.pi / 2, dim_x)
        self.ci = (1.5 + torch.sin(xi)) / dim_x

    @staticmethod
    def x_shift(t, x):
        return t + x - torch.tensor(0.5)

    def mu_pil(self, _t, x):
        return torch.zeros_like(x)

    def mu_sys(self, _t, x, v):
        return -1. + 0.5 * v * torch.ones_like(x)

    def sgm_pil(self, _t, x, dw):
        return torch.full_like(x, 2**0.5) * dw

    def sgm_sys(self, _t, x, v, dw):
        sinx = torch.sin(x)
        cosx = torch.cos(x)
        val = cosx * dw.mean(-1, keepdims=True)
        val = val + v * (sinx * dw).mean(-1, keepdims=True)
        return val

    def v(self, t, x):
        x = self.x_shift(t, x)
        cosx = torch.cos(x)
        cosx_r1 = torch.roll(cosx, -1, dims=-1)
        x_r1 = torch.roll(x, -1, dims=-1)
        s = torch.sin(x + cosx_r1 + x_r1 * cosx)
        return (self.ci * s).sum(-1, keepdims=True)

    def v_term(self, x):
        return self.v(self.te, x)

    def d_v(self, t, x):
        x = self.x_shift(t, x)
        sinx = torch.sin(x)
        cosx = torch.cos(x)
        sinx_r1 = torch.roll(sinx, -1, dims=-1)
        cosx_r1 = torch.roll(cosx, -1, dims=-1)
        x_r1 = torch.roll(x, -1, dims=-1)
        x_inner = x + cosx_r1 + x_r1 * cosx

        sin_xinn = torch.sin(x_inner)
        cos_xinn = torch.cos(x_inner)

        dx_xinn1 = 1. - x_r1 * sinx
        dx_xinn2 = cosx - sinx_r1
        dx_xinn = dx_xinn1 + dx_xinn2
        sumdx_v = (self.ci * cos_xinn * dx_xinn).sum(-1, keepdims=True)

        dxx_xinn1 = -dx_xinn1.pow(2) - dx_xinn2.pow(2)
        dxx_xinn2 = -x_r1 * cosx - cosx_r1
        sxinn_cxinn = sin_xinn * dxx_xinn1 + cos_xinn * dxx_xinn2

        v = (self.ci * sin_xinn).sum(-1, keepdims=True)
        sgm_1 = (v * sinx).pow(2).mean(-1, keepdims=True)
        sgm_2 = v * sinx.mean(-1, keepdims=True)

        sgm2_ii = cosx.pow(2) + sgm_1 + sgm_2 * (2 * cosx)
        sgm2_ir1 = cosx * cosx_r1 + sgm_1 + sgm_2 * (cosx + cosx_r1)

        dxixi_v = (self.ci * sgm2_ii * sxinn_cxinn).sum(-1, keepdim=True)
        dxir1_v0 = -sin_xinn * dx_xinn1 * dx_xinn2 - cos_xinn * sinx_r1
        dxir1_v = (self.ci * sgm2_ir1 * dxir1_v0).sum(-1, keepdim=True)
        tr_hess = (dxixi_v + 2 * dxir1_v) / self.dim_x

        d_v = 0.5 * (v * sumdx_v + tr_hess)
        return d_v, v

    def f(self, t, x, v):
        dv_val, v_true = self.d_v(t, x)
        f_val = v.pow(2) - v_true.pow(2) - dv_val
        return f_val


class EVP1(EVP):
    """
    This example is modified from an Eigenvalue problem considered by 
    https://arxiv.org/abs/2307.11942
    """

    name = 'EigenValueProblem1'
    x0_for_train = {'S1': e1_curve}
    # x0_for_train = {'S2': diag_curve}
    x0pil_range = 3.

    musys_depends_on_v = False
    sgmsys_depends_on_v = False
    f_depends_on_v = True

    true_eigenval = -1.
    rate_newlamb = 0.1
    fourier_frequency = None

    def __init__(self, *arg, **kwargs) -> None:
        super().__init__(*arg, **kwargs)
        self.te = 10

        self.it_ev = 0
        self.weight_lamb = 0.

        self.lamb_init = self.true_eigenval + 1
        self.lamb_val = torch.tensor(self.lamb_init)
        self.lamb_from_v = self.lamb_val

    def mu_pil(self, _t, x):
        # return torch.zeros_like(x)
        return x / self.dim_x

    def sgm_pil(self, _t, _x, dw):
        return dw / self.dim_x**(0.5)

    def mu_sys(self, _t, x, _v):
        return x / self.dim_x

    def sgm_sys(self, _t, _x, _v, dw):
        return dw / self.dim_x**(0.5)

    def v(self, _t, x):
        return torch.exp(-x.pow(2).sum(dim=-1, keepdim=True))

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

    def set_vnn_forward(self, vnn):

        def forward(vnn_inst, t, x):
            raw_output = vnn_inst.call(t, x)
            # x_norm = torch.norm(x, p=2, dim=-1, keepdim=True).pow(2)
            output = raw_output
            # output = raw_output / (
            #     1. + torch.norm(x, p=2, dim=-1, keepdim=True).pow(2))
            return output

        vnn.forward = types.MethodType(forward, vnn)
        return vnn


class EVPFokkerPlanck(EVP):
    # 
    
    name = 'EVPFokkerPlanck'
    # x0_for_train = {'S1': e1_curve}
    x0_for_train = {'S2': diag_curve}
    x0pil_range = math.pi

    musys_depends_on_v = False
    sgmsys_depends_on_v = False
    f_depends_on_v = True

    true_eigenval = 0.
    rate_newlamb = 1.
    fourier_frequency = (1, 5)

    def __init__(self, dim_x, **kwargs) -> None:
        super().__init__(dim_x, **kwargs)
        self.te = 1.
        # self.ci = torch.linspace(0.1, 1., dim_x)
        self.ci = torch.ones(dim_x) / dim_x

        self.lamb_init = self.true_eigenval + 1
        self.lamb_val = torch.tensor(self.lamb_init)
        self.lamb_from_v = self.lamb_val

    def mu_pil(self, _t, x):
        # return torch.zeros_like(x)
        return self.mu_sys(_t, x, None)

    def sgm_pil(self, _t, _x, dw):
        # return dw / self.dim_x**(0.5)
        return self.sgm_sys(_t, None, None, dw)

    def mu_sys(self, _t, x, _v):
        cos_cicosxi = torch.cos((self.ci * torch.cos(x)).sum(-1, keepdim=True))
        mu_val = -self.ci * torch.sin(x) * cos_cicosxi
        return mu_val

    def dxx_pot(self, x):
        ci_cosxi = self.ci * torch.cos(x)
        sum_ci_cosxi = ci_cosxi.sum(-1, keepdim=True)
        sum_ci2_sin2xi = (self.ci * torch.sin(x)).pow(2).sum(-1, keepdim=True)
        cos_cicosxi = torch.cos(sum_ci_cosxi)
        sin_cicosxi = torch.sin(sum_ci_cosxi)
        dxx_pot_val = -cos_cicosxi * sum_ci_cosxi - sin_cicosxi * sum_ci2_sin2xi
        return dxx_pot_val

    def sgm_sys(self, _t, x, _v, dw):
        return 2**0.5 * dw

    def f(self, _t, x, v):
        lamb_val = self.get_lamb()
        lap_pot_val = self.dxx_pot(x)
        f_val = (-lamb_val + lap_pot_val) * v
        return f_val

    def v(self, _t, x):
        sin_cicosxi = torch.sin((self.ci * torch.cos(x)).sum(-1, keepdim=True))
        return torch.exp(-sin_cicosxi)

    def set_vnn_forward(self, vnn):
        # def forward(vnn_inst, t, x):
        #     x = torch.cos(x)
        #     output = vnn_inst.call(t, x)
        #     output = self.softplus(output)
        #     return output

        # vnn.forward = types.MethodType(forward, vnn)
        return vnn

    def additional_loss(self, t_path, x_path, v_approx):
        x0 = self.x0_points(100)
        # x0 = torch.zeros((1, self.dim_x), device=self.t0.device)
        vappr_val = v_approx(None, x0)
        vtrue_val = self.v(None, x0)
        loss_val = (vtrue_val - vappr_val).mean().pow(2)

        # vappr_path = v_approx(t_path, x_path)
        # # self.lamb_from_v, lamb_var = self.comput_lamb(t_path, vappr_path)
        # # loss_val = loss_val + 10*lamb_var
        # xnorm_mask = (x_path.norm(dim=-1, keepdim=True) > 3.)
        # loss_val = loss_val + (xnorm_mask * vappr_path.pow(2)).mean()

        return loss_val

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
                'ev_error': (self.true_eigenval -
                             self.lamb_val).abs().item(),  # *************
                'rel_l1err': l1err.item(),
            }
            if self.record_linf_error is True:
                linf_err = abs_err.max() / vtrue_ontest_linf
                log['rel_linferr'] = linf_err.item()
            return log

        return log_func
