"""Numerical examples of the SOC-MartNet paper (arXiv:2405.03169v3, Section 4).

Two problem families cover all of Section 4:

  LinearParabolicSin -- Sec. 4.1: (dt + 1/2 Delta) v - f = 0 with manufactured
      solution v(t, x) = 1 + (1/d) sum_i sin(t + x_i); solved in PDE mode
      (Algorithm 3.2). Pilot SDE: mu = 0, sigma = I. (v3 code name: Counter.)

  NonDegHJB -- Secs. 4.2-4.3: the Bachouch et al. / E-Han-Jentzen example.
      PDE mode (Sec. 4.2, Alg. 3.2): (dt + Delta) v - |grad v|^2 = 0,
          terminal 1 + g; driver f = -|grad v|^2.
      SOC mode (Sec. 4.3, Alg. 3.1): dX = 2u dt + sqrt(2) dB,
          J(u) = 1 + E[ int |u|^2 ds + g(X_T) ], u* = -grad v;
          Hamiltonian H = 2 k^T grad v + |k|^2; pilot SDE mu = 0,
          sigma = sqrt(2) I.
      Analytic solution (both modes): v(t, x) = 1 - ln E[exp(-g(x + sqrt(2)
      sgm_scale B_{T-t}))], evaluated by chunked Monte-Carlo with 10^6 samples.
      Terminal variants: 'smooth' g = ln(0.5(1+|x|^2)) (Secs. 4.2.1/4.3.1),
      'oscillatory' g = (1/d) sum_i [ sin(x_i - pi/2) + sin(1/(delc/pi +
      x_i^2)) ] (Secs. 4.2.2/4.3.2; paper Eq. (4.3) writes the shift as
      delta = pi/10, the code uses delc/pi with delc = 0.1 -- see README).

Evaluation grid D_0 (paper Sec. 4): uniformly spaced points on the two
segments S1 = {s e_1} and S2 = {s (1,...,1)} (or s e_diag/||e_diag|| when
unit_ball=True), s in [-x0_scale, x0_scale].

Accepted-paper additions (Secs. 4.3-4.8, SOCMartNet_accepted.tex):

  LinearSinAC    -- Sec. 4.3: parabolic eq. with variable-coefficient
      generator L = sum_i sin(2 x_i) d_xi + 1/2 sum_i (1 + 0.5 sin(5t+x_i))^2
      d_xi^2 and Allen-Cahn-type source f = v - v^3 + f_bar, manufactured so
      that v = 1 + (1/d) sum_i sin(t + x_i).  PDE mode (Alg. 3.2).

  HJBLQ          -- Secs. 4.4-4.8: HJB eq. (4.7) with quadratic control cost,
      (dt + b.grad + delta0^2 Delta) v + inf_k {2 k.grad v + delta0^{-2}|k|^2
      [+ eps sin(1^T k)]} = 0, oscillatory terminal g(y) = osc(y - b).
      SOC mode (Alg. 3.1).  HJB-2: b=1, delta0=0.2, delc=0.3;
      HJB-3: b=1, delta0=0.1, delc=0.3.  (HJB-1 = NonDegHJB in SOC mode with
      terminal='oscillatory', delc=0.1, sgm_scale=1.)

  ShiftTargetHJB -- Sec. 4.6: HJBLQ with b=0, delta0=0.1 and log terminal
      g(x) = coeff_g ln(0.5 (1 + |x - target|^2)), target = 3*1_d.

The paper's epsilon_0 is written 0.1*pi / 0.3*pi; the authors' code uses
delc/pi with delc = 0.1 / 0.3 (the reproduction follows the code).
"""

import torch


def segment_grid(num_points, dim_x, x0_scale=1., unit_ball=False):
    """D_0: half the points on the diagonal segment, half on the 1st axis."""
    num_half = max(num_points // 2, 1)
    e_diag = torch.ones([dim_x])
    if unit_ball:
        e_diag = e_diag / torch.norm(e_diag)
    s = torch.linspace(-x0_scale, x0_scale, num_half)
    x_diag = torch.outer(s, e_diag)

    e_1st = torch.zeros([dim_x])
    e_1st[0] = 1
    x_1st = torch.outer(s, e_1st)
    return torch.concatenate([x_diag, x_1st], dim=0)


def relative_l1(v_pred, v_true):
    """Paper's RE: sum|v_pred - v_true| / sum|v_true| over D_0 (mean form)."""
    return (v_pred - v_true).abs().mean() / v_true.abs().mean()


class LinearParabolicSin:
    """Sec. 4.1 linear parabolic problem (manufactured sin solution)."""

    name = 'Counter_Example'
    H_depends_on_vx = True
    f_depends_on_vxx = False

    def __init__(self, dim_x, t0=torch.tensor(0.), te=torch.tensor(1.),
                 x0_scale=torch.tensor(1.), unit_ball=False):
        self.dim_x = dim_x
        self.dim_w = dim_x
        self.dim_u = 1
        self.t0 = t0
        self.te = te
        self.x0_scale = x0_scale
        self.unit_ball = unit_ball

    def mu(self, _t, x):
        return torch.zeros_like(x)

    def sigma(self, _t, x):
        return torch.ones_like(x)

    def v_exact(self, t, x):
        return 1. + torch.sin(t + x).mean(-1, keepdim=True)

    def v_term(self, x):
        return 1. + torch.sin(self.te + x).mean(-1, keepdim=True)

    def f(self, t, x, _v, _vx, _vxx):
        """PDE driver: f = -(dt + 1/2 Delta) v of the manufactured solution."""
        dt_v = torch.cos(t + x).mean(-1, keepdim=True)
        dxx_v = -torch.sin(t + x).mean(-1, keepdim=True)
        return -dt_v - 0.5 * dxx_v

    def H(self, t, x, k, _v, vx, _vxx):
        # SOC-mode form with a dummy scalar control (not used in Sec. 4.1,
        # which is run in PDE mode); kept for completeness.
        u2_vx = k.pow(2).mean(-1, keepdim=True) * vx.abs().mean(-1)
        return u2_vx + self.f(t, x, _v, vx, _vxx)

    def x0_points(self, num_points):
        return segment_grid(num_points, self.dim_x, float(self.x0_scale),
                            self.unit_ball)

    def test_points(self, num_points):
        return self.x0_points(num_points)


class NonDegHJB:
    """Secs. 4.2-4.3: non-degenerate HJB / associated semilinear parabolic PDE.

    terminal: 'smooth'      g = ln(0.5 (1 + |x|^2))           (Secs. 4.2.1/4.3.1)
              'oscillatory' g = (1/d) sum_i [sin(x_i - pi/2)
                              + sin(1/(delc/pi + x_i^2))]     (Secs. 4.2.2/4.3.2)
    """

    H_depends_on_vx = True
    f_depends_on_vxx = False

    def __init__(self, dim_x, t0=torch.tensor(0.), te=torch.tensor(1.),
                 terminal='smooth', delc=0.1, nsamp_mc=10**6,
                 x0_scale=torch.tensor(1.), sgm_scale=torch.tensor(1.),
                 unit_ball=False):
        self.dim_x = dim_x
        self.dim_w = dim_x
        self.dim_u = dim_x
        self.t0 = t0
        self.te = te
        self.terminal = terminal
        self.delc = delc
        self.delcpi = delc / torch.pi
        self.nsamp_mc = nsamp_mc
        self.x0_scale = x0_scale
        self.sgm_scale = sgm_scale
        self.unit_ball = unit_ball
        if terminal == 'smooth':
            self.name = 'Non-deg_HJB'
        else:
            self.name = f'Non-deg_HJBv3_delc{delc}_te{float(te)}'

    # ---- pilot SDE coefficients (control-free; Remark 3.6) ----
    def mu(self, _t, x):
        return torch.zeros_like(x)

    def sigma(self, _t, x):
        return (2**0.5 * float(self.sgm_scale)) * torch.ones_like(x)

    # ---- terminal functions ----
    def g(self, x):
        if self.terminal == 'smooth':
            return torch.log(0.5 * (1 + (x**2).sum(-1, keepdim=True)))
        return (torch.sin(x - torch.pi / 2) +
                torch.sin(1 / (self.delcpi + x.pow(2)))).mean(-1, keepdim=True)

    def v_term(self, x):
        return 1 + self.g(x)

    # ---- SOC mode: Hamiltonian H = 2 k^T grad v + |k|^2  (Sec. 4.3) ----
    def H(self, _t, _x, k, _v, vx, _vxx):
        kux = torch.einsum('...l, ...ml -> ...m', k, vx)
        return 2 * kux + k.pow(2).sum(-1, keepdim=True)

    # running cost of the associated SOCP (c1 = eps1^-2 = 1 for HJB-1),
    # mirroring HJBLQ.f_cost; this completion lets
    # NonDegHJB enter the FD-residual + controlled-pool pipeline (Sec. 4.4
    # fig2_hjb2 cells), instead of falling back to the autograd pilot path.
    def f_cost(self, k):
        return k.pow(2).sum(-1, keepdim=True)

    # ---- PDE mode: driver f = -|grad v|^2  (Sec. 4.2) ----
    def f(self, _t, _x, _v, vx, _vxx):
        return -(vx**2).sum(-1)

    # ---- analytic solution by chunked Monte-Carlo on the Cole-Hopf form ----
    def _v_chunked(self, t, x):
        std_dw = (2 * (self.te - t.squeeze(-1)))**0.5
        chunksize = min(max(int(10**7 / self.dim_w), 1), self.nsamp_mc)
        nchunk = int(self.nsamp_mc // chunksize) + 1
        exp_list = []
        for _ in range(nchunk):
            rand_mc = torch.normal(torch.zeros([chunksize, self.dim_w]),
                                   std=float(self.sgm_scale))
            dw = torch.einsum('..., ij -> ...ij', std_dw, rand_mc)
            exp_list.append(torch.exp(-self.g(x.unsqueeze(-2) + dw)).mean(-2))
        return 1 - torch.log(torch.stack(exp_list).mean(0))

    def v_exact(self, t, x, chunk_size=None):
        # t: [T-grid] or broadcastable to x[..., :1]; x: [..., points, dim]
        if chunk_size is None:
            chunk_size = min(100, int(1000 / self.dim_x**2) + 1)
        assert (t.ndim == 1) or (t.ndim == x.ndim)
        assert x.ndim >= 2

        t_expand = t.expand_as(x[..., :1])
        num_points = x.shape[0]
        chunk_size = min(num_points, chunk_size)

        out_list = []
        idx = 0
        for _ in range(num_points // chunk_size):
            out_list.append(
                self._v_chunked(t_expand[..., idx:idx + chunk_size, :],
                                x[..., idx:idx + chunk_size, :]))
            idx += chunk_size
        if num_points % chunk_size > 0:
            out_list.append(
                self._v_chunked(t_expand[..., idx:, :], x[..., idx:, :]))
        return torch.concatenate(out_list, dim=-2)

    # ---- training/evaluation grids ----
    def x0_points(self, num_points):
        return segment_grid(num_points, self.dim_x, float(self.x0_scale),
                            self.unit_ball)

    def test_points(self, num_points):
        return self.x0_points(num_points)


# ---------------------------------------------------------------------------
# Point generators for the D_0 sets of the accepted paper (Sec. 4).
# Each returns (s_coord, x) with x[i] the point at parameter s_coord[i].
# ---------------------------------------------------------------------------

def e1_points(dim_x, num_points, radius=1.):
    """S_1 = {s e_1: s in [-radius, radius]}  (Eq. (4.1))."""
    s_coord = torch.linspace(-radius, radius, num_points)
    e_1st = torch.zeros([dim_x])
    e_1st[0] = 1.
    return s_coord, torch.outer(s_coord, e_1st)


def diag_points(dim_x, num_points, radius=1.):
    """S_2 = {s 1_d: s in [-radius, radius]}  (Eq. (4.2))."""
    s_coord = torch.linspace(-radius, radius, num_points)
    return s_coord, torch.outer(s_coord, torch.ones([dim_x]))


def manifold_points(dim_x, num_points, radius=1.):
    """S_3 = {l(s)}: l_i(s) = s sgn(sin i) + cos(i + pi s), i = 1..d
    (Eq. (4.11); authors' examples.py verbatim)."""
    s_coord = torch.linspace(-radius, radius, num_points)
    e_vec = torch.arange(1, dim_x + 1)
    x = torch.outer(s_coord, torch.sign(torch.sin(e_vec))) \
        + torch.cos(e_vec + s_coord.unsqueeze(-1) * torch.pi)
    return s_coord, x


def origin_point(dim_x, num_points, _radius=1.):
    """Single-point D_0 = {0} (Secs. 4.8 / Table 1-2), repeated num_points
    times so that batching machinery applies unchanged."""
    return torch.zeros([num_points]), torch.zeros([num_points, dim_x])


def region_grid(region, num_points, dim_x, radius=1.):
    """Training/eval grid D_0 as a [num_points, dim_x] tensor.

    region: 's1s2' = S1 ∪ S2 (Secs. 4.1-4.2), 's2s3' = S2 ∪ S3 (Sec. 4.4),
            's2' = S2 only (Sec. 4.6), 'point' = {0} (Sec. 4.8).
    """
    if region == 'point':
        return origin_point(dim_x, num_points)[1]
    if region == 's2':
        return diag_points(dim_x, num_points, radius)[1]
    gens = {'s1s2': (e1_points, diag_points),
            's2s3': (diag_points, manifold_points)}.get(region)
    if gens is None:
        raise ValueError(f'unknown region {region!r}')
    half = max(num_points // 2, 1)
    return torch.cat([g(dim_x, half, radius)[1] for g in gens], dim=0)


class HJBLQ:
    """Secs. 4.4-4.8: non-degenerate HJB with quadratic control cost
    (paper Eq. (4.7); the authors' accepted-version classes
    HJB0/HJB2/HJB2b/HJB2c).

        (dt + b.grad + delta0^2 Delta) v
            + inf_k {2 k.grad v + delta0^{-2} |k|^2 [+ eps sin(1_d^T k)]} = 0,
        v(T, x) = g(x),
        g(y) = (1/d) sum_i [ sin(y_i - b - pi/2)
                             + sin(1 / (delc/pi + (y_i - b)^2)) ]

    Associated SOCP: dX = (b + 2u) dt + delta0 sqrt(2) dB, running cost
    delta0^{-2} |u|^2 [+ eps sin(1_d^T u)].  Pilot SDE = uncontrolled drift b
    with the same diffusion (Remark 3.6).  Exact solution (eps = 0):
    v(t, x) = -ln E[exp(-g(x + b(T-t) + delta0 sqrt(2) B_{T-t}))] by chunked
    Monte-Carlo with nsamp_mc samples.  For eps_purb != None (Sec. 4.7) no
    closed form exists; v_exact then returns the UNPERTURBED reference, which
    is what the paper's Fig. 9 evaluates against.

    Named instances: HJB-2 = (b=1, delta0=0.2, delc=0.3) [authors: HJB2b];
    HJB-3 = (b=1, delta0=0.1, delc=0.3) [authors: HJB2c].  HJB-1 (b=0,
    delta0=1, delc=0.1, terminal 1+g) is NonDegHJB in SOC mode.

    Note the terminal here has NO leading 1 (v_term = g), unlike NonDegHJB.
    """

    H_depends_on_vx = True
    f_depends_on_vxx = False

    def __init__(self, dim_x, t0=torch.tensor(0.), te=torch.tensor(1.),
                 b_val=1., delta0=0.2, delc=0.3, eps_purb=None,
                 nsamp_mc=10**6, x0_scale=1., x0_mode='region',
                 region='s2s3'):
        self.dim_x = dim_x
        self.dim_w = dim_x
        self.dim_u = dim_x
        self.t0 = t0
        self.te = te
        self.b_val = float(b_val)
        self.delta0 = float(delta0)
        self.delc = float(delc)
        self.delcpi = self.delc / torch.pi
        self.eps_purb = eps_purb
        self.nsamp_mc = nsamp_mc
        self.x0_scale = float(x0_scale)
        self.x0_mode = x0_mode
        self.region = region

        self.name = (f'HJBLQ_b{self.b_val}_delta{self.delta0}'
                     f'_delc{self.delc}')
        if eps_purb is not None:
            self.name += f'_eps{eps_purb}'
        if x0_mode == 'point':
            self.name += '_1p'

    # ---- pilot SDE (control-free drift b; Remark 3.6) ----
    def mu(self, _t, x):
        return torch.full_like(x, self.b_val)

    def sigma(self, _t, x):
        return (self.delta0 * 2**0.5) * torch.ones_like(x)

    # ---- terminal ----
    def g(self, x):
        xs = x - self.b_val
        return (torch.sin(xs - torch.pi / 2) +
                torch.sin(1 / (self.delcpi + xs.pow(2)))
                ).mean(-1, keepdim=True)

    def v_term(self, x):
        return self.g(x)

    # ---- SOC mode: Hamiltonian (without the pilot part b.grad v, which the
    # pilot paths supply; cf. NonDegHJB.H) ----
    def H(self, _t, _x, k, _v, vx, _vxx):
        kux = torch.einsum('...l, ...ml -> ...m', k, vx)
        ham = 2 * kux + self.delta0**(-2) * k.pow(2).sum(-1, keepdim=True)
        if self.eps_purb is not None:
            ham = ham + self.eps_purb * torch.sin(k.sum(-1, keepdim=True))
        return ham

    # running cost of the associated SOCP (for J(u) tracking; Sec. 4.6)
    def f_cost(self, k):
        f_val = self.delta0**(-2) * k.pow(2).sum(-1, keepdim=True)
        if self.eps_purb is not None:
            f_val = f_val + self.eps_purb * torch.sin(k.sum(-1,
                                                           keepdim=True))
        return f_val

    # ---- exact solution by chunked Monte-Carlo (eps = 0 reference) ----
    def _v_chunked(self, t, x):
        c_sgm = self.delta0 * 2**0.5
        std_dw = c_sgm * (self.te - t.squeeze(-1))**0.5
        x_bdt = x + self.b_val * (self.te - t)
        chunksize = min(max(int(10**7 / self.dim_w), 1), self.nsamp_mc)
        nchunk = int(self.nsamp_mc // chunksize) + 1
        exp_list = []
        for _ in range(nchunk):
            rand_mc = torch.normal(torch.zeros([chunksize, self.dim_w]),
                                   std=1.)
            dw = torch.einsum('..., ij -> ...ij', std_dw, rand_mc)
            exp_list.append(torch.exp(-self.g(x_bdt.unsqueeze(-2) + dw)
                                      ).mean(-2))
        return -torch.log(torch.stack(exp_list).mean(0))

    def v_exact(self, t, x, chunk_size=None):
        # t: [T-grid] or broadcastable to x[..., :1]; x: [..., points, dim]
        if chunk_size is None:
            chunk_size = min(100, int(1000 / self.dim_x**2) + 1)
        assert (t.ndim == 1) or (t.ndim == x.ndim)
        assert x.ndim >= 2

        t_expand = t.expand_as(x[..., :1])
        num_points = x.shape[0]
        chunk_size = min(num_points, chunk_size)

        out_list = []
        idx = 0
        for _ in range(num_points // chunk_size):
            out_list.append(
                self._v_chunked(t_expand[..., idx:idx + chunk_size, :],
                                x[..., idx:idx + chunk_size, :]))
            idx += chunk_size
        if num_points % chunk_size > 0:
            out_list.append(
                self._v_chunked(t_expand[..., idx:, :], x[..., idx:, :]))
        return torch.concatenate(out_list, dim=-2)

    # ---- J(u) by Monte-Carlo (authors' comput_cost; leading 1 of the
    # paper's J(u) := 1 + E[...] is omitted in the archived configuration) ----
    def comput_cost(self, u_func, x0, num_dt=100):
        """E[ int f_cost(u) ds + g(X_T) ] over paths of the controlled SDE
        starting from x0 ([num_paths, dim]); returns a scalar tensor."""
        dt = float(self.te - self.t0) / num_dt
        xt = x0
        run_cost = torch.zeros([x0.shape[0], 1])
        for n in range(num_dt):
            tn = torch.full([x0.shape[0], 1], float(self.t0) + dt * n)
            ut = u_func(tn, xt)
            run_cost = run_cost + self.f_cost(ut) * dt
            dw = torch.normal(torch.zeros_like(xt), std=dt**0.5)
            xt = xt + (self.b_val + 2 * ut) * dt \
                + (self.delta0 * 2**0.5) * dw
        return run_cost.mean() + self.v_term(xt).mean()

    # ---- training/evaluation grids ----
    def x0_points(self, num_points):
        if self.x0_mode == 'point':
            return origin_point(self.dim_x, num_points)[1]
        return region_grid(self.region, num_points, self.dim_x,
                           self.x0_scale)

    def test_points(self, num_points):
        if self.x0_mode == 'point':
            return origin_point(self.dim_x, 1)[1]
        return self.x0_points(num_points)


class ShiftTargetHJB(HJBLQ):
    """Sec. 4.6: SOCP with a shifted target (paper Eq. (4.13); authors'
    HJB1ShiftTarget2).

        b = 0, delta0 = 0.1,
        g(x) = coeff_g ln(0.5 (1 + |x - target|^2)),
        coeff_g = 10, target = 3 * 1_d (paper text), D_0 = S_2.

    The accepted-version archived runs (SOCMN154) used the unshifted form
    (target = 0) with the same coeff_g, and SOCMN147 used a quadratic
    mean((x-3)^2) with coeff 1.  The reproduction takes the PAPER's form
    (target = 3*1_d, log, coeff_g = 10) as authoritative and the default;
    pass target_shift=0 to recover the SOCMN154 configuration.
    """

    def __init__(self, dim_x, coeff_g=10., target_shift=3., **kwargs):
        kwargs.setdefault('region', 's2')
        super().__init__(dim_x, b_val=0., delta0=0.1, **kwargs)
        self.coeff_g = float(coeff_g)
        self.target_shift = float(target_shift)
        self.target = torch.full((dim_x,), self.target_shift)
        self.name = (f'ShiftTarget_cg{self.coeff_g}_tg{self.target_shift}'
                     + ('_1p' if self.x0_mode == 'point' else ''))

    def g(self, x):
        return self.coeff_g * torch.log(
            0.5 * (1 + ((x - self.target)**2).sum(-1, keepdim=True)))


class LinearSinAC:
    """Sec. 4.3 (time-convergence test, Fig. 3): parabolic eq. with

        L = sum_i sin(2 x_i) d_xi
            + 1/2 sum_i (1 + 0.5 sin(5 t + x_i))^2 d_xi^2,
        f(t, x, v) = v - v^3 + f_bar(t, x),

    manufactured so that v(t, x) = 1 + (1/d) sum_i sin(t + x_i) (Eq. (4.2)).
    PDE mode (Alg. 3.2); pilot SDE = the coefficients of L.  D_0 = S_2
    (authors' LinearSin trains on the diagonal segment).

    The reproduction follows the PAPER's configuration -- the Allen-Cahn
    source v - v^3 is kept explicit in the driver, so allen_cahn=True is
    the default.  The paper's base equation reads (dt + L)v + f = 0
    (eq. (4.1): the source enters with a PLUS; Sec. 4.1's form
    "(dt + 1/2 D)v - f = 0" would instead call for f_code = -f), so the
    driver is f itself:

        f_code(t, x, v) = (v - v^3) + f_bar(t, x),
        f_bar = -(dt + L)v_manuf - (v_manuf - v_manuf^3)
              = f_manuf - (v_manuf - v_manuf^3),

    with f_manuf = -(v_t + L v)_manuf, i.e. the implemented
    f_manuf + (v - v^3) - (v_manuf - v_manuf^3); v_manuf then solves the
    equation exactly.

    allen_cahn=False recovers the archived SOCMN179 LinearSin (driver =
    f_manuf only, v-independent, update_f=false offline).  That archived
    class is likely NOT the code behind the paper's Fig. 3 (the true
    producing implementation may be lost); it is kept for reference only.
    Both forms have the same residual at the true solution.
    """

    name = 'LinearSinAC'
    H_depends_on_vx = False
    f_depends_on_vxx = False
    c_in_sgm = 5.

    def __init__(self, dim_x, t0=torch.tensor(0.), te=torch.tensor(1.),
                 allen_cahn=True, x0_scale=1., **_ignored):
        self.dim_x = dim_x
        self.dim_w = dim_x
        self.dim_u = 1
        self.t0 = t0
        self.te = te
        self.allen_cahn = allen_cahn
        self.x0_scale = float(x0_scale)
        if not allen_cahn:
            self.name = 'LinearSinAC_comp'

    def mu(self, _t, x):
        return torch.sin(2 * x)

    def sigma(self, t, x):
        return 1. + 0.5 * torch.sin(self.c_in_sgm * t + x)

    def v_exact(self, t, x):
        return 1. + torch.sin(t + x).mean(-1, keepdim=True)

    def v_term(self, x):
        return 1. + torch.sin(self.te + x).mean(-1, keepdim=True)

    def f(self, t, x, v, _vx, _vxx):
        sintx = torch.sin(t + x)
        costx = torch.cos(t + x)
        dt_v = costx.mean(-1, keepdim=True)
        mu_vx = (torch.sin(2 * x) * costx).mean(-1, keepdim=True)
        sgm_val = 1. + 0.5 * torch.sin(self.c_in_sgm * t + x)
        tr_sgm_vxx = -(sgm_val.pow(2) * sintx).mean(-1, keepdim=True)
        f_manuf = -dt_v - mu_vx - 0.5 * tr_sgm_vxx
        if not self.allen_cahn:
            return f_manuf
        # f_code = f exactly as in the paper: the base equation is
        # (dt + L)v + f = 0 (eq. (4.1)), whose source enters with a plus --
        # unlike Sec. 4.1's "(dt + 1/2 D)v - f = 0", which maps to
        # f_code = -f.  So f_code = (v - v^3) + f_bar with
        # f_bar = -(dt + L)v_manuf - (v_manuf - v_manuf^3), implemented as
        # f_manuf + (v - v^3) - (v_manuf - v_manuf^3).  The alternative
        # f_manuf - (v - v^3) + (v_manuf - v_manuf^3) has the same residual
        # at v = v_manuf but carries the Allen-Cahn term with the opposite
        # sign -- a different equation, anti-damping near v ~ 1 (training
        # stalls); keeping the v-dependent part equal to the paper's
        # (v - v^3) is what fixes it.
        v_manuf = 1. + sintx.mean(-1, keepdim=True)
        return f_manuf + (v - v.pow(3)) - (v_manuf - v_manuf.pow(3))

    def H(self, t, x, k, _v, vx, _vxx):
        # SOC-mode form with a dummy scalar control (not used in Sec. 4.3,
        # which is run in PDE mode); kept for completeness.
        u2_vx = k.pow(2).mean(-1, keepdim=True) * vx.abs().mean(-1)
        return u2_vx + self.f(t, x, _v, vx, _vxx)

    def x0_points(self, num_points):
        return diag_points(self.dim_x, num_points, self.x0_scale)[1]

    def test_points(self, num_points):
        return self.x0_points(num_points)
