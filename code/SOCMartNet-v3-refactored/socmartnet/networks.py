"""Network architectures of the SOC-MartNet paper (arXiv:2405.03169v3).

Paper notation:
  u_alpha(t, x)  -- control network (Eq. (3.24)). All Section-4 examples use the
                    unbounded control space U = R^d, so the ReLU6 clamp for
                    bounded U in Eq. (3.24) is not applied.
  v_theta(t, x)  -- value network. The terminal condition v_theta(T, x) = g(x)
                    is enforced by replacement inside the solver, so this module
                    is phi_theta(t, x) for t < T.
  rho_eta(t, x)  -- adversarial (test-function) network
                    rho_eta(t, x) = sin(W1 t + W2 x + b) in R^r   (Eq. (3.25)).

All three share the additive (t, x) input trunk: the first hidden
pre-activation is W_t t + W_x x + b (two layer biases add into one effective
bias, so Eq. (3.25)'s single-bias form is the same hypothesis class).
"""

import torch
from torch import nn


class DNNtx(nn.Module):
    """Fully connected network on (t, x) with additive time/space input layers.

    dims: [dim_x] + [width] * num_hidden + [dim_out]
    forward: tx = tlayer(t) + xlayer(x); y = out_func(layers(tx))
    """

    def __init__(self, dim_x, dim_out, width, num_hidden,
                 act_func=nn.ReLU, out_func=None):
        super().__init__()
        self.dim_x = dim_x
        self.dim_out = dim_out
        self.num_hidden = num_hidden

        self.tlayer = nn.Linear(1, width)
        self.xlayer = nn.Linear(dim_x, width)

        layers = [act_func()]
        for _ in range(1, num_hidden):
            layers += [nn.Linear(width, width), act_func()]
        if num_hidden > 0:
            layers.append(nn.Linear(width, dim_out))
        else:  # rho_eta: no hidden layers, sin applied directly on tx
            layers = [nn.Identity()]
        self.layers = nn.Sequential(*layers)

        self.out_func = out_func if out_func is not None else (lambda t, y: y)

    def forward(self, t, x):
        tx = self.tlayer(t) + self.xlayer(x)
        return self.out_func(t, self.layers(tx))

    def dx(self, t, x):
        """(grad_x v, v) via autograd with graph kept (used for H_n terms)."""
        is_req = x.requires_grad
        x.requires_grad = True
        val = self.forward(t, x)
        dx_val = torch.autograd.grad(val, x,
                                     grad_outputs=torch.ones_like(val),
                                     create_graph=True)[0]
        x.requires_grad = is_req
        return dx_val.unsqueeze(-2), val


def control_net(dim_x, dim_u, width, num_hidden):
    """u_alpha: [0,T] x R^d -> R^m."""
    return DNNtx(dim_x, dim_u, width, num_hidden, act_func=nn.ReLU)


def value_net(dim_x, width, num_hidden):
    """phi_theta: [0,T] x R^d -> R (scalar value; terminal g applied by solver)."""
    return DNNtx(dim_x, 1, width, num_hidden, act_func=nn.ReLU)


def test_net(dim_x, r):
    """rho_eta(t, x) = sin(W1 t + W2 x + b) in R^r (Eq. (3.25))."""
    return DNNtx(dim_x, r, r, 0, out_func=lambda _t, y: torch.sin(y))
