"""Full Lyapunov spectrum from forward-mode AD tangent propagation.

Method
------
Benettin's algorithm needs the action of the one-step Jacobian on K
tangent vectors.  Instead of finite-differencing shadow trajectories we
propagate each tangent EXACTLY through the discrete flow with
`torch.func.jvp` (forward-mode AD), i.e.

    D_{n+1} = J(x_n) D_n,        J = d(step)/dx evaluated by AD,

followed by QR reorthonormalization; the Lyapunov exponents are

    lambda_i = sum_n log r_{i}(n) / T .

Because the simulator's contact model is a smooth (C^inf softplus) force
law, the same machinery runs through impacts with NO saltation
corrections -- the variational equation of the compliant flow is valid
at heel-strike events by construction.

Why this matters
----------------
* one implementation serves lambda_1 ... lambda_n and any system the
  simulator can express (no separate FD plumbing per model),
* exactness: no delta0 truncation error, no shadow-trajectory drift,
* differentiable end-to-end -> the SPECTRUM itself is a differentiable
  function of morphology parameters (grad-of-grad experiments).
"""
from __future__ import annotations

import math

import torch


def push_jacobian(step_fn, x: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """Exact J(x) @ D via vectorized forward-mode AD.

    step_fn: [n] -> [n] pure function (one integrator substep).
    x: [n], D: [n,K]. Returns [n,K].
    """
    def single(t):
        return torch.func.jvp(step_fn, (x,), (t,))[1]
    # out_dims=1: stack per-column results back into columns -> [n,K]
    return torch.func.vmap(single, in_dims=1, out_dims=1)(D)


def jacobian(step_fn, x: torch.Tensor) -> torch.Tensor:
    """Dense exact Jacobian J(x) [n,n] via forward-mode."""
    n = x.numel()
    eye = torch.eye(n, dtype=x.dtype, device=x.device)
    return push_jacobian(step_fn, x, eye)


@torch.no_grad()
def _qr_renormalize(D: torch.Tensor):
    """QR with positive diagonal; returns (Q, log-diag R)."""
    Q, R = torch.linalg.qr(D)
    sg = torch.sign(torch.diagonal(R, dim1=-2, dim2=-1))
    sg = torch.where(sg == 0, torch.ones_like(sg), sg)
    Q = Q * sg.unsqueeze(-2)
    log_r = torch.log(torch.diagonal(R, dim1=-2, dim2=-1).abs()
                      .clamp_min(1e-300))
    return Q, log_r


def lyapunov_spectrum(step_fn, x0: torch.Tensor, dt_per_step: float,
                      n_steps: int, qr_every: int = 10, k: int | None = None,
                      seed: int = 0, hist_every: int = 0):
    """Full Lyapunov spectrum of a discrete map (descending order).

    x0: [n] initial state.  Tangent frame is seeded as a random orthogonal
    basis (seeded RNG => reproducible); results are independent of it for
    ergodic systems (checked in tests).

    Returns dict(lams [k] descending, hist list[(t, lams)], sum_lam).
    """
    n = x0.numel()
    k = k or n
    assert k <= n
    g = torch.Generator().manual_seed(seed)
    D = torch.randn(n, k, generator=g, dtype=x0.dtype)
    Q, _ = _qr_renormalize(D)                       # random orthonormal frame
    x = x0.clone()

    log_acc = torch.zeros(k, dtype=torch.float64)
    t_tot = 0.0
    hist = []
    for i in range(n_steps):
        # push tangents through the SAME step the state takes:
        # D <- J(x_i) D,  x <- step(x_i)
        Dp = push_jacobian(step_fn, x, Q)
        x = step_fn(x)
        Q, lr = _qr_renormalize(Dp)
        log_acc += lr.double()
        t_tot += dt_per_step
        if hist_every and (i + 1) % hist_every == 0:
            hist.append((t_tot, (log_acc / t_tot).clone()))
    lams, _ = torch.sort(log_acc / t_tot, descending=True)

    # feasibility: explicit-integrator exponents cannot act faster than dt
    assert bool((lams.abs() * dt_per_step < 1.0).all()), (
        f"exponent timescale below substep: {lams} -- estimator bug")

    return {"lams": lams.to(x0.dtype), "hist": hist,
            "sum": float(lams.sum())}


def trajectory_divergence(step_fn, x0: torch.Tensor, n_steps: int):
    """Mean phase-space contraction <tr Df> along a trajectory,
    computed exactly with forward-mode jvps over unit basis vectors."""
    n = x0.numel()
    eye = torch.eye(n, dtype=x0.dtype, device=x0.device)
    x = x0.clone()
    tot = 0.0
    for _ in range(n_steps):
        tot += float(torch.diagonal(push_jacobian(step_fn, x, eye)).sum())
        x = step_fn(x)
    return tot / n_steps


# --------------------------------------------------------------------- #
# differentiable spectrum: exponents as functions of system parameters
# --------------------------------------------------------------------- #
#
# WHY NOT COTANGENTS?  Propagating W <- J^T W forward-in-time with LQ
# filtering computes the ADJOINT (left) exponents, which for strongly
# NON-NORMAL maps differ wildly from the Lyapunov spectrum even though
# svd(J^N) = svd((J^T)^N): transient growth redistributes expansion
# between successive QR steps (measured on the damped double pendulum:
# sigma_max(J) = 1.11 while sigma_max(J^50) = 2.45 despite rho(J) < 1).
# Forward tangent propagation is the correct estimator.
#
# WHY NOT torch.func.jvp / jacrev INSIDE autograd?  Reverse-mode
# differentiation through functorch forward transforms (jvp) and through
# jacrev's vmap+vjp machinery produces NaN grads (reverse-over-forward
# gaps), so d(lambda)/d(theta) cannot flow through them.  We instead
# build each one-step Jacobian column-by-column with PLAIN autograd
# grad(..., create_graph=True) -- reverse-over-reverse is fully
# supported -- and propagate tangents as differentiable matmuls; the
# whole Benettin accumulation stays connected so autograd yields
# dlambda/dtheta directly.

def _dense_jacobian_autograd(fn, x: torch.Tensor) -> torch.Tensor:
    """Exact dense Jacobian of fn at x, connected to any parameters fn
    closes over (create_graph=True). One reverse pass per output comp."""
    n_out = None
    xc = x.detach().requires_grad_(True)
    fx = fn(xc)
    n_out = fx.numel()
    rows = []
    for j in range(n_out):
        (g,) = torch.autograd.grad(fx.reshape(-1)[j], xc,
                                   create_graph=True, retain_graph=True)
        rows.append(g.reshape(1, -1))
    return torch.cat(rows, 0).t()


def lyapunov_spectrum_diff(step_fn, x0: torch.Tensor, dt_per_step: float,
                           n_steps: int, qr_every: int = 10,
                           k: int | None = None):
    """Differentiable Lyapunov spectrum: returns per-frame exponents whose

        d(lambda_i)/d(theta)

    is obtained by `torch.autograd.grad(lams[i], theta)` for any parameter
    theta captured by step_fn (stiffness, damping, morphology...).

    Component i tracks the i-th column of the seeded orthonormal frame
    (no sorting); after transient alignment each component sits on a
    distinct Oseledets subspace.  For degenerate (complex-pair) subspace
    components individual values fluctuate around the true exponent while
    pair-means converge -- see tests.
    """
    n = x0.numel()
    k = k or n
    g = torch.Generator().manual_seed(0)
    Q, _ = _qr_renormalize(torch.randn(n, k, generator=g, dtype=x0.dtype))
    Q = Q.detach()
    x = x0.detach()

    log_acc = torch.zeros(k, dtype=x0.dtype)
    for i in range(n_steps):
        J = _dense_jacobian_autograd(step_fn, x)
        Dp = J @ Q                              # frame path kept connected
        with torch.no_grad():                   # frozen trajectory
            x = step_fn(x)
        Qn, Rn = torch.linalg.qr(Dp)            # differentiable QR
        sg = torch.sign(torch.diagonal(Rn))
        sg = torch.where(sg == 0, torch.ones_like(sg), sg)
        log_acc = log_acc + torch.log(torch.diagonal(Rn).abs()
                                      .clamp_min(1e-300))
        Q = Qn * sg.unsqueeze(-2)
    lams_unsorted = log_acc / (n_steps * dt_per_step)
    return {"lams": lams_unsorted, "sum": float(lams_unsorted.detach().sum())}
