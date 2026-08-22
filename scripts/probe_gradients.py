"""Gradient-health probe: how trustworthy are BPTT gradients through
contact-rich humanoid simulation, as a function of horizon?

Method
------
Parameters theta (15-dim PD-target offsets) enter a smooth stabilization
cost accumulated over an H-step rollout under fixed PD gains.

    g_bptt = dJ/dtheta   via backprop-through-simulation (exact AD)
    g_fd   = dJ/dtheta   via central finite differences (step 1e-5)

We report ||g||, cosine similarity, and relative error for each horizon.
This is the empirical instrument for the known differentiable-sim tradeoff:
gradients are clean for short horizons and degrade (chaos + contact
discontinuities) as horizon grows -- motivating short-horizon AC (SHAC)
and, later, shadowing-based corrections.
"""
import sys
import time

import torch

sys.path.insert(0, '/Code/0x-differentiable-sim-project')
from diffsim.humanoid import make_soma_humanoid, initial_pose   # noqa: E402
from diffsim import build_geoms_compat                           # noqa: E402
from diffsim.sim import DiffSim, SimConfig, ContactConfig        # noqa: E402

DT = torch.float64


def make_sim():
    model, gspec, feet = make_soma_humanoid()
    cc = ContactConfig(k_ground=1.5e4, k_pair=8e3, damping=200.0)
    sim = DiffSim(model, build_geoms_compat(gspec),
                  SimConfig(dt=5e-4, n_substeps=8, contact=cc), dtype=DT)
    return model, sim


def rollout_cost(sim, model, q0, w0, theta, H, kp=80., kd=10.):
    """Dense stabilization cost over H control steps; differentiable in theta."""
    qt = torch.zeros(1, 15, dtype=DT) + theta.reshape(1, 15)
    q, w = q0.clone(), w0.clone()
    total = torch.zeros((), dtype=DT)
    for _ in range(H):
        tau = sim.pd_torques(q, w, qt, kp=kp, kd=kd)
        r = sim.step(q, w, tau_ext=tau, train_mode=True)
        q, w = r.q, r.w if hasattr(r, "w") else r.qd
        com_z = r.com_z[0]
        joint_pen = (q[0, 7:] ** 2).sum()
        vel_pen = (w[0] ** 2).sum() * 1e-3
        total = total + (com_z - 0.8557) ** 2 * 10 + joint_pen * 0.1 + vel_pen
    return total


def main():
    model, sim = make_sim()
    q0, w0 = initial_pose(model, 1)
    # small persistent perturbation so gradients are non-trivial
    w0 = w0 + 0.05 * torch.randn(1, model.v_dim, dtype=DT)
    torch.manual_seed(0)
    theta0 = 0.02 * torch.randn(15, dtype=DT)

    fd_eps = 1e-5
    print(f"{'H':>5} {'|g_bptt|':>11} {'|g_fd|':>11} {'cos':>9} {'rel_err':>10} {'sec':>7}")
    for H in [8, 32, 64, 128]:
        th = theta0.clone().requires_grad_(True)
        t0 = time.time()
        J = rollout_cost(sim, model, q0, w0, th, H)
        (g_bptt,) = torch.autograd.grad(J, th)
        ad_time = time.time() - t0

        g_fd = torch.zeros_like(theta0)
        for i in range(theta0.numel()):
            tp = theta0.clone(); tp[i] += fd_eps
            tm = theta0.clone(); tm[i] -= fd_eps
            with torch.no_grad():
                gp = rollout_cost(sim, model, q0, w0, tp, H)
                gm = rollout_cost(sim, model, q0, w0, tm, H)
            g_fd[i] = (gp - gm) / (2 * fd_eps)

        cos = float(torch.nn.functional.cosine_similarity(
            g_bptt, g_fd, dim=0))
        rel = float((g_bptt - g_fd).norm() / g_fd.norm().clamp(min=1e-12))
        print(f"{H:>5} {float(g_bptt.norm()):>11.4f} {float(g_fd.norm()):>11.4f} "
              f"{cos:>9.5f} {rel:>10.4f} {ad_time:>7.1f}")


if __name__ == "__main__":
    main()
