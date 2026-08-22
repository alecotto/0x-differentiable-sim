"""Gradient-health probe v2 — addresses review of v1.

Fixes vs v1:
  * seed established BEFORE any randomness (v1 perturbed unseeded)
  * E=64 matched initial-condition batch shared by BPTT and FD legs
    (v1 used E=1)
  * fd_eps plateau sweep {1e-3..1e-6} at reference horizon; plateau center
    used for headline numbers (v1 used a fixed 1e-5 with no justification)
  * multi-seed (3 seeds): median + worst-case reported, not n=1
  * config MATCHED to training defaults (k_ground etc.) so numbers are
    comparable across documents
"""
import os
import sys
import time

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from diffsim.humanoid import make_soma_humanoid, initial_pose   # noqa: E402
from diffsim import build_geoms_compat                           # noqa: E402
from diffsim.sim import DiffSim, SimConfig, ContactConfig        # noqa: E402

DT = torch.float64


def build_sim():
    model, gspec, _ = make_soma_humanoid()
    cc = ContactConfig()          # DEFAULTS -- matched everywhere
    sim = DiffSim(model, build_geoms_compat(gspec),
                  SimConfig(dt=5e-4, n_substeps=8, contact=cc), dtype=DT)
    return model, sim


def rollout_cost(sim, model, q0, w0, theta, H):
    qt = theta.reshape(1, -1).expand(q0.shape[0], -1)
    q, w = q0.clone(), w0.clone()
    total = torch.zeros(q0.shape[0], dtype=q0.dtype, device=q0.device)
    for _ in range(H):
        tau = sim.pd_torques(q, w, qt, kp=400., kd=50.)
        r = sim.step(q, w, tau_ext=tau, train_mode=True)
        q, w = r.q, r.qd
        R_t = r.R_w[:, 1]
        up = R_t[:, 2, 2].clamp(-1., 1.)
        ce = (r.com_z - 0.8573).clamp(-0.5, 0.5)
        total = total + 2.0 * up + 0.25 - 0.002 * (w ** 2).sum(-1) \
            - 0.05 * ce ** 2 - 2.0 * ((r.com_z < 0.45).to(DT))
    return total.mean()


def main():
    torch.set_num_threads(max(1, torch.get_num_threads()))
    model, sim = build_sim()
    n_theta = 15
    eps_list = [1e-3, 1e-4, 1e-5, 1e-6]
    H_list = [8, 32, 64, 128]
    seeds = [101, 202, 303]

    # ---- fd_eps plateau sweep at H=32, seed 101 --------------------------
    print("== fd_eps plateau sweep (H=32) ==")
    torch.manual_seed(101)
    q0, w0 = sample_init_batch(model, 64)
    theta0 = 0.02 * torch.randn(n_theta, dtype=DT)

    def g_fd(eps):
        g = torch.zeros_like(theta0)
        for i in range(n_theta):
            tp = theta0.clone(); tp[i] += eps
            tm = theta0.clone(); tm[i] -= eps
            with torch.no_grad():
                g[i] = (rollout_cost(sim, model, q0, w0, tp, 32)
                        - rollout_cost(sim, model, q0, w0, tm, 32)) / (2 * eps)
        return g

    fd_table = {}
    for eps in eps_list:
        t0 = time.time()
        g = g_fd(eps)
        fd_table[eps] = g
        print(f"  eps={eps:.0e}: |g|={float(g.norm()):.4f} ({time.time()-t0:.0f}s)")
    # plateau: consecutive relative differences
    print("  consecutive rel-diff:",
          [f"{float((fd_table[a]-fd_table[b]).norm()/max(fd_table[b].norm(),1e-12)):.3f}"
           for a, b in zip(eps_list, eps_list[1:])])

    # ---- headline sweep ---------------------------------------------------
    print(f"{'H':>5} {'seed':>5} {'|g_bptt|':>11} {'|g_fd|':>11} {'cos':>9} {'rel_err':>9}")
    results = {}
    for H in H_list:
        rows = []
        for seed in seeds:
            torch.manual_seed(seed)
            q0, w0 = sample_init_batch(model, 64)
            theta0 = 0.02 * torch.randn(n_theta, dtype=DT)

            th = theta0.clone().requires_grad_(True)
            J = rollout_cost(sim, model, q0, w0, th, H)
            (g_b,) = torch.autograd.grad(J, th)

            eps = eps_list[2] if H <= 32 else eps_list[1]   # plateau interior
            g_f = g_fd_generic(sim, model, q0, w0, theta0, H, eps)

            cos = float(torch.nn.functional.cosine_similarity(g_b, g_f, dim=0))
            rel = float((g_b - g_f).norm() / g_f.norm().clamp(min=1e-12))
            rows.append((float(g_b.norm()), float(g_f.norm()), cos, rel))
            print(f"{H:>5} {seed:>5} {rows[-1][0]:>11.4f} {rows[-1][1]:>11.4f} "
                  f"{cos:>9.5f} {rel:>9.4f}", flush=True)
        results[H] = rows
        meds = torch.tensor(rows).median(dim=0).values
        print(f"  -> H={H}: median |g_b|={meds[0]:.4f} |g_fd|={meds[1]:.4f} "
              f"cos={meds[2]:.5f} rel={meds[3]:.4f}")


def sample_init_batch(model, E):
    q, w = initial_pose(model, E)
    s = 0.5
    jp = (q[:, 7:] + 0.06 * s * torch.randn(E, 15, dtype=DT)).clamp(-0.3, 0.3)
    jp[:, [7, 12]] = jp[:, [7, 12]].clamp(max=0.04)
    q[:, 7:] = jp
    q[:, model.q_free_start + 6] += s * ((-0.03) + 0.04 * torch.rand(E, dtype=DT))
    w[:, :3] += 0.3 * s * torch.randn(E, 3, dtype=DT)
    w[:, 6:] += 0.3 * s * torch.randn(E, 15, dtype=DT)
    w[:, 3:5] += 0.4 * s * torch.randn(E, 2, dtype=DT)
    return q, w


def g_fd_generic(sim, model, q0, w0, theta0, H, eps):
    g = torch.zeros_like(theta0)
    for i in range(theta0.numel()):
        tp = theta0.clone(); tp[i] += eps
        tm = theta0.clone(); tm[i] -= eps
        with torch.no_grad():
            g[i] = (rollout_cost(sim, model, q0, w0, tp, H)
                    - rollout_cost(sim, model, q0, w0, tm, H)) / (2 * eps)
    return g


if __name__ == "__main__":
    main()
