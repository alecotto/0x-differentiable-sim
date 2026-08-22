"""Lyapunov-exponent estimation for contact-rich rollouts.

Two independent estimators of the largest Lyapunov exponent lambda:

1. Benettin pair-divergence (fp64): two trajectories started delta0 apart
   in velocity space, evolved in lockstep; each `renorm` steps the offset
   is measured and renormalized back to delta0.  lambda = sum(log g)/T.

2. fp32-vs-fp64 divergence: identical initial states, different precision;
   the precision noise acts as a continuous ~1e-7 perturbation.  The
   trajectory of ||q32 - q64|| is the raw divergence curve -- reported as
   an e-folding rate fitted over the pre-saturation window.

Cross-checking (1) and (2), plus sweeping k_ground, separates real chaos
from contact-stiffness artifacts:
    lambda invariant under k  -> genuine chaos
    lambda ~ sqrt(k)          -> contact spring (design parameter)
"""
from __future__ import annotations

import torch

from .humanoid import make_soma_humanoid, initial_pose
from . import build_geoms_compat
from .sim import DiffSim, SimConfig, ContactConfig


DT = torch.float64


def make_stand_sim(k_ground=None, device="cpu", dtype=torch.float64):
    model, gspec, _ = make_soma_humanoid()
    cc = ContactConfig()
    if k_ground is not None:
        cc.k_ground = k_ground
    sim = DiffSim(model, build_geoms_compat(gspec),
                  SimConfig(dt=5e-4, n_substeps=8, contact=cc),
                  device=device, dtype=dtype)
    return model, sim


@torch.no_grad()
def _step_batch(sim, q, w):
    tau = sim.pd_torques(q, w, torch.zeros(q.shape[0], 15, dtype=q.dtype,
                                           device=q.device), kp=400., kd=50.)
    r = sim.step(q, w, tau_ext=tau)
    return r.q, r.qd


@torch.no_grad()
def benettin_lambda(k_ground=None, E=16, settle=40, steps=200,
                    renorm=5, delta0=1e-9, kick=(0., 0., 0.6),
                    seed=0, device="cpu"):
    """Largest-Lyapunov estimate via renormalized pair divergence.

    Returns (lambda_per_second, history list of log-growth increments).
    """
    model, sim = make_stand_sim(k_ground, device, DT := torch.float64)
    mm = model.masses
    g = torch.Generator(device="cpu").manual_seed(seed)

    def init():
        q, w = initial_pose(model, E)
        jp = (q[:, 7:] + 0.05 * torch.randn(E, 15, generator=g).to(DT)).clamp(-0.25, 0.25)
        q[:, 7:] = jp.to(device=device, dtype=DT)
        w = w.to(device=device, dtype=DT)
        w[:, :3] += torch.tensor(kick, dtype=DT, device=device)
        w += (0.2 * torch.randn(E, model.v_dim, generator=g).to(DT).to(device))
        return q, w

    q1, w1 = init()
    # settle both onto attractor identically
    for _ in range(settle):
        q1, w1 = _step_batch(sim, q1, w1)
    q2, w2 = q1.clone(), w1.clone()

    # kick trajectory 2 in velocity space
    kick_vec = torch.randn(E, model.v_dim, generator=g).to(device, DT)
    kick_vec = kick_vec / kick_vec.norm(dim=-1, keepdim=True) * delta0
    w2 = w2 + kick_vec

    lam_sum = torch.zeros(E, dtype=DT, device=device)
    t_total = 0.0
    hist = []
    for i in range(steps):
        q1, w1 = _step_batch(sim, q1, w1)
        q2, w2 = _step_batch(sim, q2, w2)
        t_total += sim.cfg.dt * sim.cfg.n_substeps

        dq = q2 - q1
        dw = w2 - w1
        nrm = torch.sqrt(dq.pow(2).sum(-1) + dw.pow(2).sum(-1))  # [E]
        lam_sum += torch.log(nrm.clamp(min=1e-300))

        if (i + 1) % renorm == 0:
            hist.append(float(lam_sum.mean() / t_total))
            s = (delta0 / nrm.clamp(min=1e-300)).unsqueeze(-1)
            # rescale offset back to delta0 (linear tangent-space approx;
            # quaternion rows re-normalized by integrate() next step anyway)
            q2 = q1 + s * dq
            w2 = w1 + s * dw

    return float((lam_sum / t_total).mean()), hist


@torch.no_grad()
def fp32_fp64_divergence(k_ground=None, E=16, settle=40, steps=100,
                         kick=(0., 0., 0.6), seed=0, device="cpu"):
    """Divergence curve ||q32 - q64|| per control step + fitted e-fold rate.

    Returns (curve list, fitted lambda_per_second or None).
    """
    m64, sim64 = make_stand_sim(k_ground, device, DT)
    m32, sim32 = make_stand_sim(k_ground, device, torch.float32)
    g = torch.Generator(device="cpu").manual_seed(seed)

    def init(mdl, dtype):
        q, w = initial_pose(mdl, E)
        jp = (q[:, 7:] + 0.05 * torch.randn(E, 15, generator=g).to(DT)).clamp(-0.25, 0.25)
        q[:, 7:] = jp
        w[:, :3] += torch.tensor(kick, dtype=DT)
        w += 0.2 * torch.randn(E, mdl.v_dim, generator=g).to(DT)
        return q.to(dtype=dtype, device=device), w.to(dtype=dtype, device=device)

    q64, w64 = init(m64, torch.float64)
    q32, w32 = init(m32, torch.float32)

    for _ in range(settle):
        q64, w64 = _step_batch(sim64, q64, w64)
        q32, w32 = _step_batch(sim32, q32, w32)

    curve = []
    for i in range(steps):
        q64, w64 = _step_batch(sim64, q64, w64)
        q32, w32 = _step_batch(sim32, q32, w32)
        d = (q32.double() - q64).pow(2).sum(-1) + \
            (w32.double() - w64).pow(2).sum(-1)
        curve.append(float(torch.sqrt(d).mean()))     # mean over batch

    # fit slope on log curve within pre-saturation window
    import math
    lg = [math.log(max(v, 1e-300)) for v in curve]
    n = len(lg)
    seg_dt = 4e-3                                   # ctrl-step seconds
    # primary window: first quarter (earliest = least contaminated)
    slope = (lg[n // 4] - lg[0]) / max(n // 4, 1)
    lam = slope / seg_dt
    # cross-check window: second quarter
    slope2 = (lg[n // 2] - lg[n // 4]) / max(n - n // 4 - n // 4, 1)
    lam_x = slope2 / seg_dt
    return curve, lam, lam_x
