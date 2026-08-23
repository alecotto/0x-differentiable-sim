"""Lyapunov-exponent estimation.

Benettin algorithm, implemented correctly:
  * accumulate the GROWTH RATIO log(||delta_k|| / delta0) at
    renormalization events ONLY;
  * lambda = accumulated sum / total elapsed time;
  * feasibility assertion: |lambda| * dt_substep < 1 -- nothing in an
    explicit integrator decays faster than its own timestep;
  * delta0-invariance check: a correct estimator gives the same lambda
    for any offset magnitude.  A lambda tracking log(delta0)/dt is an
    accumulator bug (this exact bug produced a spurious -5192/s in an
    earlier version).
"""
from __future__ import annotations

import math

import torch

from .humanoid import make_soma_humanoid, initial_pose
from . import build_geoms_compat
from .sim import DiffSim, SimConfig, ContactConfig

DT = torch.float64


# --------------------------------------------------------------------- #
# generic core
# --------------------------------------------------------------------- #

@torch.no_grad()
def benettin_generic(step_fn, x1: torch.Tensor, x2: torch.Tensor,
                     dt_substep: float, steps: int, renorm: int,
                     delta0: float):
    """Largest Lyapunov exponent of a discrete map.

    step_fn(x) -> x' advances ONE integrator substep (flat state tensor).
    x2 must satisfy ||x2 - x1|| == delta0 on entry.
    Accumulates log(||delta||/delta0) at renorm events only.

    Returns (lambda_per_unit_time, history).
    """
    assert x1.shape == x2.shape
    lam_sum = 0.0
    t_total = 0.0
    hist = []

    for i in range(steps):
        x1 = step_fn(x1)
        x2 = step_fn(x2)
        t_total += dt_substep
        if (i + 1) % renorm == 0:
            d = x2 - x1
            nrm = float(torch.linalg.vector_norm(d))
            lam_sum += math.log(max(nrm, 1e-300) / delta0)
            hist.append(lam_sum / t_total)
            s = delta0 / max(nrm, 1e-300)
            x2 = x1 + s * d                      # renormalize to delta0

    lam = lam_sum / t_total

    # feasibility: no explicit-integrator exponent may have a timescale
    # below one substep
    assert abs(lam) * dt_substep < 1.0, (
        f"|lambda|={abs(lam):.3g}/s (timescale {1e3/abs(lam):.3f} ms) is "
        f"faster than substep {dt_substep*1e3:.3f} ms -- estimator bug")

    return lam, hist


@torch.no_grad()
def delta0_invariance_check(step_fn, x_base: torch.Tensor,
                            dt_substep: float, steps: int, renorm: int,
                            kick_dir: torch.Tensor, deltas=(1e-6, 1e-9),
                            rel_tol: float = 0.35):
    """A correct estimator is invariant to the choice of delta0.  A broken
    accumulator returns log(delta0)/dt and tracks delta0 exactly."""
    kick_dir = kick_dir / float(torch.linalg.vector_norm(kick_dir))
    lams = []
    for d0 in deltas:
        x2 = x_base + kick_dir * d0
        lam, _, _ = benettin_generic(step_fn, x_base.clone(), x2.clone(),
                                     dt_substep, steps, renorm, d0)
        lams.append(lam)
    spread = (max(lams) - min(lams)) / max(abs(min(lams)), 1e-12)
    assert spread < rel_tol, f"lambda tracks delta0: {lams} -- accumulator bug"
    return lams


# --------------------------------------------------------------------- #
# humanoid standing wrapper (k-sweep target; gait later)
# --------------------------------------------------------------------- #

def make_stand_sim(k_ground=None, device="cpu", dtype=DT):
    model, gspec, _ = make_soma_humanoid()
    cc = ContactConfig()
    if k_ground is not None:
        cc.k_ground = k_ground
    sim = DiffSim(model, build_geoms_compat(gspec),
                  SimConfig(dt=5e-4, n_substeps=8, contact=cc),
                  device=device, dtype=dtype)
    return model, sim


@torch.no_grad()
def standing_step_fn_factory(sim):
    art = sim.art

    def step(x):
        nq = art.nq
        q = x[:nq].reshape(1, -1)
        w = x[nq:].reshape(1, -1)
        tau = sim.pd_torques(q, w, torch.zeros(1, 15, dtype=x.dtype,
                                               device=x.device), kp=400., kd=50.)
        r = sim.step(q, w, tau_ext=tau)
        return torch.cat([r.q.reshape(-1), r.qd.reshape(-1)])
    return step
