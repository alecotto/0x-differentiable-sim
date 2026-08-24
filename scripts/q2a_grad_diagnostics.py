"""Q2a diagnostics: do exact gradients survive contact transitions?

Instrument: short BPTT rollouts of the push-recovery task. At every
control step we extract the LOCAL policy-output gradient g_t =
d/d(a_t) [ r_t + gamma V(s_{t+1}) ] via backprop-through-simulation over
the remaining window (this is what SHAC actually consumes), then align
gradient statistics with contact events (large changes in expected
active-contact count or foot normal force).

Metrics:
  * per-step |g_t| time series vs contact-event indicator
  * cosine(g_t, g_{t+1}) ACROSS events vs WITHIN stable contact phases
  * distribution shift summary written to benchmarks/

If |g| spikes by orders of magnitude at transitions or cosines collapse,
Adam-style updates will destabilize exactly at heelstrikes -- that is
the Q2a failure mode we are testing for.
"""
from __future__ import annotations

import json
import math
import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from diffsim.humanoid import make_soma_humanoid, initial_pose  # noqa
from diffsim import build_geoms_compat  # noqa
from diffsim.sim import DiffSim, SimConfig, ContactConfig  # noqa
from diffsim.algo.shac import ActorCritic  # noqa

DT = torch.float64


def obs_of(R_w, q, w):
    up_z = R_w[:, 1][:, 2, 2].unsqueeze(-1)
    return torch.cat([up_z, w[:, 3:6], q[:, 7:], w[:, 6:]], dim=-1)


def main(E=8, H=48, seed=7, out="benchmarks/q2a_grad_transitions.json",
         push=0.25, lean=0.05, tag=""):
    torch.manual_seed(seed)
    model, gspec, feet = make_soma_humanoid()
    cc = ContactConfig(k_ground=1.5e4, k_pair=8e3, damping=200.0)
    feet_idx = [i for i, g in enumerate(gspec) if "foot" in g["name"]]
    sim = DiffSim(model, build_geoms_compat(gspec),
                  SimConfig(dt=5e-4, n_substeps=8, contact=cc), dtype=DT,
                  feet_geoms=feet_idx)
    ac = ActorCritic(34, 15, hidden=64)

    q, w = initial_pose(model, E, dtype=DT)
    q[:, 9] += lean                       # slight lean: guarantees dynamics
    w[:, 3:5] += push                     # small push

    grads = []          # [H, E, 15]
    rewards = []
    ncont = []          # [H, E]
    fnz = []            # [H, E]
    ctx = torch.enable_grad()
    with ctx:
        for t in range(H):
            R_w, _ = sim.art.kinematics(q)
            o = obs_of(R_w, q, w)
            a = ac.act(o)
            tau = sim.pd_torques(q, w, a, kp=80., kd=10.)
            r = sim.step(q, w, tau_ext=tau, train_mode=True)
            q, w = r.q, r.qd
            R_next, _ = sim.art.kinematics(q)
            o_next = obs_of(R_next, q, w)
            upright = R_next[:, 1][:, 2, 2].clamp(-1., 1.)
            rw = 2.0 * upright + 0.25 - 0.002 * (w ** 2).sum(-1) \
                - 0.05 * (r.com_z - 0.8557).clamp(-0.5, 0.5) ** 2
            # one-step-TD local objective; its grad wrt a_t is what SHAC
            # backpropagates (short-horizon version uses longer sums)
            obj = rw.sum() + 0.99 * ac.value(o_next).sum()
            ga = torch.autograd.grad(obj, a, retain_graph=False,
                                     allow_unused=True)[0]
            grads.append(ga.detach().clone() if ga is not None
                         else torch.zeros_like(a))
            rewards.append(float(rw.mean()))
            _, cif = sim.contact_forces(q, qd=w)
            fz = cif["feet_fnz"]
            loaded = (fz > 10.0).to(q.dtype) if fz is not None \
                else torch.zeros(E, 2, dtype=q.dtype)
            ncont.append(loaded.sum(-1).detach().cpu())
            fnz.append(fz.detach().cpu() if fz is not None else torch.zeros(E))

    G = torch.stack(grads)                # [H,E,15]
    NC = torch.stack(ncont)               # [H,E]
    FZ = torch.stack(fnz)                 # [H,E,2]
    gnorm = G.norm(dim=-1)                # [H,E]
    # event: binary loaded-contact set changed for that env that step
    ev = NC.diff(dim=0).abs() >= 0.5                    # [H-1,E]
    cos = torch.nn.functional.cosine_similarity(G[:-1], G[1:], dim=-1)
    live = (gnorm[:-1] > 1e-9) & (gnorm[1:] > 1e-9)     # both grads alive
    across = cos[ev & live]
    within = cos[~ev & live]
    gn_event = gnorm[1:][ev & live]
    gn_within = gnorm[1:][~ev & live]

    res = {
        "tag": tag, "push": push, "E": E, "H": H, "seed": seed,
        "grad_norm_median": float(gnorm.median()),
        "grad_norm_p95": float(gnorm.quantile(0.95)),
        "grad_norm_max": float(gnorm.max()),
        "n_events": int((ev & live).sum()),
        "n_stable_pairs": int((~ev & live).sum()),
        "cos_across_events_median": float(across.median()) if across.numel() else None,
        "cos_within_median": float(within.median()) if within.numel() else None,
        "gnorm_at_events_over_within":
            float(gn_event.median() / max(gn_within.median(), 1e-30))
            if gn_event.numel() and gn_within.numel() else None,
        "per_step_mean_reward": rewards,
    }
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    import sys as _sys
    quiet = "--quiet" in _sys.argv
    main(E=8, H=64, push=0.0 if quiet else 0.25,
         lean=0.0 if quiet else 0.05,
         out="benchmarks/q2a_grad_transitions%s.json" % (
             "_quiet" if quiet else ""),
         tag="quiet-standing" if quiet else "pushed")
