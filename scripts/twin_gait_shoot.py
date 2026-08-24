"""Differentiable gait discovery for the compliant walker twin (Q1c/Q2).

The rigid Garcia orbit is infeasible for point feet on real ground (its
swing leg travels underground mid-stride -- the paper's scuffing
fiction).  The twin's own limit cycle must be FOUND.

Method: fixed-horizon shooting with the mirror-swap periodicity
condition.  For a period-one gait of a leg-symmetric machine, one stride
later every labeled quantity obeys

    th_a(N) = -th_a(0)      th_b(N) = -th_b(0)
    om_a(N) =  om_b(0)      om_b(N) =  om_a(0)

(the legs exchange roles; angles negate because the walk direction is
preserved while each leg moves to the other side of vertical).  No event
detection, no relabeling -- pure smooth composition of simulator steps,
so autograd gives exact gradients THROUGH the heelstrike events.

Parameterization: x = (th, om_a, om_b) with th_b(0) = -th_a(0) (the
mirror pose at start-of-stance; th < 0 = stance leg behind pivot).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from diffsim.walker import make_walker, build_geoms_simple, WALKER_P  # noqa
from diffsim.sim import DiffSim, SimConfig, ContactConfig  # noqa


def build(gamma_deg, k=2.5e4, b=400., mu=3.0, dt=1e-4):
    model, gspec, feet, aux = make_walker()
    model.gravity = aux["slope_gravity"](gamma_deg)
    cc = ContactConfig(k_ground=k, damping=b, mu=mu, margin=0.0)
    sim = DiffSim(model, build_geoms_simple(gspec),
                  SimConfig(dt=dt, n_substeps=1, contact=cc),
                  dtype=torch.float64)
    sim.pair_i = sim.pair_i[:0]
    return sim


def _steps(sim, dt):
    def f(q, w):
        qdd = sim.forward_dynamics(q, w)
        w = w + dt * qdd
        q = sim.art.integrate(q, w, dt)
        return q, w
    return f


def init_state(sim, th_a, th_b, delta):
    """Start-of-stance pose with COMPLIANT offsets: rear foot pressed in
    by penetration delta, front foot starting clear."""
    q = torch.zeros(1, sim.art.m.q_dim, dtype=torch.float64)
    w = torch.zeros(1, sim.art.m.v_dim, dtype=torch.float64)
    fs = sim.art.m.q_free_start
    q[0, fs] = 1.0
    l, r = WALKER_P["l"], WALKER_P["r_foot"]
    q[0, fs + 6] = l * math.cos(float(th_a)) + r - float(delta)
    return q, w


def rollout(sim, q, w, N, dt, chunk=200, n_ckpt=8):
    """N differentiable steps with per-chunk checkpointing so the BPTT
    graph stays bounded (backward recomputes each chunk once).
    Returns (q_N, w_N, [ckpt states evenly spaced])."""
    f = _steps(sim, dt)
    ckpts = []
    stride = max(1, N // n_ckpt)
    done = 0
    while done < N:
        k = min(chunk, N - done)
        if k == stride and len(ckpts) < n_ckpt:
            q, w = torch.utils.checkpoint.checkpoint(f, q, w,
                                                     use_reentrant=False)
            ckpts.append((q, w))
        else:
            q, w = f(q, w)
        done += k
    return q, w, ckpts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=float, default=0.009)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--N", type=int, default=9000, help="horizon steps")
    ap.add_argument("--dt", type=float, default=1e-4)
    ap.add_argument("--seed", type=float, nargs=4,
                    default=[-0.19, 0.16, 0.85, -0.30],
                    help="initial (th_a, th_b, om_a, om_b)")
    ap.add_argument("--k", type=float, default=2.5e4)
    ap.add_argument("--b", type=float, default=400.)
    ap.add_argument("--mu", type=float, default=3.0)
    ap.add_argument("--out", type=str,
                    default="benchmarks/twin_gait_shoot.json")
    args = ap.parse_args()

    sim = build(args.gamma, args.k, args.b, args.mu, args.dt)
    x = torch.nn.Parameter(torch.tensor(args.seed))   # (th_a, th_b, om_a, om_b)
    opt = torch.optim.Adam([x], lr=args.lr)
    history = []
    best = (float("inf"), None)
    delta = WALKER_P["M"] * 9.81 / args.k     # static sag of loaded foot
    for it in range(args.iters):
        th_a, th_b, om_a, om_b = x[0], x[1], x[2], x[3]
        q0, w0 = init_state(sim, th_a, th_b, delta)
        # re-inject parameters so the graph starts at the leaves
        q0 = torch.cat([q0[:, :sim.art._qs[1]],
                        th_a.reshape(1, 1),
                        th_b.reshape(1, 1)], dim=1).requires_grad_(True)
        w0 = torch.cat([w0[:, :sim.art._vs[1]],
                        om_a.reshape(1, 1),
                        om_b.reshape(1, 1)], dim=1).requires_grad_(True)

        qN, wN, ckpts = rollout(sim, q0, w0, args.N, args.dt)
        th_a_N = qN[:, sim.art._qs[1]][0]
        th_b_N = qN[:, sim.art._qs[2]][0]
        om_a_N = wN[:, sim.art._vs[1]][0]
        om_b_N = wN[:, sim.art._vs[2]][0]
        # period-1 condition: legs exchange roles componentwise
        # (validated oracle relabel map: y' = [th2, vp1, th1, vp0])
        loss = ((th_a_N - th_b) ** 2 + (th_b_N - th_a) ** 2
                + (om_a_N - om_b) ** 2 * 0.05
                + (om_b_N - om_a) ** 2 * 0.05)
        # anti-tumble: keep leg angles in the sane range along the whole
        # trajectory and require real forward advance of the hip
        for qc, wc in ckpts:
            ta = qc[:, sim.art._qs[1]][0]
            tb = qc[:, sim.art._qs[2]][0]
            loss = loss + 2.0 * ((torch.relu(ta.abs() - 0.5) ** 2)
                                 + torch.relu(tb.abs() - 0.5) ** 2)
        x0_ = q0[:, sim.art.m.q_free_start + 3][0]
        xN_ = qN[:, sim.art.m.q_free_start + 3][0]
        advance = 2.0 * WALKER_P["l"] * math.sin(max(abs(float(th_a)),
                                                     abs(float(th_b))))
        loss = loss + 4.0 * torch.relu(advance - (xN_ - x0_)) ** 2
        opt.zero_grad()
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_([x], 5.0)
        opt.step()
        resid = float(loss.detach())
        if resid < best[0]:
            best = (resid, x.detach().clone())
        history.append({"iter": it, "loss": resid,
                        "gnorm": float(gnorm),
                        "x": x.detach().tolist()})
        if it % 5 == 0 or it == args.iters - 1:
            print(f"iter {it:3d} loss={resid:.6f} |g|={float(gnorm):.4f} "
                  f"x={np.round(x.detach().numpy(), 4)}", flush=True)
        if resid < 1e-7:
            print("converged.", flush=True)
            break

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"args": vars(args), "best_loss": best[0],
                   "best_x": None if best[1] is None else
                   best[1].tolist(),
                   "history": history}, fh, indent=1)
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
