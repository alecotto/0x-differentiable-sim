"""Frontier sweep: Delta-cos vs strike velocity v_n (Q3 + Delta(v_n) law).

The passive walker cannot probe the Pi_ramp frontier because its
touchdowns are slow by design (v_n ~ 0.01-0.4 m/s -> Pi_ramp huge).
This script FORCES strike speed: seed the mid-stance state with extra
base downward velocity so the swing tip reaches the ground at a
controlled rate, sweeping Pi_ramp from ~50 down through the predicted
~2.5 transition to <1 -- a clean single-axis pass through the frontier
with only ONE knob (v_n) varying.

Per configuration:
  * measured v_n at the event (from clearance-descent rate)
  * Pi_ramp = eps_ramp/(v_n*dt)
  * cos_A / cos_B / delta_cos (analytic BPTT vs smoothed FD)
  * ||g_fd|| at the standard epsilon (for Delta = 2*||g_fd||*eps)
Pre-registered reading: if delta_cos transitions near Pi_ramp ~ 2.5,
the collapse generalises beyond the configurations it was fit on.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import diffsim.walker as W  # noqa: E402
from diffsim.walker import make_walker, build_geoms_simple, WALKER_P  # noqa
from diffsim.sim import DiffSim, SimConfig, ContactConfig  # noqa

DT_TYPE = torch.float64


def build(gamma, k, b, mu, dt, beta_soft):
    params = dict(WALKER_P, beta=0.02)
    model, gspec, feet, aux = make_walker(params)
    model.gravity = aux["slope_gravity"](gamma)
    cc = ContactConfig(k_ground=k, damping=b, mu=mu, margin=0.0,
                       beta_soft=beta_soft)
    sim = DiffSim(model, build_geoms_simple(gspec),
                  SimConfig(dt=dt, n_substeps=1, contact=cc),
                  dtype=DT_TYPE)
    sim.pair_i = sim.pair_i[:0]
    return sim


def pack(sim, q, w):
    fs = sim.art.m.q_free_start
    s = torch.cat([q[:, fs + 4:fs + 7],
                   q[:, sim.art._qs[1]].reshape(1, 1),
                   q[:, sim.art._qs[2]].reshape(1, 1)], dim=1)
    v = torch.cat([w[:, sim.art._vs[0]:sim.art._vs[0] + 6],
                   w[:, sim.art._vs[1]].reshape(1, 1),
                   w[:, sim.art._vs[2]].reshape(1, 1)], dim=1)
    return torch.cat([s, v], dim=1)


def unpack(sim, x):
    fs = sim.art.m.q_free_start
    q = torch.zeros(1, sim.art.m.q_dim, dtype=x.dtype)
    w = torch.zeros(1, sim.art.m.v_dim, dtype=x.dtype)
    q[0, fs] = 1.0
    q[0, fs + 4:fs + 7] = x[:, 0:3]
    q[0, sim.art._qs[1]] = x[:, 3]
    q[0, sim.art._qs[2]] = x[:, 4]
    w[:, sim.art._vs[0]:sim.art._vs[0] + 6] = x[:, 5:11]
    w[:, sim.art._vs[1]] = x[:, 11]
    w[:, sim.art._vs[2]] = x[:, 12]
    return q, w


def advance(sim, q, w, n):
    """Differentiable rollout segment (no no_grad: BPTT needs graph)."""
    dt = sim.cfg.dt
    for _ in range(n):
        qdd = sim.forward_dynamics(q, w)
        w = w + dt * qdd
        q = sim.art.integrate(q, w, dt)
    return q, w


def cos(u, v):
    nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if nu < 1e-300 or nv < 1e-300:
        return float("nan")
    return float(np.dot(u, v) / (nu * nv))


def grad_window(sim, x, n_steps, eps, R=6, seed=7):
    xa = x.clone().requires_grad_(True)
    q, w = unpack(sim, xa)
    q, w = advance(sim, q, w, n_steps)
    obj = q[0, sim.art.m.q_free_start + 4]        # hip_x (px) -- CORRECTED
    g_ana = torch.autograd.grad(obj, xa)[0].detach().numpy().reshape(-1)
    rng = np.random.default_rng(seed)
    dim = x.shape[1]
    G = np.zeros((R, dim))
    for r in range(R):
        jit = rng.normal(0, eps * 0.3, size=dim)
        for j in range(dim):
            e = np.zeros(dim)
            e[j] = eps
            vals = []
            for sg in (+1., -1.):
                xt = torch.tensor(x.detach().numpy() + sg * e + jit,
                                  dtype=DT_TYPE)
                qq, ww = unpack(sim, xt)
                with torch.no_grad():
                    qq, ww = advance(sim, qq, ww, n_steps)
                vals.append(float(qq[0, sim.art.m.q_free_start + 4]))
            G[r, j] = (vals[0] - vals[1]) / (2 * eps)
    return g_ana, G.mean(axis=0), cos(G[0::2].sum(0), G[1::2].sum(0))


@torch.no_grad()
def find_event_with_vn(sim, q, w, max_steps=40000):
    """First descending crossing of EITHER foot; returns (step, v_n)."""
    dt = sim.cfg.dt
    prev = {"a": None, "b": None}
    armed = {"a": False, "b": False}
    for i in range(max_steps):
        R_w, p_w = sim.art.kinematics(q)
        cen, _ = sim.art.geoms_world(q, sim.geoms, R_w, p_w)
        for lg, nm in (("a", "foot_a"), ("b", "foot_b")):
            cl = float(cen[:, sim.geoms.names.index(nm), 2]
                       - WALKER_P["r_foot"])
            p = prev[lg]
            if p is not None:
                if not armed[lg] and p > 5e-4:
                    armed[lg] = True
                elif armed[lg] and cl <= 0.0:
                    return i, (p - cl) / dt
            prev[lg] = cl
        qdd = sim.forward_dynamics(q, w)
        w = w + dt * qdd
        q = sim.art.integrate(q, w, dt)
    return None, None


def measure(vn_target, gamma=0.009, k=2.5e5, b=400., mu=0.9, dt=1e-4,
            beta_soft=1e4, W=200, eps=2e-5, R=6):
    """DROP PROTOCOL: release the machine from height h so first touch
    occurs at v_n = sqrt(v0^2 + 2 g h) -- the only way to control strike
    speed on a machine whose natural landings are all ~0.04 m/s."""
    sim = build(gamma, k, b, mu, dt, beta_soft)
    fs = sim.art.m.q_free_start
    l, r = WALKER_P["l"], WALKER_P["r_foot"]
    # pose: modest split, both feet CLEAR of ground
    th_a, th_b = 0.12, -0.10          # absolute angles (rad)
    h_clear = min(0.05, max(2e-4,
            0.5 * vn_target ** 2 / 9.81 + 2e-4))
    q = torch.zeros(1, sim.art.m.q_dim, dtype=DT_TYPE)
    w = torch.zeros(1, sim.art.m.v_dim, dtype=DT_TYPE)
    q[0, fs] = 1.0
    q[0, fs + 4] = 0.001 * vn_target            # slight forward drift
    q[0, fs + 6] = l * math.cos(min(abs(th_a), abs(th_b))) \
        + WALKER_P["r_foot"] + h_clear
    q[0, sim.art._qs[1]] = th_a
    q[0, sim.art._qs[2]] = th_b
    w[0, sim.art._vs[0] + 5] = -float(math.sqrt(max(
        vn_target ** 2 - 2 * 9.81 * h_clear, 0.0)))   # base sink rate
    w[0, sim.art._vs[1]] = -0.3                    # legs swinging gently
    w[0, sim.art._vs[2]] = 0.25

    i_ev, v_n = find_event_with_vn(sim, q, w)
    if i_ev is None:
        return {"vn_target": vn_target,
                "error": f"event {i_ev} (h_clear={h_clear:.3f})"}
    # adaptive window: drops strike fast, so shrink W to fit
    W = min(W, max(60, i_ev // 2 - 10))
    if i_ev < 2 * W:
        return {"vn_target": vn_target,
                "error": f"event too early ({i_ev})"}
    out = {"vn_target": vn_target, "i_event": i_ev, "h_clear": h_clear,
           "W": W}
    starts = {"A": i_ev - 2 * W, "B": i_ev - W // 2}
    floors = {}
    for name, i0 in starts.items():
        qq, ww = advance(sim, q.clone(), w.clone(), i0)
        x = pack(sim, qq, ww).requires_grad_(True)
        g_ana, g_fd, floor = grad_window(sim, x, W, eps=eps, R=R)
        out[f"cos_{name}"] = cos(g_ana, g_fd)
        out[f"gnorm_fd_{name}"] = float(np.linalg.norm(g_fd))
        floors[name] = floor
    out["v_n"] = v_n
    out["Pi_ramp"] = (1.0 / beta_soft) / (v_n * dt)
    out["delta_cos"] = out["cos_A"] - out["cos_B"]
    out["floor_mean"] = 0.5 * (floors["A"] + floors["B"])
    out["Delta_jump_est"] = 2 * out["gnorm_fd_B"] * eps
    return out


if __name__ == "__main__":
    rows = []
    print("=== v_n frontier sweep ===", flush=True)
    for vt in (0.06, 0.12, 0.25, 0.5, 1.0, 2.0, 3.2):
        try:
            r = measure(vt)
            rows.append(r)
            print(f"vt={vt:5.2f}: vn={r.get('v_n', float('nan')):.3f} "
                  f"Pi_ramp={r.get('Pi_ramp', float('nan')):8.1f} "
                  f"dcos={r.get('delta_cos', float('nan')):+.3f} "
                  f"cosB={r.get('cos_B', float('nan')):.3f} "
                  f"Delta={r.get('Delta_jump_est', float('nan')):.2e} "
                  f"floor={r.get('floor_mean', float('nan')):.2f}", flush=True)
        except Exception as e:
            print(f"vt={vt}: ERROR {e}", flush=True)
        with open("benchmarks/vn_frontier_sweep.json", "w") as fh:
            json.dump(rows, fh, indent=1)
    print("saved benchmarks/vn_frontier_sweep.json", flush=True)
