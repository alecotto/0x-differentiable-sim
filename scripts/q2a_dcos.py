"""Delta-cos scaling experiment (Q2a proof / Q3 frontier / saltation threshold).

One instrumented sweep emitting three results per configuration:
  1. Delta-cos : drop in cos(analytic, FD-reference) when a contact event
                 enters the differentiation window
  2. Q3        : gradient-fidelity vs stiffness point
  3. Saltation : whether compliant tangent propagation approaches the
                 rigid-limit answer as k -> infinity

Definitions
-----------
Window A: horizon ending W/2 before heelstrike (no contact force).
Window B: same-length horizon straddling heelstrike.
Per window: g_ana = BPTT gradient of phi(s_N)=hip_x(N) wrt the 13-dim
window-start state (px,py,pz,th_a,th_b; wx,wy,wz,vx,vy,vz,om_a,om_b);
g_fd = smoothed central differences (R replicates of sub-epsilon jitter).
Split-sample floor: cos between odd/even replicate halves of the FD
estimate -- the self-consistency ceiling of the reference.  Delta-cos is
signal only where it exceeds that floor.

Falsifier (pre-registered): if Delta-cos tracks dt alone at fixed
Pi_1 = sqrt(m_eff/k)/dt, the effect is an integration artifact and the
finding is withdrawn.
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

from diffsim.walker import make_walker, build_geoms_simple, WALKER_P  # noqa
from diffsim.sim import DiffSim, SimConfig, ContactConfig  # noqa

DT_TYPE = torch.float64


def build(gamma, k, b, mu, dt, beta=None, implicit=False):
    params = None
    if beta is not None:
        params = dict(WALKER_P)
        params["beta"] = beta
    model, gspec, feet, aux = make_walker(params)
    model.gravity = aux["slope_gravity"](gamma)
    cc = ContactConfig(k_ground=k, damping=b, mu=mu, margin=0.0,
                       implicit_damping=implicit)
    sim = DiffSim(model, build_geoms_simple(gspec),
                  SimConfig(dt=dt, n_substeps=1, contact=cc),
                  dtype=DT_TYPE)
    sim.pair_i = sim.pair_i[:0]
    return sim


def seed_state(sim, th_a=-0.19, th_b=0.16):
    q = torch.zeros(1, sim.art.m.q_dim, dtype=DT_TYPE)
    w = torch.zeros(1, sim.art.m.v_dim, dtype=DT_TYPE)
    fs = sim.art.m.q_free_start
    q[0, fs] = 1.0
    l, r = WALKER_P["l"], WALKER_P["r_foot"]
    delta = WALKER_P["M"] * 9.81 / sim.cfg.contact.k_ground
    q[0, fs + 6] = l * math.cos(th_a) + r - delta
    q[0, sim.art._qs[1]] = th_a
    q[0, sim.art._qs[2]] = th_b
    w[0, sim.art._vs[1]] = 0.85
    w[0, sim.art._vs[2]] = -0.30
    return q, w


def advance(sim, q, w, n):
    dt = sim.cfg.dt
    for _ in range(n):
        q, w = sim.step_substep(q, w)
    return q, w


@torch.no_grad()
def max_penetration(sim, q0, w0, i_start, n):
    """Max ground penetration depth over a window (post-hoc rerun)."""
    q, w = q0.clone(), w0.clone()
    if i_start > 0:
        q, w = advance(sim, q, w, i_start)
    worst = 0.0
    r = WALKER_P["r_foot"]
    for _ in range(n):
        q, w = advance(sim, q, w, 1)
        R_w, p_w = sim.art.kinematics(q)
        cen, _ = sim.art.geoms_world(q, sim.geoms, R_w, p_w)
        for nm in ("foot_a", "foot_b"):
            tip = cen[:, sim.geoms.names.index(nm)]
            worst = max(worst, float(r - tip[:, 2]))
    return worst


@torch.no_grad()
def find_event(sim, q0, w0, max_steps=20000):
    """Heelstrike substep + strike speed v_n (clearance descent rate)."""
    dt = sim.cfg.dt
    q, w = q0.clone(), w0.clone()
    armed = False
    prev_cl = None
    for i in range(max_steps):
        with torch.no_grad():
            q, w = advance(sim, q, w, 1)
        R_w, p_w = sim.art.kinematics(q)
        cen, _ = sim.art.geoms_world(q, sim.geoms, R_w, p_w)
        tip = cen[:, sim.geoms.names.index("foot_b")]
        cl = float(tip[:, 2] - WALKER_P["r_foot"])
        if cl > 5e-4:
            armed = True
        elif armed and cl <= 0.0 and prev_cl is not None:
            return i, (prev_cl - cl) / dt      # v_n > 0 descending
        prev_cl = cl
    return None, None


POS_IDX = None   # set per-model: 5 position slots, 8 velocity slots


def pack(sim, q, w):
    """Window-start state vector (requires_grad caller's choice).

    positions: px,py,pz, th_a, th_b          (quat pinned to identity --
        valid because planar motion keeps the base upright; monitored via
        the drift check inside one_stride_twin elsewhere)
    velocities: wx,wy,wz, vx,vy,vz, om_a, om_b
    """
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


def objective(sim, x, n_steps):
    q, w = unpack(sim, x)
    q, w = advance(sim, q, w, n_steps)
    return q[0, sim.art.m.q_free_start + 4]      # hip_x readout (px; fs+4..6 = px,py,pz)


def grad_window(sim, x, n_steps, eps=2e-5, R=4, seed=0):
    """Analytic gradient + smoothed central-FD reference + split halves."""
    xa = x.clone().requires_grad_(True)
    obj = objective(sim, xa, n_steps)
    g_ana = torch.autograd.grad(obj, xa)[0].detach().numpy().reshape(-1)

    rng = np.random.default_rng(seed)
    dim = x.shape[1]
    G = np.zeros((R, dim))
    for r in range(R):
        jit = rng.normal(0.0, eps * 0.3, size=dim)
        for j in range(dim):
            e = np.zeros(dim)
            e[j] = eps
            vals = []
            for sign in (+1.0, -1.0):
                xp = x.detach().numpy() + sign * e + jit
                xt = torch.tensor(xp, dtype=DT_TYPE)
                vals.append(float(objective(sim, xt, n_steps)))
            G[r, j] = (vals[0] - vals[1]) / (2 * eps)
    g_fd = G.mean(axis=0)
    floor = cos(G[0::2].sum(axis=0), G[1::2].sum(axis=0)) if R >= 4 \
        else float("nan")
    return g_ana, g_fd, floor


def cos(u, v):
    nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if nu < 1e-300 or nv < 1e-300:
        return float("nan")
    return float(np.dot(u, v) / (nu * nv))


def measure_config(gamma=0.009, k=2.5e4, b=400., mu=0.9, dt=1e-4,
                   beta=0.02, implicit=False, W=200, eps=2e-5, R=4,
                   seed_state_fn=seed_state):
    sim = build(gamma, k, b, mu, dt, beta=beta, implicit=implicit)
    q0, w0 = seed_state(sim)
    i_ev, v_n = find_event(sim, q0, w0)
    if i_ev is None:
        return {"error": "event not found"}
    # windows need: A starts at i_ev-2W >= 0 and B starts >= 0
    W = min(W, max(40, i_ev // 2 - 20))

    out = {"gamma": gamma, "k": k, "b": b, "mu": mu, "dt": dt,
           "beta": beta, "implicit": implicit, "i_event": int(i_ev),
           "v_n": float(v_n), "W": W}
    starts = {"A": i_ev - 2 * W, "B": i_ev - W // 2}
    for name, i0 in starts.items():
        q, w = advance(sim, q0.clone(), w0.clone(), i0)
        x = pack(sim, q, w).requires_grad_(True)
        g_ana, g_fd, floor = grad_window(sim, x, W, eps=eps, R=R,
                                         seed=42)
        out[f"cos_{name}"] = cos(g_ana, g_fd)
        out[f"gnorm_ana_{name}"] = float(np.linalg.norm(g_ana))
        out[f"gnorm_fd_{name}"] = float(np.linalg.norm(g_fd))
        out[f"fd_floor_{name}"] = floor
    if "cos_A" in out and "cos_B" in out:
        out["delta_cos"] = out["cos_A"] - out["cos_B"]
        out["floor_mean"] = 0.5 * (out.get("fd_floor_A", float("nan"))
                                   + out.get("fd_floor_B", float("nan")))
    out["pen_max_B"] = max_penetration(sim, q0, w0,
                                       starts["B"], W)
    for m_name, m_val in (("mf", beta * WALKER_P["M"]),
                          ("M", WALKER_P["M"])):
        out[f"Pi1_{m_name}"] = math.sqrt(m_val / k) / dt
    cc = sim.cfg.contact
    # Pi_ramp = delta_pen * beta_soft  (penetration vs softplus width;
    # >>1 = hard-ramp regime, <<1 = ramp-smoothed)
    out["Pi_ramp_static"] = out["pen_max_B"] * cc.beta_soft
    # DYNAMIC ramp-crossing group: substeps spent traversing the ramp
    if v_n is not None and v_n > 1e-9:
        out["Pi_ramp"] = (1.0 / cc.beta_soft) / (v_n * dt)
    else:
        out["Pi_ramp"] = float("nan")
    out["Pi_static_res"] = out["pen_max_B"] * cc.beta_soft  # delta/eps_ramp
    # gate group: static sag relative to ACTIVATION-GATE width 'smooth'
    delta_sag = WALKER_P["M"] * 9.81 / (2.0 * k)
    out["gate_ratio"] = delta_sag / cc.smooth
    return out


if __name__ == "__main__":
    rows = []
    print("=== Delta-cos sweep ===", flush=True)
    grid = [
        dict(k=2.5e4, implicit=False),
        dict(k=2.5e5, implicit=True),
        dict(k=2.5e6, implicit=True),
        dict(k=2.5e5, dt=2e-4, implicit=True),   # falsifier pair:
        dict(k=2.5e5, dt=5e-5, implicit=True),   # same k, varied dt
    ]
    for cfg in grid:
        try:
            r = measure_config(**cfg)
            rows.append(r)
            print(json.dumps({kk: (round(vv, 4)
                                   if isinstance(vv, float) else vv)
                              for kk, vv in r.items()}), flush=True)
        except Exception as e:
            print(f"{cfg}: ERROR {e}", flush=True)
        with open("benchmarks/q2a_dcos_sweep.json", "w") as fh:
            json.dump(rows, fh, indent=1)
    print("saved benchmarks/q2a_dcos_sweep.json", flush=True)
