"""Q3 FRONTIER SWEEP: walking-mechanism vs gradient-cleanliness on one axis.

At k=1e6 (where the twin achieves one stride), sweep contact damping b
and ramp width eps_ramp (=1/beta_soft) -- the two knobs governing how
fast the old foot unloads (double-support overlap duration).  Per
configuration measure BOTH:

  (a) retained swing rate omega_a / oracle prediction, backswing stall,
      double-support overlap duration (both feet normal force > 5% BW)
  (b) Delta-cos across the same heelstrike event

Prediction under test: retained omega_a rises monotonically as overlap
duration falls.  If narrowing eps_ramp recovers the backswing while
DEGRADING Delta-cos (per the jump finding), the walking-cleanliness
conflict is confirmed on this second independent axis.
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

sys.path.insert(0, os.path.join(_ROOT, "scripts"))
from walker_oracle import continuation_fp  # noqa: E402

DT_TYPE = torch.float64
BW = WALKER_P["M"] * 9.81


def build(gamma, k, b, beta_soft, mu, dt, implicit=True):
    params = dict(WALKER_P, beta=0.02)
    model, gspec, feet, aux = make_walker(params)
    model.gravity = aux["slope_gravity"](gamma)
    cc = ContactConfig(k_ground=k, damping=b, mu=mu, margin=0.0,
                       beta_soft=beta_soft, implicit_damping=implicit)
    feet_idx = [i for i, g in enumerate(gspec) if "foot" in g["name"]]
    sim = DiffSim(model, build_geoms_simple(gspec),
                  SimConfig(dt=dt, n_substeps=1, contact=cc),
                  dtype=DT_TYPE, feet_geoms=feet_idx)
    sim.pair_i = sim.pair_i[:0]
    return sim


def seed_mid(sim):
    import walker_oracle as wo
    orc, y_mid, _t, fp = wo.midstance_state(0.009)
    assert y_mid is not None
    q = torch.zeros(1, sim.art.m.q_dim, dtype=DT_TYPE)
    w = torch.zeros(1, sim.art.m.v_dim, dtype=DT_TYPE)
    fs = sim.art.m.q_free_start
    l, r = WALKER_P["l"], WALKER_P["r_foot"]
    delta = BW / (2 * sim.cfg.contact.k_ground)
    q[0, fs] = 1.0
    q[0, fs + 6] = l * math.cos(float(y_mid[0])) + r - delta
    q[0, sim.art._qs[1]] = float(y_mid[0])
    q[0, sim.art._qs[2]] = float(y_mid[2])
    w[0, sim.art._vs[1]] = float(y_mid[1])
    w[0, sim.art._vs[2]] = float(y_mid[3])
    return q, w


@torch.no_grad()
def advance(sim, q, w, n):
    dt = sim.cfg.dt
    for _ in range(n):
        qdd = sim.forward_dynamics(q, w)
        w = w + dt * qdd
        q = sim.art.integrate(q, w, dt)
    return q, w


def telemetry(sim, q, w, ms=200.):
    """Post-strike leg-a rate trace + overlap duration."""
    dt = sim.cfg.dt
    n = int(ms * 1e-3 / dt)
    trace = []
    overlap_steps = 0
    thresh = 0.05 * BW
    for i in range(n):
        _, cif = sim.contact_forces(q, qd=w)
        fz = cif["feet_fnz"][0] if cif["feet_fnz"] is not None \
            else torch.zeros(2)
        if bool((fz > thresh).all()):
            overlap_steps += 1
        R_w, p_w = sim.art.kinematics(q)
        cen, _ = sim.art.geoms_world(q, sim.geoms, R_w, p_w)
        hip = p_w[:, 0]
        tip = cen[:, sim.geoms.names.index("foot_a")]
        d = hip - tip
        th_a = float(np.arctan2(d[:, 0], d[:, 2]))
        om_a = float(w[:, sim.art._vs[1]])
        trace.append((i * dt * 1e3, th_a, om_a))
        qdd = sim.forward_dynamics(q, w)
        w = w + dt * qdd
        q = sim.art.integrate(q, w, dt)
    arr = np.array(trace)
    om0 = float(arr[:, 2].max())
    below = arr[arr[:, 2] < 0.1 * max(om0, 1e-9)]
    tau = float(below[0, 0]) if len(below) else None
    bi = int(np.argmax(arr[:, 1]))
    tip_rise = WALKER_P["l"] * (
        math.cos(abs(float(arr[0, 1]))) - math.cos(arr[bi, 1])) * 1e3
    return {"peak_rate": om0, "decay_10pct_ms": tau,
            "stall_rad": float(arr[bi, 1]),
            "backswing_rise_mm": tip_rise,
            "overlap_ms": overlap_steps * dt * 1e3}


def march_to_strike(sim, q, w, max_steps=60000):
    dt = sim.cfg.dt
    armed = False
    prev = None
    for i in range(max_steps):
        R_w, p_w = sim.art.kinematics(q)
        cen, _ = sim.art.geoms_world(q, sim.geoms, R_w, p_w)
        cl = float(cen[:, sim.geoms.names.index("foot_b"), 2]
                   - WALKER_P["r_foot"])
        if cl > 5e-4:
            armed = True
        elif armed and cl <= 0.0 and prev is not None:
            return i, (prev - cl) / dt
        prev = cl
        qdd = sim.forward_dynamics(q, w)
        w = w + dt * qdd
        q = sim.art.integrate(q, w, dt)
    return None, None


# ---- Delta-cos (same event, separate standard-config sim) -------------

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


def objective(sim, x, n_steps):
    q, w = unpack(sim, x)
    q, w = advance(sim, q, w, n_steps)
    return q[0, sim.art.m.q_free_start + 4]     # hip_x (px)


def cos(u, v):
    nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if nu < 1e-300 or nv < 1e-300:
        return float("nan")
    return float(np.dot(u, v) / (nu * nv))


def dcos_at_event(sim, q, w, i_ev, W=200, eps=2e-5, R=6):
    starts = {"A": max(0, i_ev - 2 * W), "B": max(0, i_ev - W // 2)}
    out = {}
    for name, i0 in starts.items():
        qq, ww = advance(sim, q.clone(), w.clone(), i0)
        x = pack(sim, qq, ww).requires_grad_(True)
        obj = objective(sim, x, W)
        g_ana = torch.autograd.grad(obj, x)[0].detach().numpy().reshape(-1)
        rng = np.random.default_rng(7)
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
                    with torch.no_grad():
                        vals.append(float(objective(sim, xt, W)))
                G[r, j] = (vals[0] - vals[1]) / (2 * eps)
        g_fd = G.mean(axis=0)
        out[f"cos_{name}"] = cos(g_ana, g_fd)
        out[f"floor_{name}"] = cos(G[0::2].sum(0), G[1::2].sum(0))
    out["delta_cos"] = out.get("cos_A", float('nan')) - \
        out.get("cos_B", float('nan'))
    return out


def main():
    gamma = 0.009
    k, dt = 1e6, 2e-5
    orc_o, s_fp = continuation_fp(gamma, 0.02)[:2]
    oracle_rate = abs(s_fp[2])

    rows = []
    grid = []
    for b in (400., 800., 1600.):
        grid.append({"b": b, "beta_soft": 1e4})
    for bs in (5e3, 2e4):
        grid.append({"b": 800., "beta_soft": bs})

    for cfg in grid:
        sim = build(gamma, k, cfg["b"], cfg["beta_soft"], 0.9, dt)
        q, w = seed_mid(sim)
        i_ev, v_n = march_to_strike(sim, q, w)
        if i_ev is None:
            print(f"{cfg}: no strike", flush=True)
            continue
        tel = telemetry(sim, q, w, ms=200.)
        tel.update(cfg)
        tel["k"] = k
        tel["v_n"] = v_n
        tel["retained_fraction"] = tel["peak_rate"] / oracle_rate
        tel["Pi_ramp"] = (1.0 / cfg["beta_soft"]) / (v_n * dt)
        # dcos on an identically-parameterized standard sim
        try:
            sim2 = build(gamma, k, cfg["b"], cfg["beta_soft"], 0.9, dt)
            q2, w2 = seed_mid(sim2)
            i2, _ = march_to_strike(sim2, q2, w2)
            dc = dcos_at_event(sim2, q2, w2, i2)
            tel.update(dc)
        except Exception as e:
            tel["dcos_error"] = str(e)
        rows.append(tel)
        print(json.dumps({kk: (round(vv, 4) if isinstance(vv, float) else vv)
                          for kk, vv in tel.items()}), flush=True)

    os.makedirs("benchmarks", exist_ok=True)
    with open("benchmarks/q3_frontier_sweep.json", "w") as fh:
        json.dump(rows, fh, indent=1)
    print("saved benchmarks/q3_frontier_sweep.json", flush=True)


if __name__ == "__main__":
    main()
