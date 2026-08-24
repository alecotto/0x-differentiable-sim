"""Search for compliant-contact parameters where the DiffSim twin walks.

For each (k_ground, damping) pair, seed E parallel copies at the oracle
mid-stance state with rate scales around 1.0 and count clean
heelstrikes.  Batched over E: python-loop overhead dominates, so E is
nearly free.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from diffsim.walker import make_walker, build_geoms_simple, WALKER_P  # noqa
from diffsim.sim import DiffSim, SimConfig, ContactConfig  # noqa
import walker_oracle as wo  # noqa: E402

DT = torch.float64


def build(gamma, k, b, mu=3.0, dt=1e-4):
    model, gspec, feet, aux = make_walker()
    model.gravity = aux["slope_gravity"](gamma)
    cc = ContactConfig(k_ground=k, damping=b, mu=mu, margin=0.0)
    sim = DiffSim(model, build_geoms_simple(gspec),
                  SimConfig(dt=dt, n_substeps=1, contact=cc), dtype=DT)
    sim.pair_i = sim.pair_i[:0]
    sim.pair_j = sim.pair_j[:0]
    return sim


def leg_state(sim, q, w):
    R_w, p_w = sim.art.kinematics(q)
    centers, _ = sim.art.geoms_world(q, sim.geoms, R_w, p_w)
    hip = p_w[:, 0]
    out = {}
    for k, nm in (("a", "foot_a"), ("b", "foot_b")):
        gi = sim.geoms.names.index(nm)
        tip = centers[:, gi]
        d = hip - tip
        out[f"th_{k}"] = torch.atan2(d[:, 0], d[:, 2])
        out[f"clear_{k}"] = tip[:, 2] - WALKER_P["r_foot"]
        out[f"om_{k}"] = w[:, sim.art._vs[1 if k == "a" else 2]]
    return out


def run_case(gamma, k, b, scales=(0.95, 1.0, 1.05, 1.15, 1.3),
             T=2.6, dt=1e-4, mu=3.0):
    orc, y_mid, _t, fp = wo.midstance_state(gamma)
    assert y_mid is not None, f"no oracle gait at gamma={gamma}"
    E = len(scales)
    sim = build(gamma, k, b, mu=mu, dt=dt)
    q = torch.zeros(E, sim.art.m.q_dim, dtype=DT)
    w = torch.zeros(E, sim.art.m.v_dim, dtype=DT)
    fs = sim.art.m.q_free_start
    q[:, fs] = 1.0
    l, r = WALKER_P["l"], WALKER_P["r_foot"]
    q[:, fs + 6] = l * math.cos(float(y_mid[0])) + r
    q[:, sim.art._qs[1]] = float(y_mid[0])
    q[:, sim.art._qs[2]] = float(y_mid[2])
    for j, s_ in enumerate(scales):
        w[j, sim.art._vs[1]] = y_mid[1] * s_
        w[j, sim.art._vs[2]] = y_mid[3] * s_
    n_steps = int(T / dt)
    strikes = np.zeros(E)
    last_leg = [None] * E
    prev_cl = np.full((E, 2), np.nan)
    armed = np.zeros((E, 2), dtype=bool)
    fell = np.zeros(E, dtype=bool)
    for i in range(n_steps):
        qdd = sim.forward_dynamics(q, w)
        w = w + dt * qdd
        q = sim.art.integrate(q, w, dt)
        st = leg_state(sim, q, w)
        cl = torch.stack([st["clear_a"], st["clear_b"]], dim=1).cpu().numpy()
        th = torch.stack([st["th_a"], st["th_b"]], dim=1).cpu().numpy()
        bad = (np.abs(th) > 1.2).any(axis=1)
        fell |= bad & ~np.isnan(cl).any(axis=1)
        for j in range(E):
            if fell[j]:
                continue
            for lg in range(2):
                p = prev_cl[j, lg]
                if np.isnan(p):
                    prev_cl[j, lg] = cl[j, lg]
                    continue
                if not armed[j, lg]:
                    armed[j, lg] = cl[j, lg] > 5e-4
                elif cl[j, lg] <= 0.0 and lg != last_leg[j]:
                    strikes[j] += 1
                    last_leg[j] = lg
                    armed[j, lg] = False
                prev_cl[j, lg] = cl[j, lg]
        if bool((fell | (strikes >= 3)).all()):
            break
    return {"k": k, "b": b, "gamma": gamma,
            "scales": list(scales), "strikes": strikes.tolist(),
            "fell": fell.tolist()}


if __name__ == "__main__":
    gam = float(sys.argv[1]) if len(sys.argv) > 1 else 0.009
    grid = [(2.5e4, 400.), (2.5e4, 100.), (1e5, 200.), (2e5, 300.),
            (4e5, 400.), (1e6, 600.)]
    print(f"=== twin walk search, gamma={gam} ===", flush=True)
    for k, b in grid:
        r = run_case(gam, k, b)
        print(f"k={k:8.0f} b={b:6.0f} strikes={r['strikes']} "
              f"fell={r['fell']}", flush=True)
