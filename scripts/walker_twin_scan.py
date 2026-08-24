"""Attractor landscape scan for the compliant walker twin (Q1c).

Instead of aiming optimizers blind, map what long-term behaviors exist:
batch E initial conditions around the plausible gait region, roll T
seconds, classify each trajectory:

  walk    : >=3 alternating clean heelstrikes AND net forward advance
  shuffle : periodic-ish but no alternating strikes (braced crawl)
  fall    : leg angle escapes |th|>1.2

Batched over E (python overhead dominates -> E nearly free).
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


def build(gamma, k=2.5e4, b=400., mu=0.9, dt=1e-4):
    model, gspec, feet, aux = make_walker()
    model.gravity = aux["slope_gravity"](gamma)
    cc = ContactConfig(k_ground=k, damping=b, mu=mu, margin=0.0)
    sim = DiffSim(model, build_geoms_simple(gspec),
                  SimConfig(dt=dt, n_substeps=1, contact=cc),
                  dtype=torch.float64)
    sim.pair_i = sim.pair_i[:0]
    return sim


def scan(gamma, E=64, T=3.0, dt=1e-4, k=2.5e4, b=400., mu=0.9, seed=0):
    rng = np.random.default_rng(seed)
    sim = build(gamma, k, b, mu, dt)
    fs = sim.art.m.q_free_start
    l, r = WALKER_P["l"], WALKER_P["r_foot"]
    delta = WALKER_P["M"] * 9.81 / k

    tha = rng.uniform(-0.28, -0.06, E)
    thb = rng.uniform(-0.10, 0.22, E)
    oma = rng.uniform(0.55, 1.35, E)
    omb = rng.uniform(-1.6, -0.1, E)

    q = torch.zeros(E, sim.art.m.q_dim, dtype=torch.float64)
    w = torch.zeros(E, sim.art.m.v_dim, dtype=torch.float64)
    q[:, fs] = 1.0
    q[:, fs + 6] = torch.from_numpy(l * np.cos(tha)) + r - delta
    q[:, sim.art._qs[1]] = torch.from_numpy(tha)
    q[:, sim.art._qs[2]] = torch.from_numpy(thb)
    w[:, sim.art._vs[1]] = torch.from_numpy(oma)
    w[:, sim.art._vs[2]] = torch.from_numpy(omb)

    x0 = q[:, fs + 3].clone()
    strikes = np.zeros(E)
    last = np.full(E, -1)
    armed = np.zeros((E, 2), dtype=bool)
    prev = np.full((E, 2), np.nan)
    fell = np.zeros(E, dtype=bool)
    n_steps = int(T / dt)
    for i in range(n_steps):
        qdd = sim.forward_dynamics(q, w)
        w = w + dt * qdd
        q = sim.art.integrate(q, w, dt)
        R_w, p_w = sim.art.kinematics(q)
        cen, _ = sim.art.geoms_world(q, sim.geoms, R_w, p_w)
        hip = p_w[:, 0]
        cl = np.empty((E, 2))
        ths = np.empty((E, 2))
        for jx, nm in ((0, "foot_a"), (1, "foot_b")):
            tip = cen[:, sim.geoms.names.index(nm)]
            d = hip - tip
            cl[:, jx] = (tip[:, 2] - r).cpu().numpy()
            ths[:, jx] = np.arctan2(d[:, 0].cpu().numpy(),
                                    d[:, 2].cpu().numpy())
        alive = ~fell
        bad = (np.abs(ths) > 1.2).any(axis=1)
        fell |= alive & bad
        for j in range(E):
            if fell[j]:
                continue
            for lg in range(2):
                p = prev[j, lg]
                if np.isnan(p):
                    prev[j, lg] = cl[j, lg]
                    continue
                if not armed[j, lg]:
                    armed[j, lg] = p > 5e-4
                elif cl[j, lg] <= 0.0 and lg != last[j]:
                    strikes[j] += 1
                    last[j] = lg
                    armed[j, lg] = False
                prev[j, lg] = cl[j, lg]
        if bool(fell.all()):
            break
    adv = (q[:, fs + 3] - x0).cpu().numpy()
    labels = []
    for j in range(E):
        if fell[j]:
            labels.append("fall")
        elif strikes[j] >= 3 and adv[j] > 0.15:
            labels.append("walk")
        elif strikes[j] >= 1:
            labels.append("steps-then-fall" if adv[j] < 0.05 else "walk?")
        else:
            labels.append("shuffle")
    return {"gamma": gamma, "k": k, "b": b, "mu": mu,
            "labels": {lab: int(sum(x == lab for x in labels))
                       for lab in set(labels)},
            "strikes": strikes.tolist(), "advance": adv.tolist(),
            "ics": {"tha": tha.tolist(), "thb": thb.tolist(),
                    "oma": oma.tolist(), "omb": omb.tolist()},
            "labels_raw": labels}


if __name__ == "__main__":
    out = {}
    for gam in (0.009, 0.012, 0.015):
        r = scan(gam, E=64, T=3.0)
        out[str(gam)] = r
        print(f"gamma={gam}: {r['labels']}", flush=True)
        walk_idx = [i for i, x in enumerate(r["labels_raw"])
                    if x.startswith("walk")]
        for i in walk_idx[:5]:
            print(f"   walk IC: tha={r['ics']['tha'][i]:+.3f} "
                  f"thb={r['ics']['thb'][i]:+.3f} "
                  f"oma={r['ics']['oma'][i]:+.3f} "
                  f"omb={r['ics']['omb'][i]:+.3f} "
                  f"strikes={r['strikes'][i]:.0f} "
                  f"adv={r['advance'][i]:+.3f}", flush=True)
    with open("benchmarks/twin_attractor_scan.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print("saved benchmarks/twin_attractor_scan.json", flush=True)
