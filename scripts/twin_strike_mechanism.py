"""High-k walking + strike-mechanism experiment (Q1c headline evidence).

Regenerates every number quoted in README/RESEARCH_LOG for the high-k
twin result:

  A. dt-convergence of the k=1e6 stride (dt in {5e-5, 2e-5})
  B. stiffness axis: net advance at k in {2.5e4, 4e5, 1e6}
  C. post-strike telemetry: retained swing rate vs rigid-orbit
     prediction, decay time, backswing stall amplitude (twin), and the
     oracle's own backswing amplitude (rigid reference)

Stride counting rule: alternating-leg touchdown events separated by
>= 50 ms from the previous count (suppresses stiff-contact chatter).
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
from walker_oracle import WalkerOracle, continuation_fp  # noqa: E402

DT_TYPE = torch.float64


def build(gamma, k, b=400., mu=0.9, dt=1e-4, beta=None,
          implicit=True):
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


def midstance_seed(sim, gamma):
    """Oracle-derived mid-stance IC (the ONLY known entry into the
    narrow walking basin)."""
    import walker_oracle as wo
    orc, y_mid, _t, fp = wo.midstance_state(gamma)
    assert y_mid is not None, f"no oracle gait at gamma={gamma}"
    q = torch.zeros(1, sim.art.m.q_dim, dtype=DT_TYPE)
    w = torch.zeros(1, sim.art.m.v_dim, dtype=DT_TYPE)
    fs = sim.art.m.q_free_start
    l, r = WALKER_P["l"], WALKER_P["r_foot"]
    delta = WALKER_P["M"] * 9.81 / sim.cfg.contact.k_ground
    q[0, fs] = 1.0
    q[0, fs + 6] = l * math.cos(float(y_mid[0])) + r - delta
    q[0, sim.art._qs[1]] = float(y_mid[0])
    q[0, sim.art._qs[2]] = float(y_mid[2])
    w[0, sim.art._vs[1]] = float(y_mid[1])
    w[0, sim.art._vs[2]] = float(y_mid[3])
    return q, w, fp


def foot_state(sim, q, w):
    R_w, p_w = sim.art.kinematics(q)
    cen, _ = sim.art.geoms_world(q, sim.geoms, R_w, p_w)
    hip = p_w[:, 0]
    out = {}
    for lg, nm in (("a", "foot_a"), ("b", "foot_b")):
        tip = cen[:, sim.geoms.names.index(nm)]
        d = hip - tip
        out[f"x_{lg}"] = float(tip[:, 0])
        out[f"cl_{lg}"] = float(tip[:, 2] - WALKER_P["r_foot"])
        out[f"th_{lg}"] = float(np.arctan2(d[:, 0], d[:, 2]))
    out["hip_x"] = float(q[0, sim.art.m.q_free_start + 4])
    out["om_a"] = float(w[:, sim.art._vs[1]])
    out["om_b"] = float(w[:, sim.art._vs[2]])
    return out


def run_stride(sim, q, w, T=6.0):
    """March until fall or horizon; count alternating strikes."""
    dt = sim.cfg.dt
    x0 = q[0, sim.art.m.q_free_start + 4].item()
    last = None
    armed = {"a": False, "b": False}
    prev = {"a": None, "b": None}
    strikes = []
    maxcl = {"a": -9., "b": -9.}
    fell_t = None
    n = int(T / dt)
    for i in range(n):
        qdd = sim.forward_dynamics(q, w)
        w = w + dt * qdd
        q = sim.art.integrate(q, w, dt)
        st = foot_state(sim, q, w)
        t = i * dt
        for lg in ("a", "b"):
            maxcl[lg] = max(maxcl[lg], st[f"cl_{lg}"])
            p = prev[lg]
            if p is not None:
                if not armed[lg]:
                    if p > 5e-4:
                        armed[lg] = True
                elif st[f"cl_{lg}"] <= 0.0 and lg != last:
                    if not strikes or t - strikes[-1][0] >= 0.05:
                        strikes.append((round(t, 4), lg))
                    last = lg
                    armed[lg] = False
            prev[lg] = st[f"cl_{lg}"]
        if abs(st["th_a"]) > 1.2 or abs(st["th_b"]) > 1.2:
            fell_t = t
            break
    return {"advance": st["hip_x"] - x0, "fell_t": fell_t,
            "n_strikes": len(strikes),
            "strikes": strikes,
            "max_clearance_mm": {k: v * 1e3 for k, v in maxcl.items()}}


def post_strike_telemetry(sim, q, w, ms=200.0):
    """From a just-struck state: sample leg-a (new swing) rate & angle."""
    dt = sim.cfg.dt
    n = int(ms * 1e-3 / dt)
    trace = []
    for i in range(n):
        st = foot_state(sim, q, w)
        trace.append((i * dt * 1e3, st["th_a"], st["om_a"]))
        qdd = sim.forward_dynamics(q, w)
        w = w + dt * qdd
        q = sim.art.integrate(q, w, dt)
    return trace


def march_to_first_strike(sim, q, w, max_steps=40000):
    dt = sim.cfg.dt
    armed = False
    prev = None
    for i in range(max_steps):
        st = foot_state(sim, q, w)
        cl = st["cl_b"]
        if cl > 5e-4:
            armed = True
        elif armed and cl <= 0.0 and prev is not None:
            return q, w, i, (prev - cl) / dt
        prev = cl
        qdd = sim.forward_dynamics(q, w)
        w = w + dt * qdd
        q = sim.art.integrate(q, w, dt)
    return None, None, None, None


def main():
    gamma = 0.009
    results = {"gamma": gamma}

    # ---- oracle rigid reference -------------------------------------
    orc, s_fp = continuation_fp(gamma, WALKER_P["beta"])[:2]
    y = np.array([s_fp[0], s_fp[1], -s_fp[0], s_fp[2]])
    om_pred = abs(s_fp[2])
    bs = -9.
    t_o = 0.
    for _ in range(1200):
        y = orc.flow_step(y)
        t_o += orc.h
        bs = max(bs, y[2])
    tip_rise_oracle = WALKER_P["l"] * (math.cos(abs(s_fp[0]))
                                       - math.cos(bs))
    results["oracle_reference"] = {
        "post_strike_rate_prediction": om_pred,
        "backswing_peak_rad": bs,
        "backswing_tip_rise_mm": tip_rise_oracle * 1e3,
    }

    # ---- A/B: dt convergence + stiffness axis ------------------------
    runs = {}
    for k, dt in ((1e6, 5e-5), (1e6, 2e-5),
                  (4e5, 5e-5), (2.5e4, 5e-5)):
        sim = build(gamma, k=k, dt=dt, beta=WALKER_P["beta"])
        q, w, fp = midstance_seed(sim, gamma)
        r = run_stride(sim, q, w, T=6.0)
        runs[f"k{k:.0e}_dt{dt:.0e}"] = r
        print(f"k={k:.0e} dt={dt:.0e}: adv={r['advance']:+.4f} "
              f"fell={r['fell_t']} strikes={r['n_strikes']} "
              f"maxclear={r['max_clearance_mm']}", flush=True)
    results["stride_runs"] = runs

    # ---- C: post-strike telemetry at k=1e6 ---------------------------
    sim = build(gamma, k=1e6, dt=5e-5, beta=WALKER_P["beta"])
    q, w, fp = midstance_seed(sim, gamma)
    q, w, i_ev, vn = march_to_first_strike(sim, q, w)
    trace = post_strike_telemetry(sim, q, w, ms=200.)
    arr = np.array(trace)
    om0 = float(arr[:, 2].max())          # peak within first window
    below = arr[arr[:, 2] < 0.1 * om0]
    tau = float(below[0, 0]) if len(below) else None
    bs_idx = int(np.argmax(arr[:, 1]))
    stall_amp = float(arr[bs_idx, 1])
    th_swing_start = float(arr[0, 1])   # leg-a angle AT strike
    tip_rise = WALKER_P["l"] * (math.cos(abs(th_swing_start))
                                - math.cos(abs(stall_amp))) * 1e3
    results["post_strike_telemetry"] = {
        "v_n_at_event": vn,
        "twin_peak_rate": om0,
        "oracle_predicted_rate": om_pred,
        "retained_fraction": om0 / om_pred,
        "decay_to_10pct_ms": tau,
        "backswing_stall_rad": stall_amp,
        "backswing_stall_ms": float(arr[bs_idx, 0]),
        "backswing_tip_rise_mm": tip_rise,
        "trace_ms_rad_omega": [[float(a), float(b), float(c)]
                                for a, b, c in trace[::4]],
    }
    print(f"telemetry: peak om={om0:.3f} vs oracle {om_pred:.3f} "
          f"(retained {om0/om_pred:.2f}), decay tau={tau}ms, "
          f"stall={stall_amp:+.4f} rad ({tip_rise:.2f} mm rise)", flush=True)

    os.makedirs("benchmarks", exist_ok=True)
    with open("benchmarks/twin_strike_mechanism.json", "w") as fh:
        json.dump(results, fh, indent=1)
    print("saved benchmarks/twin_strike_mechanism.json", flush=True)


if __name__ == "__main__":
    main()
