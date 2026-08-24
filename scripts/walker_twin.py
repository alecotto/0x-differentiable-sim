"""DiffSim soft-contact twin of the Garcia simplest walker (Q1c phase 2).

The twin (diffsim/walker.py) is simulated with the FULL DiffSim engine --
compliant contacts, regularized friction, semi-implicit Euler -- walking
down a slope realized as tilted gravity.  This script answers:

    does the compliant-contact twin reproduce the rigid-limit hybrid
    dynamics of the independent sympy oracle (scripts/walker_oracle.py)?

Protocol
--------
1. Build twin at slope gamma; map oracle-B's period-one fixed point into
   twin coordinates (absolute leg angles = hinge angles while the base
   quaternion stays identity; rates in rad/s).
2. Iterate stride-to-stride: detect heelstrike as the swing sphere's
   ground clearance crossing zero while descending; sample the section
   state at that substep and relabel legs.
3. Report per-stride return deviation (twin vs oracle fixed point),
   stride period, contact-force peaks, energy decay, planarity drift.

Deviations from the rigid oracle are EXPECTED (phantom penetration
ln2/beta_soft, mg/k static sag, foot radius r=1mm, impact smearing over
the contact time constant) -- quantifying them is precisely the Q1
measurement: our contact model is validated if deviations stay small and
the gait remains attracting with matching multipliers.
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
import walker_oracle as wo  # noqa: E402

DT = torch.float64


def build_twin(gamma_deg: float, k_ground=2.5e4, damping=400.0, mu=3.0,
               dt=2e-4):
    model, gspec, feet, aux = make_walker()
    model.gravity = aux["slope_gravity"](gamma_deg)
    cc = ContactConfig(k_ground=k_ground, damping=damping, mu=mu,
                       margin=0.0)
    sim = DiffSim(model, build_geoms_simple(gspec), SimConfig(dt=dt,
                   n_substeps=1, contact=cc), dtype=DT,
                  feet_geoms=None)
    # Garcia's model declares scuffing nonexistent; the twin must too.
    sim.pair_i = sim.pair_i[:0]
    sim.pair_j = sim.pair_j[:0]
    return sim


def leg_state(sim, q, w):
    """Absolute angles/rates of both legs + foot clearances [E,...]."""
    R_w, p_w = sim.art.kinematics(q)
    centers, _ = sim.art.geoms_world(q, sim.geoms, R_w, p_w)
    hip = p_w[:, 0]
    out = {}
    for k, nm in (("a", "foot_a"), ("b", "foot_b")):
        gi = sim.geoms.names.index(nm)
        tip = centers[:, gi]
        d = hip - tip
        out[f"th_{k}"] = torch.atan2(d[:, 0], d[:, 2])
        out[f"tip_{k}"] = tip
        out[f"clear_{k}"] = tip[:, 2] - WALKER_P["r_foot"]
        out[f"om_{k}"] = w[:, sim.art._vs[1 if k == "a" else 2]]
    return out


@torch.no_grad()
def one_stride_twin(sim, q0, w0, max_steps=12000):
    """March until heelstrike: ANY non-stance leg's clearance crosses
    from clearly positive (> tol_above) down through zero -- tracked as
    a per-leg descending event, immune to the double-support instant
    where both feet sit near zero clearance.

    Stance identification at t=0 uses the deeper-penetrating leg.
    Returns dict with RELABELED section state (touched leg first).
    """
    dt = sim.cfg.dt
    q, w = q0.clone(), w0.clone()
    st = leg_state(sim, q, w)
    cl_a = float(st["clear_a"][0])
    cl_b = float(st["clear_b"][0])
    stance = "a" if cl_a < cl_b else "b"
    armed = {"a": False, "b": False}   # arm only after leaving ground
    armed[stance] = False
    armed["a" if stance == "b" else "b"] = True
    prev = {"a": cl_a, "b": cl_b}
    TOL_ABOVE = 2e-4                   # must rise above this before arming
    for step in range(max_steps):
        qdd = sim.forward_dynamics(q, w)
        w = w + dt * qdd
        q = sim.art.integrate(q, w, dt)
        st = leg_state(sim, q, w)
        for leg in ("a", "b"):
            cl = float(st[f"clear_{leg}"][0])
            if not armed[leg]:
                if cl > TOL_ABOVE:
                    armed[leg] = True
            elif cl <= 0.0:
                # heelstrike on `leg`; sample & relabel
                other = "a" if leg == "b" else "b"
                return {
                    "ok": True, "steps": step + 1,
                    "t": (step + 1) * dt,
                    "s": (float(st[f"th_{leg}"][0]),
                          float(st[f"om_{leg}"][0]),
                          float(st[f"om_{other}"][0])),
                    "full": (float(st[f"th_{leg}"][0]),
                             float(st[f"om_{leg}"][0]),
                             float(st[f"th_{other}"][0]),
                             float(st[f"om_{other}"][0])),
                    "touched_leg": leg,
                }
            prev[leg] = cl
        R_w, _ = sim.art.kinematics(q)
        quat = q[:, sim.art.m.q_free_start:sim.art.m.q_free_start + 4]
        drift = float((quat - torch.tensor([1., 0., 0., 0.],
                                           dtype=DT)).norm())
        if drift > 1e-6:
            return {"ok": False, "steps": step + 1,
                    "t": (step + 1) * dt, "reason": f"planarity {drift:.2e}"}
        if abs(float(st["th_a"][0])) > 1.5 or abs(float(st["th_b"][0])) > 1.5:
            return {"ok": False, "steps": step + 1, "t": (step + 1) * dt,
                    "reason": "fell"}
    return {"ok": False, "steps": max_steps, "t": max_steps * dt,
            "reason": "timeout"}


def oracle_fp_seed(gamma, go_mod=None):
    """Oracle-B fixed point at the twin's beta via beta-continuation,
    plus the Garcia-mapped small-beta seed for reference."""
    orc, fp = wo.continuation_fp(gamma, WALKER_P["beta"])
    sG = None
    if fp is not None and go_mod is not None:
        fpG = go_mod.shoot_fixed_point(gamma, go_mod.analytic_seed(
            gamma, "long"))
        if fpG is not None:
            thG, wG = fpG["theta"], fpG["omega"]
            wr = math.sqrt(9.81 / WALKER_P["l"])
            c2 = math.cos(2 * thG)
            sG = np.array([-thG, -wG * wr,
                           -wG * wr + c2 * (1 - c2) * (wG * wr / c2)])
    return orc, fp, sG


def main(gammas=(0.009,), n_strides=12, out="benchmarks/walker_twin.json"):
    import garcia_oracle as go
    results = {}
    for gam in gammas:
        print(f"\n=== gamma={gam:.4f} ===")
        orc, fp, sG = oracle_fp_seed(gam, go)
        assert fp is not None, "oracle FP required to seed the twin"
        print(f"oracle-B FP: {np.round(fp, 6)}  "
              f"(rho={orc.find_fixed_point(fp)['rho']:.3f})")
        sim = build_twin(gam)
        th1, d1, d2 = fp
        q0 = torch.zeros(1, sim.art.m.q_dim, dtype=DT)
        w0 = torch.zeros(1, sim.art.m.v_dim, dtype=DT)
        fs = sim.art.m.q_free_start
        q0[:, fs] = 1.0
        l = WALKER_P["l"]
        r_f = WALKER_P["r_foot"]
        # hip placed so the STANCE tip rests at z=r (touching); stance
        # angle th1 -> hip z = l cos(th1) + r ; x offset consistent
        q0[:, fs + 6] = l * math.cos(th1) + r_f
        q0[:, sim.art._qs[1]] = th1          # stance leg (leg_a)
        q0[:, sim.art._qs[2]] = -th1         # swing mirrored
        w0[:, sim.art._vs[1]] = d1
        w0[:, sim.art._vs[2]] = d2
        # NOTE: after relabeling conventions the twin starts with leg_a as
        # stance at th1<0 (behind pivot) and leg_b swinging forward.
        devs, periods, ok_n = [], [], 0
        q, w = q0, w0
        s_ref = fp
        for k in range(n_strides):
            r = one_stride_twin(sim, q, w)
            if not r["ok"]:
                print(f"  stride {k}: FAILED at t={r['t']:.3f}s "
                      f"({r.get('reason', '?')})")
                break
            s_tw = np.array(r["s"][:3])
            dev = float(np.max(np.abs(s_tw[:2] - s_ref[:2])))
            devs.append(dev)
            periods.append(r["t"])
            ok_n += 1
            print(f"  stride {k}: t={r['t']:.4f}s  "
                  f"s=({s_tw[0]:+.5f},{s_tw[1]:+.5f})  "
                  f"|dev(th,om)|={dev:.4f}")
            # rebuild exact state from full sample for next stride
            q, w = state_from_full(sim, np.array(r["full"]))
        results[str(gam)] = {
            "oracle_fp": list(map(float, fp)),
            "garcia_mapped": None if sG is None else list(
                map(float, sG)),
            "n_strides_completed": ok_n,
            "periods_s": periods,
            "return_dev_max": max(devs) if devs else None,
            "period_mean_s": float(np.mean(periods)) if periods else None,
        }
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"\nwrote {out}")
    return results


def state_from_full(sim, full):
    q0 = torch.zeros(1, sim.art.m.q_dim, dtype=torch.float64)
    w0 = torch.zeros(1, sim.art.m.v_dim, dtype=torch.float64)
    fs = sim.art.m.q_free_start
    q0[:, fs] = 1.0
    l = WALKER_P["l"]
    r_f = WALKER_P["r_foot"]
    q0[:, fs + 6] = l * math.cos(full[0]) + r_f
    q0[:, sim.art._qs[1]] = full[0]
    q0[:, sim.art._qs[2]] = full[2]
    w0[:, sim.art._vs[1]] = full[1]
    w0[:, sim.art._vs[2]] = full[3]
    return q0, w0


if __name__ == "__main__":
    main()
