"""Decisive split for humanoid tumble divergence: force correctness vs
integrator stability.

Runs RK4 (tiny dt) and semi-implicit Euler on the SAME closed-loop ODE
(zero gravity, no contacts, joints initialized within limits) and tracks
mechanical energy E = 1/2 qd^T M(q) qd.

* RK4 conserves, Euler explodes -> integrator stability problem.
* RK4 also drifts              -> the force field itself is non-conservative.
"""
import sys

import torch

sys.path.insert(0, '/Code/0x-differentiable-sim-project')
from diffsim.humanoid import make_soma_humanoid          # noqa: E402
from diffsim import build_geoms_compat                    # noqa: E402
from diffsim.sim import DiffSim, SimConfig                # noqa: E402

DT = torch.float64


def main():
    mh, gspec, feet = make_soma_humanoid()
    mh.gravity = (0., 0., 0.)
    geoms = build_geoms_compat(gspec)
    cfg = SimConfig(dt=1e-4, n_substeps=1)
    cfg.max_vel = None
    sim = DiffSim(mh, geoms, cfg, dtype=DT)
    sim.ground_idx = sim.ground_idx[:0]
    sim.ground_body = sim.ground_body[:0]
    sim.pair_i = sim.pair_i[:0]
    sim.pair_j = sim.pair_j[:0]
    art = sim.art

    def deriv(q, qd):
        R, p = art.kinematics(q)
        sub = art.subspace_terms(q, R, p)
        _, _, Iw = art._world_inertias(q, R, p)
        M = art.mass_matrix(q, R, p, sub, Iw)
        h = art.bias_forces_analytic(q, qd, R, p, sub, Iw)
        qdd = torch.linalg.solve(M, (-h).unsqueeze(-1)).squeeze(-1)
        return art._qspace_rate(q, qd), qdd

    def energy(q, qd):
        R, p = art.kinematics(q)
        M = art.mass_matrix(q, R, p)
        return float((0.5 * (qd.unsqueeze(1) @ M @ qd.unsqueeze(2)).reshape(())))

    torch.manual_seed(5)
    q0 = torch.zeros(1, mh.q_dim, dtype=DT)
    q0[:, mh.q_free_start] = 1.0
    q0[:, mh.q_free_start + 6] = 1.2
    q0[:, 7:] = (torch.rand(1, mh.q_dim - 7, dtype=DT) * 2 - 1) * 0.08
    w0 = torch.zeros(1, mh.v_dim, dtype=DT)
    w0[0, :3] = torch.tensor([1.0, 2.0, 3.0])
    E0 = energy(q0, w0)
    print(f"E0 = {E0:.6f}")

    T = 0.02   # short horizon: the Euler run reached +900 J by 10 ms

    # ---- RK4 ----------------------------------------------------------
    for dtv in [2e-4, 5e-5]:
        qq, ww = q0.clone(), w0.clone()
        n = int(T / dtv)
        mx = 0.0
        for i in range(n):
            k1q, k1w = deriv(qq, ww)
            k2q, k2w = deriv(qq + 0.5 * dtv * k1q, ww + 0.5 * dtv * k1w)
            k3q, k3w = deriv(qq + 0.5 * dtv * k2q, ww + 0.5 * dtv * k2w)
            k4q, k4w = deriv(qq + dtv * k3q, ww + dtv * k3w)
            qq = qq + dtv / 6 * (k1q + 2 * k2q + 2 * k3q + k4q)
            ww = ww + dtv / 6 * (k1w + 2 * k2w + 2 * k3w + k4w)
            mx = max(mx, abs(energy(qq, ww) - E0))
            if not torch.isfinite(qq).all():
                print(f"RK4 dt={dtv:.0e}: BLEW UP at i={i}")
                break
        else:
            print(f"RK4  dt={dtv:.0e} T={T}s: max|dE|={mx:.3e}")

    # ---- semi-implicit Euler ------------------------------------------
    qq, ww = q0.clone(), w0.clone()
    dtv = 1e-4
    n = int(0.01 / dtv)
    for i in range(n):
        _, qdd = deriv(qq, ww)
        ww = ww + dtv * qdd
        qq = art.integrate(qq, ww, dtv)
    print(f"Euler dt={dtv:.0e} T=0.01s: dE={energy(qq,ww)-E0:+.3e}")


if __name__ == "__main__":
    main()
