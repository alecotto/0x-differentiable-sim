import os
"""Throughput benchmark: environment-steps/sec vs batch size E.

Measures steady-state control-step time (8 physics substeps each) with the
production PD controller attached, reporting:

    E      ms/ctrl-step   env-steps/s

Also compares the analytic Coriolis path against the autodiff oracle.
fp64 CPU here; the tensor code is device-agnostic (CUDA-ready).
"""
import sys
import time

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from diffsim.humanoid import make_soma_humanoid, initial_pose   # noqa: E402
from diffsim import build_geoms_compat                           # noqa: E402
from diffsim.sim import DiffSim, SimConfig                       # noqa: E402

DT = torch.float64


def bench(E, n_steps=30, analytic=True):
    model, gspec, _ = make_soma_humanoid()
    sim = DiffSim(model, build_geoms_compat(gspec),
                  SimConfig(dt=5e-4, n_substeps=1,
                            use_analytic_bias=analytic), dtype=DT)
    sim.ground_idx = sim.ground_idx[:0]
    sim.ground_body = sim.ground_body[:0]
    sim.pair_i = sim.pair_i[:0]
    sim.pair_j = sim.pair_j[:0]

    q, w = initial_pose(model, E)
    qt = torch.zeros(E, 15, dtype=DT)

    # warmup
    for _ in range(5):
        tau = sim.pd_torques(q, w, qt, kp=80., kd=10.)
        r = sim.step(q, w, tau_ext=tau)
        q, w = r.q, r.qd

    t0 = time.time()
    for _ in range(n_steps):
        tau = sim.pd_torques(q, w, qt, kp=80., kd=10.)
        r = sim.step(q, w, tau_ext=tau)
        q, w = r.q, r.qd
    dt = (time.time() - t0) / n_steps
    return dt * 1e3, E / dt


def main():
    print(f"{'E':>6} {'ms/ctrl':>10} {'env-steps/s':>14}   "
          f"{'(oracle ms)':>11}")
    for E in [64, 256, 1024, 4096]:
        ms_a, eps_a = bench(E, analytic=True)
        if E <= 1024:
            ms_o, _ = bench(E, n_steps=10, analytic=False)
            extra = f"   ({ms_o:11.1f})"
        else:
            extra = f"   ({'—':>11})"
        print(f"{E:>6} {ms_a:>10.2f} {eps_a:>14,.0f}   {extra}")


if __name__ == "__main__":
    main()
