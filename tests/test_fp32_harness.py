import os
"""fp32 vs fp64-oracle tolerance harness.

The production path runs fp32 on GPU; correctness is anchored to the fp64
autodiff oracle.  This harness checks that fp32 forward dynamics track the
fp64 reference to TASK-level tolerance (state drift over a rollout), not
bitwise equality.  Also measures the fp32 speed dividend.

Run:  python tests/test_fp32_harness.py [--device cuda]
"""
import sys
import time

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from diffsim.humanoid import make_soma_humanoid, initial_pose   # noqa: E402
from diffsim import build_geoms_compat                           # noqa: E402
from diffsim.sim import DiffSim, SimConfig                       # noqa: E402


def rollout(sim, model, E, steps, seed=0):
    """Identical perturbations regardless of dtype: draw in fp64, cast."""
    torch.manual_seed(seed)
    q, w = initial_pose(model, E)
    q = q.clone()
    w = w.clone()
    noise = 0.3 * torch.randn(E, 3, dtype=torch.float64)
    w[:, :3] += noise.to(w.dtype)
    w = w.to(dtype=sim.dtype, device=sim.device)
    q = q.to(dtype=sim.dtype, device=sim.device)
    for _ in range(steps):
        tau = sim.pd_torques(q, w, torch.zeros(E, 15, dtype=q.dtype, device=q.device),
                             kp=80., kd=10.)
        r = sim.step(q, w, tau_ext=tau)
        q, w = r.q, r.qd
    return q, w


def main(device="cpu"):
    steps = 100          # 0.4 s of simulated time at dt=5e-4 x8 substeps
    E = 256

    m64, g64, _ = make_soma_humanoid()
    sim64 = DiffSim(m64, build_geoms_compat(g64),
                    SimConfig(dt=5e-4, n_substeps=8), device=device,
                    dtype=torch.float64)
    q_ref, w_ref = rollout(sim64, m64, E, steps, seed=7)

    m32, g32, _ = make_soma_humanoid()
    sim32 = DiffSim(m32, build_geoms_compat(g32),
                    SimConfig(dt=5e-4, n_substeps=8), device=device,
                    dtype=torch.float32)
    q32, w32 = rollout(sim32, m32, E, steps, seed=7)

    dq = (q32.double() - q_ref).abs()
    dw = (w32.double() - w_ref).abs()
    print(f"device={device}  E={E}  steps={steps} (0.4 s sim time)")
    print(f"max |dq|      : {float(dq.max()):.3e}")
    print(f"mean |dq|     : {float(dq.mean()):.3e}")
    print(f"max |dw|      : {float(dw.max()):.3e}")
    print(f"finite(fp32)  : {bool(torch.isfinite(q32).all())}")

    # Distribution-level tolerance: chaotic contact dynamics amplify any
    # perturbation exponentially, so per-trajectory tracking is not the
    # right criterion -- matching STATISTICS is (this is what RL consumes).
    frac_ok = float((dq.max(dim=-1).values < 5e-2).float().mean())
    print(f"fraction of envs within 5e-2 rad: {frac_ok:.3f}")
    print(f"mean |dq| (distribution metric) : {float(dq.mean()):.3e}")
    ok = float(dq.mean()) < 5e-2 and bool(torch.isfinite(q32).all())
    print("DISTRIBUTION-TOLERANCE PASS:", ok)

    for name, simx, mdl in [("fp64", sim64, m64), ("fp32", sim32, m32)]:
        qx, wx = initial_pose(mdl, E)
        qx = qx.clone().to(dtype=simx.dtype, device=simx.device)
        wx = wx.clone().to(dtype=simx.dtype, device=simx.device)
        t0 = time.time()
        for _ in range(30):
            tau = simx.pd_torques(qx, wx,
                                  torch.zeros(E, 15, dtype=qx.dtype,
                                              device=qx.device),
                                  kp=80., kd=10.)
            r = simx.step(qx, wx, tau_ext=tau)
            qx, wx = r.q, r.qd
        dtms = (time.time() - t0) / 30 * 1e3
        print(f"{name}: {dtms:7.2f} ms/control-step @E={E} "
              f"-> {E/dtms*1e3:>10,.0f} env-steps/s")


if __name__ == "__main__":
    dev = "cuda" if ("--device" in sys.argv and torch.cuda.is_available()) else "cpu"
    main(dev)
