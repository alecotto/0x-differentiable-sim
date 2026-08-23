import os
"""Validate the Benettin estimator against KNOWN answers.

1. Damped double pendulum at stable equilibrium: largest Lyapunov exponent
   equals max Re(eig(A)) of the linearized system -- closed-form ground truth.
2. Undamped chaotic regime: lambda_1 > 0; Benettin agrees with independent
   naive divergence slope using the SAME dt.
"""
import math

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys_path_tmp = os.path.join(_ROOT, 'tests')

import sys
sys.path.insert(0, _ROOT)
sys.path.insert(0, sys_path_tmp)

from diffsim.articulation import Articulation          # noqa: E402
from diffsim.lyapunov import benettin_generic, delta0_invariance_check  # noqa: E402
from test_dynamics import _build_pendulum              # noqa: E402

DT = torch.float64


def make_pendulum_step(art, dt=1e-3, damping=0.0):
    @torch.no_grad()
    def step(x):
        q = x[:2].reshape(1, -1)
        qd = x[2:].reshape(1, -1)
        R, p = art.kinematics(q)
        sb = art.subspace_terms(q, R, p)
        _, _, Iw = art._world_inertias(q, R, p)
        M = art.mass_matrix(q, R, p, sb, Iw)
        h = art.bias_forces_analytic(q, qd, R, p, sb, Iw)
        tg = art.gravity_genforce(q, R, p, sb)
        tld = (-damping * qd.reshape(-1)) if damping else torch.zeros(2, dtype=q.dtype)
        qdd = torch.linalg.solve(M, (tg - h).unsqueeze(-1) + tld.unsqueeze(-1)).squeeze(-1).reshape(-1)
        qd_new = qd.reshape(-1) + dt * qdd
        q_new = q.reshape(-1) + dt * qd_new
        return torch.cat([q_new, qd_new])
    return step


def analytic_max_exponent(art, model, damping):
    """max Re eig of linearized A at hanging equilibrium."""
    q = torch.zeros(1, 2, dtype=DT)
    M = art.mass_matrix(q)[0]
    eps = 1e-6
    Kg = torch.zeros(2, 2, dtype=DT)
    for i in range(2):
        qp = q.clone(); qp[:, i] += eps
        qm = q.clone(); qm[:, i] -= eps
        Kg[:, i] = (art.gravity_genforce(qp) - art.gravity_genforce(qm)) / (2 * eps)
    D = damping * torch.eye(2, dtype=DT)
    Minv = torch.linalg.inv(M)
    A = torch.cat([torch.cat([torch.zeros(2, 2), torch.eye(2)], 1),
                   torch.cat([+Minv @ Kg, -Minv @ D], 1)], 0)
    eigs = torch.linalg.eigvals(A)
    return float(eigs.real.max())


def test_benettin_matches_analytic_damped_equilibrium():
    """Damped pendulum at rest: lambda_1 == max Re(linearized eig).

    Uses dt=1e-3 for BOTH the step function and benettin_generic so the
    time bookkeeping is consistent.
    """
    damping = 0.8
    dt = 1e-4
    model = _build_pendulum(2)
    art = Articulation(model)
    # NOTE: do NOT set model.damping — damping is applied ONLY via the
    # step function's explicit `tld` term; the analytic ground truth must
    # use the same single source of damping.
    step = make_pendulum_step(art, dt=dt, damping=damping)

    x_base = torch.zeros(4, dtype=DT)
    kick_dir = torch.tensor([0., 0., 1., 0.5], dtype=DT)

    lam_true = analytic_max_exponent(art, model, damping)
    print(f"  analytic lam_max_re = {lam_true:+.5f} /s")

    lams = []
    for d0 in [1e-6, 1e-9]:
        x2 = x_base + kick_dir / kick_dir.norm() * d0
        lam, _ = benettin_generic(step, x_base.clone(), x2.clone(),
                                  dt_substep=dt, steps=500, renorm=25,
                                  delta0=d0)
        lams.append(lam)
        print(f"  benettin(d0={d0:.0e}) = {lam:+.5f} /s")
        assert abs(lam - lam_true) < max(0.05 * abs(lam_true), 0.05), (
            f"benettin {lam:.5f} vs analytic {lam_true:.5f}")

    # delta0 invariance
    spread = (max(lams) - min(lams)) / max(abs(min(lams)), 1e-12)
    assert spread < 0.35, f"lambda tracks delta0: spread={spread:.2f}"
    print(f"  delta0 invariance: spread={spread:.3f} PASS")


def test_chaotic_regime_positive_lambda():
    model = _build_pendulum(2)
    art = Articulation(model)
    dt = 1e-4
    step = make_pendulum_step(art, dt=dt, damping=0.0)

    x1 = torch.tensor([0.6, 0.9, 2.5, -1.5], dtype=DT)
    kick = torch.tensor([0., 0., 1e-9, 0.], dtype=DT)

    lam, hist = benettin_generic(step, x1.clone(), x1 + kick,
                                 dt_substep=dt, steps=20000, renorm=20,
                                 delta0=1e-9)
    print(f"  chaotic lambda_1 = {lam:+.4f} /s")
    assert lam > 0.1

    # independent naive divergence slope (same dt!)
    y1 = x1.clone()
    y2 = x1 + kick
    t = 0.0
    slopes = []
    for i in range(60000):
        y1 = step(y1); y2 = step(y2); t += dt
        if (i + 1) % 5000 == 0:
            d = float(torch.linalg.vector_norm(y2 - y1))
            if d > 1e2:
                break
            slopes.append(math.log(d / 1e-9) / t)
    naive = slopes[-1] if slopes else float('nan')
    print(f"  naive divergence slope ~ {naive:+.4f} /s")
    assert abs(naive - lam) < max(0.35 * abs(lam), 0.5)


if __name__ == "__main__":
    test_benettin_matches_analytic_damped_equilibrium()
    test_chaotic_regime_positive_lambda()
    print("ALL LYAPUNOV VALIDATION TESTS PASSED")
