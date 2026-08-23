"""Validate the Benettin estimator against KNOWN answers.

1. Damped double pendulum at stable equilibrium: the largest Lyapunov
   exponent equals max Re(eigenvalue) of the linearized system -- exactly
   computable from M, K, D.  This is a closed-form ground truth.
2. Undamped chaotic regime: lambda_1 must be strictly positive, and the
   Benettin estimate must agree with an independent naive divergence-
   slope measurement.

These are the missing ladder rung: an estimator that has never been
validated against a known value is not an estimator.
"""
import math
import sys

import torch

sys.path.insert(0, '/Code/0x-differentiable-sim-project')
sys.path.insert(0, '/Code/0x-differentiable-sim-project/tests')

from diffsim.articulation import Articulation          # noqa: E402
from diffsim.lyapunov import benettin_generic, delta0_invariance_check  # noqa: E402
from test_dynamics import _build_pendulum              # noqa: E402

DT = torch.float64


def make_pendulum_step(art, damping=0.0):
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
        tld = (-damping * qd.reshape(-1)) if damping else 0.0
        qdd = torch.linalg.solve(M, (tg - h).unsqueeze(-1) + tld.unsqueeze(-1)).squeeze(-1).reshape(-1)
        return torch.cat([qd.reshape(-1), qdd])
    return step


def analytic_max_exponent(art, model, damping):
    """max Re eig of linearized A at hanging equilibrium."""
    q = torch.zeros(1, 2, dtype=DT)
    R, p = art.kinematics(q)
    sb = art.subspace_terms(q, R, p)
    M = art.mass_matrix(q, R, p, sb)[0]
    eps = 1e-6
    Kg = torch.zeros(2, 2, dtype=DT)
    for i in range(2):
        qp = q.clone(); qp[:, i] += eps
        qm = q.clone(); qm[:, i] -= eps
        Kg[:, i] = (art.gravity_genforce(qp) - art.gravity_genforce(qm)) / (2 * eps)
    D = damping * torch.eye(2, dtype=DT)
    # A = [[0, I], [-M^-1 Kg, -M^-1 (D + joint_damping)]]
    Dtot = D + (model.damping.diag() if model.damping is not None else 0)
    Minv = torch.linalg.inv(M)
    A = torch.cat([torch.cat([torch.zeros(2, 2), torch.eye(2)], 1),
                   torch.cat([-Minv @ Kg, -Minv @ Dtot], 1)], 0)
    eigs = torch.linalg.eigvals(A)
    return float(eigs.real.max())


def test_benettin_matches_analytic_damped_equilibrium():
    """Damped pendulum at rest: lambda_1 == max Re(linearized eig)."""
    damping = 0.8
    model = _build_pendulum(2)
    model.damping = damping * torch.ones(2, dtype=DT)
    art = Articulation(model)
    step = make_pendulum_step(art, damping=damping + float(model.damping[0]))

    dt_sub = 1e-3
    x_base = torch.zeros(4, dtype=DT)          # hanging equilibrium

    kick_dir = torch.tensor([0., 0., 1., 0.5], dtype=DT)
    lams = delta0_invariance_check(step, x_base, dt_sub, steps=400,
                                   renorm=10, kick_dir=kick_dir)

    lam_true = analytic_max_exponent(art, model, damping)
    for lam in lams:
        assert abs(lam - lam_true) < 0.05 * max(abs(lam_true), 0.5), (
            f"benettin {lam:.4f} vs analytic {lam_true:.4f}")
    print(f"  [ok] damped equilibrium: benettin={lams} analytic={lam_true:.5f}")


def test_chaotic_regime_positive_lambda():
    model = _build_pendulum(2)
    art = Articulation(model)
    step = make_pendulum_step(art, damping=0.0)

    # energetic chaotic initial condition
    x1 = torch.tensor([0.6, 0.9, 0.0, 0.0], dtype=DT)
    E_target = None
    # pump energy: set velocities so total energy is high
    x1[2:] = torch.tensor([2.5, -1.5], dtype=DT)

    dt_sub = 1e-4
    renorm = 20                                 # every 2 ms
    delta0 = 1e-9

    lam, hist = benettin_generic(step, x1.clone(),
                                 x1 + torch.tensor([0., 0., 1e-9, 0.], dtype=DT),
                                 dt_sub, steps=20000, renorm=renorm,
                                 delta0=delta0)
    print(f"  [info] chaotic regime lambda_1 = {lam:+.4f} /s")
    assert lam > 0.1, f"expected positive lambda for chaotic double pendulum, got {lam}"

    # independent cross-check: naive two-trajectory slope over pre-saturation
    y1 = x1.clone()
    y2 = x1 + torch.tensor([0., 0., 1e-12, 0.], dtype=DT)
    slopes = []
    t = 0.0
    for i in range(60000):
        y1 = step(y1); y2 = step(y2); t += 1e-5
        if (i + 1) % 5000 == 0:
            d = float(torch.linalg.vector_norm(y2 - y1))
            if d > 1e2:
                break
            slopes.append(math.log(d / 1e-12) / t)
    naive = slopes[-1] if slopes else float('nan')
    print(f"  [info] naive divergence slope ~ {naive:+.4f} /s")
    assert abs(naive - lam) < max(0.35 * abs(lam), 0.5), (
        f"benettin {lam:.4f} vs naive {naive:.4f}")


if __name__ == "__main__":
    test_benettin_matches_analytic_damped_equilibrium()
    test_chaotic_regime_positive_lambda()
    print("ALL LYAPUNOV VALIDATION TESTS PASSED")
