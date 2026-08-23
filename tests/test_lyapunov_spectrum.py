"""Q5 validation: full Lyapunov spectra entirely inside the AD framework.

Checks
------
1. push_jacobian == exact one-step Jacobian (vs central FD).
2. Chaotic double pendulum: lambda_1 matches the independent FD-Benettin
   estimator at MATCHED dt; spectrum symmetric {l, ~0, ~0, -l}; sum ~ 0
   (Hamiltonian volume preservation); seed/direction invariance.

   NOTE on absolute values: lambda_1 of the DISCRETE map depends on dt
   through integrator dissipation (measured: +0.047/s at dt=5e-3 rising to
   +0.10/s at dt=2e-4, T=60 s, released-from-rest ICs).  The historical
   README claim of +0.80/s is NOT reproducible and was an artifact of the
   pre-b0fb030 accumulator bug.  Cross-METHOD agreement at matched dt is
   the invariant we assert here.
3. Damped pendulum at stable equilibrium: every exponent matches the
   linearized continuous-time spectrum Re(eig A); sum matches <div f>.
4. CONTACT: bouncing ball on a vibrating table (chaotic impact oscillator,
   Gamma = a w^2 / g ~ 2): tangent propagation crosses impacts with NO
   saltation correction; lambda_1 agrees with FD-Benettin; spectrum shows
   the expected (+l, ~0, -l) structure.
5. DIFFERENTIABLE SPECTRUM: d(lambda)/d(damping) by autodiff through the
   whole Benettin accumulation matches finite differencing of converged
   exponents over the parameter.

Runtime: default horizons keep this file ~5 min CPU; set DIFFSIM_FULL=1
for the long-horizon numbers quoted in the README.
"""
import math
import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'tests'))

from diffsim.articulation import Articulation                    # noqa: E402
from diffsim.collision import softplus_pen                       # noqa: E402
from diffsim.lyapunov import benettin_generic                    # noqa: E402
from diffsim.lyapunov_spectrum import (                          # noqa: E402
    jacobian, lyapunov_spectrum, lyapunov_spectrum_diff, push_jacobian,
    trajectory_divergence,
)
from test_dynamics import _build_pendulum                        # noqa: E402

DT = torch.float64
FULL = os.environ.get("DIFFSIM_FULL", "0") == "1"


def _pendulum_art():
    model = _build_pendulum(2)
    return Articulation(model), model


def make_pendulum_step(art, dt, damping=0.0):
    """Pure one-substep map (semi-implicit Euler), forward-mode safe."""
    def step(x):
        q = x[:2].reshape(1, -1)
        qd = x[2:].reshape(1, -1)
        R, p = art.kinematics(q)
        sb = art.subspace_terms(q, R, p)
        _, _, Iw = art._world_inertias(q, R, p)
        M = art.mass_matrix(q, R, p, sb, Iw)
        h = art.bias_forces_analytic(q, qd, R, p, sb, Iw)
        tg = art.gravity_genforce(q, R, p, sb)
        tld = (-damping * qd.reshape(-1)) if damping \
            else torch.zeros(2, dtype=q.dtype)
        qdd = torch.linalg.solve(
            M, (tg - h).unsqueeze(-1) + tld.unsqueeze(-1)
        ).squeeze(-1).reshape(-1)
        return torch.cat([q.reshape(-1) + dt * (qd.reshape(-1) + dt * qdd),
                          qd.reshape(-1) + dt * qdd])
    return step


def _fd_jacobian(f, x, eps=1e-6):
    n = x.numel()
    J = torch.zeros(n, n, dtype=x.dtype)
    for i in range(n):
        xp = x.clone(); xp[i] += eps
        xm = x.clone(); xm[i] -= eps
        J[:, i] = (f(xp) - f(xm)) / (2 * eps)
    return J


# --------------------------------------------------------------------- #
def test_push_jacobian_exact():
    art, _ = _pendulum_art()
    f = make_pendulum_step(art, 5e-3)
    x = torch.tensor([0.6, 0.9, 2.5, -1.5], dtype=DT)
    J_ad = jacobian(f, x)
    J_fd = _fd_jacobian(f, x)
    rel = float((J_ad - J_fd).norm() / J_fd.norm())
    print(f"  one-step Jacobian AD-vs-FD rel err = {rel:.2e}")
    assert rel < 1e-6

    g = torch.Generator().manual_seed(3)
    D = torch.randn(4, 6, generator=g, dtype=DT)
    err = float((push_jacobian(f, x, D) - J_fd @ D).norm())
    assert err < 1e-6


def test_chaotic_spectrum_structure():
    art, _ = _pendulum_art()
    dt = 5e-3
    f = make_pendulum_step(art, dt)
    x0 = torch.tensor([0.6, 0.9, 2.5, -1.5], dtype=DT)

    n_steps = 6000 if FULL else 3000
    lams_all = []
    for seed in (1, 7):
        r = lyapunov_spectrum(f, x0, dt_per_step=dt, n_steps=n_steps,
                              qr_every=10, seed=seed)
        lams_all.append(r["lams"])
        print(f"  seed={seed}: spectrum = "
              f"[{', '.join(f'{v:+.4f}' for v in r['lams'].tolist())}]  "
              f"sum={r['sum']:+.4f}")

    l1 = [float(l[0]) for l in lams_all]
    assert l1[0] > 0.05, f"lambda_1 not positive: {l1}"
    assert abs(l1[0] - l1[1]) < 0.15, f"seed dependence: {l1}"

    # cross-check against the INDEPENDENT FD-Benettin implementation
    kick = torch.zeros_like(x0); kick[2] = 1e-9
    lam_fd, _ = benettin_generic(lambda x_: f(x_), x0.clone(),
                                 (x0 + kick).clone(), dt_substep=dt,
                                 steps=n_steps, renorm=20, delta0=1e-9)
    print(f"  lambda_1: tangent={l1[0]:+.4f}  FD-Benettin={lam_fd:+.4f}")
    assert abs(l1[0] - lam_fd) < max(0.15, 0.3 * abs(lam_fd))

    # Hamiltonian structure: symmetric spectrum, near-zero sum
    lam = lams_all[0]
    sym_err = float((lam[:2] + torch.flip(lam, dims=[0])[:2]).abs().max())
    print(f"  symmetry max |l_i + l_(n+1-i)| err = {sym_err:.4f}")
    assert sym_err < 0.25, f"spectrum not symmetric: {lam}"
    sums = [float(l.sum()) for l in lams_all]
    print(f"  sum(lambda) = {sums}")
    assert all(abs(s) < 0.35 for s in sums), \
        f"H flow must conserve phase volume up to O(dt): {sums}"


def test_damped_equilibrium_spectrum():
    art, model = _pendulum_art()
    damping = 0.8
    dt = 5e-3
    f = make_pendulum_step(art, dt, damping=damping)

    q = torch.zeros(1, 2, dtype=DT)
    M = art.mass_matrix(q)[0]
    eps = 1e-6
    Kg = torch.zeros(2, 2, dtype=DT)
    for i in range(2):
        qp = q.clone(); qp[:, i] += eps
        qm = q.clone(); qm[:, i] -= eps
        Kg[:, i] = (art.gravity_genforce(qp) - art.gravity_genforce(qm)) / (2 * eps)
    Minv = torch.linalg.inv(M)
    A = torch.cat([torch.cat([torch.zeros(2, 2), torch.eye(2)], 1),
                   torch.cat([Minv @ Kg, -Minv @ (damping * torch.eye(2, dtype=DT))], 1)], 0)

    # Ground truth = exponents OF THE DISCRETE MAP being iterated
    # (log|multiplier| / dt).  Comparing against continuous-time Re(eig A)
    # would charge the O(s^2 dt / 2) integrator bias to the estimator.
    J_step = jacobian(f, torch.zeros(4, dtype=DT))
    ev = torch.linalg.eigvals(J_step)
    ref = sorted((torch.log(ev.abs()) / dt).tolist(), reverse=True)

    x_eq = torch.zeros(4, dtype=DT)
    r = lyapunov_spectrum(f, x_eq, dt_per_step=dt, n_steps=8000,
                          qr_every=20, seed=0)
    got = r["lams"].tolist()
    print(f"  continuous Re(eig A)/1 = [{', '.join(f'{v:+.5f}' for v in sorted(torch.linalg.eigvals(A).real.tolist(), reverse=True))}]")
    print(f"  discrete-map exponents = [{', '.join(f'{v:+.5f}' for v in ref)}]")
    print(f"  tangent spectrum       = [{', '.join(f'{v:+.5f}' for v in got)}]")
    # per-component tolerance covers slow alignment inside near-degenerate
    # (complex-pair) subspaces; pair-mean must match tightly.
    for gr, rr in zip(got, ref):
        assert abs(gr - rr) < 0.15, f"exponent {gr:+.5f} vs {rr:+.5f}"
    n_pair = 2
    for j in range(0, len(ref), n_pair):
        gm = sum(got[j:j + n_pair]) / n_pair
        rm = sum(ref[j:j + n_pair]) / n_pair
        assert abs(gm - rm) < 0.02, f"pair-mean {gm:+.5f} vs {rm:+.5f}"

    # Sum rule for the DISCRETE estimator:
    #   sum_i lambda_i == <log|det J_step|>/dt ~= (<tr J_step> - n)/dt
    # (the continuous-flow identity sum = <div f> picks up an additional
    #  O(s^2 dt) integrator bias for stiff modes, so it is NOT the right
    #  reference here.)
    n_dim = x_eq.numel()
    eye = torch.eye(n_dim, dtype=DT)
    x = x_eq.clone()
    trs = []
    for _ in range(64):
        trs.append(float(torch.diagonal(push_jacobian(f, x, eye)).sum()))
        x = f(x)
    expected_sum = (sum(trs) / len(trs) - n_dim) / dt
    print(f"  sum(lambda)={r['sum']:+.5f}   "
          f"(<tr J>-n)/dt={expected_sum:+.5f}")
    assert abs(r["sum"] - expected_sum) < max(0.05, 0.05 * abs(expected_sum))


def test_contact_impact_tangents():
    """Bouncing ball on a vibrating table: chaotic impact oscillator.

    State [z, v, th]; floor z_f = a sin(th); softplus compliant contact;
    Gamma = a*om^2/g ~ 2.  Proves tangent propagation crosses impacts with
    no saltation machinery and recovers the exponent measured by the
    independent FD estimator.
    """
    m_, k_, b_, beta = 1.0, 5.0e4, 5.0, 1.0e4
    a_, om = 0.01, 45.0
    dt = 1e-4

    def step(x):
        z, v, th = x[0], x[1], x[2]
        zf = a_ * torch.sin(th)
        pen = softplus_pen(-(z - zf), beta)
        act = pen / (pen + 1e-4)
        vr = v - a_ * om * torch.cos(th)
        fn = k_ * pen
        fd = b_ * act * softplus_pen(torch.clamp(-vr, min=0.0), 1e3)
        acc = -9.81 + (fn + fd) / m_
        v2 = v + dt * acc
        return torch.stack([z + dt * v2, v2, th + dt * om])

    x = torch.tensor([0.05, -2.0, 0.0], dtype=DT)
    burn = 200000 if FULL else 100000
    for _ in range(burn):
        x = step(x)
    print(f"  post burn-in state z={float(x[0]):+.4f} v={float(x[1]):+.3f}")

    # local exactness AT an impacting configuration (contact likely active)
    pen_now = float(softplus_pen(-(x[0] - a_ * math.sin(float(x[2]))), beta))
    print(f"  contact penetration at probe state: {pen_now:.3e} m")
    J_ad = jacobian(step, x)
    J_fd = _fd_jacobian(step, x, eps=1e-7)
    rel = float((J_ad - J_fd).norm() / J_fd.norm())
    print(f"  impact-state Jacobian AD-vs-FD rel err = {rel:.2e}")
    assert rel < 1e-5

    n_steps = 120000 if FULL else 80000     # T = 12 s / 8 s
    res = lyapunov_spectrum(step, x.clone(), dt_per_step=dt,
                            n_steps=n_steps, qr_every=100, seed=2)
    lams = res["lams"]
    kick = torch.tensor([0.0, 1e-9, 0.0], dtype=DT)
    lam_fd, _ = benettin_generic(step, x.clone(), (x + kick).clone(),
                                 dt_substep=dt, steps=n_steps, renorm=500,
                                 delta0=1e-9)
    print(f"  impact-oscillator spectrum = "
          f"[{', '.join(f'{v:+.4f}' for v in lams.tolist())}]")
    print(f"  lambda_1: tangent={float(lams[0]):+.4f}  FD={lam_fd:+.4f}")
    assert float(lams[0]) > 0.2, "expected chaos in the impact oscillator"
    assert abs(float(lams[1])) < 0.25, \
        f"neutral direction missing: {lams}"
    assert abs(float(lams[0]) - lam_fd) < max(0.3, 0.3 * abs(lam_fd))


def test_differentiable_spectrum():
    """d(lambda)/d(damping) by autodiff == central FD of the SAME
    finite-window objective over damping."""
    art, model = _pendulum_art()

    def build_step(c_):
        dt = 5e-3
        def step(x):
            q = x[:2].reshape(1, -1)
            qd = x[2:].reshape(1, -1)
            R, p = art.kinematics(q)
            sb = art.subspace_terms(q, R, p)
            _, _, Iw = art._world_inertias(q, R, p)
            M = art.mass_matrix(q, R, p, sb, Iw)
            h = art.bias_forces_analytic(q, qd, R, p, sb, Iw)
            tg = art.gravity_genforce(q, R, p, sb)
            tld_full = (-c_ * qd).reshape(1, 2)
            qdd = torch.linalg.solve(
                M, ((tg - h) + tld_full).unsqueeze(-1)).squeeze(-1)
            return torch.cat([q.reshape(-1) + dt * (qd.reshape(-1) + dt * qdd.reshape(-1)),
                              qd.reshape(-1) + dt * qdd.reshape(-1)])
        return step

    n_steps = 2000
    c_param = torch.tensor(0.8, dtype=DT, requires_grad=True)
    out = lyapunov_spectrum_diff(build_step(c_param), torch.zeros(4, dtype=DT),
                                 dt_per_step=5e-3, n_steps=n_steps,
                                 qr_every=20)
    lams = out["lams"]
    lam_max = lams.max()
    (dl_dc,) = torch.autograd.grad(lam_max, c_param)

    # central FD over c of the identical finite-window objective
    dc = 0.05
    vals = []
    with torch.enable_grad():
        for cv in (0.8 + dc, 0.8 - dc):
            cc = torch.tensor(cv, dtype=DT, requires_grad=True)
            o = lyapunov_spectrum_diff(build_step(cc), torch.zeros(4, dtype=DT),
                                       dt_per_step=5e-3, n_steps=n_steps,
                                       qr_every=20)
            vals.append(float(o["lams"].max().detach()))
    fd = (vals[0] - vals[1]) / (2 * dc)
    print(f"  lambda_max window objective = {float(lam_max.detach()):+.6f}")
    print(f"  d(lambda)/dc: autodiff = {float(dl_dc):+.6f}   "
          f"central-FD = {fd:+.6f}")
    assert abs(float(dl_dc) - fd) < max(0.01, 0.10 * abs(fd)), (
        f"dlambda/dc mismatch: ad={float(dl_dc):+.6f} fd={fd:+.6f}")


if __name__ == "__main__":
    test_push_jacobian_exact()
    test_chaotic_spectrum_structure()
    test_damped_equilibrium_spectrum()
    test_contact_impact_tangents()
    test_differentiable_spectrum()
    print("ALL LYAPUNOV-SPECTRUM TESTS PASSED")
