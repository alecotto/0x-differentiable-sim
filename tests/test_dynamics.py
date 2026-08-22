"""Energy-conservation and analytic checks for articulated dynamics.

A conservative system integrated with the internal force computation must
conserve E = KE + PE up to symplectic-Euler bounded oscillation.  The
single pendulum exercises M and gravity; the double pendulum additionally
exercises the Coriolis/gyroscopic bias path (h != 0), which is where most
sign errors in spatial dynamics live.
"""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diffsim.articulation import Articulation, Model

DT = torch.float64


def _build_pendulum(n_links: int):
    """Chain of n_links hanging rods, hinged about +y, attached to a fixed
    anchor body. Rod i has length L, mass m; COM at local [0,0,-L/2]."""
    L, m = 1.0, 1.0
    Icom = torch.diag(torch.tensor([m * L * L / 12.0] * 2 + [1e-4], dtype=DT))
    B = []
    B.append(("anchor", -1, 0, (0.0, 1.0, 0.0), torch.zeros(3, dtype=DT),
              1e6, torch.zeros(3, dtype=DT), torch.eye(3, dtype=DT) * 1e6))
    parent = [-1]
    for i in range(n_links):
        # first rod hangs from the anchor; subsequent rods attach at the
        # previous rod's tip (-L below its frame origin)
        offset = torch.tensor([0.0, 0.0, -(L if i > 0 else 0.0)], dtype=DT)
        B.append((f"rod{i}", len(B) - 1, 1, (0.0, 1.0, 0.0),
                  offset, m, torch.tensor([0.0, 0.0, -L / 2], dtype=DT), Icom))
        parent.append(len(B) - 2)
    model = Model(
        n_bodies=len(B), parent=parent, body_names=[b[0] for b in B],
        fix_R=torch.stack([torch.eye(3, dtype=DT)] * len(B)),
        fix_p=torch.stack([b[4] for b in B]),
        j_type=torch.tensor([b[2] for b in B]),
        j_axis=torch.tensor([b[3] for b in B], dtype=DT),
        masses=torch.tensor([b[5] for b in B], dtype=DT),
        com=torch.stack([b[6] for b in B]),
        inertia_com=torch.stack([b[7] for b in B]),
        q_dim=n_links, v_dim=n_links,
        dof_body=list(range(1, n_links + 1)),
        body_dof_start=[-1] + list(range(n_links)),
        q_free_start=-1,
    )
    return model


def energy(art, model, q, qd):
    R, p = art.kinematics(q)
    M = art.mass_matrix(q, R, p)
    cw = art.com_positions(q, R, p)
    gz = -9.81
    KE = 0.5 * (qd.unsqueeze(1) @ M @ qd.unsqueeze(2)).squeeze(-1).squeeze(-1)
    # U = -m g.c with g = (0,0,gz): only the z component contributes
    PE = -(cw[..., 2] * model.masses * gz).sum(dim=-1)
    return KE + PE


def integrate(art, model, q, qd, T, dtv):
    n = int(T / dtv)
    Es = []
    for _ in range(n):
        R, p = art.kinematics(q)
        M = art.mass_matrix(q, R, p)
        h = art.bias_forces(q, qd, R, p)
        tg = art.gravity_genforce(q, R, p)
        qdd = torch.linalg.solve(M, (tg - h).unsqueeze(-1)).squeeze(-1)
        qd = qd + dtv * qdd
        q = q + dtv * qd
        Es.append(float(energy(art, model, q, qd)[0].detach()))
    return q, qd, Es


def test_single_pendulum_conserves_energy():
    torch.manual_seed(0)
    model = _build_pendulum(1)
    art = Articulation(model)
    # analytic checks at a generic state
    q = torch.tensor([[0.6]], dtype=DT)
    qd = torch.tensor([[1.7]], dtype=DT)
    R, p = art.kinematics(q)
    M = art.mass_matrix(q, R, p)
    assert abs(M.item() - 1.0 / 3.0) < 1e-12
    tg = art.gravity_genforce(q, R, p)
    assert abs(tg.item() - (-9.81 * 0.5 * math.sin(0.6))) < 1e-12

    _, _, Es = integrate(art, model, q.clone(), qd.clone(), T=0.5, dtv=5e-5)
    drift = abs(Es[-1] - Es[0]) / max(abs(Es[0]), 1e-12)
    assert drift < 5e-4, f"single pendulum energy drift {drift:.2e}"


def test_double_pendulum_conserves_energy():
    """Chaotic regime: bias forces (Coriolis/gyroscopic) are strongly active.
    Energy conservation here validates the full h(q,qd) pipeline."""
    torch.manual_seed(0)
    model = _build_pendulum(2)
    art = Articulation(model)
    q = torch.tensor([[0.9, 0.4]], dtype=DT)
    qd = torch.tensor([[1.3, -0.8]], dtype=DT)

    R, p = art.kinematics(q)
    M = art.mass_matrix(q, R, p)
    h = art.bias_forces(q, qd, R, p)
    assert float(h.abs().max()) > 1e-6, "double pendulum should have nonzero bias"

    # symmetry of mass matrix
    assert torch.allclose(M, M.transpose(-1, -2), atol=1e-12)
    # positive definiteness
    eigs = torch.linalg.eigvalsh(M[0])
    assert float(eigs.min()) > 0.0

    # NOTE: windowed max-deviation, NOT endpoint drift -- the double
    # pendulum is chaotic, so an endpoint sample is phase-sensitive.
    # Healthy dynamics show a bounded, non-secular excursion (~1e-3
    # relative at this horizon); broken Coriolis terms blow up 20-80%.
    _, _, Es = integrate(art, model, q.clone(), qd.clone(), T=0.25, dtv=5e-5)
    E0 = Es[0]
    max_dev = max(abs(e - E0) for e in Es) / max(abs(E0), 1e-12)
    end_dev = abs(Es[-1] - E0) / max(abs(E0), 1e-12)
    assert max_dev < 5e-3, f"double pendulum energy excursion {max_dev:.2e}"
    assert end_dev < 5e-3, f"double pendulum endpoint drift {end_dev:.2e}"


def test_bias_matches_finite_difference_of_momentum():
    """h(q,qd) must equal the Coriolis term implied by d/dt(M qd) along the
    unforced flow: check via central differences of the momentum map."""
    torch.manual_seed(1)
    model = _build_pendulum(2)
    art = Articulation(model)
    q = torch.tensor([[0.7, -0.3]], dtype=DT)
    qd = torch.tensor([[0.9, 1.1]], dtype=DT)
    eps = 1e-6

    def momentum(q_, qd_):
        R, p = art.kinematics(q_)
        M = art.mass_matrix(q_, R, p)
        return M @ qd_.unsqueeze(-1)

    # dM/dt qd along flow, computed by FD on M alone:
    # h = Md qdd + Mdot qd - tau  => with zero applied torque and qdd from
    # forward dynamics, energy conservation already covers this; instead we
    # verify h against the Lagrangian identity via numerical differentiation
    # of kinetic energy: d/dq_i (1/2 qd^T M(q) qd) relates to Christoffel form.
    def ke(q_):
        R, p = art.kinematics(q_)
        M = art.mass_matrix(q_, R, p)
        return 0.5 * (qd @ M @ qd.transpose(-1, -2)).squeeze()

    grad_ke = torch.zeros_like(q)
    for i in range(q.shape[1]):
        qp = q.clone(); qp[:, i] += eps
        qm = q.clone(); qm[:, i] -= eps
        grad_ke[:, i] = (ke(qp) - ke(qm)) / (2 * eps)

    # Christoffel identity: h_j = sum_k (dM_jk/dt) qd_k - 1/2 d/dq_j (qd^T M qd)
    # compute dM/dt by chain rule through FD on q:
    nq_ = q.shape[1]
    dMdq = torch.zeros(1, nq_, art.nv, art.nv, dtype=DT)
    for i in range(nq_):
        qp = q.clone(); qp[:, i] += eps
        qm = q.clone(); qm[:, i] -= eps
        dMdq[:, i] = (art.mass_matrix(qp) - art.mass_matrix(qm)) / (2 * eps)
    Mdot = (dMdq * qd.view(1, -1, 1, 1)).sum(dim=1)          # [1,nv,nv]
    h = art.bias_forces(q, qd)
    # Christoffel identity: C = (dM/dt) qd - d/dq KE, where the gradient
    # of KE = 1/2 qd^T M qd already carries the factor 1/2.
    lhs = (Mdot @ qd.unsqueeze(-1)).squeeze(-1) - grad_ke
    assert torch.allclose(lhs, h, atol=5e-5), (
        f"bias vs Christoffel mismatch: {lhs} vs {h}"
    )
