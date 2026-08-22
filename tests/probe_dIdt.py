"""Probe: validate closed-form dI/dt blocks + per-body IVdot vs FD truth."""
import sys

import torch

sys.path.insert(0, '/Code/0x-differentiable-sim-project')
sys.path.insert(0, '/Code/0x-differentiable-sim-project/tests')

from test_dynamics import _build_pendulum
from diffsim.articulation import Articulation

DT = torch.float64
EPS = 1e-7


def skewb(v):
    x, y, z = v.unbind(-1)
    o = torch.zeros_like(x)
    return torch.stack([o, -z, y, z, o, -x, -y, x, o], -1).reshape(*v.shape[:-1], 3, 3)


def world_inertia_perbody(art, model, Q):
    """[1,nb,6,6] spatial inertia about world origin, per body."""
    R_, p_ = art.kinematics(Q)
    cw = art.com_positions(Q, R_, p_)                      # [1,nb,3]
    Jw = torch.matmul(torch.matmul(
        R_, model.inertia_com.expand(1, -1, -1, -1)), R_.transpose(-1, -2))
    me = model.masses.view(1, model.n_bodies, 1, 1)        # [1,nb,1,1]
    Cx = skewb(cw)                                         # [1,nb,3,3]
    top = torch.cat([Jw - me * (Cx @ Cx), me * Cx], dim=-1)        # [1,nb,3,6]
    Mb = torch.diag_embed(model.masses.view(1, model.n_bodies, 1).expand(1, -1, 3))
    bot = torch.cat([-me * Cx, Mb], dim=-1)                        # [1,nb,3,6]
    return torch.cat([top, bot], dim=-2)                            # [1,nb,6,6]


def main():
    m2 = _build_pendulum(2)
    art = Articulation(m2)
    torch.manual_seed(0)
    q = (torch.rand(1, 2, dtype=DT) * 2 - 1) * 0.8
    qd = (torch.rand(1, 2, dtype=DT) * 2 - 1)

    nb = m2.n_bodies

    # ---- FD dI/dt per body -------------------------------------------
    dIdt_fd = (world_inertia_perbody(art, m2, q + qd * EPS)
               - world_inertia_perbody(art, m2, q)) / EPS   # [1,nb,6,6]

    # ---- analytic dI/dt per body -------------------------------------
    R, p = art.kinematics(q)
    sub = art.subspace_terms(q, R, p)
    S = sub["S"]
    Sdq = art.sdot_terms(q, qd, R, p, sub)

    V = torch.zeros(1, nb, 6, dtype=DT)
    Av = torch.zeros_like(V)
    for b in range(nb):
        ds = art._body_dofs[b]
        if ds:
            dst = torch.tensor(ds)
            V[:, b] = (S.index_select(1, dst) * qd.index_select(1, dst).unsqueeze(-1)).sum(1)
            Av[:, b] = (Sdq.index_select(1, dst) * qd.index_select(1, dst).unsqueeze(-1)).sum(1)

    w, vo = V[..., :3], V[..., 3:]
    cw = art.com_positions(q, R, p)
    cdot = vo + torch.linalg.cross(w, cw, dim=-1)
    Jw = torch.matmul(torch.matmul(R, m2.inertia_com.expand(1, -1, -1, -1)),
                      R.transpose(-1, -2))
    Om = skewb(w)
    Cx = skewb(cw)
    Cxdot = skewb(cdot)
    me = m2.masses.view(1, nb, 1, 1)
    Jdot = Om @ Jw - Jw @ Om
    Itop_dot = Jdot - me * (Cxdot @ Cx + Cx @ Cxdot)
    TRdot = me * Cxdot
    analytic = torch.cat([
        torch.cat([Itop_dot, TRdot], dim=-1),
        torch.cat([-TRdot, torch.zeros_like(TRdot)], dim=-1),
    ], dim=-2)                                              # [1,nb,6,6]

    print("dI/dt blockwise diffs (per body):")
    for b in range(nb):
        d = (dIdt_fd[0, b] - analytic[0, b]).abs()
        print(f"  b{b}: TL={float(d[:3,:3].max()):.2e} TR={float(d[:3,3:].max()):.2e} "
              f"BL={float(d[3:,:3].max()):.2e} BR={float(d[3:,3:].max()):.2e}")

    # ---- momentum-rate residual vs full IVdot ------------------------
    def mom(Q):
        Ip = world_inertia_perbody(art, m2, Q)
        R_, p_ = art.kinematics(Q)
        s_ = art.subspace_terms(Q, R_, p_)
        S_ = s_["S"]
        Vv = torch.zeros(1, nb, 6, dtype=DT)
        for b in range(nb):
            ds = art._body_dofs[b]
            if ds:
                dst = torch.tensor(ds)
                Vv[:, b] = (S_.index_select(1, dst) * qd.index_select(1, dst)
                            .unsqueeze(-1)).sum(1)
        return torch.matmul(Ip, Vv.unsqueeze(-1)).squeeze(-1)

    _, _, Iw = art._world_inertias(q, R, p)
    IA = torch.matmul(Iw, Av.unsqueeze(-1)).squeeze(-1)
    resid = (mom(q + qd * EPS) - mom(q)) / EPS - IA         # true gyroscopic part

    IV = torch.cat([
        (Itop_dot @ w.unsqueeze(-1)).squeeze(-1) + (TRdot @ vo.unsqueeze(-1)).squeeze(-1),
        (-TRdot @ w.unsqueeze(-1)).squeeze(-1),
    ], dim=-1)

    print("per-body |resid - IVdot|:")
    for b in range(nb):
        print(f"  b{b}: {float((resid[0,b]-IV[0,b]).abs().max()):.3e}  "
              f"resid={[round(float(x),6) for x in resid[0,b].tolist()]}  "
              f"IV={[round(float(x),6) for x in IV[0,b].tolist()]}")


if __name__ == "__main__":
    main()
