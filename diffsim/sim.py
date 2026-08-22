"""DiffSim engine: batched differentiable physics step.

Design notes
------------
Contact forces enter the equations of motion through the Jacobian-transpose
identity  tau_c = sum_k J_k^T f_k,  with the point-Jacobian obtained in
closed form from the world-frame joint subspaces:

    dp/dq_d = S_w[d,:3] x p + S_w[d,3:]

so no nested autograd graphs are needed anywhere -- every quantity is a
plain differentiable tensor op and backprop-through-time across thousands
of steps stays cheap and exact.
"""

from __future__ import annotations

import dataclasses
from typing import List, NamedTuple, Optional

import torch

from .articulation import Articulation, Model
from .collision import Geoms, eval_ground, eval_pairs, smooth_ramp, softplus_pen





@dataclasses.dataclass
class ContactConfig:
    k_ground: float = 2.5e4          # normal stiffness [N/m]
    k_pair: float = 1.5e4            # self-collision stiffness
    damping: float = 400.0           # normal damping [N s/m]
    mu: float = 0.9                  # friction coefficient (regularized viscous)
    margin: float = 0.0              # contact activation distance
    beta: float = 200.0              # (legacy) softplus sharpness
    smooth: float = 1e-4             # contact ramp smoothing width [m]
    v_reg: float = 0.05              # tangential velocity regularization [m/s]
    limit_k: float = 5.0e4           # joint-limit spring stiffness
    limit_beta: float = 100.0        # joint-limit softness


@dataclasses.dataclass
class SimConfig:
    dt: float = 0.004                # physics timestep
    n_substeps: int = 4              # physics substeps per control step
    contact: ContactConfig = dataclasses.field(default_factory=ContactConfig)
    max_vel: float = 30.0            # per-substep |qd| safety clamp
    use_analytic_bias: bool = True   # closed-form Coriolis (default);
                                     # False = autodiff oracle (reference)


class StepResult(NamedTuple):
    q: torch.Tensor                  # [E,Nq]
    qd: torch.Tensor                 # [E,Nv]
    qdd: torch.Tensor                # [E,Nv]
    n_contacts: torch.Tensor         # [E] expected active-contact count
    com_z: torch.Tensor              # [E] center-of-mass height
    R_w: Optional[torch.Tensor]      # [E,nb,3,3] link orientations
    p_w: Optional[torch.Tensor]      # [E,nb,3] link origins


class DiffSim:
    """Differentiable, GPU-batched articulated-body simulator."""

    def __init__(self, model: Model, geoms: Geoms, cfg: Optional[SimConfig] = None,
                 device="cpu", dtype=torch.float64,
                 skip_adjacent_pairs: bool = True,
                 feet_geoms: Optional[List[int]] = None):
        self.art = Articulation(model, device=device, dtype=dtype)
        m = self.m = self.art.m
        self.geoms = geoms.to(device, dtype)
        self.cfg = cfg or SimConfig()
        self.device, self.dtype = device, dtype
        self.nv, self.nq = self.art.nv, self.art.nq
        self._feet_geoms = list(feet_geoms or [])

        G = len(self.geoms)
        pairs = []
        for i in range(G):
            for j in range(i + 1, G):
                bi, bj = int(self.geoms.body[i]), int(self.geoms.body[j])
                if bi == bj:
                    continue
                if skip_adjacent_pairs and (
                    m.parent[bi] == bj or m.parent[bj] == bi
                ):
                    continue
                pairs.append((i, j))
        if pairs:
            ii, jj = zip(*pairs)
            self.pair_i = torch.tensor(ii, dtype=torch.long, device=device)
            self.pair_j = torch.tensor(jj, dtype=torch.long, device=device)
        else:
            self.pair_i = torch.zeros(0, dtype=torch.long, device=device)
            self.pair_j = torch.zeros(0, dtype=torch.long, device=device)
        self.pair_body_i = self.geoms.body[self.pair_i]      # [P]
        self.pair_body_j = self.geoms.body[self.pair_j]      # [P]

        self.ground_idx = torch.nonzero(self.geoms.collide_ground, as_tuple=True)[0]
        self.ground_body = self.geoms.body[self.ground_idx]  # [g]

    # ------------------------------------------------------------------ #
    # Jacobian helpers
    # ------------------------------------------------------------------ #

    def _point_dir(self, sub, d: int, pts: torch.Tensor):
        """Velocity direction of dof d evaluated at world points.
        Returns [E,K,3]:  S[:3] x p + S[3:]."""
        Sd = sub["S"][:, d]                                   # [E,6]
        w, vo = Sd[..., :3], Sd[..., 3:]
        return torch.linalg.cross(w.unsqueeze(1), pts, dim=-1) + vo.unsqueeze(1)

    def _point_velocities(self, qd, sub, pts: torch.Tensor, body_of_pt: torch.Tensor):
        """Velocity of body-attached world points. pts [E,K,3], bodies [K]."""
        vel = torch.zeros_like(pts)
        for b in torch.unique(body_of_pt).tolist():
            mask = body_of_pt == int(b)
            ds = self.art._body_dofs[int(b)]
            pk = pts[:, mask]
            acc = torch.zeros_like(pk)
            for d in ds:
                acc = acc + qd[:, d:d + 1, None] * self._point_dir(sub, d, pk)
            vel[:, mask] = acc
        return vel

    def _point_genforce(self, sub, pts: torch.Tensor, forces: torch.Tensor,
                        body_of_pt: torch.Tensor):
        """tau_c [E,nv] = sum_k J_k^T f_k."""
        E = pts.shape[0]
        tau = torch.zeros(E, self.nv, dtype=pts.dtype, device=pts.device)
        for b in torch.unique(body_of_pt).tolist():
            mask = body_of_pt == int(b)
            ds = self.art._body_dofs[int(b)]
            if not ds:
                continue
            pk = pts[:, mask]
            fk = forces[:, mask]
            for d in ds:
                dir_pk = self._point_dir(sub, d, pk)
                tau[:, d] = tau[:, d] + (fk * dir_pk).sum(dim=(-1, -2))
        return tau

    # ------------------------------------------------------------------ #
    # contacts
    # ------------------------------------------------------------------ #

    def contact_forces(self, q, R_w=None, p_w=None, sub=None, qd=None):
        """Assemble all contact point-forces -> (tau_c [E,nv], info dict)."""
        cc = self.cfg.contact
        if R_w is None or p_w is None:
            R_w, p_w = self.art.kinematics(q)
        if sub is None:
            sub = self.art.subspace_terms(q, R_w, p_w)

        centers, Rg = self.art.geoms_world(q, self.geoms, R_w, p_w)
        E = q.shape[0]

        pts_all, f_all, bod_all = [], [], []
        n_active = torch.zeros(E, dtype=q.dtype, device=q.device)
        feet_fnz: Optional[torch.Tensor] = None

        up = torch.tensor([0.0, 0.0, 1.0], dtype=q.dtype, device=q.device)

        # ---- ground -------------------------------------------------------
        ng = int(self.ground_idx.numel())
        if ng > 0:
            gi = self.ground_idx
            res = eval_ground(self.geoms, centers, Rg)
            dist = res["dist"][:, gi]                          # [E,g]
            p_body = res["p_body"][:, gi]                      # [E,g,3] body pt
            pen = smooth_ramp(cc.margin - dist, cc.smooth)
            act = torch.where(pen > 0, pen / (pen + 1e-9), torch.zeros_like(pen))

            vb = self._point_velocities(qd, sub, p_body, self.ground_body) \
                if qd is not None else None
            n_hat = up.expand_as(p_body)
            fn = cc.k_ground * pen                             # [E,g]
            f_vec = fn.unsqueeze(-1) * n_hat                   # push up
            if vb is not None:
                vn = (vb * n_hat).sum(-1)                      # <0 approaching
                vn_c = torch.clamp(-vn, max=2.0)                # cap approach speed
                fn = fn + cc.damping * act * smooth_ramp(vn_c, 1e-3)
                vt = vb - vn.unsqueeze(-1) * n_hat
                vtn = torch.linalg.vector_norm(vt, dim=-1)
                ft = -(cc.mu * fn).unsqueeze(-1) * vt / (vtn.unsqueeze(-1) + cc.v_reg)
                f_vec = fn.unsqueeze(-1) * n_hat + ft

            n_active = n_active + act.sum(-1)
            pts_all.append(p_body)
            f_all.append(f_vec)
            bod_all.append(self.ground_body.unsqueeze(0).expand(E, -1))
            if getattr(self, "_feet_geoms", None):
                pos_in_g = {int(gi[k]): k for k in range(ng)}
                cols = [pos_in_g[g] for g in self._feet_geoms if g in pos_in_g]
                if cols:
                    feet_fnz = fn[:, torch.tensor(cols, device=q.device)]

        # ---- self-collision pairs ------------------------------------------
        np_ = int(self.pair_i.numel())
        if np_ > 0:
            res = eval_pairs(self.geoms, centers, Rg, self.pair_i, self.pair_j)
            dist = res["dist"]
            p1, p2 = res["p1"], res["p2"]
            pen = smooth_ramp(cc.margin - dist, cc.smooth)
            act = torch.where(pen > 0, pen / (pen + 1e-9), torch.zeros_like(pen))

            diff = p1 - p2
            n_hat = diff / torch.clamp(
                torch.linalg.vector_norm(diff, dim=-1, keepdim=True), min=1e-9
            )
            fn = cc.k_pair * pen
            pc = 0.5 * (p1 + p2)
            bodies = torch.cat([self.pair_body_i, self.pair_body_j])  # [2P]

            f_on_1 = fn.unsqueeze(-1) * n_hat
            f_on_2 = -f_on_1
            if qd is not None:
                Pb = torch.cat([pc, pc], dim=1)                # [E,2P,3]
                vall = self._point_velocities(qd, sub, Pb, bodies)
                P = p1.shape[1]
                vrel = vall[:, :P] - vall[:, P:]               # [E,P,3]
                vn = (vrel * n_hat).sum(-1)
                vn_c = torch.clamp(-vn, max=2.0)                # cap approach speed
                fn = fn + cc.damping * act * smooth_ramp(vn_c, 1e-3)
                vt = vrel - vn.unsqueeze(-1) * n_hat
                vtn = torch.linalg.vector_norm(vt, dim=-1)
                ft = -(cc.mu * fn / (vtn + cc.v_reg)).unsqueeze(-1) * vt
                f_on_1 = fn.unsqueeze(-1) * n_hat + ft
                f_on_2 = -(fn.unsqueeze(-1) * n_hat + ft)

            n_active = n_active + act.sum(-1)
            pts_all.append(torch.cat([pc, pc], dim=1))
            f_all.append(torch.cat([f_on_1, f_on_2], dim=1))
            bod_all.append(bodies.unsqueeze(0).expand(E, -1))

        if not pts_all:
            zero = torch.zeros(E, self.nv, dtype=q.dtype, device=q.device)
            return zero, {"n_contacts": n_active, "feet_fnz": feet_fnz}

        pts = torch.cat(pts_all, dim=1)
        fs = torch.cat(f_all, dim=1)
        bods = torch.cat(bod_all, dim=1)[0]                    # [K] shared across E
        tau_c = self._point_genforce(sub, pts, fs, bods)
        return tau_c, {"n_contacts": n_active, "feet_fnz": feet_fnz}

    # ------------------------------------------------------------------ #
    # limits / damping
    # ------------------------------------------------------------------ #

    def limit_and_damping_forces(self, q, qd):
        cc = self.cfg.contact
        m = self.m
        tau = torch.zeros_like(qd)
        if m.limit_dof_idx is not None and m.limit_dof_idx.numel() > 0:
            idx = m.limit_dof_idx
            ql = q.index_select(1, idx)
            pen_lo = softplus_pen(m.joint_limit_lo - ql, cc.limit_beta)
            pen_hi = softplus_pen(ql - m.joint_limit_hi, cc.limit_beta)
            tau_lim = cc.limit_k * (pen_lo - pen_hi)
            tau = tau.index_copy(1, idx, tau.index_select(1, idx) + tau_lim)
        if m.damping is not None:
            tau = tau - m.damping * qd
        return tau

    # ------------------------------------------------------------------ #
    # forward dynamics
    # ------------------------------------------------------------------ #

    def forward_dynamics(self, q, qd, tau_ext=None, want_aux=False):
        """Compute accelerations qdd [E,nv]. Fully differentiable."""
        art = self.art
        m = self.m
        R_w, p_w = art.kinematics(q)
        sub = art.subspace_terms(q, R_w, p_w)
        _, _, Iw = art._world_inertias(q, R_w, p_w)

        M = art.mass_matrix(q, R_w, p_w, sub, Iw)
        if self.cfg.use_analytic_bias:
            h = art.bias_forces_analytic(q, qd, R_w, p_w, sub, Iw)
        else:
            h = art.bias_forces(q, qd, R_w, p_w, sub, Iw)
        t_g = art.gravity_genforce(q, R_w, p_w, sub)
        t_c, cinfo = self.contact_forces(q, R_w, p_w, sub, qd=qd)
        t_ld = self.limit_and_damping_forces(q, qd)

        rhs = t_g - h + t_c + t_ld
        if tau_ext is not None:
            rhs = rhs + tau_ext
        if m.armature > 0.0:
            M = M + m.armature * torch.eye(self.nv, dtype=M.dtype, device=M.device)

        qdd = torch.linalg.solve(M, rhs.unsqueeze(-1)).squeeze(-1)

        if want_aux:
            com_w = art.com_positions(q, R_w, p_w)
            com_xyz = (com_w * m.masses.unsqueeze(-1)).sum(1) / m.masses.sum()
            aux = {"n_contacts": cinfo["n_contacts"], "com_z": com_xyz[..., 2],
                   "M": M, "R_w": R_w, "p_w": p_w}
            return qdd, aux
        return qdd

    # ------------------------------------------------------------------ #
    # stepping
    # ------------------------------------------------------------------ #

    def step(self, q, qd, tau_ext=None, train_mode: bool = False) -> StepResult:
        """Advance one control step (`n_substeps` physics substeps).

        train_mode=True keeps the autograd graph across steps so losses can
        be backpropagated through the whole rollout (BPTT / SHAC).  By
        default the rollout runs under no_grad for speed.
        """
        dt = self.cfg.dt
        ctx = torch.enable_grad() if train_mode else torch.no_grad()
        with ctx:
            qc, qdc = q, qd
            qdd, aux = None, None
            for _ in range(self.cfg.n_substeps):
                qdd, aux = self.forward_dynamics(qc, qdc, tau_ext, want_aux=True)
                qdc = qdc + dt * qdd
                if self.cfg.max_vel is not None:
                    nrm = qdc.norm(dim=-1, keepdim=True)
                    qdc = qdc * torch.clamp(self.cfg.max_vel / (nrm + 1e-12), max=1.0)
                qc = self.art.integrate(qc, qdc, dt)
            if not train_mode:
                qc, qdc, qdd = qc.detach(), qdc.detach(), qdd.detach()
        return StepResult(q=qc, qd=qdc, qdd=qdd,
                          n_contacts=aux["n_contacts"].detach(),
                          com_z=aux["com_z"].detach(),
                          R_w=aux["R_w"], p_w=aux["p_w"])

    def pd_torques(self, q, qd, q_pos_target, kp, kd):
        """PD controller on actuated (hinge/slide) dofs.
        Returns tau [E,NV] with zeros on unactuated dofs.
        q_pos_target [E,n_act], kp/kd scalars or broadcastable."""
        m = self.m
        idx = []
        for i in range(self.art.nb):
            if int(m.j_type[i]) in (1, 2):  # hinge/slide
                idx.append(self.art._vs[i])
        idx_t = torch.tensor(idx, device=q.device)
        q_cur = q.index_select(1, idx_t)
        qd_cur = qd.index_select(1, idx_t)
        n_act = len(idx)
        if not torch.is_tensor(kp):
            kp = torch.tensor(float(kp), dtype=q.dtype, device=q.device)
        if not torch.is_tensor(kd):
            kd = torch.tensor(float(kd), dtype=q.dtype, device=q.device)
        tau_act = kp * (q_pos_target - q_cur) - kd * qd_cur
        tau = torch.zeros_like(qd)
        tau = tau.index_copy(1, idx_t, tau_act.to(tau.dtype))
        return tau
