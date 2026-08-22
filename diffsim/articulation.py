"""Differentiable articulated rigid-body dynamics.

Formulation
-----------
Generalized coordinates with a floating base represented as a free joint
(quaternion + position).  All dynamics quantities (joint motion subspaces,
spatial inertias, composite inertias, twists) are expressed in WORLD
coordinates about the world origin.  This makes the Composite Rigid Body
Algorithm (CRBA) and the recursive Newton-Euler bias computation almost
transform-free: composite inertias of subtrees become plain sums, and
"propagation" up the kinematic tree becomes identity.

Everything is implemented in batched PyTorch ops: an arbitrary number of
parallel environments E is carried as the leading dimension of every
tensor, and the whole pipeline is differentiable end-to-end (gradients
flow through the mass matrix, bias forces, contacts, and integrator).

State layout
------------
q  : [E, Nq]   hinge/slide -> 1 (angle / displacement)
                free joint -> 7  [qw qx qy qz px py pz]
qd : [E, Nv]   hinge/slide -> 1 (rate)
                free joint -> 6  [wx wy wz vx vy vz]
                (base angular velocity in WORLD frame; linear velocity of
                 the base body-frame origin, WORLD frame)

Convention: at most one free joint, and it must be the root body (index 0).
"""

from __future__ import annotations

import dataclasses
from typing import List, Optional

import torch

from .linalg import exp_so3, quat_integrate, quat_mul, quat_to_matrix, skew
from .spatial import force_cross, spatial_inertia_world

# joint types
J_FIXED, J_HINGE, J_SLIDE, J_FREE = 0, 1, 2, 3


@dataclasses.dataclass
class Model:
    """Static description of an articulated system (batch-size independent)."""

    n_bodies: int
    parent: List[int]                      # parent body index (-1 for root)
    body_names: List[str]
    # fixed transform parent -> predecessor (joint) frame, parent coords
    fix_R: torch.Tensor                    # [nb,3,3]
    fix_p: torch.Tensor                    # [nb,3]
    j_type: torch.Tensor                   # [nb] int
    j_axis: torch.Tensor                   # [nb,3] unit axis, predecessor coords
    # link inertial properties, LINK frame (origin at the joint anchor)
    masses: torch.Tensor                   # [nb]
    com: torch.Tensor                      # [nb,3]
    inertia_com: torch.Tensor              # [nb,3,3] about COM, link axes
    q_dim: int                             # Nq
    v_dim: int                             # Nv
    dof_body: List[int]                    # dof idx -> body idx
    body_dof_start: List[int]              # body idx -> first dof (-1 if none)
    q_free_start: int                      # q index of free joint (-1 if none)
    gravity: tuple = (0.0, 0.0, -9.81)
    # optional actuated-dof properties
    joint_limit_lo: Optional[torch.Tensor] = None   # [n_lim] (-inf ok)
    joint_limit_hi: Optional[torch.Tensor] = None   # [n_lim]
    limit_dof_idx: Optional[torch.Tensor] = None    # [n_lim] dof indices
    damping: Optional[torch.Tensor] = None          # [nv] viscous damping
    armature: float = 0.0                           # added diag inertia [nv]

    def to(self, device, dtype=torch.float64):
        def _t(x):
            return x.to(device=device, dtype=dtype) if torch.is_tensor(x) else x
        return dataclasses.replace(
            self,
            fix_R=_t(self.fix_R), fix_p=_t(self.fix_p),
            j_type=self.j_type.to(device),
            j_axis=_t(self.j_axis),
            masses=_t(self.masses), com=_t(self.com), inertia_com=_t(self.inertia_com),
            joint_limit_lo=_t(self.joint_limit_lo),
            joint_limit_hi=_t(self.joint_limit_hi),
            limit_dof_idx=(self.limit_dof_idx.to(device) if self.limit_dof_idx is not None else None),
            damping=_t(self.damping),
        )


class Articulation:
    """Batched differentiable kinematics + dynamics for a `Model`."""

    def __init__(self, model: Model, device="cpu", dtype=torch.float64):
        self.m = model.to(device, dtype)
        self.device = device
        self.dtype = dtype
        m = self.m

        self.nv = m.v_dim
        self.nq = m.q_dim
        self.nb = m.n_bodies

        # ---- bookkeeping ------------------------------------------------
        # per-body q / qd start indices
        self._qs = [-1] * self.nb      # hinge/slide q index == v index
        self._fs = [-1] * self.nb      # free joint q start
        self._vs = [-1] * self.nb      # free joint v start
        qi = 0
        vi = 0
        for i in range(self.nb):
            jt = int(m.j_type[i])
            if jt in (J_HINGE, J_SLIDE):
                self._qs[i] = qi
                self._vs[i] = vi
                qi += 1
                vi += 1
            elif jt == J_FREE:
                self._fs[i] = qi
                self._vs[i] = vi
                qi += 7
                vi += 6

        # ancestor chains (padded with -1); body order assumed topological
        anc = torch.full((self.nb, self.nb), -1, dtype=torch.long)
        depth = torch.zeros(self.nb, dtype=torch.long)
        for i in range(self.nb):
            chain = []
            j = m.parent[i]
            while j != -1:
                chain.append(j)
                j = m.parent[j]
            for d, j in enumerate(chain):
                anc[i, d] = j
            depth[i] = len(chain)
        self._anc = anc.to(device)
        self._depth = depth.to(device)

        # reverse-topological order (children before parents)
        order = sorted(range(self.nb), key=lambda i: -int(depth[i]))
        self._rev_order = [i for i in order]

        # per-body ancestor DOF list (as python lists for fast slicing).
        # NOTE: multi-dof joints (free) contribute ALL their dofs.
        self._body_dofs: List[List[int]] = []
        for b in range(self.nb):
            ds = []
            j = b
            while j != -1:
                dj = m.body_dof_start[j]
                if dj >= 0:
                    nd = 6 if int(m.j_type[j]) == J_FREE else 1
                    ds.extend(range(dj, dj + nd))
                j = m.parent[j]
            self._body_dofs.append(sorted(ds))

        # per-DOF ancestor DOF lists (padded tensor)
        nd = self.nv
        dof_anc = torch.full((nd, nd), -1, dtype=torch.long)
        for d in range(nd):
            bi = m.dof_body[d]
            for k, dd in enumerate(self._body_dofs[bi]):
                dof_anc[d, k] = dd
        self._dof_anc = dof_anc.to(device)
        self._dof_body_t = torch.tensor(m.dof_body, dtype=torch.long, device=device)

        self._free_i = next(
            (i for i in range(self.nb) if int(m.j_type[i]) == J_FREE), -1
        )

    # ------------------------------------------------------------------ #
    # kinematics
    # ------------------------------------------------------------------ #

    def kinematics(self, q: torch.Tensor):
        """Forward kinematics. q [E,Nq] -> R_w [E,nb,3,3], p_w [E,nb,3].

        Body frame origin coincides with its joint anchor; fixed offsets
        live in geom/inertial local frames.
        """
        m = self.m
        E = q.shape[0]
        R_w = torch.zeros(E, self.nb, 3, 3, dtype=q.dtype, device=q.device)
        p_w = torch.zeros(E, self.nb, 3, dtype=q.dtype, device=q.device)
        for i in range(self.nb):
            jt = int(m.j_type[i])
            par = m.parent[i]
            if jt == J_FREE:
                f = self._fs[i]
                R = quat_to_matrix(q[:, f:f + 4])
                p = q[:, f + 4:f + 7]
            elif par == -1:
                R = m.fix_R[i].expand(E, 3, 3)
                p = m.fix_p[i].expand(E, 3)
            elif jt == J_FIXED:
                Rp, pp = R_w[:, par], p_w[:, par]
                R = Rp @ m.fix_R[i]
                p = pp + Rp @ m.fix_p[i]
            else:
                Rp, pp = R_w[:, par], p_w[:, par]
                Rpre = Rp @ m.fix_R[i]
                ppre = pp + Rp @ m.fix_p[i]
                qi = self._qs[i]
                if jt == J_HINGE:
                    R = Rpre @ quat_to_matrix(exp_so3(m.j_axis[i] * q[:, qi:qi + 1]))
                    p = ppre
                else:  # SLIDE
                    R = Rpre
                    p = ppre + torch.einsum("eij,ej->ei", Rpre, m.j_axis[i] * q[:, qi:qi + 1])
            R_w[:, i], p_w[:, i] = R, p
        return R_w, p_w

    def com_positions(self, q, R_w=None, p_w=None):
        """World COM positions [E,nb,3]."""
        m = self.m
        if R_w is None or p_w is None:
            R_w, p_w = self.kinematics(q)
        E = q.shape[0]
        return torch.einsum("ebij,ebj->ebi", R_w, m.com.expand(E, -1, -1)) + p_w

    def geoms_world(self, q, g, R_w=None, p_w=None):
        """World poses of collision geoms.

        g: Geoms container. Returns centers [E,G,3] and orientations
        [E,G,3,3] of geom local frames.
        """
        if R_w is None or p_w is None:
            R_w, p_w = self.kinematics(q)
        E = q.shape[0]
        Rb = R_w[:, g.body]          # [E,G,3,3]
        pb = p_w[:, g.body]          # [E,G,3]
        c = torch.einsum("egij,egj->egi", Rb, g.local_p.expand(E, -1, -1)) + pb
        Rg = torch.einsum("egij,egjk->egik", Rb, g.local_R.expand(E, -1, -1, -1))
        return c, Rg

    # ------------------------------------------------------------------ #
    # world-frame joint subspaces
    # ------------------------------------------------------------------ #

    def subspace_terms(self, q: torch.Tensor, R_w=None, p_w=None):
        """Per-DOF world motion subspaces.

        Returns dict:
          S      [E,nv,6]   subspace about world origin
          axis_w [E,nv,3]   world axis (hinge/slide; unit z for free)
          anchor [E,nv,3]   world anchor point (hinge/slide; zeros for free)
          kind   [nv]       long tensor of joint types per dof
        """
        m = self.m
        if R_w is None or p_w is None:
            R_w, p_w = self.kinematics(q)
        E = q.shape[0]
        S = torch.zeros(E, self.nv, 6, dtype=q.dtype, device=q.device)
        axis_w = torch.zeros(E, self.nv, 3, dtype=q.dtype, device=q.device)
        anchor = torch.zeros(E, self.nv, 3, dtype=q.dtype, device=q.device)
        for d in range(self.nv):
            bi = int(m.dof_body[d])
            jt = int(m.j_type[bi])
            if jt == J_FREE:
                # free joint has 6 dofs: [wx wy wz vx vy vz]; column of dof d
                # is the twist about the world origin for a unit rate of that
                # component:  ang_i: [e_i; o_b x e_i],  lin_j: [0; e_j]
                pb = p_w[:, bi]                              # base origin (world)
                k = d - self._vs[bi]
                if k < 3:
                    S[:, d, k] = 1.0
                    S[:, d, 3:] = torch.linalg.cross(pb, torch.eye(3, dtype=q.dtype, device=q.device)[k].expand(E, 3), dim=-1)
                else:
                    S[:, d, 3 + k - 3] = 1.0
                axis_w[:, d, 2] = 1.0
            elif jt == J_HINGE:
                par = m.parent[bi]
                Rpre = R_w[:, par] @ m.fix_R[bi]
                ppre = p_w[:, par] + R_w[:, par] @ m.fix_p[bi]
                a_w = Rpre @ m.j_axis[bi]
                S[:, d, :3] = a_w
                S[:, d, 3:] = torch.linalg.cross(ppre, a_w, dim=-1)
                axis_w[:, d] = a_w
                anchor[:, d] = ppre
            else:  # SLIDE
                par = m.parent[bi]
                Rpre = R_w[:, par] @ m.fix_R[bi]
                d_w = Rpre @ m.j_axis[bi]
                S[:, d, 3:] = d_w
                axis_w[:, d] = d_w
        kind = torch.tensor(
            [int(m.j_type[int(b)]) for b in m.dof_body], device=q.device
        )
        return {"S": S, "axis_w": axis_w, "anchor": anchor, "kind": kind}

    def sdot_terms(self, q, qd, R_w, p_w, sub=None):
        """Time derivative of each world subspace contracted with own rate:
        returns Sdot_qd [E,nv,6]."""
        m = self.m
        if sub is None:
            sub = self.subspace_terms(q, R_w, p_w)
        S, axis_w, anchor = sub["S"], sub["axis_w"], sub["anchor"]
        E = q.shape[0]
        dev = q.device
        dt_ = q.dtype
        out = torch.zeros_like(S)
        zero3 = torch.zeros(E, 3, dtype=dt_, device=dev)
        for d in range(self.nv):
            bi = int(m.dof_body[d])
            jt = int(m.j_type[bi])
            # predecessor-frame twist: dofs of the PARENT body chain only,
            # strictly excluding this joint's own dof
            par = m.parent[bi]
            anc_list = self._body_dofs[par] if par != -1 else []
            if anc_list:
                Sa = S[:, anc_list, :]                       # [E,k,6]
                qa = qd[:, anc_list]                         # [E,k]
                Vp = (Sa * qa.unsqueeze(-1)).sum(dim=1)      # [E,6]
                wp, vp = Vp[:, :3], Vp[:, 3:]
            else:
                wp, vp = zero3, zero3
            if jt == J_HINGE:
                a_w = axis_w[:, d]
                pa = anchor[:, d]
                adot = torch.linalg.cross(wp, a_w, dim=-1)
                pdot = vp + torch.linalg.cross(wp, pa, dim=-1)
                bot = torch.linalg.cross(pdot, a_w, dim=-1) + torch.linalg.cross(pa, adot, dim=-1)
                out[:, d, :3] = adot * qd[:, d:d + 1]
                out[:, d, 3:] = bot * qd[:, d:d + 1]
            elif jt == J_SLIDE:
                out[:, d, 3:] = torch.linalg.cross(wp, axis_w[:, d], dim=-1) * qd[:, d:d + 1]
            else:  # FREE: only angular columns are time-dependent
                # d/dt col_k = [0 ; v_o x e_k], so Sdot@qd = [0 ; v_o x w]
                # restricted to this dof's own component.
                bi_vs = self._vs[bi]
                k = d - bi_vs
                if k < 3:
                    vw = qd[:, bi_vs + 3:bi_vs + 6]          # v_o (base origin)
                    ek = torch.eye(3, dtype=q.dtype, device=q.device)[k].expand(E, 3)
                    out[:, d, 3:] = torch.linalg.cross(vw, ek, dim=-1) * qd[:, d:d + 1]
        return out

    # ------------------------------------------------------------------ #
    # dynamics
    # ------------------------------------------------------------------ #

    def _world_inertias(self, q, R_w, p_w):
        m = self.m
        E = q.shape[0]
        com_w = self.com_positions(q, R_w, p_w)
        Jc = torch.einsum(
            "ebij,ebjk,ebkl->ebil",
            R_w, m.inertia_com.expand(E, -1, -1, -1), R_w.transpose(-1, -2),
        )
        Iw = spatial_inertia_world(m.masses.expand(E, -1), com_w, Jc)  # [E,nb,6,6]
        return com_w, Jc, Iw

    def _accumulate_subtrees(self, per_body: torch.Tensor) -> torch.Tensor:
        """Sum child values into parents over reverse-topological order."""
        FS = per_body.clone()
        for i in self._rev_order:
            pi = self.m.parent[i]
            if pi != -1:
                FS[:, pi] = FS[:, pi] + FS[:, i]
        return FS

    def mass_matrix(self, q: torch.Tensor, R_w=None, p_w=None, sub=None,
                    Iw=None) -> torch.Tensor:
        """CRBA in world coordinates. Returns M [E,nv,nv] (symmetric pd)."""
        m = self.m
        if R_w is None or p_w is None:
            R_w, p_w = self.kinematics(q)
        if sub is None:
            sub = self.subspace_terms(q, R_w, p_w)
        if Iw is None:
            _, _, Iw = self._world_inertias(q, R_w, p_w)
        E = q.shape[0]
        dev, dt_ = q.device, q.dtype

        IC = self._accumulate_subtrees(Iw)                        # [E,nb,6,6]
        S = sub["S"]

        M = torch.zeros(E, self.nv, self.nv, dtype=dt_, device=dev)
        for k in range(self.nv):
            Fk = torch.einsum(
                "eij,ej->ei",
                IC[:, int(self.m.dof_body[k])],
                S[:, k],
            )                                                      # [E,6]
            anc = [int(x) for x in self._dof_anc[k].tolist() if x >= 0]
            # NOTE: keep the two writes disjoint -- (k,k) must be written
            # exactly once or index_put_ backward will double-count the
            # gradient flowing into Fk.
            M[:, k, k] = torch.einsum("ei,ei->e", S[:, k], Fk)
            if anc:
                Hanc = torch.einsum("eci,ei->ec", S[:, anc, :], Fk)
                M[:, anc, k] = Hanc                                # column k, ancestor rows
                M[:, k, anc] = Hanc                                # row k, ancestor columns
        return M

    def gravity_genforce(self, q, R_w=None, p_w=None, sub=None):
        """Generalized gravity force tau_g [E,nv]."""
        m = self.m
        if R_w is None or p_w is None:
            R_w, p_w = self.kinematics(q)
        gv = torch.tensor(m.gravity, dtype=q.dtype, device=q.device)
        com_w = self.com_positions(q, R_w, p_w)
        fg_lin = m.masses.unsqueeze(-1).expand_as(com_w) * gv
        fg_tau = torch.linalg.cross(com_w, fg_lin, dim=-1)
        Fg = torch.cat([fg_tau, fg_lin], dim=-1)                  # [E,nb,6]
        FS = self._accumulate_subtrees(Fg)
        if sub is None:
            sub = self.subspace_terms(q, R_w, p_w)
        return torch.einsum("evi,evi->ev", sub["S"], FS[:, self.m.dof_body])

    def _twists_and_inertias(self, q, qd, R_w=None, p_w=None, sub=None, Iw=None):
        """Per-body twists V about the world origin [E,nb,6] plus shared
        intermediates."""
        m = self.m
        if R_w is None or p_w is None:
            R_w, p_w = self.kinematics(q)
        if sub is None:
            sub = self.subspace_terms(q, R_w, p_w)
        if Iw is None:
            _, _, Iw = self._world_inertias(q, R_w, p_w)
        E = q.shape[0]
        V = torch.zeros(E, self.nb, 6, dtype=q.dtype, device=q.device)
        S = sub["S"]
        for b in range(self.nb):
            ds = self._body_dofs[b]
            if ds:
                dst = torch.tensor(ds, device=q.device)
                V[:, b] = (
                    S.index_select(1, dst) * qd.index_select(1, dst).unsqueeze(-1)
                ).sum(dim=1)
        return V, R_w, p_w, sub, Iw

    def _momentum_flat(self, q, qd):
        """Spatial momenta P_b = I_w(q) V(q,qd), flattened [E, nb*6]."""
        V, _, _, _, Iw = self._twists_and_inertias(q, qd=qd)
        P = torch.matmul(Iw, V.unsqueeze(-1)).squeeze(-1)
        return P.reshape(q.shape[0], -1)

    def _qspace_rate(self, q, qd):
        """Time derivative of the CONFIGURATION vector q [E,Nq].

        Hinge/slide dofs equal qd; the free-joint quaternion obeys
        q_dot = 1/2 (0, omega_world) (x) q and its position follows the
        base linear velocity.
        """
        m = self.m
        qr = torch.zeros_like(q)
        for i in range(self.nb):
            jt = int(m.j_type[i])
            if jt in (J_HINGE, J_SLIDE):
                qi, vi = self._qs[i], self._vs[i]
                qr[:, qi] = qd[:, vi]
            elif jt == J_FREE:
                f, v = self._fs[i], self._vs[i]
                w = qd[:, v:v + 3]
                half_w = 0.5 * torch.cat(
                    [torch.zeros_like(w[:, :1]), w], dim=-1
                )
                qr[:, f:f + 4] = quat_mul(half_w, q[:, f:f + 4])
                qr[:, f + 4:f + 7] = qd[:, v + 3:v + 6]
        return qr

    def bias_forces(self, q, qd, R_w=None, p_w=None, sub=None, Iw=None):
        """Coriolis/centrifugal generalized forces h [E,nv]:
        M qdd = tau_g - h (+ other forces).

        Computed from the Lagrangian identity
            h = (dM/dt) qd - 1/2 d/dq (qd^T M qd)
        with BOTH pieces evaluated by exact autodiff through the validated
        CRBA mass matrix:
            dM/dt   : forward-mode jvp of M along (dq/dt=qd, d(qd)/dt=0)
            grad_ke : reverse-mode gradient of kinetic energy w.r.t. q
        Gravity enters separately through `gravity_genforce` (analytically
        validated). This avoids every hand-derived world-frame spatial
        algebra pitfall; correctness rests on M (independently validated
        against finite-differenced link-geometry kinetic energy).
        """
        from torch.func import jvp

        E = q.shape[0]

        (_, Mdot) = jvp(lambda q_: self.mass_matrix(q_), (q,),
                        (self._qspace_rate(q, qd),))

        # h = (dM/dt) qd - A^T d(KE)/dq, where A = d(q_dot)/d(qd) is the
        # configuration-rate map (identity for hinge/slide dofs, the
        # quaternion kinematic map for the free joint).  The pullback is
        # computed by reverse-mode through `_qspace_rate`, so no hand
        # derived quaternion Jacobians are needed.
        need_graph = torch.is_grad_enabled()
        qd_in = qd if (qd.requires_grad and need_graph) else qd.detach().requires_grad_(True)

        def ke_fn(q_):
            M_ = self.mass_matrix(q_)
            return 0.5 * torch.matmul(
                torch.matmul(qd.unsqueeze(1), M_), qd.unsqueeze(2)
            ).reshape(E)

        with torch.enable_grad():
            # gradient of KE wrt the configuration vector (Q-space)
            q_for_ke = q if (q.requires_grad and need_graph) else q.detach().requires_grad_(True)
            (grad_ke,) = torch.autograd.grad(
                ke_fn(q_for_ke).sum(), q_for_ke, create_graph=need_graph
            )
            # pull back to velocity coordinates through the rate map
            rate = self._qspace_rate(q, qd_in)
            (grad_pull,) = torch.autograd.grad(
                (rate * grad_ke).sum(), qd_in, create_graph=need_graph
            )

        return torch.matmul(Mdot, qd.unsqueeze(-1)).squeeze(-1) - grad_pull

    def bias_forces_analytic(self, q, qd, R_w=None, p_w=None, sub=None, Iw=None):
        """Closed-form Coriolis vector (fast path).

        Per body, the momentum rate about the world origin splits as
            dP/dt = I_w A + (dI_w/dt) V ,
        where A = sum over ancestor+self dofs of (Sdot_jw qd_j) is the
        velocity-only acceleration and dI/dt follows from
            c_dot = v_o + w x c ,   J_dot = [w]x J - J [w]x :
            dI/dt = [[J_dot - m(Cx'Cx + CxCx'), m Cx'],
                     [-m Cx'                        , 0]]
        with Cx=[c]x.  NOTE the bottom-left block multiplies the ANGULAR
        component of V.

        Must agree with `bias_forces` (autodiff oracle) to ~1e-9; that
        agreement is asserted in tests.
        """
        m = self.m
        if R_w is None or p_w is None:
            R_w, p_w = self.kinematics(q)
        if sub is None:
            sub = self.subspace_terms(q, R_w, p_w)
        if Iw is None:
            _, _, Iw = self._world_inertias(q, R_w, p_w)
        E = q.shape[0]
        dev, dt_ = q.device, q.dtype

        S = sub["S"]
        Sdq = self.sdot_terms(q, qd, R_w, p_w, sub)

        V = torch.zeros(E, self.nb, 6, dtype=dt_, device=dev)
        Av = torch.zeros(E, self.nb, 6, dtype=dt_, device=dev)
        for b in range(self.nb):
            ds = self._body_dofs[b]
            if ds:
                dst = torch.tensor(ds, device=dev)
                V[:, b] = (
                    S.index_select(1, dst) * qd.index_select(1, dst).unsqueeze(-1)
                ).sum(dim=1)
                # NOTE: Sdq rows are already contracted with their OWN rate
                # (sdot_terms returns Sdot_jw @ qd_j) -- do NOT multiply by
                # qd again here.
                Av[:, b] = Sdq.index_select(1, dst).sum(dim=1)

        com_w = self.com_positions(q, R_w, p_w)
        w, vo = V[..., :3], V[..., 3:]
        com_vel = vo + torch.linalg.cross(w, com_w, dim=-1)      # c_dot

        Om = skew(w)                                             # [E,nb,3,3]
        Cx = skew(com_w)
        Cxdot = skew(com_vel)

        # rotational inertia in world coords and its rate
        R_b = R_w
        Jw = torch.matmul(torch.matmul(R_b, m.inertia_com.expand(E, -1, -1, -1)),
                          R_b.transpose(-1, -2))
        Jdot = Om @ Jw - Jw @ Om

        me = m.masses.unsqueeze(-1).unsqueeze(-1)                # [E,nb,1,1]
        Itop_dot = Jdot - me * (Cxdot @ Cx + Cx @ Cxdot)
        TRdot = me * Cxdot                                       # d/dt (m c~x)

        IVdot_ang = (Itop_dot @ w.unsqueeze(-1)).squeeze(-1) \
            + (TRdot @ vo.unsqueeze(-1)).squeeze(-1)
        IVdot_lin = -(TRdot @ w.unsqueeze(-1)).squeeze(-1)
        IVdot = torch.cat([IVdot_ang, IVdot_lin], dim=-1)        # [E,nb,6]

        IA = torch.matmul(Iw, Av.unsqueeze(-1)).squeeze(-1)
        Fb = IA + IVdot

        FS = self._accumulate_subtrees(Fb)
        return torch.einsum("evi,evi->ev", sub["S"], FS[:, self.m.dof_body])

    # ------------------------------------------------------------------ #
    # integration helpers
    # ------------------------------------------------------------------ #

    def integrate(self, q, qd, dt: float):
        """Position update after velocity update (semi-implicit Euler)."""
        m = self.m
        qn = q.clone()
        for i in range(self.nb):
            jt = int(m.j_type[i])
            if jt in (J_HINGE, J_SLIDE):
                qi, vi = self._qs[i], self._vs[i]
                qn[:, qi] = q[:, qi] + dt * qd[:, vi]
            elif jt == J_FREE:
                f, v = self._fs[i], self._vs[i]
                qw = qd[:, v:v + 3]
                vw = qd[:, v + 3:v + 6]
                pos = q[:, f + 4:f + 7] + dt * vw
                R = quat_to_matrix(q[:, f:f + 4])
                wb = torch.einsum("eji,ej->ei", R, qw)             # R^T w (world->body)
                quat = quat_integrate(q[:, f:f + 4], wb, dt)
                qn[:, f:f + 4] = quat
                qn[:, f + 4:f + 7] = pos
        return qn
