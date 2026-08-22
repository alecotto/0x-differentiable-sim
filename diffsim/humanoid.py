"""SOMA-class humanoid model construction.

`make_soma_humanoid()` builds a full-size humanoid (floating base, 15
actuated dofs) with capsule/sphere collision geometry, joint limits,
damping, and realistic-ish segment inertias.  It serves as the default
training asset and as a template for loading real URDF/MJCF assets.

Topology (20 bodies, topological order):

    pelvis(FREE)
      torso(HINGE y) ── head(FIXED)
        l/r_upper_arm(HINGE x @shoulder) ── l/r_lower_arm(HINGE y @elbow)
      l_hip_pitch(HINGE y) ── l_hip_roll(HINGE x, massless frame)
          ── l_thigh(FIXED) ── l_shin(HINGE y @knee)
              ── l_ankle_pitch(HINGE y, massless) ── l_foot(FIXED)
      (right leg mirrored)

Zero-mass "frame" bodies implement multi-dof joints since every body
carries at most one hinge/slide/free joint.
"""

from __future__ import annotations

import math

import torch

from .articulation import J_FIXED, J_FREE, J_HINGE, Model
from .collision import CAPSULE, Geoms, SPHERE


def _eye(n):
    return torch.eye(n)


def _rot(axis, ang=0.0):
    """Axis-index rotation matrix."""
    c, s = math.cos(ang), math.sin(ang)
    if axis == 0:
        return torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=torch.float64)
    if axis == 1:
        return torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float64)
    return torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=torch.float64)


def _diag_inertia(ixx, iyy, izz):
    return torch.diag(torch.tensor([ixx, iyy, izz], dtype=torch.float64))


def make_soma_humanoid(scale: float = 1.0) -> tuple:
    """Build the default humanoid.

    Returns (model: Model, geoms_spec, feet_geoms) where geoms_spec is a
    list of dicts ready for `build_geoms`.
    """
    s = scale

    # ------------------------------------------------------------------
    # body table: (name, parent, jtype, axis, fix_p, fix_R, m, com, Icom_diag)
    # ------------------------------------------------------------------
    B = []
    A = (1.0, 0.0, 0.0)
    Y = (0.0, 1.0, 0.0)

    def cap_inertia(m_, r_, hl_):
        # capsule about center-of-mass (approx: cylinder + hemispheres)
        mc = m_
        ixx = mc * (0.25 * r_ ** 2 + (2 * hl_) ** 2 / 12.0 + 0.5 * r_ ** 2) * 0.5 \
            + mc * r_ ** 2 * 0.4
        return (
            0.5 * mc * r_ ** 2 + mc * hl_ ** 2 / 3.0,
            0.5 * mc * r_ ** 2 + mc * hl_ ** 2 / 3.0,
            mc * r_ ** 2 * 0.5,
        )

    def add(name, parent, jt, axis, fix_p, m_, com_z, ir, ihl=None, shape="capsule"):
        B.append((name, parent, jt, axis, torch.tensor(fix_p, dtype=torch.float64),
                  m_ * s, torch.tensor([0.0, 0.0, com_z]) if com_z is not None else None,
                  ir, ihl))

    hip_y, knee_y, ankle_y = -0.05 * s, -0.36 * s, -0.40 * s

    add("pelvis", -1, J_FREE, Y, [0, 0, 0], 10.0, 0.0, (0.09, 0.06))
    add("torso", 0, J_HINGE, Y, [0, 0, 0.08], 16.0, 0.13, (0.095, 0.14))
    add("head", 1, J_FIXED, Y, [0, 0, 0.30], 3.5, 0.06, (0.085, 0.08))

    for side, sx in (("l", 1.0), ("r", -1.0)):
        add(f"{side}_upper_arm", 1, J_HINGE, A, [sx * 0.19, 0, 0.24],
            1.6, -0.10, (0.04, 0.12))
        add(f"{side}_lower_arm", len(B) - 1, J_HINGE, Y, [0, 0, -0.26],
            1.1, -0.11, (0.035, 0.11))

    for side, sx in (("l", 1.0), ("r", -1.0)):
        base = len(B)
        add(f"{side}_hip_pitch", 0, J_HINGE, Y, [sx * 0.09, 0, -0.05], 0.30, 0.0, (0.03, 0.03))
        add(f"{side}_hip_roll", base, J_HINGE, A, [0, 0, 0.0], 0.30, 0.0, (0.03, 0.03))
        add(f"{side}_thigh", base + 1, J_FIXED, Y, [0, 0, 0.0], 4.5, -0.17, (0.062, 0.17))
        add(f"{side}_shin", base + 2, J_HINGE, Y, [0, 0, knee_y], 2.6, -0.18, (0.05, 0.18))
        add(f"{side}_ankle_pitch", base + 3, J_HINGE, Y, [0, 0, ankle_y], 0.15, 0.0, (0.025, 0.025))
        add(f"{side}_ankle_roll", base + 4, J_HINGE, A, [0, 0, 0.0], 0.15, 0.0, (0.025, 0.025))
        add(f"{side}_foot", base + 5, J_FIXED, Y, [0, 0, 0.0], 0.9, 0.0, (0.038, 0.10))

    names = [b[0] for b in B]
    nb = len(B)
    parent = [(b[1] if b[1] >= 0 else -1) for b in B]
    fix_R = torch.stack([_rot(2, 0.0) for _ in B])
    fix_p = torch.stack([b[4] for b in B])
    j_type = torch.tensor([b[2] for b in B])
    j_axis = torch.tensor([b[3] for b in B], dtype=torch.float64)
    masses = torch.tensor([b[5] for b in B], dtype=torch.float64)

    com_list = []
    for i, b in enumerate(B):
        cz = b[6]
        if cz is None:
            com_list.append(torch.zeros(3, dtype=torch.float64))
        elif int(b[2]) == J_FREE:
            com_list.append(torch.tensor([0.0, 0.0, 0.05], dtype=torch.float64))
        else:
            com_list.append(cz.to(torch.float64))
    com = torch.stack(com_list)

    icom = torch.stack([
        _diag_inertia(*cap_inertia(b[5], b[7][0], b[7][1]))
        if b[5] > 0.05 else _diag_inertia(3e-3, 3e-3, 3e-3)
        for b in B
    ])

    # dof bookkeeping ----------------------------------------------------
    dof_body, body_dof_start = [], []
    q_free_start = -1
    qi, vi = 0, 0
    for i, b in enumerate(B):
        jt = int(b[2])
        if jt == J_FREE:
            q_free_start = qi
            body_dof_start.append(vi)
            dof_body.extend([i] * 6)
            qi += 7
            vi += 6
        elif jt == J_HINGE:
            body_dof_start.append(vi)
            dof_body.append(i)
            qi += 1
            vi += 1
        else:
            body_dof_start.append(-1)
    nq, nv = qi, vi

    # limits & damping on actuated dofs -----------------------------------
    actuated = [d for d, bi in enumerate(dof_body) if int(B[bi][2]) == J_HINGE]
    lo = torch.full((len(actuated),), -2.6, dtype=torch.float64)
    hi = torch.full((len(actuated),), 2.6, dtype=torch.float64)
    # knees cannot hyperextend: leg hinge dofs start after torso+arms
    first_leg_dof = actuated[1 + 4]          # skip torso (1) + 2x2 arm dofs
    knee_idx = [k for k, d in enumerate(actuated)
                if d >= first_leg_dof + 2]   # hip_pitch, hip_roll, then knee
    for k in knee_idx:
        lo[k] = -math.pi / 2 * 0.95
        hi[k] = 0.05
    damping = torch.full((nv,), 0.6, dtype=torch.float64)
    damping[:6] = 0.05  # base dofs: light damping only

    model = Model(
        n_bodies=nb, parent=parent, body_names=names,
        fix_R=fix_R, fix_p=fix_p, j_type=j_type, j_axis=j_axis,
        masses=masses, com=com, inertia_com=icom,
        q_dim=nq, v_dim=nv, dof_body=dof_body, body_dof_start=body_dof_start,
        q_free_start=q_free_start,
        joint_limit_lo=lo, joint_limit_hi=hi,
        limit_dof_idx=torch.tensor(actuated, dtype=torch.long),
        damping=damping, armature=0.01,
    )

    # ------------------------------------------------------------------
    # collision geoms
    # ------------------------------------------------------------------
    gspec = []

    def g(name, body_name, p, shape, r, hl=0.0, rot_axis=None, ang=0.0, ground=True):
        gspec.append(dict(name=name, body=names.index(body_name),
                          p=torch.tensor(p, dtype=torch.float64),
                          shape=shape, r=r * s, hl=hl * s,
                          R=_rot(rot_axis, ang) if rot_axis is not None else _eye(3),
                          ground=ground))

    g("pelvis_c", "pelvis", [0, 0, 0.02], CAPSULE, 0.09, 0.07)
    g("torso_c", "torso", [0, 0, 0.13], CAPSULE, 0.095, 0.14)
    g("head_s", "head", [0, 0, 0.06], SPHERE, 0.085)
    for side in ("l", "r"):
        sx = 1.0 if side == "l" else -1.0
        g(f"{side}_uarm", f"{side}_upper_arm", [0, 0, -0.11], CAPSULE, 0.04, 0.11)
        g(f"{side}_farm", f"{side}_lower_arm", [0, 0, -0.12], CAPSULE, 0.035, 0.10)
        g(f"{side}_thigh_c", f"{side}_thigh", [0, 0, -0.17], CAPSULE, 0.062, 0.165)
        g(f"{side}_shin_c", f"{side}_shin", [0, 0, -0.18], CAPSULE, 0.05, 0.175)
        # foot: horizontal capsule pointing forward (+x)
        g(f"{side}_foot_c", f"{side}_foot", [0.03, sx * 0.0, -0.035], CAPSULE,
          0.038, 0.10, rot_axis=1, ang=math.pi / 2)

    feet_geoms = [gs["name"] for gs in gspec if gs["name"].endswith("_foot_c")]
    return model, gspec, feet_geoms


def build_geoms(gspec, device="cpu", dtype=torch.float64) -> Geoms:
    from .collision import CAPSULE, SPHERE

    body = torch.tensor([g["body"] for g in gspec], dtype=torch.long)
    local_p = torch.stack([g["p"] for g in gspec]).to(dtype)
    local_R = torch.stack([g["R"] for g in gspec]).to(dtype)
    gtype = torch.tensor(
        [CAPSULE if g["shape"] == "capsule" else SPHERE for g in gspec], dtype=torch.long
    )
    radius = torch.tensor([g["r"] for g in gspec], dtype=dtype)
    half_len = torch.tensor([g.get("hl", 0.0) for g in gspec], dtype=dtype)
    collide_ground = torch.tensor([bool(g.get("ground", True)) for g in gspec])
    return Geoms(body=body, local_p=local_p, local_R=local_R, gtype=gtype,
                 radius=radius, half_len=half_len, collide_ground=collide_ground,
                 names=[g["name"] for g in gspec])


def initial_pose(model: Model, batch: int, device="cpu",
                 dtype=torch.float64) -> tuple:
    """Neutral standing state (q, qd): legs straight, feet resting on z=0.

    Pelvis origin height = hip->ankle drop + foot geom clearance below
    the ankle (geom center offset 0.035 + radius 0.038).
    """
    q0 = torch.zeros(batch, model.q_dim, dtype=dtype, device=device)
    q0[:, model.q_free_start] = 1.0                               # identity quat
    foot_drop = 0.05 + 0.36 + 0.40                                # pelvis->ankle
    sole_clearance = 0.035 + 0.038
    q0[:, model.q_free_start + 6] = foot_drop + sole_clearance    # pelvis origin z
    qd = torch.zeros(batch, model.v_dim, dtype=dtype, device=device)
    return q0, qd
