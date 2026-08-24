"""Garcia-style simplest-walker TWIN built inside DiffSim (Q1c).

Morphology (single parameter source shared with the sympy oracle in
scripts/walker_oracle.py):

    hip   : point mass M (free joint), tiny rotational inertia I_hip
            (numerical regularizer only; dynamics ~ point mass)
    leg_a : massless rod carrying a POINT mass beta*M at its tip
            (hinge about world-y at the hip; com sits at the tip)
    leg_b : mirrored second leg

The walker descends a slope of angle gamma WITHOUT any slope geometry:
gravity is tilted to ghat = g*(sin(gamma), 0, -cos(gamma)), which is the
Galilean-equivalent flat-ground problem.  Zero changes to validated
collision code.

Feet are SPHERE geoms of radius r_foot << l centered at the point masses;
they provide the soft contact (k_ground, damping, friction) whose ability
to reproduce rigid-limit hybrid walking dynamics is precisely what Q1
tests.

Model choices documented for rigor:
  * no joint limits, no damping, no armature  (passive machine)
  * leg-leg self-collision DISABLED after construction: Garcia's model
    declares scuffing nonexistent; the twin must not silently resolve it
  * hip inertia 1e-4 kg m^2 ~ 2e-5 * (total m l^2): keeps the free-joint
    quaternion well conditioned without measurably changing dynamics
"""
from __future__ import annotations

import math

import torch

from .articulation import J_FREE, J_HINGE, Model
from .collision import Geoms, SPHERE

# ---- shared morphology parameters (oracle imports this dict) ------------
WALKER_P = {
    "M": 10.0,          # hip point mass [kg]
    "beta": 0.02,       # foot point mass = beta * M
    "l": 0.5,           # leg length [m]
    "r_foot": 1e-3,     # foot sphere radius [m]
    "I_hip": 1.0e-3,    # hip rotational inertia regularizer [kg m^2]
}
# NOTE on beta: Garcia's asymptotics are for beta -> 0; the TWIN keeps a
# finite foot mass because the shared contact law (k=2.5e4, b=400) is
# only explicitly stable when m_foot/b >> dt (dt=5e-5 needs m_foot/b
# >= ~2.5e-4 kg s).  Measured stability boundary of the ORBIT itself:
# at gamma=0.012 the period-one FP destabilizes between beta=0.02
# (rho=0.58) and beta=0.035 (rho=2.18), absent at beta=0.05 -- so
# beta=0.02 sits in the last both-stable window.  Twin and oracle share
# WALKER_P; external anchoring runs through the O(beta) accel check.


def make_walker(params=None):
    """Build (model, gspec, feet_geoms, aux) for the passive biped."""
    p = dict(WALKER_P if params is None else params)
    M, mf, l, r, Ih = p["M"], p["beta"] * p["M"], p["l"], p["r_foot"], p["I_hip"]

    Y = torch.tensor([0.0, 1.0, 0.0])
    eye = torch.eye(3, dtype=torch.float64)

    names = ["hip", "leg_a", "leg_b"]
    parent = [-1, 0, 0]
    j_type = torch.tensor([J_FREE, J_HINGE, J_HINGE])
    j_axis = torch.stack([Y, Y, Y])
    fix_R = torch.stack([eye, eye, eye])
    fix_p = torch.stack([torch.zeros(3, dtype=torch.float64),
                         torch.zeros(3, dtype=torch.float64),
                         torch.zeros(3, dtype=torch.float64)])
    masses = torch.tensor([M, mf, mf], dtype=torch.float64)
    # link-frame origins sit at the hip; leg com at the tip (point mass)
    com = torch.stack([torch.zeros(3, dtype=torch.float64),
                       torch.tensor([0.0, 0.0, -l], dtype=torch.float64),
                       torch.tensor([0.0, 0.0, -l], dtype=torch.float64)])
    tiny = torch.diag(torch.tensor([1e-9, 1e-9, 1e-9], dtype=torch.float64))
    ih = torch.diag(torch.tensor([Ih, Ih, Ih], dtype=torch.float64))
    inertia_com = torch.stack([ih, tiny, tiny])

    # dof bookkeeping (mirrors humanoid.py conventions)
    dof_body, body_dof_start = [], []
    qi = vi = 0
    q_free_start = -1
    for i, jt in enumerate(j_type.tolist()):
        if jt == int(J_FREE):
            q_free_start = qi
            body_dof_start.append(vi)
            dof_body.extend([i] * 6)
            qi += 7
            vi += 6
        elif jt == int(J_HINGE):
            body_dof_start.append(vi)
            dof_body.append(i)
            qi += 1
            vi += 1
        else:
            body_dof_start.append(-1)

    model = Model(
        n_bodies=3, parent=parent, body_names=names,
        fix_R=fix_R, fix_p=fix_p, j_type=j_type, j_axis=j_axis,
        masses=masses, com=com, inertia_com=inertia_com,
        q_dim=qi, v_dim=vi, dof_body=dof_body,
        body_dof_start=body_dof_start, q_free_start=q_free_start,
        gravity=(0.0, 0.0, -9.81),       # overwritten by slope helpers
        limit_dof_idx=None, damping=None, armature=0.0,
    )

    gspec = []
    for i, nm in enumerate(("foot_a", "foot_b")):
        gspec.append(dict(name=nm, body=i + 1,
                          p=torch.tensor([0.0, 0.0, -l], dtype=torch.float64),
                          shape="sphere", r=r, hl=0.0, R=eye, ground=True))
    feet_geoms = [g["name"] for g in gspec]

    def slope_gravity(gamma_deg: float):
        """Return the gravity tuple for a downhill slope gamma (degrees)."""
        gam = math.radians(gamma_deg)
        return (9.81 * math.sin(gam), 0.0, -9.81 * math.cos(gam))

    aux = {"params": p, "slope_gravity": slope_gravity}
    return model, gspec, feet_geoms, aux


def build_geoms_simple(gspec, device="cpu", dtype=torch.float64) -> Geoms:
    from .collision import CAPSULE

    body = torch.tensor([g["body"] for g in gspec], dtype=torch.long)
    local_p = torch.stack([g["p"] for g in gspec]).to(dtype)
    local_R = torch.stack([g["R"] for g in gspec]).to(dtype)
    gtype = torch.tensor(
        [CAPSULE if g["shape"] == "capsule" else SPHERE for g in gspec],
        dtype=torch.long)
    radius = torch.tensor([g["r"] for g in gspec], dtype=dtype)
    half_len = torch.tensor([g.get("hl", 0.0) for g in gspec], dtype=dtype)
    collide_ground = torch.tensor(
        [bool(gg.get("ground", True)) for gg in gspec])
    return Geoms(body=body, local_p=local_p, local_R=local_R, gtype=gtype,
                 radius=radius, half_len=half_len,
                 collide_ground=collide_ground.expand(len(gspec)),
                 names=[g["name"] for g in gspec])


def walker_state(model, th_a: float, om_a: float, th_b: float, om_b: float,
                 batch: int = 1, device="cpu", dtype=torch.float64):
    """Full (q, qd) from absolute leg angles/rates, hip at rest, upright.

    Leg hinge angles ARE absolute world angles while the base orientation is
    identity (planar motion keeps it there; `planarity_error` monitors).
    Hip placed so the lower tip of leg_a rests exactly on z=0.
    """
    q = torch.zeros(batch, model.q_dim, dtype=dtype, device=device)
    w = torch.zeros(batch, model.v_dim, dtype=dtype, device=device)
    fs = model.q_free_start
    q[:, fs] = 1.0
    qa = model_q_index(model, "leg_a")
    qb = model_q_index(model, "leg_b")
    q[:, qa] = th_a
    q[:, qb] = th_b
    # place hip so foot_a tip touches z = r_foot exactly
    r = WALKER_P["r_foot"]
    q[:, fs + 6] = WALKER_P["l"] * math.cos(th_a) + r
    va = model_v_index(model, "leg_a")
    vb = model_v_index(model, "leg_b")
    w[:, va] = om_a
    w[:, vb] = om_b
    return q, w


def model_q_index(model, body_name: str) -> int:
    b = model.body_names.index(body_name)
    return model.body_dof_start[b]


def model_v_index(model, body_name: str) -> int:
    return model_q_index(model, body_name)


def planarity_error(sim, q) -> torch.Tensor:
    """Max deviation of the base rotation from identity (must stay ~0)."""
    R_w, _ = sim.art.kinematics(q)
    R0 = R_w[:, 0]
    err = R0 - torch.eye(3, dtype=R0.dtype, device=R0.device).unsqueeze(0)
    return err.abs().reshape(q.shape[0], -1).max(dim=-1).values


def leg_angles_world(sim, q):
    """Absolute leg angles + tips from world FK (no reliance on base frame).

    Returns dict with th_[ab] [E], tip_[ab] [E,3] (sphere centers)."""
    R_w, p_w = sim.art.kinematics(q)
    centers, _ = sim.art.geoms_world(q, sim.geoms, R_w, p_w)
    hip = p_w[:, 0]
    out = {}
    for k, nm in (("a", "foot_a"), ("b", "foot_b")):
        gi = sim.geoms.names.index(nm)
        tip = centers[:, gi]
        d = hip - tip
        out[f"th_{k}"] = torch.atan2(d[..., 0], d[..., 2])
        out[f"tip_{k}"] = tip
    return out
