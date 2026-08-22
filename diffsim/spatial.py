"""Spatial (6D) algebra, Featherstone convention.

Spatial motion vector:  v = [angular(3); linear(3)]
Spatial force vector:   f = [torque(3);  force(3)]
Spatial inertia about frame origin O with COM at c (frame coords):

    I = [[ J_c - m c~x c~x , m c~x ],
         [ -m c~x          , m 1  ]]

where c~x = skew(c) and J_c is the rotational inertia about the COM.

Motion transform X(R, p) maps a motion vector from a child frame {B}
(origin at p in parent coords, orientation R = R_parent<-child) to the
parent frame:

    X = [[ R        , 0 ],
         [ -R p~x   , R ]]

Inertia transforms as I_parent = X^{-T} I_child X^{-1}; we never build
X explicitly for that -- `spatial_inertia_world` applies the parallel-axis
theorem directly in world coordinates.
"""

from __future__ import annotations

import torch

from .linalg import skew


def motion_transform(R: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """[...,3,3], [...,3] -> [...,6,6] motion transform child -> parent."""
    *B, _ = p.shape
    Z = torch.zeros(*B, 3, 3, device=p.device, dtype=p.dtype)
    top = torch.cat([R, Z], dim=-1)
    bottom = torch.cat([-R @ skew(p), R], dim=-1)
    return torch.cat([top, bottom], dim=-2)


def spatial_inertia(mass: torch.Tensor, com: torch.Tensor, inertia_com: torch.Tensor) -> torch.Tensor:
    """Build spatial inertia about the body-frame origin.

    mass [...], com [...,3], inertia_com [...,3,3] -> [...,6,6]
    """
    cx = skew(com)
    Ibar = inertia_com - mass[..., None, None] * (cx @ cx)
    B = mass[..., None, None] * cx
    Zm = -mass[..., None, None] * cx
    Mblock = torch.diag_embed(mass.unsqueeze(-1).expand(*mass.shape, 3))
    top = torch.cat([Ibar, B], dim=-1)
    bottom = torch.cat([Zm, Mblock], dim=-1)
    return torch.cat([top, bottom], dim=-2)


def spatial_inertia_world(
    mass: torch.Tensor,
    com_w: torch.Tensor,
    rot_inertia_w: torch.Tensor,
) -> torch.Tensor:
    """Spatial inertia about the WORLD ORIGIN with world orientation.

    mass [...], com position in world [...,3], rotational inertia about
    COM expressed in world [...,3,3].

    Structure (ordering [angular; linear]):
        I = [[ J_c - m c~x c~x ,  m c~x ],
             [ -m c~x          ,  m 1   ]]
    e.g. linear momentum p = m (v_o - c x w) = m v_com.
    """
    cx = skew(com_w)
    Iang = rot_inertia_w - mass[..., None, None] * (cx @ cx)
    B = mass[..., None, None] * cx
    Mblock = torch.diag_embed(mass.unsqueeze(-1).expand(*mass.shape, 3))
    top = torch.cat([Iang, B], dim=-1)
    bottom = torch.cat([B.transpose(-1, -2), Mblock], dim=-1)   # = -m c~
    return torch.cat([top, bottom], dim=-2)


def transform_spatial_inertia(X: torch.Tensor, I: torch.Tensor) -> torch.Tensor:
    """I_new = X^{-T} I X^{-1}: express inertia given in child frame in parent frame."""
    Xi = torch.linalg.inv(X)
    return Xi.transpose(-1, -2) @ I @ Xi


def motion_cross(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Spatial motion cross product a x b ([...,6])."""
    wa, va = a[..., :3], a[..., 3:]
    wb, vb = b[..., :3], b[..., 3:]
    ang = torch.linalg.cross(wa, wb, dim=-1)
    lin = torch.linalg.cross(wa, vb, dim=-1) + torch.linalg.cross(va, wb, dim=-1)
    return torch.cat([ang, lin], dim=-1)


def force_cross(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Spatial force cross product a x* b ([...,6])."""
    wa, va = a[..., :3], a[..., 3:]
    fb, ff = b[..., :3], b[..., 3:]
    torque = torch.linalg.cross(wa, fb, dim=-1) + torch.linalg.cross(va, ff, dim=-1)
    force = torch.linalg.cross(wa, ff, dim=-1)
    return torch.cat([torque, force], dim=-1)


def rotate_motion(S: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Rotate both halves of a motion subspace matrix [...,6,K] by R [...,3,3]."""
    return torch.cat([R @ S[..., :3, :], R @ S[..., 3:, :]], dim=-3)


def twist_of_point(V: torch.Tensor, point_w: torch.Tensor) -> torch.Tensor:
    """Linear velocity of the material point at world position `point_w`
    given twist V about the world origin [...,6]. Returns [...,3]."""
    w, v = V[..., :3], V[..., 3:]
    return v + torch.linalg.cross(w, point_w, dim=-1)
