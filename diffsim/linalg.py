"""Batched, fully differentiable rigid-body math primitives.

Conventions
-----------
* Quaternions are [w, x, y, z] (scalar-first).
* Rotation matrices R map body-local vectors to world coordinates (R = R_wb).
* All functions are pure torch ops, broadcast over arbitrary leading batch dims,
  and differentiable end-to-end.
"""

from __future__ import annotations

import torch

EPS = 1e-9


def skew(v: torch.Tensor) -> torch.Tensor:
    """[..., 3] -> [..., 3, 3] skew-symmetric cross-product matrix."""
    x, y, z = v.unbind(-1)
    o = torch.zeros_like(x)
    return torch.stack(
        [o, -z, y,
         z, o, -x,
         -y, x, o],
        dim=-1,
    ).reshape(*v.shape[:-1], 3, 3)


def quat_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / torch.clamp(torch.linalg.vector_norm(q, dim=-1, keepdim=True), min=EPS)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product; supports broadcasting."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def quat_conj(q: torch.Tensor) -> torch.Tensor:
    qc = q.clone()
    qc[..., 1:] = -qc[..., 1:]
    return qc


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors v by quaternions q."""
    t = 2.0 * torch.linalg.cross(q[..., 1:], v, dim=-1)
    return v + q[..., :1] * t + torch.linalg.cross(q[..., 1:], t, dim=-1)


def quat_rotate_inv(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return quat_rotate(quat_conj(q), v)


def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """[...4] -> [...3,3]. Assumes unit quaternion (normalizes internally)."""
    q = quat_normalize(q)
    w, x, y, z = q.unbind(-1)
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    m = torch.stack(
        [
            1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
            2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
            2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy),
        ],
        dim=-1,
    )
    return m.reshape(*q.shape[:-1], 3, 3)


def matrix_to_quat(R: torch.Tensor) -> torch.Tensor:
    """Robust Shepperd-style conversion [...,3,3] -> [...,4]."""
    m = R
    tr = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]

    # Candidate raw quaternions for each of the four branches.
    q_w = torch.stack(
        [1 + tr, m[..., 2, 1] - m[..., 1, 2], m[..., 0, 2] - m[..., 2, 0], m[..., 1, 0] - m[..., 0, 1]],
        dim=-1,
    )
    q_x = torch.stack(
        [m[..., 2, 1] - m[..., 1, 2], 1 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2],
         m[..., 0, 1] + m[..., 1, 0], m[..., 0, 2] + m[..., 2, 0]],
        dim=-1,
    )
    q_y = torch.stack(
        [m[..., 0, 2] - m[..., 2, 0], m[..., 0, 1] + m[..., 1, 0],
         1 + m[..., 1, 1] - m[..., 0, 0] - m[..., 2, 2], m[..., 1, 2] + m[..., 2, 1]],
        dim=-1,
    )
    q_z = torch.stack(
        [m[..., 1, 0] - m[..., 0, 1], m[..., 0, 2] + m[..., 2, 0],
         m[..., 1, 2] + m[..., 2, 1], 1 + m[..., 2, 2] - m[..., 0, 0] - m[..., 1, 1]],
        dim=-1,
    )

    s_w = torch.sqrt(torch.clamp(1.0 + tr, min=EPS)) * 2.0
    s_x = torch.sqrt(torch.clamp(1.0 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2], min=EPS)) * 2.0
    s_y = torch.sqrt(torch.clamp(1.0 + m[..., 1, 1] - m[..., 0, 0] - m[..., 2, 2], min=EPS)) * 2.0
    s_z = torch.sqrt(torch.clamp(1.0 + m[..., 2, 2] - m[..., 0, 0] - m[..., 1, 1], min=EPS)) * 2.0

    use_z = (m[..., 2, 2] > m[..., 0, 0]) & (m[..., 2, 2] > m[..., 1, 1]) & (tr <= 0.0)
    use_y = (m[..., 1, 1] > m[..., 0, 0]) & ~use_z & (tr <= 0.0)

    q = q_x / s_x.unsqueeze(-1)
    q = torch.where(use_y.unsqueeze(-1), q_y / s_y.unsqueeze(-1), q)
    q = torch.where(use_z.unsqueeze(-1), q_z / s_z.unsqueeze(-1), q)
    q = torch.where((tr > 0).unsqueeze(-1), q_w / s_w.unsqueeze(-1), q)
    return quat_normalize(q)


def exp_so3(w: torch.Tensor) -> torch.Tensor:
    """Rotation vector -> quaternion (Rodrigues via half-angle)."""
    theta = torch.linalg.vector_norm(w, dim=-1, keepdim=True)
    half = 0.5 * theta
    s = torch.where(theta > EPS, torch.sin(half) / torch.clamp(theta, min=EPS), 0.5 - theta.pow(2) / 48.0)
    q = torch.cat([torch.cos(half), s * w], dim=-1)
    return quat_normalize(q)


def log_so3(q: torch.Tensor) -> torch.Tensor:
    """Quaternion -> rotation vector."""
    q = quat_normalize(q)
    w = torch.clamp(q[..., :1], min=-1.0, max=1.0)
    vec = q[..., 1:]
    sin_half = torch.linalg.vector_norm(vec, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half, w)
    scale = torch.where(sin_half > EPS, angle / torch.clamp(sin_half, min=EPS), 2.0 * torch.ones_like(angle))
    return scale * vec


def quat_integrate(q: torch.Tensor, omega_body: torch.Tensor, dt: torch.Tensor | float) -> torch.Tensor:
    """Integrate quaternion by body-frame angular velocity over dt (exact exp map)."""
    if not torch.is_tensor(dt):
        dt = torch.as_tensor(float(dt))
    dq = exp_so3(omega_body * dt)
    return quat_normalize(quat_mul(q, dq))


def transform_points(R: torch.Tensor, p: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """R [...,3,3], p [...,3], pts [...,N,3] -> world points."""
    return torch.einsum("...ij,...nj->...ni", R, pts) + p.unsqueeze(-2)


def make_transform(R: torch.Tensor | None = None, p: torch.Tensor | None = None):
    if R is None:
        R = torch.eye(3)
    if p is None:
        p = torch.zeros(3)
    return R, p


def xform_apply(R: torch.Tensor, p: torch.Tensor, other_R: torch.Tensor, other_p: torch.Tensor):
    """Compose transforms: result applies `other` first then (R, p)."""
    return R @ other_R, R @ other_p + p
