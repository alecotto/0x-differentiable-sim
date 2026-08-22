"""Differentiable collision detection and contact force model.

Primitives
----------
SPHERE   : center + radius
CAPSULE  : segment (a,b) + radius  (capsule axis = local z of geom frame)
PLANE    : ground plane z <= 0 (infinite, world-fixed)

Distances are exact and piecewise-smooth; contact activation uses a C^inf
softplus ramp so penetration depth, normals, and forces are differentiable
everywhere -- a hard requirement for backprop-through-simulation.

Contact model
-------------
Normal force : f_n = k * softplus_pen(margin - d)   (+ normal damping term)
Friction     : regularized viscous Coulomb in the tangent plane,
               scaled by the (smoothed) normal impulse.
Action/reaction is enforced by construction: equal and opposite point
forces are applied at shared witness points.
"""

from __future__ import annotations

import dataclasses
from typing import List, Optional, Tuple

import torch

SPHERE, CAPSULE = 0, 1


@dataclasses.dataclass
class Geoms:
    """Flat array of collision geoms attached to bodies."""

    body: torch.Tensor        # [G] long
    local_p: torch.Tensor     # [G,3]
    local_R: torch.Tensor     # [G,3,3]
    gtype: torch.Tensor       # [G] long
    radius: torch.Tensor      # [G]
    half_len: torch.Tensor    # [G] (capsules; 0 for spheres)
    collide_ground: torch.Tensor  # [G] bool
    names: List[str]

    def to(self, device, dtype=torch.float64):
        return dataclasses.replace(
            self,
            body=self.body.to(device),
            local_p=self.local_p.to(device=device, dtype=dtype),
            local_R=self.local_R.to(device=device, dtype=dtype),
            gtype=self.gtype.to(device),
            radius=self.radius.to(device=device, dtype=dtype),
            half_len=self.half_len.to(device=device, dtype=dtype),
            collide_ground=self.collide_ground.to(device),
        )

    def __len__(self):
        return int(self.body.shape[0])

    def endpoints(self, centers: torch.Tensor, Rg: torch.Tensor):
        """Capsule segment endpoints for every geom: ([E,G,3],[E,G,3]).
        Spheres return their center for both endpoints."""
        hl = self.half_len.unsqueeze(0).unsqueeze(-1)          # [1,G,1]
        off = Rg[..., :, 2] * hl                               # [E,G,3] along local z
        return centers - off, centers + off


def softplus_pen(x: torch.Tensor, beta: float = 50.0) -> torch.Tensor:
    """C^inf approximation of relu(x); x may be any sign."""
    return torch.nn.functional.softplus(x * beta) / beta


def smooth_ramp(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Smooth hinge: exactly 0 at x<=~0, linear for x >> eps.

    r(x) = 0.5*(x + sqrt(x^2 + eps^2)) - 0.5*eps

    Unlike a shifted softplus this has *zero* value AND near-zero force at
    touching (x=0), removing phantom pre-load forces.
    """
    return 0.5 * (x + torch.sqrt(x * x + eps * eps)) - 0.5 * eps


# --------------------------------------------------------------------- #
# primitive distances: return (dist, p_on_A [...,3], p_on_B [...,3])
# --------------------------------------------------------------------- #

def dist_sphere_sphere(c1, r1, c2, r2):
    diff = c1 - c2
    dist = torch.linalg.vector_norm(diff, dim=-1) - r1 - r2
    return dist, c1, c2


def dist_sphere_plane(c, r):
    dist = c[..., 2] - r
    pw = torch.stack([c[..., 0], c[..., 1], torch.zeros_like(dist)], dim=-1)
    return dist, c, pw


def _closest_point_on_segment(p, a, b):
    ab = b - a
    t = torch.einsum("...i,...i->...", ab, p - a) / torch.clamp(
        torch.einsum("...i,...i->...", ab, ab), min=1e-12
    )
    t = torch.clamp(t, 0.0, 1.0)
    return a + t.unsqueeze(-1) * ab


def dist_sphere_capsule(p, r, a, b, rc):
    cp = _closest_point_on_segment(p, a, b)
    diff = p - cp
    dist = torch.linalg.vector_norm(diff, dim=-1) - r - rc
    return dist, p, cp


def dist_capsule_capsule(a1, b1, r1, a2, b2, r2):
    """Exact segment-segment distance (Ericson), witness points returned."""
    d1 = b1 - a1
    d2 = b2 - a2
    r = a1 - a2
    aa = (d1 * d1).sum(-1)
    ee = (d2 * d2).sum(-1)
    f = (d2 * r).sum(-1)
    EPSV = 1e-10
    c = (d1 * r).sum(-1)
    bb = (d1 * d2).sum(-1)
    denom = aa * ee - bb * bb
    safe_denom = torch.where(denom > EPSV, denom, torch.ones_like(denom))
    s = torch.where(denom > EPSV, torch.clamp((bb * f - ee * c) / safe_denom, 0.0, 1.0),
                    torch.zeros_like(denom))
    # handle degenerate (parallel / zero-length) segments
    parallel = denom <= EPSV
    s = torch.where(parallel & (aa > EPSV), torch.clamp(-c / torch.clamp(aa, min=EPSV), 0.0, 1.0), s)
    t = torch.where(ee > EPSV, torch.clamp((bb * s + f) / torch.clamp(ee, min=EPSV), 0.0, 1.0),
                    torch.zeros_like(ee))
    # recompute s for clamped t (only where segment 1 non-degenerate)
    s2 = torch.where(aa > EPSV, torch.clamp((bb * t - c) / torch.clamp(aa, min=EPSV), 0.0, 1.0), s)
    s = torch.where(~parallel, s2, s)
    del EPSV

    p1 = a1 + s.unsqueeze(-1) * d1
    p2 = a2 + t.unsqueeze(-1) * d2
    dist = torch.linalg.vector_norm(p1 - p2, dim=-1) - r1 - r2
    return dist, p1, p2


# --------------------------------------------------------------------- #
# pair evaluation
# --------------------------------------------------------------------- #

def eval_pairs(g: Geoms, centers, Rg, idx_i, idx_j):
    """Evaluate a fixed list of geom pairs. Returns dict of tensors [E,P]."""
    ci, cj = centers[:, idx_i], centers[:, idx_j]
    ti, tj = g.gtype[idx_i], g.gtype[idx_j]
    ri, rj = g.radius[idx_i], g.radius[idx_j]
    ai_all, bi_all = g.endpoints(centers, Rg)
    ai, bi = ai_all[:, idx_i], bi_all[:, idx_i]
    aj, bj = ai_all[:, idx_j], bi_all[:, idx_j]

    # dispatch: assume homogeneous groups provided by caller; here we
    # implement the general case via masks.
    E, P = ci.shape[0], len(idx_i)
    dev, dt = ci.device, ci.dtype
    dist = torch.zeros(E, P, dtype=dt, device=dev)
    p1 = torch.zeros(E, P, 3, dtype=dt, device=dev)
    p2 = torch.zeros(E, P, 3, dtype=dt, device=dev)

    m_ss = (ti == SPHERE) & (tj == SPHERE)
    m_sc = (ti == SPHERE) & (tj == CAPSULE)
    m_cs = (ti == CAPSULE) & (tj == SPHERE)
    m_cc = (ti == CAPSULE) & (tj == CAPSULE)

    if m_ss.any():
        d, w1, w2 = dist_sphere_sphere(ci[:, m_ss], ri[m_ss], cj[:, m_ss], rj[m_ss])
        dist[:, m_ss], p1[:, m_ss], p2[:, m_ss] = d, w1, w2
    if m_sc.any():
        d, w1, w2 = dist_sphere_capsule(ci[:, m_sc], ri[m_sc], aj[:, m_sc], bj[:, m_sc], rj[m_sc])
        dist[:, m_sc], p1[:, m_sc], p2[:, m_sc] = d, w1, w2
    if m_cs.any():
        d, w2, w1 = dist_sphere_capsule(cj[:, m_cs], rj[m_cs], ai[:, m_cs], bi[:, m_cs], ri[m_cs])
        dist[:, m_cs], p1[:, m_cs], p2[:, m_cs] = d, w1, w2
    if m_cc.any():
        d, w1, w2 = dist_capsule_capsule(ai[:, m_cc], bi[:, m_cc], ri[m_cc],
                                         aj[:, m_cc], bj[:, m_cc], rj[m_cc])
        dist[:, m_cc], p1[:, m_cc], p2[:, m_cc] = d, w1, w2
    return {"dist": dist, "p1": p1, "p2": p2}


def eval_ground(g: Geoms, centers, Rg):
    """Ground-plane contacts for every geom, TWO points per capsule.

    Each capsule segment endpoint generates its own independent contact
    (own signed distance, own witness point) -- restoring a real support
    polygon under flat feet and eliminating the argmin endpoint switch.
    Spheres degenerate to coincident endpoints; callers apply a static
    weight mask so they are counted once.

    Returns dict with:
      dist    [E,G,2]  signed distance per endpoint
      p_body  [E,G,2,3] closest point ON THE BODY under each endpoint
      p_world [E,G,2,3] projection on the plane
    """
    ai, bi = g.endpoints(centers, Rg)
    r = g.radius.unsqueeze(-1)                                     # [G,1]
    # stack endpoints: index -1 => point "a", index -2 => point "b"
    ex = torch.stack([ai, bi], dim=-2)                             # [E,G,2,3]
    dist = ex[..., 2] - r                                          # [E,G,2]
    p_body = torch.cat([ex[..., :2],
                        (ex[..., 2] - r).unsqueeze(-1)], dim=-1)   # [E,G,2,3]
    p_world = torch.stack([ex[..., 0], ex[..., 1],
                           torch.zeros_like(ex[..., 0])], dim=-1)
    return {"dist": dist, "p_body": p_body, "p_world": p_world}
