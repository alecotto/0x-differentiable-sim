"""Compatibility helper: build Geoms from spec dicts (used by humanoid.py)."""

from __future__ import annotations

import torch

from .collision import Geoms


def build_geoms_compat(gspec, device="cpu", dtype=torch.float64) -> Geoms:
    from .collision import CAPSULE, SPHERE

    body = torch.tensor([g["body"] for g in gspec], dtype=torch.long, device=device)
    local_p = torch.stack([g["p"] for g in gspec]).to(dtype)
    local_R = torch.stack([g["R"] for g in gspec]).to(dtype)
    gtype = torch.tensor(
        [CAPSULE if g["shape"] == "capsule" else SPHERE for g in gspec],
        dtype=torch.long, device=device,
    )
    radius = torch.tensor([g["r"] for g in gspec], dtype=dtype, device=device)
    half_len = torch.tensor([g.get("hl", 0.0) for g in gspec], dtype=dtype, device=device)
    collide_ground = torch.tensor([bool(g.get("ground", True)) for g in gspec], device=device)
    return Geoms(body=body, local_p=local_p, local_R=local_R, gtype=gtype,
                 radius=radius, half_len=half_len,
                 collide_ground=collide_ground,
                 names=[g["name"] for g in gspec])
