"""DiffSim: fully differentiable, GPU-accelerated articulated-body simulation."""

from .articulation import Articulation, Model, J_FIXED, J_FREE, J_HINGE, J_SLIDE
from .build import build_geoms_compat
from .collision import CAPSULE, Geoms, SPHERE
from .sim import ContactConfig, DiffSim, SimConfig

__all__ = [
    "Articulation", "Model", "J_FIXED", "J_FREE", "J_HINGE", "J_SLIDE",
    "Geoms", "CAPSULE", "SPHERE", "build_geoms_compat",
    "DiffSim", "SimConfig", "ContactConfig",
]
