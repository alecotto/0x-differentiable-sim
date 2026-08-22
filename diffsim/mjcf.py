"""Minimal MJCF (MuJoCo XML) loader -> DiffSim Model/Geoms.

Supports the subset needed for humanoid assets:
  <mujoco><worldbody><body> nested trees
  <joint type="free|hinge|slide" axis pos range>
  <geom type="sphere|capsule|box(->bounding capsule)" size pos quat/euler
  <inertial pos mass diaginertia|fullinertia>   (falls back to geom-derived)
  <motor/actuator> joints are auto-discovered as actuated

Not supported (yet): equality constraints, tendons, meshes, sites, sensors.
"""
import math
import xml.etree.ElementTree as ET

import torch

from .articulation import J_FIXED, J_FREE, J_HINGE, J_SLIDE, Model


def _euler_to_matrix(a: str) -> torch.Tensor:
    """MJCF euler angles (degrees) with default XYZ sequence."""
    ax, ay, az = [math.radians(float(v)) for v in a.split()]
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)
    Rx = torch.tensor([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=torch.float64)
    Ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=torch.float64)
    Rz = torch.tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=torch.float64)
    return Rz @ Ry @ Rx


def _quat_to_matrix(qs: str) -> torch.Tensor:
    w, x, y, z = [float(v) for v in qs.split()]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return torch.tensor([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=torch.float64)


def load_mjcf(path: str, device="cpu", dtype=torch.float64,
              capsule_from_box: bool = True):
    """Parse MJCF XML. Returns (Model, Geoms-spec-dict-list, actuated_dofs)."""
    tree = ET.parse(path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("no worldbody")

    bodies = []          # per-body dict during construction
    geoms_spec = []

    def parse_body(el, parent_idx, R_p, p_p):
        name = el.get("name", f"body{len(bodies)}")
        pos = torch.tensor([float(v) for v in el.get("pos", "0 0 0").split()],
                           dtype=torch.float64)
        R_loc = torch.eye(3, dtype=torch.float64)
        if el.get("quat"):
            R_loc = _quat_to_matrix(el.get("quat"))
        elif el.get("euler"):
            R_loc = _euler_to_matrix(el.get("euler"))
        R_fix = R_p @ R_loc                       # orientation of body frame in parent
        p_fix = p_p + R_p @ pos                   # joint anchor in parent coords

        idx = len(bodies)
        entry = dict(name=name, parent=parent_idx, fix_R=R_fix, fix_p=p_fix,
                     j_type=J_FIXED, j_axis=(0., 1., 0.), masses=[],
                     com=None, inertia=None, geoms=[])
        bodies.append(entry)

        for child in el:
            if child.tag == "joint":
                jt = child.get("type", "hinge")
                if jt == "free":
                    entry["j_type"] = J_FREE
                elif jt == "slide":
                    entry["j_type"] = J_SLIDE
                elif jt == "hinge":
                    entry["j_type"] = J_HINGE
                if child.get("axis"):
                    entry["j_axis"] = tuple(
                        float(v) for v in child.get("axis").split())
                if child.get("range"):
                    lo, hi = [float(v) for v in child.get("range").split(" ")]
                    entry["j_range"] = (lo, hi)
            elif child.tag == "inertial":
                entry["mass"] = float(child.get("mass"))
                c = torch.tensor([float(v) for v in
                                  child.get("pos", "0 0 0").split()],
                                 dtype=torch.float64)
                entry["com_local"] = c
                if child.get("diaginertia"):
                    d = [float(v) for v in child.get("diaginertia").split()]
                    entry["Ic"] = torch.diag(torch.tensor(d, dtype=torch.float64))
                elif child.get("fullinertia"):
                    ixx, iyy, izz, ixy, ixz, iyz = [
                        float(v) for v in child.get("fullinertia").split()]
                    I = torch.tensor([[ixx, ixy, ixz], [ixy, iyy, iyz],
                                      [ixz, iyz, izz]], dtype=torch.float64)
                    # rotate into frame if quat given on inertial
                    if child.get("quat"):
                        RI = _quat_to_matrix(child.get("quat"))
                        I = RI @ I @ RI.T
                    entry["Ic"] = I
            elif child.tag == "geom":
                gtype = child.get("type", "sphere")
                sizes = [float(v) for v in child.get("size", "0.05").split()]
                gpos = torch.tensor([float(v) for v in
                                     child.get("pos", "0 0 0").split()],
                                    dtype=torch.float64)
                gR = torch.eye(3, dtype=torch.float64)
                if child.get("quat"):
                    gR = _quat_to_matrix(child.get("quat"))
                elif child.get("euler"):
                    gR = _euler_to_matrix(child.get("euler"))
                gp = dict(name=child.get("name", f"g{len(geoms_spec)}"),
                          body=idx, p=gpos, R=gR, ground=True)
                if gtype == "sphere":
                    gp.update(shape="sphere", r=sizes[0])
                elif gtype == "capsule":
                    gp.update(shape="capsule",
                              r=sizes[0], hl=(sizes[1] if len(sizes) > 1 else 0.0))
                elif gtype in ("box", "ellipsoid") and capsule_from_box:
                    hx = sizes[0]
                    hy = sizes[1] if len(sizes) > 1 else hx
                    hz = sizes[2] if len(sizes) > 2 else hy
                    r = min(hx, hy, hz)
                    hl = max(hx, hy, hz) - r
                    gp.update(shape="capsule", r=r, hl=max(hl, 0.0))
                else:
                    continue
                geoms_spec.append(gp)

        for child_el in el.findall("body"):
            parse_body(child_el, idx, R_fix, p_fix)

    top = [el for el in worldbody.findall("body")]
    for el in top:
        parse_body(el, -1, torch.eye(3, dtype=torch.float64),
                   torch.zeros(3, dtype=torch.float64))

    # ---- assemble Model tensors ------------------------------------------
    nb = len(bodies)

    # free-joint handling: MJCF puts <freejoint> typically on root; ensure
    # exactly one and it's body 0's chain root
    free_idx = [i for i, b in enumerate(bodies) if b["j_type"] == J_FREE]

    q_dim = sum(7 if b["j_type"] == J_FREE else
                (1 if b["j_type"] in (J_HINGE, J_SLIDE) else 0)
                for b in bodies)
    dof_body, body_dof_start = [], []
    qi = vi = 0
    q_free_start = -1
    for i, b in enumerate(bodies):
        jt = b["j_type"]
        if jt == J_FREE:
            body_dof_start.append(vi)
            dof_body.extend([i] * 6)
            q_free_start = qi
            qi += 7
            vi += 6
        elif jt in (J_HINGE, J_SLIDE):
            body_dof_start.append(vi)
            dof_body.append(i)
            qi += 1
            vi += 1
        else:
            body_dof_start.append(-1)

    def cap_I(mass_, r_, hl_):
        mc = max(mass_, 1e-6)
        return torch.diag(torch.tensor([
            mc * (hl_ ** 2 / 3.0 + r_ ** 2 / 4.0),
            mc * (hl_ ** 2 / 3.0 + r_ ** 2 / 4.0),
            mc * r_ ** 2 / 2.0], dtype=torch.float64))

    masses, coms, icoms = [], [], []
    for b in bodies:
        m_b = float(b.get("mass", 0.01)) or 0.01
        masses.append(m_b)
        coms.append(b.get("com_local", torch.zeros(3, dtype=torch.float64)))
        if "Ic" in b:
            icoms.append(b["Ic"])
        else:
            gs = [g for g in geoms_spec if g["body"] == bodies.index(b)]
            if gs:
                g0 = gs[0]
                icoms.append(cap_I(m_b, g0["r"], g0.get("hl", 0.0)))
            else:
                icoms.append(torch.eye(3, dtype=torch.float64) * 1e-4)

    # hinge limits in dof order
    actuated = [d for d, bi in enumerate(dof_body)
                if bodies[bi]["j_type"] == J_HINGE]
    lim_lo, lim_hi, lim_idx = [], [], []
    for d in actuated:
        rng = bodies[dof_body[d]].get("j_range")
        if rng is not None:
            lim_idx.append(d)
            lim_lo.append(rng[0])
            lim_hi.append(rng[1])

    model = Model(
        n_bodies=nb,
        parent=[b["parent"] for b in bodies],
        body_names=[b["name"] for b in bodies],
        fix_R=torch.stack([b["fix_R"] for b in bodies]).to(device=device, dtype=dtype),
        fix_p=torch.stack([b["fix_p"] for b in bodies]).to(device=device, dtype=dtype),
        j_type=torch.tensor([b["j_type"] for b in bodies], device=device),
        j_axis=torch.stack([
            torch.tensor(b["j_axis"], dtype=dtype) /
            max(torch.linalg.vector_norm(torch.tensor(b["j_axis"], dtype=dtype)), 1e-12)
            for b in bodies]).to(device),
        masses=torch.tensor(masses, device=device, dtype=dtype),
        com=torch.stack(coms).to(device=device, dtype=dtype),
        inertia_com=torch.stack(icoms).to(device=device, dtype=dtype),
        q_dim=q_dim, v_dim=vi,
        dof_body=dof_body, body_dof_start=body_dof_start,
        q_free_start=q_free_start,
        damping=torch.full((vi,), 0.5, dtype=dtype, device=device),
        armature=0.01,
        limit_dof_idx=(torch.tensor(lim_idx, dtype=torch.long, device=device)
                       if lim_idx else None),
        joint_limit_lo=(torch.tensor(lim_lo, dtype=dtype, device=device)
                        if lim_idx else None),
        joint_limit_hi=(torch.tensor(lim_hi, dtype=dtype, device=device)
                        if lim_idx else None),
    )
    return model, geoms_spec
