import sys, torch
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from diffsim.humanoid import make_soma_humanoid, initial_pose
from diffsim import build_geoms_compat
from diffsim.sim import DiffSim, SimConfig, ContactConfig

model, gspec, _ = make_soma_humanoid()
geoms = build_geoms_compat(gspec)
device = "cuda"
cc = ContactConfig(k_ground=1.5e4, k_pair=8e3, damping=200.)
cc.limit_k = 2000.; cc.limit_beta = 50.

for name, dtype in [("fp64", torch.float64), ("fp32", torch.float32)]:
    sim = DiffSim(model, geoms,
                  SimConfig(dt=5e-4, n_substeps=8, use_analytic_bias=True,
                            contact=cc),
                  device='cuda', dtype=dtype)
    q, w = initial_pose(model, 1)
    q = q.to(device=device, dtype=dtype)
    w = w.to(device=device, dtype=dtype)
    qt = torch.zeros(1, 15, dtype=dtype, device=device)
    mm_d = model.masses.to(dtype=dtype, device=device)
    total_m = float(mm_d.sum())

    coms = []
    for i in range(100):
        R, p = sim.art.kinematics(q)
        c = (sim.art.com_positions(q, R, p) * mm_d.unsqueeze(-1)).sum(1) / mm_d.sum()
        cz = float(c[0, 2])
        coms.append(round(cz, 4))
        tau = sim.pd_torques(q, w, qt, kp=400., kd=50.)
        r = sim.step(q, w, tau_ext=tau)
        q, w = r.q, r.qd

    print(f"{name}: com_z = {coms[::20]}...{coms[-3:]}")
    del sim
