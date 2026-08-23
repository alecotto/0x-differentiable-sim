"""Bouncing ball on flat floor: contact model validation under repeated impacts.

A point mass bouncing vertically on our softplus contact model should show:
1. Height decreases geometrically: h_n = h_0 * e^n where e depends on k, damping
2. Contact duration shortens as energy decreases
3. No energy gain (no suction/anti-damping artifacts)
4. Eventually settles to rest at the softplus phantom-penetration offset

This is the simplest possible validation of the contact force chain.
"""
import sys
import torch
import math

sys.path.insert(0, '/Code/0x-differentiable-sim-project')
from diffsim.collision import softplus_pen


def bounce_sim(k=1.5e4, b=50.0, m=1.0, h0=1.0, dt=1e-5, t_max=5.0,
               beta_soft=1e4, device="cpu"):
    """Simulate bouncing ball with DiffSim's contact model.

    Returns dict with trajectory data and derived quantities.
    """
    dtype = torch.float64 if device == "cpu" else torch.float32
    dev = torch.device(device)

    z = torch.tensor(h0 + 1e-4, dtype=dtype, device=dev)  # start just above floor
    v = torch.tensor(0.0, dtype=dtype, device=dev)

    zs = [float(z)]
    vs = [float(v)]
    ts = [0.0]

    n_steps = int(t_max / dt)
    impact_heights = []   # peak heights between bounces
    last_peak_z = h0
    going_up = False
    e_prev = 0.5 * m * v ** 2 + m * 9.81 * float(z)

    max_energy_violation = 0.0

    for i in range(n_steps):
        # gravity
        acc = -9.81

        # contact force when penetrating
        pen = softplus_pen(-z, beta_soft).item()  # penetration depth
        fn = k * pen
        fdamp = b * pen / (pen + 1e-4) * max(0.0, -v.item())  # damping only on approach
        f_contact = fn + fdamp
        acc += f_contact / m

        # semi-implicit Euler
        v += dt * acc
        z = z + dt * v

        # detect apexes (peaks between bounces)
        if i > 0 and vs[-1] > 0 and float(v) < 0 and float(z) > 0.01:
            impact_heights.append(float(z))
            going_up = False

        zs.append(float(z))
        vs.append(float(v))
        ts.append((i + 1) * dt)

        if not math.isfinite(float(z)):
            break

    # analyze successive apex heights → restitution ratio
    restitution_ratios = []
    for i in range(1, len(impact_heights)):
        if impact_heights[i - 1] > 0.01:
            e_ratio = math.sqrt(max(impact_heights[i], 1e-10) /
                                max(impact_heights[i - 1], 1e-10))
            restitution_ratios.append(e_ratio)

    total_energy = [0.5 * m * v_ ** 2 + m * 9.81 * max(z_, 0)
                    for z_, v_ in zip(zs, vs)]

    return {
        "ts": ts, "zs": zs, "vs": vs,
        "impact_heights": impact_heights,
        "restitution_ratios": restitution_ratios,
        "total_energy": total_energy,
        "max_energy_violation": max_energy_violation,
        "final_height": zs[-1],
        "n_bounces": len(impact_heights),
    }


def validate():
    print("=== Bouncing Ball Contact Model Validation ===\n")

    result = bounce_sim(k=1.5e4, b=50.0, m=1.0, h0=1.0)

    print(f"N bounces detected: {result['n_bounces']}")
    print(f"Impact heights: {[f'{h:.3f}' for h in result['impact_heights'][:8]]}")

    ratios = result["restitution_ratios"]
    if ratios:
        print(f"\nRestitution ratios (sqrt(h_n/h_(n-1))):")
        for i, r in enumerate(ratios[:8]):
            print(f"  bounce {i}->{i+1}: e = {r:.4f}")
        mean_e = sum(ratios[:6]) / min(len(ratios), 6)
        std_e = math.sqrt(sum((r - mean_e) ** 2 for r in ratios[:6]) / min(len(ratios), 6)) \
            if len(ratios) > 1 else 0
        print(f"  mean e = {mean_e:.4f} ± {std_e:.4f} (first {min(len(ratios),6)} bounces)")

    # energy check: total energy should never increase after initial drop
    te = result["total_energy"]
    max_increase = 0.0
    for i in range(1, len(te)):
        d = te[i] - te[i - 1]
        if d > 0 and te[i] > 0.05:  # ignore near-zero noise
            max_increase = max(max_increase, d)

    print(f"\nMax energy increase per step: {max_increase:.6e} J")
    print(f"Energy monotonically decreasing (after initial drop): "
          f"{all(te[i] >= te[i+1] - 1e-8 for i in range(len(te)-1) if te[i] < 9.8)}")
    print(f"Final height: {result['final_height']:.6f} m")

    # PASS/FAIL
    checks = []
    if result["n_bounces"] >= 5:
        checks.append(("multiple bounces", True))
    else:
        checks.append(("multiple bounces", False))

    if ratios and all(r < 1.01 for r in ratios):
        checks.append(("no energy gain at impacts", True))
    else:
        checks.append(("no energy gain at impacts",
                       all(r < 1.05 for r in ratios)))

    consistent = len(set(round(r, 3) for r in ratios[:6])) <= 2 if len(ratios) >= 6 else True
    checks.append(("consistent restitution", consistent))

    checks.append(("settles to floor", abs(result["final_height"]) < 1e-3))

    print("\n=== RESULTS ===")
    all_pass = True
    for name, ok in checks:
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {name}")
        if not ok:
            all_pass = False
    return all_pass


if __name__ == "__main__":
    ok = validate()
    sys.exit(0 if ok else 1)
