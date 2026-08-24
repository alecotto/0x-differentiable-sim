"""Garcia et al. 1998 simplest-walker ORACLE -- paper-faithful core.

Implements the published equations VERBATIM (Garcia, Chatterjee, Ruina &
Coleman, "The simplest walking model", ASME J Biomech Eng 120(2):281,
1998; equations/impact matrix extracted directly from the paper PDF):

Flow (dimensionless time tau = t sqrt(g/l), beta -> 0):
    theta_ddot = sin(theta - gamma)                          (paper eq 1)
    phi_ddot   = theta_ddot
               + (theta_dot^2 - cos(theta - gamma)) sin(phi) (paper eq 2)

Heelstrike condition (paper eq 3):   phi - 2 theta = 0
with the stance leg past vertical (theta < 0), approached from above.

Jump map (paper eq 4, exact matrix for beta = 0):
    theta+  = -theta-
    thetadot+ = cos(2 theta-) thetadot-
    phi+    = -2 theta-
    phidot+ = cos(2 theta-) (1 - cos(2 theta-)) thetadot-

Poincare section: just after heelstrike, reduced to (theta+, thetadot+)
(paper's 2D reduction).

External anchors to reproduce:
  * stable period-1 gaits only for 0 < gamma < 0.0151
  * period doubling ('limping') near gamma = 0.017; cascade done by
    gamma ~ 0.019; walker falls above that (Feigenbaum ratios 5.9, 5.2,...)
  * theta* ~ gamma^(1/3); two period-one branches (short & long period)
"""
from __future__ import annotations

import math

import torch

DT = torch.float64


def accel(y: torch.Tensor, gamma: float) -> torch.Tensor:
    th, om, ph, et = y[..., 0], y[..., 1], y[..., 2], y[..., 3]
    th_dd = torch.sin(th - gamma)
    ph_dd = th_dd + (om * om - torch.cos(th - gamma)) * torch.sin(ph)
    return torch.stack([om, th_dd, et, ph_dd], dim=-1)


def rk4_step(y: torch.Tensor, gamma: float, h: float) -> torch.Tensor:
    k1 = accel(y, gamma)
    k2 = accel(y + 0.5 * h * k1, gamma)
    k3 = accel(y + 0.5 * h * k2, gamma)
    k4 = accel(y + h * k3, gamma)
    return y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


@torch.no_grad()
def strike_paper(y: torch.Tensor, gamma: float, h: float = 5e-4,
                 max_tau: float = 25.0):
    """Integrate one swing phase of the paper system until heelstrike,
    apply jump eq 4.  y [N,4] post-impact states. Returns (y_next, ok).

    TD = phi - 2 theta = 0 crossed from BELOW (g: - -> 0+), with
    theta < 0 (stance past vertical).  The earlier DOWN-crossing near
    leg-parallel configuration is scuffing -- ignored per the paper.
    Validated against the analytic fixed point: from Garcia's Table-1
    asymptotic FP the flow returns to -theta* at tau ~= 3.88 (gamma=.009).
    """
    N = y.shape[0]
    done = torch.zeros(N, dtype=torch.bool)
    fell = torch.zeros(N, dtype=torch.bool)
    g = lambda z: z[..., 2] - 2 * z[..., 0]
    gp = g(y)
    t = 0.0
    while bool((~done & ~fell).any()) and t < max_tau:
        yn = rk4_step(y, gamma, h)
        t += h
        gn = g(yn)
        cross = (~done & ~fell) & (gp < 0) & (gn >= 0) \
            & (yn[..., 0] < 0) & (t > 0.5)
        idxs = torch.nonzero(cross, as_tuple=True)[0].tolist()
        for i in idxs:
            lo, hi = 0.0, 1.0
            ya, yb = y[i], yn[i]
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                if float(g((ya + (yb - ya) * mid).unsqueeze(0))) < 0:
                    lo = mid
                else:
                    hi = mid
            yh = ya + (yb - ya) * hi
            a_, o_ = float(yh[0]), float(yh[1])
            c2 = math.cos(2 * a_)
            y[i] = torch.tensor(
                [-a_, c2 * o_, -2 * a_, c2 * (1 - c2) * o_], dtype=DT)
            done[i] = True
        fell |= (~done) & (yn[..., 0].abs() > 2.0)
        upd = (~done & ~fell)
        y = torch.where(upd.unsqueeze(-1), yn, y)
        gp = torch.where(upd, gn, gp)
    return y, done


def section_state(s: torch.Tensor) -> torch.Tensor:
    """s [N,2]=(theta+, thetadot+) -> full state via the jump relations."""
    th, om = s[:, 0], s[:, 1]
    c2 = torch.cos(2 * th)
    return torch.stack([th, om, 2 * th, c2 * (1 - c2) * om], dim=-1)


def return_map(s: torch.Tensor, gamma: float, h: float = 5e-4):
    y, ok = strike_paper(section_state(s), gamma, h=h)
    return y[..., :2], ok


@torch.no_grad()
def shoot_fixed_point(gamma: float, guess, iters=600, tol=1e-11, h=5e-4,
                      fd_eps=1e-6):
    """Fixed point by map ITERATION (the long-period branch is attracting,
    |mu| ~= 0.6, so contraction converges geometrically), then
    central-FD multipliers of the section map at the converged point.
    (Safeguarded Newton proved unreliable here: P(s) appears to have a
    discontinuity ridge near the basin boundary, which poisons FD-J.)
    """
    import numpy as np
    s = np.array(guess, dtype=float)
    if not np.isfinite(s).all():
        return None

    def P(sv):
        p, ok = return_map(torch.tensor(np.asarray(sv).reshape(1, 2),
                                        dtype=DT), gamma, h=h)
        return p[0].numpy(), bool(ok[0])

    p, ok = P(s)
    if not ok:
        return None
    for _ in range(iters):
        d = np.max(np.abs(p - s))
        if d < tol:
            break
        s = p
        p, ok = P(s)
        if not ok:
            return None
    resid = np.max(np.abs(p - s))
    if resid > 1e-7:
        return None

    cols = []
    for j in range(2):
        sp_ = s.copy(); sp_[j] += fd_eps
        sm_ = s.copy(); sm_[j] -= fd_eps
        pp, o1 = P(sp_)
        pm, o2 = P(sm_)
        if not (o1 and o2):
            return None
        cols.append((pp - pm) / (2 * fd_eps))
    M = np.stack(cols, axis=1)
    ev = np.sort(np.abs(np.linalg.eigvals(M)))[::-1]
    return {"theta": float(s[0]), "omega": float(s[1]), "resid": float(resid),
            "mult": [float(x) for x in ev]}


# Garcia Table 1 (Appendix A.2): asymptotic period-one gait data
TABLE1 = {
    "short": dict(tau0=math.pi, tau1=-0.907496, Th0=0.943976,
                  Th1=-0.264561, alpha=-1.090331, c1=0.866610),
    "long": dict(tau0=3.812092, tau1=1.579129, Th0=0.970956,
                 Th1=-0.270837, alpha=-1.045203, c1=1.062895),
}


def analytic_seed(gamma: float, branch="long"):
    t = TABLE1[branch]
    g13 = gamma ** (1 / 3.)
    th = t["Th0"] * g13 + t["Th1"] * gamma
    om = t["alpha"] * t["Th0"] * g13 + (t["alpha"] * t["Th1"] + t["c1"]) * gamma
    return th, om


def find_gait(gamma: float, verbose=False):
    """Find period-one fixed points of both branches at slope gamma."""
    out = []
    seen = []
    for branch in ("long", "short"):
        th0, om0 = analytic_seed(gamma, branch)
        seeds = [(th0, om0)]
        # small neighborhood as insurance
        for dth in (-0.02, 0.0, 0.02):
            for dw in (0.85, 1.0, 1.15):
                if (dth, dw) == (0.0, 1.0):
                    continue
                seeds.append((th0 + dth, om0 * dw))
        for sd in seeds:
            try:
                r = shoot_fixed_point(gamma, sd)
            except Exception:
                r = None
            if r is None:
                continue
            key = (round(r["theta"], 6), round(r["omega"], 6))
            if any(abs(key[0] - k[0]) + abs(key[1] - k[1]) < 1e-4
                   for k in seen):
                continue
            seen.append(key)
            r["branch"] = branch if len(seen) == 1 else \
                ("short" if r["theta"] < 0.7 * th0 else branch)
            out.append(r)
            if verbose:
                print(f"    [{r['branch']}] FP th={r['theta']:+.5f} "
                      f"om={r['omega']:+.5f} "
                      f"|m|=({r['mult'][0]:.4f},{r['mult'][1]:.4f})")
            break   # one per branch is enough
    return out


def attractor_periods(gamma: float, s0, n_steps=300, tail=120,
                      h=5e-4, tol=1e-6):
    """Iterate the map from s0; classify the tail attractor as period-k."""
    import numpy as np
    s = np.array(s0, dtype=float)
    hist = []
    for _ in range(n_steps):
        p, ok = return_map(torch.tensor(s.reshape(1, 2), dtype=DT),
                           gamma, h=h)
        if not bool(ok[0]):
            return {"gamma": gamma, "fell": True}
        s = p[0].numpy()
        hist.append(s.copy())
    tail_pts = hist[-tail:]
    # find minimal period p such that successive p-shifts agree
    for per in range(1, 33):
        seg = tail_pts[-per * 4:]
        if len(seg) < per * 2:
            continue
        diffs = [np.max(np.abs(seg[i] - seg[i + per]))
                 for i in range(len(seg) - per)]
        if max(diffs) < tol * max(1.0, per):
            ths = sorted(set(round(float(x[0]), 6)
                             for x in tail_pts[-per * 3:]))
            return {"gamma": gamma, "period": per, "fell": False,
                    "points": ths}
    return {"gamma": gamma, "period": None, "fell": False,
            "chaotic": True,
            "spread": float(np.ptp([x[0] for x in tail_pts]))}


if __name__ == "__main__":
    import json
    import os

    results = {"fixed_points": {}, "attractors": {}}
    gammas = [0.004, 0.008, 0.010, 0.012, 0.013, 0.014, 0.0145, 0.015,
              0.0151, 0.0155, 0.016, 0.017, 0.018, 0.019, 0.020]

    print("== period-one branch continuation ==")
    prev = {}
    for g in gammas:
        fps = find_gait(g)
        line = f"gamma={g:.4f}:"
        for fp in fps:
            stable = "STABLE" if fp["mult"][0] < 1 else "unstable"
            line += (f"  [{fp['branch']}] th*={fp['theta']:+.5f} "
                     f"om*={fp['omega']:+.5f} m=({fp['mult'][0]:+.4f},"
                     f"{fp['mult'][1]:+.4f}) {stable};")
            results["fixed_points"].setdefault(str(g), {})[fp["branch"]] = fp
        print(line)

    print("\n== attractor classification (long-period ICs) ==")
    for g in [0.010, 0.015, 0.0155, 0.016, 0.0165, 0.017, 0.018, 0.019,
              0.020]:
        th0, om0 = analytic_seed(g, "long")
        att = attractor_periods(g, (th0, om0))
        results["attractors"][str(g)] = {
            k: v for k, v in att.items() if k != "points"}
        print(f"gamma={g:.4f}: {att}")

    os.makedirs("benchmarks", exist_ok=True)
    with open("benchmarks/garcia_oracle.json", "w") as fh:
        json.dump(results, fh, indent=1)
    print("\nwrote benchmarks/garcia_oracle.json")
