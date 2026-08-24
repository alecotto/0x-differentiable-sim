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
    """Integrate one swing phase of the paper system until heelstrike
    (phi - 2 theta = 0 from above, theta<0, t>0.3), apply jump eq 4.
    y [N,4] post-impact states. Returns (y_next, ok)."""
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
        cross = (~done & ~fell) & (gp > 0) & (gn <= 0) \
            & (yn[..., 0] < 0) & (t > 0.3)
        idxs = torch.nonzero(cross, as_tuple=True)[0].tolist()
        for i in idxs:
            lo, hi = 0.0, 1.0
            ya, yb = y[i], yn[i]
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                if float(g((ya + (yb - ya) * mid).unsqueeze(0))) > 0:
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
def shoot_fixed_point(gamma: float, guess, iters=40, tol=1e-13, h=5e-4):
    import numpy as np
    s = np.array(guess, dtype=float)

    def P(sv):
        p, ok = return_map(torch.tensor(np.asarray(sv).reshape(1, 2),
                                        dtype=DT), gamma, h=h)
        return p[0].numpy(), bool(ok[0])

    for _ in range(iters):
        p, ok = P(s)
        if not ok:
            return None
        f = p - s
        if np.max(np.abs(f)) < tol:
            break
        J = np.zeros((2, 2))
        eps = 1e-8
        cols_ok = True
        for j in range(2):
            sp_ = s.copy(); sp_[j] += eps
            sm_ = s.copy(); sm_[j] -= eps
            pp, o1 = P(sp_)
            pm, o2 = P(sm_)
            if not (o1 and o2):
                cols_ok = False
                break
            J[:, j] = (pp - pm) / (2 * eps)
        if not cols_ok:
            return None
        s = s - np.linalg.solve(J, f)
    else:
        return None
    cols = []
    eps = 1e-7
    for j in range(2):
        sp_ = s.copy(); sp_[j] += eps
        sm_ = s.copy(); sm_[j] -= eps
        pp, o1 = P(sp_)
        pm, o2 = P(sm_)
        if not (o1 and o2):
            return None
        cols.append((pp - pm) / (2 * eps))
    M = np.stack(cols, axis=1)
    ev = np.sort(np.abs(np.linalg.eigvals(M)))[::-1]
    return {"theta": float(s[0]), "omega": float(s[1]),
            "mult": [float(x) for x in ev]}


def find_gait(gamma: float, verbose=False):
    """Grid-seeded Newton for both period-1 branches at slope gamma."""
    out = []
    a0 = (3 * gamma / 4) ** (1 / 3.)
    seeds = []
    for da in (-0.06, -0.03, 0.0, 0.03, 0.06):
        for w in (0.05, 0.08, 0.11, 0.14, 0.17, 0.20, 0.25, 0.30):
            seeds.append((abs(a0) + da, -w))       # long-period branch guess
            seeds.append((-abs(a0) - da, w))       # mirrored convention
    seen = []
    for sd in seeds:
        try:
            r = shoot_fixed_point(gamma, sd)
        except Exception:
            r = None
        if r is None:
            continue
        key = (round(r["theta"], 6), round(r["omega"], 6))
        if any(abs(key[0] - k[0]) + abs(key[1] - k[1]) < 1e-4 for k in seen):
            continue
        seen.append(key)
        out.append(r)
        if verbose:
            print(f"    FP th={r['theta']:+.5f} om={r['omega']:+.5f} "
                  f"|m|={r['mult'][0]:.4f},{r['mult'][1]:.4f}")
    return out


if __name__ == "__main__":
    print("gamma=0.009 (paper fig-2 example slope):")
    fps = find_gait(0.009, verbose=True)
    for fp in fps:
        stable = "STABLE" if fp["mult"][0] < 1 else "unstable"
        print(f"  theta*={fp['theta']:+.6f} omega*={fp['omega']:+.6f} "
              f"multipliers=({fp['mult'][0]:+.4f},{fp['mult'][1]:+.4f}) "
              f"{stable}")
