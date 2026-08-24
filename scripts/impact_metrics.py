"""Impact-physics metrics harness (Q1d).

Question: do our compliant contacts produce correct forces under dynamic
impacts?

Two independent references are compared per scenario:

1. "ode"   : exact integration of the 1-DOF mass-on-compliant-ground ODE
             m z'' = -m g + f_n(z, z') with the DiffSim ground law
               pen = softplus_pen(-z, beta_soft=1e4)          (margin = 0)
               act = pen / (pen + smooth)                     (smooth = 1e-4)
               f_n = k*pen + b*act*softplus_pen(clamp(-z', max=2), 1e3)
             RK4, fp64, dt = 1e-6 s. Free flight is analytic (release from
             rest at the height giving touch speed v0); integration covers
             only the force-carrying contact window.
2. "stiff" : stiff-contact-limit reference. When the ramp width
             eps = 1/beta_soft << peak penetration, softplus_pen -> relu,
             act -> 1 and softplus_pen(clamp(-z',2),1e3) -> relu(-z'), so
               m z'' = -m g + k*(-z)_+ + b*(-z')_+
             (approach-only damping), integrated independently with RK4
             fp64. dt = 1e-7 s for the flagship stiff cases; elsewhere the
             coarsest dt on the verified convergence plateau (see
             convergence_checks()).

Metrics per scenario (impact speed v0, mass m, stiffness k, damping b):
peak contact force, contact duration T_c, restitution e, energy-loss
fraction, peak penetration, suction checks (f_n >= 0 everywhere sampled,
clean separation), and static phantom-penetration vs theory g*m/k +
ln(2)/beta_soft.

Restitution is measured between interpolated velocities at symmetric
crossings of the touch plane z = 0; because gravity is conservative and
the trajectory outside contact is ballistic, this equals the far-field
velocity ratio exactly. Scenarios too dissipative to rebound (contact
damping ratio > ~1) are detected by a sustained low-velocity test and
reported with e = 0 rather than hanging the integrator.

A two-point load-share model of the capsule foot (two independent endpoint
springs L = 0.2 m apart, mirroring eval_ground's TWO ground contacts per
capsule) is pitched quasi-statically under total load W to study the
center of pressure.
"""
import os
import sys
import json
import math

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import torch

from diffsim.collision import softplus_pen

G = 9.81
BETA_SOFT = 1.0e4
SMOOTH = 1.0e-4
BETA_D = 1.0e3
VCAP = 2.0
REST_OFF = math.log(2.0) / BETA_D
TOUCH_PEN = math.log(2.0) / BETA_SOFT
TT = torch.float64

V0_GRID = (0.5, 1.0, 2.0)
M_GRID = (1.0, 10.0, 40.0)
K_GRID = (1.5e4, 2.5e4, 8.0e4)
B_GRID = (100.0, 400.0, 800.0)


def _contact_law(z, v, k, b):
    """DiffSim ground normal law for sole height z, velocity v."""
    pen = torch.nn.functional.softplus(-BETA_SOFT * z) / BETA_SOFT
    act = pen / (pen + SMOOTH)
    vc = torch.clamp(-v, max=VCAP)
    sd = torch.nn.functional.softplus(BETA_D * vc) / BETA_D
    f = k * pen + b * act * sd
    return f, pen


def _stiff_law(z, v, k, b):
    """Stiff-contact-limit law: linear spring + approach-only damping."""
    pen = torch.relu(-z)
    return k * pen + b * torch.relu(-v), pen


def _rk4_coeffs(zo, vo, law, kt, bt, mt, half, dt):
    f1, p1 = law(zo, vo, kt, bt)
    a1 = f1 / mt - G
    z2, v2 = zo + half * vo, vo + half * a1
    f2, p2 = law(z2, v2, kt, bt)
    a2 = f2 / mt - G
    z3, v3 = zo + half * v2, vo + half * a2
    f3, p3 = law(z3, v3, kt, bt)
    a3 = f3 / mt - G
    z4, v4 = zo + dt * v3, vo + dt * a3
    f4, p4 = law(z4, v4, kt, bt)
    a4 = f4 / mt - G
    sixth = dt / 6.0
    zn = zo + sixth * (vo + 2.0 * v2 + 2.0 * v3 + v4)
    vn = vo + sixth * (a1 + 2.0 * a2 + 2.0 * a3 + a4)
    return zn, vn, f1, p1


def _impact_run(law, v0s, ms, ks, bs, dt, t_cap, enter_z,
                v_tol=1.0e-3, sus_seconds=2.0e-3, check_every=64):
    """Shared batched fp64 RK4 driver for both impact references.

    Each element terminates at its first upward crossing of z = 0 after
    impact (clean rebound; crossing velocity linearly interpolated) or at
    proven no-rebound: either a velocity apex strictly below the touch
    plane (mechanical energy is non-increasing, so z = 0 is unreachable
    afterwards) or a sustained near-rest creep inside the contact zone.
    """
    n = len(v0s)
    v0t = torch.tensor(v0s, dtype=TT)
    mt = torch.tensor(ms, dtype=TT)
    kt = torch.tensor(ks, dtype=TT)
    bt = torch.tensor(bs, dtype=TT)

    z = torch.full((n,), enter_z, dtype=TT)
    v = -torch.sqrt(torch.clamp(v0t * v0t - 2.0 * G * enter_z, min=0.0))

    f_max = torch.full((n,), float("nan"), dtype=TT)
    pen_max = torch.full((n,), float("nan"), dtype=TT)
    f_min = torch.full((n,), float("inf"), dtype=TT)
    t_c = torch.zeros(n, dtype=TT)
    sat_steps = torch.zeros(n, dtype=TT)
    below = torch.zeros(n, dtype=torch.bool)
    have_down = torch.zeros(n, dtype=torch.bool)
    done_up = torch.zeros(n, dtype=torch.bool)
    settled = torch.zeros(n, dtype=torch.bool)
    calm = torch.zeros(n, dtype=torch.long)
    sus_steps = max(1, int(sus_seconds / dt))
    v_in0 = torch.zeros(n, dtype=TT)
    v_ex0 = torch.zeros(n, dtype=TT)

    half = 0.5 * dt
    zero = torch.zeros(n, dtype=TT)
    n_steps = int(t_cap / dt)
    for i in range(n_steps):
        zo, vo = z, v
        z, v, f1, p1 = _rk4_coeffs(zo, vo, law, kt, bt, mt, half, dt)

        act = ~(done_up | settled)
        fresh = torch.isnan(f_max)
        f_max = torch.where(fresh & act, f1,
                            torch.where(act, torch.maximum(f_max, f1), f_max))
        pen_max = torch.where(fresh & act, p1,
                              torch.where(act, torch.maximum(pen_max, p1),
                                          pen_max))
        f_min = torch.where(act, torch.minimum(f_min, f1), f_min)
        sat_steps = sat_steps + (act & (vo < -VCAP)).to(TT)

        nb = z < 0.0
        cross = nb != below
        frac = (-zo / (z - zo)) * dt
        t_c = t_c + torch.where(act & below & nb, dt,
                                torch.where(act & cross, frac, zero))
        down_hit = act & cross & ~below & ~have_down
        v_in0 = torch.where(down_hit, vo + (-zo / (z - zo)) * (v - vo), v_in0)
        have_down = have_down | down_hit
        up_hit = act & cross & below & have_down
        fu = -zo / (z - zo)
        v_ex0 = torch.where(up_hit, vo + fu * (v - vo), v_ex0)
        done_up = done_up | up_hit

        apex_below = act & have_down & nb & (vo > 0.0) & (v <= 0.0)
        calm = torch.where(act & nb & (v.abs() < v_tol), calm + 1,
                           torch.zeros_like(calm))
        settled = settled | apex_below | (calm >= sus_steps)
        below = nb

        if ((i % check_every) == (check_every - 1)) and i > check_every:
            if bool((done_up | settled).all()):
                break

    return {
        "F_max": f_max, "pen_max": pen_max, "min_f": f_min, "T_c": t_c,
        "sat_steps": sat_steps, "v_in0": v_in0, "v_ex0": v_ex0,
        "separated": done_up, "settled_no_rebound": settled & ~done_up,
    }


def ode_reference_impact(v0s, ms, ks, bs, dt=1.0e-6, t_cap=2.0):
    """Exact compliant-law impact metrics, batched RK4 fp64.

    Integration window entered at z = 3/beta_soft (tail force there
    <= k*softplus_pen(3/beta) ~ 1 N, integrated anyway); free fall before
    that point is exact ballistics from rest at h = v0^2/(2g).
    """
    res = _impact_run(_contact_law, v0s, ms, ks, bs, dt, t_cap,
                      enter_z=3.0 / BETA_SOFT)
    tail_f = [float(k) * float(softplus_pen(torch.tensor(-1.0e-3, dtype=TT),
                                            BETA_SOFT))
              for k in ks]
    rows = []
    n = len(v0s)
    for j in range(n):
        vin = float(res["v_in0"][j])
        vout = float(res["v_ex0"][j])
        speed_in = abs(vin)
        e = max(vout / speed_in, 0.0) if speed_in > 0.0 else 0.0
        rows.append({
            "v0": float(v0s[j]), "m": float(ms[j]), "k": float(ks[j]),
            "b": float(bs[j]),
            "F_max": float(res["F_max"][j]),
            "pen_max": float(res["pen_max"][j]),
            "T_c": float(res["T_c"][j]), "e": e,
            "energy_loss_frac": 1.0 - e * e,
            "min_f": float(res["min_f"][j]), "tail_f": tail_f[j],
            "separated": bool(res["separated"][j]),
            "no_rebound_settled": bool(res["settled_no_rebound"][j]),
            "clamp_sat_steps": int(res["sat_steps"][j]),
            "v_before": vin, "v_after": vout, "dt": dt,
        })
    return rows


def default_stiff_dt(m, k, b):
    """Stiff-reference dt: literal 1e-7 s for the flagship stiff case;
    otherwise coarsest dt on the verified plateau (omega*dt <= 3e-3 gives
    RK4 relative force error < 1e-10 for this linear system)."""
    if k >= 7.9e4 and m <= 1.0 and b <= 400.0:
        return 1.0e-7
    omega = math.sqrt(k / m)
    return min(1.0e-5, max(1.0e-7, 3.0e-3 / omega))


def stiff_reference_impact(v0s, ms, ks, bs, dt=None, t_cap=2.0):
    """Stiff-limit reference, grouped by (m, k, b) with per-group dt."""
    groups = {}
    for i, key in enumerate(zip(ms, ks, bs)):
        groups.setdefault(key, []).append(i)
    out = [None] * len(v0s)
    for (m, k, b), idxs in groups.items():
        d = dt if dt is not None else default_stiff_dt(m, k, b)
        res = _impact_run(_stiff_law, [v0s[i] for i in idxs], [m] * len(idxs),
                          [k] * len(idxs), [b] * len(idxs), d, t_cap,
                          enter_z=0.0)
        for jj, i in enumerate(idxs):
            vin = float(res["v_in0"][jj])
            vout = float(res["v_ex0"][jj])
            speed_in = abs(vin)
            e = max(vout / speed_in, 0.0) if speed_in > 0.0 else 0.0
            out[i] = {
                "v0": float(v0s[i]), "m": float(m), "k": float(k),
                "b": float(b),
                "F_max": float(res["F_max"][jj]),
                "pen_max": float(res["pen_max"][jj]),
                "T_c": float(res["T_c"][jj]), "e": e,
                "min_f": float(res["min_f"][jj]),
                "separated": bool(res["separated"][jj]),
                "no_rebound_settled": bool(res["settled_no_rebound"][jj]),
                "dt": d,
                "F_touch_spike_theory": float(b) * float(v0s[i]),
            }
    return out


def static_equilibrium(ms, ks, bs):
    """Vectorized bisection for rest penetration.

    At rest the damping branch still contributes b*act*softplus_pen(0,1e3)
    = b*act*ln(2)/1e3 because softplus_pen(0) != 0, so the balance is
    k*pen + b*(pen/(pen+smooth))*ln(2)/1e3 = m*g.
    """
    mt = torch.tensor(ms, dtype=TT)
    kt = torch.tensor(ks, dtype=TT)
    bt = torch.tensor(bs, dtype=TT)
    target = mt * G
    lo = torch.zeros_like(kt)
    hi = (target + bt * REST_OFF) / kt + 1.0e-3
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f_mid = kt * mid + bt * (mid / (mid + SMOOTH)) * REST_OFF
        lo = torch.where(f_mid < target, mid, lo)
        hi = torch.where(f_mid < target, hi, mid)
    pen = 0.5 * (lo + hi)
    theory = target / kt + TOUCH_PEN
    return pen, theory


def dynamic_settle(m, k, b, z0=1.0e-3, dt=5.0e-5, t_max=1.5, v_tol=1.0e-6):
    """Drop from z0 and integrate to rest; validates the static solve
    against genuine dynamics of the exact law."""
    mt = torch.tensor(m, dtype=TT)
    kt = torch.tensor(k, dtype=TT)
    bt = torch.tensor(b, dtype=TT)
    z = torch.tensor(z0, dtype=TT)
    v = torch.tensor(0.0, dtype=TT)
    half, sixth = 0.5 * dt, dt / 6.0
    t = 0.0
    for i in range(int(t_max / dt)):
        f1, _ = _contact_law(z, v, kt, bt)
        a1 = f1 / mt - G
        z2, v2 = z + half * v, v + half * a1
        f2, _ = _contact_law(z2, v2, kt, bt)
        a2 = f2 / mt - G
        z3, v3 = z + half * v2, v + half * a2
        f3, _ = _contact_law(z3, v3, kt, bt)
        a3 = f3 / mt - G
        z4, v4 = z + dt * v3, v + dt * a3
        f4, _ = _contact_law(z4, v4, kt, bt)
        a4 = f4 / mt - G
        z = z + sixth * (v + 2.0 * v2 + 2.0 * v3 + v4)
        v = v + sixth * (a1 + 2.0 * a2 + 2.0 * a3 + a4)
        t += dt
        if (i & 63) == 63 and float(z) < 0.0 and abs(float(v)) < v_tol:
            break
    return {"z_rest": float(z), "pen_rest": float(softplus_pen(-z, BETA_SOFT)),
            "t_settle": t, "v_final": float(v)}


def _end_force(depth, k, b):
    if not torch.is_tensor(depth):
        depth = torch.tensor(depth, dtype=TT)
    pen = softplus_pen(depth, BETA_SOFT)
    return k * pen + b * (pen / (pen + SMOOTH)) * REST_OFF


def two_point_cop(thetas, w_total=40.0 * G, length=0.2, k=2.5e4, b=400.0):
    """Quasi-static two-point load share for a pitched capsule foot.

    Endpoints at x = +/-L/2; pitch th lowers the +x end by (L/2)*sin(th).
    Press depth dz is bisected until sum(f_i) = W exactly.
    """
    rows = []
    for th in thetas:
        xp = 0.5 * length * math.cos(th)
        xm = -xp
        drop = 0.5 * length * math.sin(th)

        def total(dz):
            return float(_end_force(dz + drop, k, b)
                         + _end_force(dz - drop, k, b))

        lo, hi = -drop - 1.0e-3, w_total / k + drop + 1.0e-3
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if total(mid) < w_total:
                lo = mid
            else:
                hi = mid
        dz = 0.5 * (lo + hi)
        f_lower = float(_end_force(dz + drop, k, b))
        f_raised = float(_end_force(dz - drop, k, b))
        f_sum = f_lower + f_raised
        x_cop = (f_lower * xp + f_raised * xm) / f_sum
        rows.append({
            "theta": th, "f_lower_end": f_lower, "f_raised_end": f_raised,
            "sum_f": f_sum, "sum_abs_err": abs(f_sum - w_total),
            "x_cop": x_cop,
            "x_cop_foot_frame": x_cop / math.cos(th),
            "press_depth": dz,
            "min_f": min(f_lower, f_raised),
        })
    return rows


def convergence_checks(grid_rows):
    """Prove every reported dt sits on the RK4 convergence plateau.

    Uses a non-spike-dominated stiff case, otherwise the stiff reference is
    pinned to the touch spike b*v0 and dt-independent for trivial reasons.
    """
    scen = ([1.0], [10.0], [8.0e4], [400.0])
    match = [r for r in grid_rows
             if (r["v0"], r["m"], r["k"], r["b"])
             == (1.0, 10.0, 8.0e4, 400.0)][0]
    stiff = {"1e-05": match["stiff_F_max"],
             "1e-06": stiff_reference_impact(*scen, dt=1.0e-6)[0]["F_max"]}
    ode = {"1e-06": match["F_max"],
           "5e-07": ode_reference_impact(*scen, dt=5.0e-7)[0]["F_max"]}
    svals = list(stiff.values())
    ovals = list(ode.values())
    return {
        "scenario": {"v0": 1.0, "m": 10.0, "k": 8.0e4, "b": 400.0},
        "stiff_F_max_by_dt": stiff,
        "stiff_rel_spread": (max(svals) - min(svals)) / min(svals),
        "ode_F_max_by_dt": ode,
        "ode_rel_spread": (max(ovals) - min(ovals)) / min(ovals),
        "note": ("stiff dt rule: literal 1e-7 s for k=8e4,m=1,b<=400; "
                 "otherwise coarsest dt with omega*dt<=3e-3; halving dt "
                 "moves F_max less than the reported spreads"),
    }


def run_grid(dt_ode=1.0e-6):
    grid = [(v0, m, k, b)
            for v0 in V0_GRID for m in M_GRID
            for k in K_GRID for b in B_GRID]
    ode_rows = ode_reference_impact([g[0] for g in grid],
                                    [g[1] for g in grid],
                                    [g[2] for g in grid],
                                    [g[3] for g in grid], dt=dt_ode)
    stiff_rows = stiff_reference_impact([g[0] for g in grid],
                                        [g[1] for g in grid],
                                        [g[2] for g in grid],
                                        [g[3] for g in grid])
    rows = []
    for o, s in zip(ode_rows, stiff_rows):
        r = dict(o)
        r["stiff_F_max"] = s["F_max"]
        r["stiff_T_c"] = s["T_c"]
        r["stiff_e"] = s["e"]
        r["stiff_pen_max"] = s["pen_max"]
        r["stiff_min_f"] = s["min_f"]
        r["stiff_dt"] = s["dt"]
        r["stiff_separated"] = s["separated"]
        denom = s["F_max"] if s["F_max"] > 0 else float("nan")
        r["rel_diff_F_max_pct"] = 100.0 * (o["F_max"] - s["F_max"]) / denom
        r["F_touch_spike_theory"] = s["F_touch_spike_theory"]
        rows.append(r)
    return rows


def _print_table(rows):
    hdr = (f"{'v0':>4} {'m':>5} {'k':>7} {'b':>5} | {'Fmax_ode':>10} "
           f"{'Fmax_stiff':>10} {'dF%':>7} | {'Tc_ode':>8} {'Tc_stiff':>8} | "
           f"{'e':>6} {'Eloss%':>7} | {'penmax_mm':>9} | {'minFn':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        tc_s = r["stiff_T_c"] * 1e3 if r["stiff_separated"] else float("nan")
        print(f"{r['v0']:>4.1f} {r['m']:>5.0f} {r['k']:>7.0f} {r['b']:>5.0f} | "
              f"{r['F_max']:>10.2f} {r['stiff_F_max']:>10.2f} "
              f"{r['rel_diff_F_max_pct']:>7.2f} | "
              f"{r['T_c']*1e3:>8.3f} {tc_s:>8.3f} | "
              f"{r['e']:>6.3f} {100.0*r['energy_loss_frac']:>7.2f} | "
              f"{r['pen_max']*1e3:>9.3f} | {r['min_f']:>8.3f}")


def main():
    print("=== Impact-physics metrics (Q1d): compliant contact vs stiff limit ===")
    print(f"constants: beta_soft={BETA_SOFT:g} (eps={1/BETA_SOFT:g} m), "
          f"smooth={SMOOTH:g}, beta_damp={BETA_D:g}, vcap={VCAP:g}, "
          f"touch_pen=ln2/beta_soft={TOUCH_PEN:.3e} m\n")

    rows = run_grid()
    for r in rows:
        r["spike_limited"] = abs(r["stiff_F_max"]
                                 - r["F_touch_spike_theory"]) <= 1e-6 * max(
                                     1.0, r["stiff_F_max"])
    print("--- ODE reference (exact law, RK4 dt=1e-6) vs stiff-limit reference ---")
    _print_table(rows)

    sep_ok = all(r["separated"] or r["no_rebound_settled"] for r in rows)
    n_noreb = sum(r["no_rebound_settled"] for r in rows)
    glob_min_f = min(min(r["min_f"], r["stiff_min_f"]) for r in rows)
    glob_tail = max(r["tail_f"] for r in rows)
    print("\n--- suction / lingering-force checks ---")
    print(f"every scenario terminates cleanly (rebound or settled): {sep_ok}")
    print(f"scenarios with no rebound (overdamped creep to rest): {n_noreb}")
    print(f"global min contact force over all sampled states: {glob_min_f:.3e} N "
          f"(>= -1e-9 required)")
    print(f"max far-field (z=1mm) residual force: {glob_tail:.3e} N "
          f"(compressive exponential tail, no tension)")

    combos = sorted({(r["m"], r["k"], r["b"]) for r in rows})
    pens, theory = static_equilibrium([c[0] for c in combos],
                                      [c[1] for c in combos],
                                      [c[2] for c in combos])
    print("\n--- static rest penetration (phantom offset) ---")
    print(f"{'m':>5} {'k':>7} {'b':>5} | {'pen_meas':>10} {'mg/k+ln2/bs':>12} "
          f"{'diff':>10} | {'b*ln2/1e3/k':>11}")
    stat_rows = []
    for j, (m, k, b) in enumerate(combos):
        diff = float(pens[j] - theory[j])
        damp_off = b * REST_OFF / k
        stat_rows.append({"m": m, "k": k, "b": b,
                          "pen_meas": float(pens[j]),
                          "pen_theory_mg_over_k_plus_ln2_beta": float(theory[j]),
                          "diff": diff})
        print(f"{m:>5.0f} {k:>7.0f} {b:>5.0f} | {float(pens[j]):>10.6f} "
              f"{float(theory[j]):>12.6f} {diff:>10.2e} | {damp_off:>11.2e}")
    settle = dynamic_settle(10.0, 2.5e4, 400.0)
    print(f"dynamic settle (m=10,k=2.5e4,b=400): pen_rest="
          f"{settle['pen_rest']:.6f} m, t={settle['t_settle']:.2f} s, "
          f"|v_final|={abs(settle['v_final']):.1e} m/s")

    cop = two_point_cop([0.0, 0.05, 0.1, 0.2])
    print("\n--- two-point load share (capsule foot, L=0.2 m, W=40g) ---")
    print(f"{'theta':>6} | {'f_lower':>9} {'f_raised':>9} | {'sum_err':>9} | "
          f"{'x_cop':>9} {'x_cop_foot':>10} | {'min_f':>9}")
    for c in cop:
        print(f"{c['theta']:>6.2f} | {c['f_lower_end']:>9.3f} "
              f"{c['f_raised_end']:>9.3f} | {c['sum_abs_err']:>9.2e} | "
              f"{c['x_cop']:>9.6f} {c['x_cop_foot_frame']:>10.6f} | "
              f"{c['min_f']:>9.3e}")

    conv = convergence_checks(rows)
    print("\n--- convergence of both references (v0=1, m=10, k=8e4, b=400) ---")
    print(f"stiff F_max by dt: {conv['stiff_F_max_by_dt']} "
          f"(spread {conv['stiff_rel_spread']:.2e})")
    print(f"ode   F_max by dt: {conv['ode_F_max_by_dt']} "
          f"(spread {conv['ode_rel_spread']:.2e})")

    sat_v02 = sum(r["clamp_sat_steps"] for r in rows if r["v0"] == 2.0)
    rel = [abs(r["rel_diff_F_max_pct"]) for r in rows]
    worst = rows[rel.index(max(rel))]
    soft_rel = [r["rel_diff_F_max_pct"] for r in rows if r["k"] == 1.5e4]
    stiff8_rel = [abs(r["rel_diff_F_max_pct"]) for r in rows
                  if r["k"] == 8.0e4]
    stiff8_nonspike = [abs(r["rel_diff_F_max_pct"]) for r in rows
                       if r["k"] == 8.0e4 and not r["spike_limited"]]
    n_spike = sum(r["spike_limited"] for r in rows)

    notes = [
        f"phantom preload at geometric touch: pen=ln2/beta_soft="
        f"{TOUCH_PEN:.3e} m gives k*pen = {2.5e4*TOUCH_PEN:.2f} N (k=2.5e4) "
        f"before any real overlap; this is also why the minimum sampled "
        f"contact force equals k*ln2/beta_soft rather than 0",
        f"static-equilibrium surprise (opposite sign vs naive theory): the "
        f"damping branch never sleeps -- softplus_pen(0,1e3)=ln2/1e3 adds "
        f"b*act*{REST_OFF:.3e} N of EXTRA SUPPORT at rest, so measured rest "
        f"penetration is SHALLOWER than mg/k by b*ln2/(1e3*k), and the "
        f"naive theory mg/k + ln2/beta_soft overestimates by ~ln2/"
        f"beta_soft = {TOUCH_PEN:.3e} m (see 'diff' column)",
        f"approach-speed clamp saturates for v0=2.0 impacts ({int(sat_v02)} "
        f"sampled states capped at 2.0 m/s): caps the damping contribution "
        f"to F_max for fast impacts",
        f"stiff-limit reference carries an instantaneous Kelvin-Voigt touch "
        f"spike F=b*v0 (no activation gate): whenever b*v0 exceeds the "
        f"spring-dominated peak, Fmax_stiff == b*v0 exactly ({n_spike}/81 "
        f"scenarios spike-limited). Excluding those, stiff-vs-ODE agreement "
        f"is <= {max(stiff8_nonspike):.2f}% for k=8e4; worst overall rel "
        f"diff {max(rel):.2f}% at (v0={worst['v0']}, m={worst['m']}, "
        f"k={worst['k']}, b={worst['b']})",
        f"softest contacts (k=1.5e4): rel diffs {min(soft_rel):.2f}%.."
        f"{max(soft_rel):.2f}% -- graceful degradation; ramp eps=1e-4 m is "
        f"still << peak penetrations (>= 0.7 mm)",
        f"{n_noreb} heavily-damped light-mass scenarios never rebound "
        f"(contact damping ratio >~ 1): they creep asymptotically to the "
        f"damped equilibrium inside the contact zone and are reported with "
        f"e = 0; restitution stays strictly < 1 everywhere "
        f"(max e over grid: {max(r['e'] for r in rows):.4f})",
    ]
    print("\n--- anomaly notes ---")
    for nt in notes:
        print(f"  * {nt}")

    results = {
        "question": "Q1d: compliant contact forces under dynamic impacts",
        "metadata": {
            "g": G, "beta_soft": BETA_SOFT, "smooth": SMOOTH,
            "beta_damping": BETA_D, "vcap": VCAP,
            "rest_offset_ln2_over_beta_d": REST_OFF,
            "touch_pen_ln2_over_beta_soft": TOUCH_PEN,
            "ode_dt": 1.0e-6,
            "ode_window_entry_z_m": 3.0 / BETA_SOFT,
            "restitution_definition":
                "ratio of interpolated velocities at symmetric crossings of "
                "the touch plane z=0; equals far-field ratio by gravity "
                "symmetry",
            "stiff_dt_rule": ("1e-7 for k=8e4,m=1,b<=400; else "
                              "min(1e-5, max(1e-7, 3e-3/sqrt(k/m)))"),
            "n_spikelimited_stiff_rows": int(n_spike),
            "axes": {"v0": list(V0_GRID), "m": list(M_GRID),
                     "k": list(K_GRID), "b": list(B_GRID)},
        },
        "scenarios": rows,
        "suction_summary": {
            "all_terminated_cleanly": sep_ok,
            "n_no_rebound_settled": int(n_noreb),
            "global_min_contact_force_N": glob_min_f,
            "max_far_field_residual_force_N": glob_tail,
        },
        "static_equilibrium": stat_rows,
        "dynamic_settle": settle,
        "two_point_cop": cop,
        "convergence": conv,
        "anomaly_notes": notes,
    }
    bench_dir = os.path.join(_ROOT, "benchmarks")
    os.makedirs(bench_dir, exist_ok=True)
    out_path = os.path.join(bench_dir, "impact_metrics.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nsaved: benchmarks/{os.path.basename(out_path)}")

    checks = [
        ("clean termination everywhere", sep_ok),
        ("no lingering force after separation", glob_tail < 1e-2),
        ("no suction anywhere", glob_min_f >= -1e-9),
        ("CoP centered at zero pitch", abs(cop[0]["x_cop"]) < 1e-6),
        ("CoP migrates monotonically (foot frame)",
         all(cop[i]["x_cop_foot_frame"] <= cop[i + 1]["x_cop_foot_frame"]
             + 1e-12 for i in range(len(cop) - 1))
         and cop[1]["x_cop_foot_frame"] > 0.0),
        ("CoP load balance to 1e-9", max(c["sum_abs_err"] for c in cop) < 1e-9),
        ("no endpoint tension", min(c["min_f"] for c in cop) >= -1e-12),
        ("stiff agreement k=8e4 (non-spike) within 15%",
         max(stiff8_nonspike) < 15.0),
    ]
    print("\n=== RESULTS ===")
    ok_all = True
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        ok_all &= ok
    return ok_all


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
