"""Compass simplest-walker ORACLE -- independent sympy derivation (Q1c).

This is the rigid-contact counterpart of the DiffSim soft-contact twin
(diffsim/walker.py).  It shares NOTHING numerically with the DiffSim
engine: the equations of motion are derived symbolically from Cartesian
kinematics via the Lagrangian, and the impact map is the generic plastic
collision projection

    v+ = v- - Minv J^T (J Minv J^T)^-1 J v-

with J the Jacobian of the contacting point (the new stance tip).
Morphology equals the twin's (diffsim.walker.WALKER_P is the single
parameter source): point-mass hip M, point masses beta*M at the feet,
massless legs of length l; slope gamma realized as tilted gravity
ghat = g(sin(gamma), -cos(gamma)) in the (x, z) plane -- walking downhill
on flat ground.

Coordinates: ABSOLUTE leg angles from world vertical, positive downhill.
    stance foot pinned at origin -> hip   = ( l*sin(th1), l*cos(th1))
    swing tip                    -> sw    = hip + l*(-sin(th2), -cos(th2))
Walking direction: downhill = +x = hip advancing over the pivot, so th1
INCREASES through stance (-th* -> +th*) and the swing angle DECREASES
(+th* -> -th*).  Heelstrike: swing tip reaches z=0 in FRONT of the pivot
<=> th1 + th2 = 0 crossed upward (g: - -> +) with th1 > 0 > th2; the
leg-parallel near-vertical crossing is scuffing (ignored, per Garcia).
Post-impact relabel: new stance angle th1' = th2_touch (< 0), new swing
th2' = th1_touch (> 0).  Section state after impact:
s = (th1', dth1', dth2')  (3 numbers; dth2' reconstructs the full state).

External anchors:
  * acceleration field must equal Garcia et al. 1998 eqs (1)-(2) up to
    O(beta) under the verified map theta_G = -th1_w,
    phi_G = th2_w - th1_w with dimensionless-time rescaling;
  * Garcia Table-1 asymptotic period-one gait (rates converted from
    dimensionless tau!) must close one stride to O(beta).
"""
from __future__ import annotations

import math
import sys

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 2)[0])

from diffsim.walker import WALKER_P  # noqa: E402  (single parameter source)

P0 = dict(WALKER_P)

# --------------------------------------------------------------------- #
# symbolic dynamics
# --------------------------------------------------------------------- #
import sympy as sp  # noqa: E402

_th1, _th2, _d1, _d2, _gam = sp.symbols("th1 th2 d1 d2 gam", real=True)
_dd1, _dd2 = sp.symbols("dd0 dd1", real=True)
_M, _mf, _l, _g = sp.symbols("M mf l g", positive=True)

_hip = sp.Matrix([_l * sp.sin(_th1), _l * sp.cos(_th1)])
_sw = _hip + sp.Matrix([-_l * sp.sin(_th2), -_l * sp.cos(_th2)])
_ghat = sp.Matrix([_g * sp.sin(_gam), -_g * sp.cos(_gam)])
_v = sp.Matrix([_d1, _d2])
_qv = sp.Matrix([_th1, _th2])

_vhip = _hip.jacobian(_qv) * _v
_vsw = _sw.jacobian(_qv) * _v
_T = sp.Rational(1, 2) * (_M * _vhip.dot(_vhip) + _mf * _vsw.dot(_vsw))
_U = -(_M * _ghat.dot(_hip) + _mf * _ghat.dot(_sw))
_L = sp.expand(_T - _U)

_dldv = [sp.diff(_L, _v[i]) for i in range(2)]
_eom = []
for i in range(2):
    d_dt = sum(sp.diff(_dldv[i], _qv[j]) * _v[j] for j in range(2)) \
        + sum(sp.diff(_dldv[i], _v[j]) * [_dd1, _dd2][j] for j in range(2))
    _eom.append(sp.expand(d_dt - sp.diff(_L, _qv[i])))
_sol = sp.solve(_eom, [_dd1, _dd2], dict=True)[0]
_ACC = sp.simplify(sp.Matrix([sp.simplify(_sol[_dd1]),
                              sp.simplify(_sol[_dd2])]))
_MINV = sp.simplify((sp.hessian(_T, _v)).inv())
_JAC_SW = _sw.jacobian(_qv)


def _subs(beta_val):
    return {_M: P0["M"], _mf: beta_val * P0["M"], _l: P0["l"], _g: 9.81}


def _lambdas(beta_val):
    sb = _subs(beta_val)
    f_minv = sp.lambdify([_th1, _th2, _d1, _d2, _gam], _MINV.subs(sb), "numpy")
    f_acc = sp.lambdify(
        [_th1, _th2, _d1, _d2, _gam, _dd1, _dd2], _ACC.subs(sb), "numpy")
    f_J = sp.lambdify([_th1, _th2], _JAC_SW.subs(sb), "numpy")
    f_T = sp.lambdify([_th1, _th2, _d1, _d2, _gam], _T.subs(sb), "numpy")
    f_U = sp.lambdify([_th1, _th2, _gam], _U.subs(sb), "numpy")
    return f_minv, f_acc, f_J, f_T, f_U


_LAM: dict[float, tuple] = {}


def lam(beta=None):
    b = P0["beta"] if beta is None else beta
    if b not in _LAM:
        _LAM[b] = _lambdas(b)
    return _LAM[b]


# --------------------------------------------------------------------- #
# oracle dynamics
# --------------------------------------------------------------------- #

class WalkerOracle:
    """Rigid-contact hybrid flow map (sympy-derived, fp64 numpy)."""

    def __init__(self, gamma: float, beta: float | None = None,
                 h: float = 5e-4):
        self.gamma = float(gamma)
        self.beta = P0["beta"] if beta is None else float(beta)
        self.h = h
        self.f_minv, self.f_acc, self.f_J, self.f_T, self.f_U = lam(self.beta)

    def mass_matrix(self, y):
        th1, _, th2, _ = y
        return np.linalg.inv(np.asarray(
            self.f_minv(th1, th2, 0.0, 0.0, self.gamma), dtype=float))

    def accel(self, y):
        """NOTE: _ACC is the SOLVED acceleration field (sympy eliminated
        qdd), so we evaluate it directly -- multiplying by Minv here would
        double-process the equations (the bug this line once had)."""
        th1, d1, th2, d2 = y
        return np.asarray(
            self.f_acc(th1, th2, d1, d2, self.gamma, 0.0, 0.0),
            dtype=float).reshape(2)

    def deriv(self, y):
        return np.array([y[1], self.accel(y)[0], y[3], self.accel(y)[1]])

    def flow_step(self, y):
        k1 = self.deriv(y)
        k2 = self.deriv(y + 0.5 * self.h * k1)
        k3 = self.deriv(y + 0.5 * self.h * k2)
        k4 = self.deriv(y + self.h * k3)
        return y + (self.h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    @staticmethod
    def g_strike(y):
        return y[0] + y[2]

    def energy(self, y):
        th1, d1, th2, d2 = y
        T = float(self.f_T(th1, th2, d1, d2, self.gamma))
        U = float(self.f_U(th1, th2, self.gamma))
        return T + U

    def swing_tip(self, y):
        th1, _, th2, _ = y
        l = P0["l"]
        hx, hz = l * math.sin(th1), l * math.cos(th1)
        return np.array([hx - l * math.sin(th2), hz - l * math.cos(th2)])

    def impact(self, y):
        """Plastic heelstrike via Newtonian impulse balance.

        Topology switch: pre-impact the STANCE tip P is pinned (its point
        mass contributes nothing); at impact the ground exerts an unknown
        impulse J at the SWING tip Q (which sticks: vQ+ = 0), the old pin
        RELEASES (no external impulse at P), and both massless rods
        transmit axial impulses N1 (hip-P) and N2 (hip-Q).

        Unknowns x = [vh+, vP+, N1, N2, Jx, Jz] (8), vQ+ == 0.
        Equations:
          hip :  M(vh - vh-) + N1 n1 + N2 n2 = 0
          P   :  m(vP - 0 ) - N1 n1           = 0      (released)
          Q   :  m(0  - vQ-) - N2 n2 - J      = 0
          rod1:  (vP - vh) . t1               = 0
          rod2:  (0  - vh) . t2               = 0     (plastic + rigid)
        where ni = unit(P/Q - hip), ti = perp(ni).  Solve 8x8 linear
        system; extract theta-dot1+ = vh.(c1,-s1)/l and
        theta-dot2+ = (-c2 vh.x + s2 vh.z)/l from rod rigidity.

        Returns (y_post, info).  KE must not increase; angular momentum
        about Q must be conserved exactly (J acts at Q).
        """
        th1, d1, th2, d2 = y
        l, M, mf = P0["l"], P0["M"], self.beta * P0["M"]
        s1, c1 = math.sin(th1), math.cos(th1)
        s2, c2 = math.sin(th2), math.cos(th2)
        # pre-impact velocities (P pinned)
        vh_m = np.array([l * d1 * c1, -l * d1 * s1])
        vQ_m = vh_m + np.array([-l * d2 * c2, l * d2 * s2])
        n1 = np.array([-s1, -c1])
        n2 = np.array([-s2, -c2])
        # x = [vhx, vhz, vPx, vPz, N1, N2, Jx, Jz]
        A = np.zeros((8, 8))
        b = np.zeros(8)
        # hip momentum
        A[0, 0], A[0, 4], A[0, 5] = M, n1[0], n2[0]
        b[0] = M * vh_m[0]
        A[1, 1], A[1, 4], A[1, 5] = M, n1[1], n2[1]
        b[1] = M * vh_m[1]
        # P momentum (released: vP- = 0)
        A[2, 2], A[2, 4] = mf, -n1[0]
        A[3, 3], A[3, 4] = mf, -n1[1]
        # Q momentum (vQ+ = 0)
        A[4, 5], A[4, 6], A[4, 7] = -n2[0], -1.0, 0.0
        b[4] = -mf * vQ_m[0]
        A[5, 5], A[5, 6], A[5, 7] = -n2[1], 0.0, -1.0
        b[5] = -mf * vQ_m[1]
        # rigid rods: endpoint relative velocity is PERPENDICULAR to the
        # rod, i.e. along the tangent <=> (v_end - v_hip) . n = 0
        A[6, 0:4] = [-n1[0], -n1[1], n1[0], n1[1]]
        A[7, 0], A[7, 1] = -n2[0], -n2[1]
        x = np.linalg.solve(A, b)
        vh = x[0:2]
        # extract generalized rates in POST-relabel geometry:
        # new stance leg angle = old th2 (its tip is pinned at Q):
        #   vh = d1+ * l*(c2,-s2)
        # new swing leg angle = old th1; its tip velocity obeys
        #   vP - vh = d2+ * l*(-c1,s1)
        d1_p = float(vh @ np.array([c2, -s2]) / l)
        d2_p = float((x[2:4] - vh) @ np.array([-c1, s1]) / l)
        yp = np.array([th2, d1_p, th1, d2_p])   # relabel legs
        ke_pre = 0.5 * M * vh_m @ vh_m \
            + 0.5 * mf * vQ_m @ vQ_m            # P pinned: no KE
        vQ_p = np.zeros(2)
        vP_p = x[2:4]
        ke_post = 0.5 * M * vh @ vh + 0.5 * mf * vP_p @ vP_p \
            + 0.5 * mf * vQ_p @ vQ_p
        pivot = self.swing_tip(y)          # Q in the pre-impact frame (P at origin)
        info = {"ke_pre": ke_pre, "ke_post": ke_post, "pivot": pivot,
                "L_pre": self.angular_momentum(y, (0.0, 0.0), pivot),
                "L_post": self.angular_momentum(yp, pivot, pivot)}
        return yp, info

    def angular_momentum(self, y, S_xy, Q_xy):
        """Total z-angular momentum about world point Q_xy.

        y is expressed in the pinned frame whose ORIGIN is the state's own
        stance foot, located at world S_xy (constant-offset shifts
        positions only; velocities unaffected).  Point masses: hip (M),
        stance tip (mf, at rest on its pin), swing tip (mf).
        """
        th1, d1, th2, d2 = y
        l, M, mf = P0["l"], P0["M"], self.beta * P0["M"]
        s1, c1, s2, c2 = math.sin(th1), math.cos(th1), \
            math.sin(th2), math.cos(th2)
        S = np.asarray(S_xy, dtype=float)
        Q = np.asarray(Q_xy, dtype=float)
        r_hip = S + np.array([l * s1, l * c1])
        v_hip = np.array([l * d1 * c1, -l * d1 * s1])
        r_sw = r_hip + np.array([-l * s2, -l * c2])
        v_sw = v_hip + np.array([-l * d2 * c2, l * d2 * s2])
        mom = M * np.cross(np.append(r_hip - Q, 0.0),
                           np.append(v_hip, 0.0))[2]
        mom += mf * np.cross(np.append(S - Q, 0.0),
                             np.zeros(3))[2]          # pinned: v=0
        mom += mf * np.cross(np.append(r_sw - Q, 0.0),
                             np.append(v_sw, 0.0))[2]
        return float(mom)

    def one_stride(self, y0, max_tau=25.0):
        """Post-impact state -> next post-impact state.

        World-coords walk direction: stance angle th1 increases through
        stance (-th* -> +th*, hip advances downhill over the pivot);
        swing th2 decreases (+th* -> -th*, tip travels forward).
        Strike condition: g = th1+th2 crosses from negative (tip below
        ground -- the Garcia fiction ignores mid-swing scuffing, so the
        tip may travel underground through mid-stance) to >= 0 with the
        TRUE-strike geometry th1 > 0 > th2 (stance past the pivot, swing
        planting in front).  The leg-parallel near-vertical crossing
        (th1 ~ -th2 ~ 0, g changing the OTHER way) is scuffing and never
        satisfies the branch test.
        """
        y = np.array(y0, dtype=float)
        t = 0.0
        gp = self.g_strike(y)
        while t < max_tau:
            yn = self.flow_step(y)
            t += self.h
            gn = self.g_strike(yn)
            if gp < 0 <= gn:
                lo, hi = 0.0, 1.0
                ya, yb = y, yn
                for _ in range(60):
                    mid = 0.5 * (lo + hi)
                    if self.g_strike(ya + (yb - ya) * mid) < 0:
                        lo = mid
                    else:
                        hi = mid
                yt = ya + (yb - ya) * hi
                if yt[0] > 0 and yt[2] < 0:
                    yp, info = self.impact(yt)
                    return yp, True, int(round(t / self.h)), info
                # scuff geometry: keep integrating
            if abs(float(yn[0])) > 3.0 or abs(float(yn[2])) > 3.0:
                return yn, False, int(round(t / self.h)), None
            y, gp = yn, gn
        return y, False, int(round(t / self.h)), None

    def section_map(self, s3):
        """s3 = (th1+, dth1+, dth2+) post-impact -> next, or (None, False)."""
        y0 = np.array([s3[0], s3[1], -s3[0], s3[2]])
        yp, ok, n, info = self.one_stride(y0)
        if not ok:
            return None, False, n, info
        return np.array([yp[0], yp[1], yp[3]]), True, n, info

    def find_fixed_point(self, seed3, iters=400, tol=1e-11):
        s = np.array(seed3, dtype=float)
        sn = s
        for _ in range(iters):
            sn, ok, _n, _i = self.section_map(s)
            if not ok:
                return None
            if np.max(np.abs(sn - s)) < tol:
                return {"s": sn, "resid": float(np.max(np.abs(sn - s))),
                        "rho": self._spectral_radius(sn)}
            s = sn
        resid = float(np.max(np.abs(sn - s)))
        if resid < 1e-7:
            return {"s": s, "resid": resid,
                    "rho": self._spectral_radius(s)}
        return None

    def _spectral_radius(self, s3, eps=1e-6):
        cols = []
        for j in range(3):
            spp = s3.copy(); spp[j] += eps
            smm = s3.copy(); smm[j] -= eps
            pp, o1, _, _ = self.section_map(spp)
            pm, o2, _, _ = self.section_map(smm)
            if not (o1 and o2):
                return float("nan")
            cols.append((pp - pm) / (2 * eps))
        Mj = np.stack(cols, axis=1)
        return float(np.max(np.abs(np.linalg.eigvals(Mj))))


# --------------------------------------------------------------------- #
# external anchor: Garcia beta->0 limit
# --------------------------------------------------------------------- #
_EOM_LAM = None


def _eom_residual(y, gamma, acc, beta_val):
    """Lagrange-equation residuals E_i(q, qd, acc) at finite beta."""
    global _EOM_LAM
    if _EOM_LAM is None:
        sb_all = {_M: P0["M"], _l: P0["l"], _g: 9.81}
        fs = [sp.lambdify([_th1, _th2, _d1, _d2, _gam, _mf, _dd1, _dd2],
                          e.subs(sb_all), "numpy") for e in _eom]
        _EOM_LAM = fs
    th1, d1, th2, d2 = y
    return [float(f(th1, th2, d1, d2, gamma, beta_val * P0["M"],
                    float(acc[0]), float(acc[1]))) for f in _EOM_LAM]


def garcia_limit_check(gamma=0.009, n=64, seed=0, betas=(1e-4, 1e-6)):
    """ACCELERATION anchor: our sympy flow == published Garcia flow.

    Compares instantaneous accelerations (NOT integrated trajectories):
    integrating near beta->0 is hopeless with explicit RK4 because the
    vanishing swing-leg inertia creates a fast mode omega ~ 1/sqrt(beta)
    (the singular-perturbation stiffness).  Accelerations are algebraic
    in the state, so they carry no stiffness -- and the published field
    IS the beta->0 limit of ours, so |a_ours(beta) - a_garcia| must
    shrink like O(beta).

    Coordinate map (VERIFIED against the published field): Garcia theta
    DECREASES through stance (omega < 0) while our world angle INCREASES
    (+x downhill), so theta_G = -th1_w, phi_G = th2_w - th1_w,
    rates omega_G = -d1, eta_G = d2 - d1; accelerations transform the
    same way plus the dimensionless-time rescaling
    a_w = (g/l) * a_G, v_G = sqrt(l/g) * v_w.
    """
    import garcia_oracle as go

    rng = np.random.default_rng(seed)
    worst_by_beta = {b: 0.0 for b in betas}
    gl = 9.81 / P0["l"]                 # real <-> dimensionless factor
    tscale = math.sqrt(P0["l"] / 9.81)  # dimless rate = real * tscale
    for _ in range(n):
        th1w, th2w = rng.uniform(-0.35, 0.35, 2)
        if abs(th1w - th2w) < 0.05:
            continue
        d1, d2 = rng.uniform(-1.0, 1.0, 2)
        y = np.array([th1w, d1, th2w, d2])
        thG = -th1w
        phiG = th2w - th1w
        om_d = -d1 * tscale              # dimensionless rates (negated)
        th_dd = math.sin(thG - gamma)
        phi_dd = th_dd + (om_d ** 2 - math.cos(thG - gamma)) \
            * math.sin(phiG)
        # a_w1 = -a_G_theta ; a_w2 = -(a_G_theta - a_G_phi) = a_G_phi - a_G_theta
        aG = np.array([-th_dd, phi_dd - th_dd]) * gl   # to real units
        for b in betas:
            a_ours = WalkerOracle(gamma, beta=b).accel(y)
            worst_by_beta[b] = max(worst_by_beta[b],
                                   float(np.max(np.abs(a_ours - aG))))
    ratios = [worst_by_beta[betas[i + 1]] / max(worst_by_beta[betas[i]],
                                                1e-300)
              for i in range(len(betas) - 1)]
    return worst_by_beta[betas[-1]], worst_by_beta, ratios


def analytic_seed_garcia(gamma):
    import garcia_oracle as go
    return go.analytic_seed(gamma, "long")


def continuation_fp(gamma, beta_target, betas=(0.001, 0.005, 0.01, 0.02,
                                              0.035, 0.05)):
    """Track the period-one fixed point in beta by continuation.

    The Garcia-mapped seed lands in the basin only for small beta; larger
    foot masses shift the gait.  Walk beta upward, seeding each solve
    with the previous FP.
    """
    import garcia_oracle as go
    fpG = go.shoot_fixed_point(gamma, go.analytic_seed(gamma, "long"))
    if fpG is None:
        return None, None
    thG, wG = fpG["theta"], fpG["omega"]
    wr = math.sqrt(9.81 / P0["l"])
    c2 = math.cos(2 * thG)
    s = np.array([-thG, -wG * wr,
                  -wG * wr + c2 * (1 - c2) * (wG * wr / c2)])
    chain = []
    for b in betas:
        if b > beta_target:
            break
        orc = WalkerOracle(gamma, beta=b)
        r = orc.find_fixed_point(s)
        if r is None:
            # retry from perturbed seeds before giving up
            for fac in (1.02, 0.98):
                r = orc.find_fixed_point(np.array([s[0] * fac, s[1] * fac,
                                                   s[2]]))
                if r is not None:
                    break
        if r is None:
            return None, chain
        s = r["s"]
        chain.append({"beta": b, "s": list(map(float, s)),
                      "rho": r["rho"]})
        if abs(b - beta_target) < 1e-12:
            return orc, s
    if abs(betas[-1] - beta_target) < 1e-12 or beta_target in betas:
        return WalkerOracle(gamma, beta=beta_target), s
    # beta_target beyond the chain end: one more step
    orc = WalkerOracle(gamma, beta=beta_target)
    r = orc.find_fixed_point(s)
    return (orc, r["s"]) if r is not None else (None, chain)


def midstance_state(gamma, beta_target=None):
    """Full state on the periodic orbit at the theta1 = 0 crossing
    (stance leg vertical, swing foot airborne).  Used to seed the twin
    AWAY from the double-support instant: compliant contacts make the
    instantaneous rigid-model liftoff sticky, so a TD-section seed falls
    into a braced two-foot standing attractor instead of walking."""
    out = continuation_fp(gamma, beta_target or P0["beta"])
    orc, s = out[0], out[1]
    if orc is None or s is None or not np.isscalar(s[0]):
        return None, None, None, None
    y = np.array([s[0], s[1], -s[0], s[2]])
    t = 0.0
    prev = y[0]
    while t < 25.0:
        y = orc.flow_step(y)
        t += orc.h
        if prev < 0 <= y[0]:          # stance crossing vertical, forward
            return orc, y, t, s
        prev = y[0]
    return orc, None, None, s


if __name__ == "__main__":
    print("== walker oracle validation ==")
    e, byb, ratios = garcia_limit_check()
    print("flow match vs Garcia oracle (max err over 64 random states):")
    for b, v in byb.items():
        print(f"  beta={b:g}: {v:.3e}")
    # PASS criterion: error vanishes LINEARLY in beta (ratio == beta ratio
    # within 20%) and residual small -- the published field is the
    # beta->0 limit, so O(1) agreement at finite beta is the theorem.
    ok = ratios and abs(ratios[0] - 1e-2) < 0.2e-2 and e < 1e-3
    print(f"  decay ratios {['%.3f' % r for r in ratios]} (expect 0.01)")
    print(f"  {'PASS' if ok else 'FAIL'}")

    gam = 0.009
    orc = WalkerOracle(gam)
    from garcia_oracle import shoot_fixed_point  # noqa: E402
    # numeric Garcia FP found previously at gamma=0.009 (theta*, omega*)
    fpG = shoot_fixed_point(gam, analytic_seed_garcia(gam))
    if fpG is None:
        raise SystemExit("garcia FP not found; cannot seed")
    th_star, w_star = fpG["theta"], fpG["omega"]      # Garcia coords
    # Garcia variables are DIMENSIONLESS (tau = t sqrt(g/l)); convert
    # rates to real rad/s, then mirror into world coords (theta_G = -th1_w)
    wr = math.sqrt(9.81 / P0["l"])
    w_star_r = w_star * wr
    c2 = math.cos(2 * th_star)
    w_pre_r = w_star_r / c2                 # pre-impact stance rate, real
    phd_post_r = c2 * (1 - c2) * w_pre_r    # paper jump eq 4, real units
    # world post-impact: th1_w = -theta_G, d1 = -omega_G(real);
    # phi_dot_G maps WITHOUT negation (difference of two negated angles):
    # eta_G = d2 - d1 -> d2 = d1 + eta_G.  NOTE: at finite beta the
    # swing-rate coordinate is slaved differently than in the beta->0
    # limit (singular perturbation), so only theta*/omega*/tau/multipliers
    # are expected to match Garcia closely -- not d2*.
    s3 = np.array([-th_star, -w_star_r,
                   -w_star_r + phd_post_r])
    yp, okc, nsteps, st = orc.one_stride(
        np.array([s3[0], s3[1], -s3[0], s3[2]]))
    print(f"one stride from mapped Garcia FP: closed={okc} "
          f"tau={nsteps * orc.h:.4f} (paper long branch ~3.88)")
    if st:
        print(f"  KE ratio post/pre = "
              f"{st['ke_post'] / max(st['ke_pre'], 1e-30):.5f}")
        print(f"  L about pivot pre/post = {st['L_pre']:.10e} / "
              f"{st['L_post']:.10e}")
    if okc:
        dev = float(np.max(np.abs(yp - np.array(
            [s3[0], s3[1], -s3[0], s3[2]]))))
        print(f"  return deviation after one stride = {dev:.6f} "
              f"(O(beta) expected)")
