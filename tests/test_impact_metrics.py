"""Impact-physics metrics invariants (Q1d).

Asserts the scientifically important properties of the compliant contact
model under normal impacts, against two independent references:
  - ODE reference : exact DiffSim ground law integrated at dt = 1e-6 s
  - stiff limit   : piecewise spring + approach-only damping reference

fp64 throughout; scenario subsets use short-window (light-mass) cases so
total runtime stays well under 120 s while every dt used is on its
verified RK4 convergence plateau (the harness reports halving-dt spreads).
"""
import os
import sys
from functools import lru_cache

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import impact_metrics as im


@lru_cache(maxsize=1)
def _mono_rows():
    scens = [(0.5, 1.0, 2.5e4, 400.0), (1.0, 1.0, 2.5e4, 400.0),
             (2.0, 1.0, 2.5e4, 400.0),
             (1.0, 1.0, 1.5e4, 100.0), (1.0, 1.0, 8.0e4, 100.0),
             (1.0, 1.0, 2.5e4, 100.0), (1.0, 1.0, 2.5e4, 800.0)]
    rows = im.ode_reference_impact([s[0] for s in scens],
                                   [s[1] for s in scens],
                                   [s[2] for s in scens],
                                   [s[3] for s in scens], dt=1.0e-6)
    return {tuple(s): r for s, r in zip(scens, rows)}


@lru_cache(maxsize=1)
def _stiff_agreement_pairs():
    """k=8e4 cases where neither reference is spike-dominated; first pair
    integrates the stiff reference at the literal spec dt = 1e-7."""
    out = []
    for stiff_dt in (1.0e-7, None):
        scens = ([1.0], [1.0 if stiff_dt else 10.0], [8.0e4],
                 [100.0 if stiff_dt else 400.0])
        ode = im.ode_reference_impact(*scens, dt=1.0e-6)[0]
        st = im.stiff_reference_impact(*scens, dt=stiff_dt)[0]
        out.append((ode, st))
    return out


@lru_cache(maxsize=1)
def _soft_pair():
    scens = ([1.0], [1.0], [1.5e4], [400.0])
    ode = im.ode_reference_impact(*scens, dt=1.0e-6)[0]
    st = im.stiff_reference_impact(*scens)[0]
    return ode, st


@lru_cache(maxsize=1)
def _spiked_pair():
    scens = ([2.0], [1.0], [8.0e4], [800.0])
    ode = im.ode_reference_impact(*scens, dt=1.0e-6)[0]
    st = im.stiff_reference_impact(*scens)[0]
    return ode, st


def test_peak_force_monotone_in_impact_speed():
    rows = _mono_rows()
    f = [rows[(v, 1.0, 2.5e4, 400.0)]["F_max"] for v in (0.5, 1.0, 2.0)]
    assert f[0] < f[1] < f[2]


def test_peak_force_monotone_in_stiffness():
    rows = _mono_rows()
    f = [rows[(1.0, 1.0, k, 100.0)]["F_max"]
         for k in (1.5e4, 2.5e4, 8.0e4)]
    assert f[0] < f[1] < f[2]


def test_restitution_strictly_dissipative_all_scenarios():
    for r in _mono_rows().values():
        assert r["separated"]
        assert 0.0 < r["e"] < 1.0
        assert 0.0 < r["energy_loss_frac"] < 1.0


def test_restitution_decreases_with_damping():
    rows = _mono_rows()
    e_lo = rows[(1.0, 1.0, 2.5e4, 100.0)]["e"]
    e_mid = rows[(1.0, 1.0, 2.5e4, 400.0)]["e"]
    e_hi = rows[(1.0, 1.0, 2.5e4, 800.0)]["e"]
    assert e_hi < e_mid < e_lo


def test_no_suction_and_no_lingering_force():
    for r in list(_mono_rows().values()) + \
            [o for o, _ in _stiff_agreement_pairs()]:
        assert r["min_f"] >= -1e-9
        assert 0.0 <= r["tail_f"] < 1e-2


def test_ode_vs_stiff_agreement_for_stiff_contacts():
    for ode, st in _stiff_agreement_pairs():
        assert st["separated"]
        rel = abs(ode["F_max"] - st["F_max"]) / st["F_max"]
        assert rel < 0.15, (ode["m"], ode["b"], rel)
        print(f"k=8e4 (v0={ode['v0']},m={ode['m']},b={ode['b']},"
              f"dt_stiff={st['dt']:g}): F_ode={ode['F_max']:.2f} N "
              f"F_stiff={st['F_max']:.2f} N rel={100*rel:.3f}%")


def test_ode_vs_stiff_soft_contact_documented_degradation():
    ode, st = _soft_pair()
    rel = abs(ode["F_max"] - st["F_max"]) / st["F_max"]
    assert rel < 0.40
    print(f"soft k=1.5e4: F_ode={ode['F_max']:.2f} N "
          f"F_stiff={st['F_max']:.2f} N rel={100*rel:.2f}% (documented)")


def test_heavily_damped_stiff_touch_spike_is_bounded_and_documented():
    ode, st = _spiked_pair()
    rel = abs(ode["F_max"] - st["F_max"]) / st["F_max"]
    assert rel < 1.0
    print(f"v0=2,b=800,k=8e4: F_ode={ode['F_max']:.2f} N "
          f"F_stiff={st['F_max']:.2f} N rel={100*rel:.2f}% "
          f"(Kelvin-Voigt touch spike b*v0={st['F_touch_spike_theory']:.0f} N)")


def test_static_phantom_penetration_matches_support_balance():
    ms, ks, bs = [], [], []
    for m in (1.0, 10.0, 40.0):
        for k in (1.5e4, 2.5e4, 8.0e4):
            for b in (100.0, 400.0, 800.0):
                ms.append(m)
                ks.append(k)
                bs.append(b)
    pens, spec_theory = im.static_equilibrium(ms, ks, bs)
    for j in range(len(ms)):
        pen = float(pens[j])
        bare = ms[j] * im.G / ks[j]
        pred = bare - bs[j] * im.REST_OFF / ks[j]
        assert abs(pen - pred) < 2e-5
        assert pen < bare
        assert spec_theory[j] > pen


def test_dynamic_settle_matches_static_solve():
    static, _ = im.static_equilibrium([10.0], [2.5e4], [400.0])
    dyn = im.dynamic_settle(10.0, 2.5e4, 400.0, t_max=1.2)
    assert abs(dyn["pen_rest"] - float(static[0])) < 2e-6
    assert dyn["pen_rest"] > 0.0


def test_two_point_cop_centered_monotone_no_tension():
    cop = im.two_point_cop([0.0, 0.05, 0.1, 0.2])
    assert abs(cop[0]["x_cop"]) < 1e-6
    xs = [c["x_cop_foot_frame"] for c in cop]
    assert all(xs[i] <= xs[i + 1] + 1e-12 for i in range(len(xs) - 1))
    assert xs[1] > xs[0] and xs[2] > 0.5 * xs[-1]
    for c in cop:
        assert c["sum_abs_err"] < 1e-9
        assert c["min_f"] >= -1e-12
    assert cop[3]["f_raised_end"] >= 0.0
