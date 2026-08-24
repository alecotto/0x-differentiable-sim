"""Validation suite for the compass-walker sympy oracle (Q1c).

Anchors:
 1. The acceleration field equals Garcia et al. 1998 eqs (1)-(2) up to
    O(beta) under the verified coordinate map (external anchor).
 2. Flow conserves mechanical energy (tilted gravity is conservative).
 3. Heelstrike impact: angular momentum about the new pivot is EXACTLY
    conserved; kinetic energy strictly decreases (plastic).
 4. A period-one fixed point exists near the mapped Garcia Table-1 gait
    and is contracting (rho < 1), matching theta*/omega* to O(beta).
"""
import importlib.util
import math
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import walker_oracle as wo  # noqa: E402


def _load_garcia():
    spec = importlib.util.spec_from_file_location(
        "garcia_oracle", os.path.join(_ROOT, "scripts", "garcia_oracle.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_garcia_acceleration_anchor():
    err, by_beta, ratios = wo.garcia_limit_check(n=32)
    assert abs(ratios[0] - 0.01) < 0.002, \
        f"error must vanish linearly in beta, ratios={ratios}"
    assert err < 1e-3


def test_flow_energy_conservation():
    orc = wo.WalkerOracle(0.009, beta=0.001)
    y = np.array([0.2, -0.25, -0.19, -0.05])
    E0 = orc.energy(y)
    for _ in range(2000):
        y = orc.flow_step(y)
        assert abs(orc.energy(y) - E0) < 1e-9


def test_impact_momentum_and_dissipation():
    for seed in range(8):
        rng = np.random.default_rng(seed)
        gam = rng.uniform(0.004, 0.015)
        th = rng.uniform(0.05, 0.25)
        orc = wo.WalkerOracle(gam, h=1e-3)
        # integrate until a genuine strike, then check the jump
        y = np.array([th * 0.9, -rng.uniform(0.5, 1.2),
                      th, -rng.uniform(0.5, 1.2)])
        yp, ok, n, info = orc.one_stride(y, max_tau=12.0)
        if not ok:
            continue
        assert info["L_pre"] != 0.0
        assert abs(info["L_post"] - info["L_pre"]) <= 1e-10 * max(
            1.0, abs(info["L_pre"])), "angular momentum must be conserved"
        assert info["ke_post"] <= info["ke_pre"], "plastic impact loses KE"
        return
    pytest.fail("no strike generated from 8 seeds")


def test_period_one_fixed_point_near_garcia():
    go = _load_garcia()
    gam = 0.009
    fpG = go.shoot_fixed_point(gam, go.analytic_seed(gam, "long"))
    assert fpG is not None, "garcia reference FP missing"
    thG, wG = fpG["theta"], fpG["omega"]
    # WALKER_P["beta"] now sits above the direct-seed basin; track the
    # orbit up the beta chain from the published asymptotic seed.
    orc, s3 = wo.continuation_fp(gam, wo.P0["beta"])
    assert s3 is not None and np.isscalar(s3[0]), \
        "oracle-B period-one FP not found via beta continuation"
    r = orc.find_fixed_point(s3)
    assert r is not None
    assert r["resid"] < 1e-9
    assert r["rho"] < 1.0, "gait must be attracting at gamma=0.009"
    wr = math.sqrt(9.81 / wo.P0["l"])
    # theta* and stance rate match the published asymptotics to O(beta)
    assert abs(r["s"][0] - (-thG)) < 2e-2
    assert abs(r["s"][1] - (-wG * wr)) < 5e-2


def test_tilted_gravity_equilibrium():
    """Slope equivalence sanity: with gravity tilted by gamma the whole
    machine standing at th1 = th2 = -gamma (both legs aligned with the
    apparent gravity) is an equilibrium of the pinned-stance flow."""
    for gam in (0.004, 0.009, 0.015):
        orc = wo.WalkerOracle(gam)
        y = np.array([-gam, 0.0, -gam, 0.0])
        a = orc.accel(y)
        assert np.max(np.abs(a)) < 1e-9
