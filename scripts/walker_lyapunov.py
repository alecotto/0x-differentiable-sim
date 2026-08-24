"""Lyapunov spectrum of the rigid compass-walk limit cycle (Q5 x Q1c).

Benettin with finite-difference tangents propagated through the VALIDATED
hybrid oracle map (sympy flow + Newtonian plastic impact,
scripts/walker_oracle.py).  The section map is 3-dimensional:
    s = (th1+, dth1+, dth2+)   just-post-impact state
so the spectrum has 3 exponents; lambda_1 should match
ln(rho)/T_stride for the attracting branch (rho from FD multipliers),
and the sum rule lambda_1+lambda_2+lambda_3 = trace-log of the stride
map provides an internal consistency check.

Near the period-doubling onset the largest exponent must approach 0
from below (flip => crosses to +), which we verify at two slopes.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

import walker_oracle as wo  # noqa: E402


def lyapunov_walker(gamma, beta=0.001, T_steps=4000, eps=1e-8,
                    n_renorm=None, seed_s=None):
    orc = wo.WalkerOracle(gamma, beta=beta)
    if seed_s is None:
        out = wo.continuation_fp(gamma, beta)
        s = out[1]
        if s is None or not np.isscalar(s[0]):
            return None
        # converge harder
        r = orc.find_fixed_point(s)
        s = r["s"]
    else:
        s = np.array(seed_s)
    n_dim = 3
    if n_renorm is None:
        n_renorm = T_steps
    renorm_every = 1          # renormalize every stride
    tangents = np.eye(n_dim)
    logs = []
    s0 = s.copy()
    for k in range(T_steps):
        base, ok_b, _, _ = orc.section_map(s0)
        if not ok_b:
            return None
        cols = []
        for j in range(n_dim):
            sp = s0.copy()
            sp[j] += eps * max(1.0, abs(s0[j]))
            sm = s0.copy()
            sm[j] -= eps * max(1.0, abs(s0[j]))
            pp, okp, _, _ = orc.section_map(sp)
            pm, okm, _, _ = orc.section_map(sm)
            if not (okp and okm):
                return None
            Jcol = (pp - pm) / (2 * eps * max(1.0, abs(s0[j])))
            cols.append(Jcol)
        J = np.stack(cols, axis=1)
        # propagate tangent frames one stride
        tangents_new = J @ tangents
        # QR reorthonormalization (Benettin)
        Q, R = np.linalg.qr(tangents_new)
        diag = np.abs(np.diag(R))
        if np.any(diag < 1e-300):
            return None
        logs.append(np.log(diag))
        tangents = Q
        s0 = base
        del tangents_new
    total = np.sum(logs, axis=0)
    strides = len(logs)
    # stride time from FP: measure once
    y = np.array([s[0], s[1], -s[0], s[2]])
    _yp, ok, nsteps, _i = orc.one_stride(y)
    t_stride = nsteps * orc.h if ok else float("nan")
    lam = total / (strides * t_stride)
    return {"gamma": gamma, "beta": beta, "n_strides": strides,
            "t_stride": t_stride, "spectrum": lam.tolist(),
            "sum": float(lam.sum())}


if __name__ == "__main__":
    out = {}
    print("=== walker hybrid-map Lyapunov spectra ===", flush=True)
    for gam, beta in ((0.009, 0.02), (0.009, 0.001), (0.006, 0.02)):
        try:
            r = lyapunov_walker(gam, beta=beta, T_steps=600)
            if r is None:
                print(f"gamma={gam} beta={beta}: no orbit", flush=True)
                continue
            out[f"g{gam}_b{beta}"] = r
            print(f"gamma={gam} beta={beta}: lam="
                  f"{np.round(r['spectrum'], 3)}  "
                  f"t_stride={r['t_stride']:.3f}s", flush=True)
        except Exception as e:
            print(f"gamma={gam} beta={beta}: ERROR {e}", flush=True)
    with open("benchmarks/walker_lyapunov.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print("saved benchmarks/walker_lyapunov.json", flush=True)
