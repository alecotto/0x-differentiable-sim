"""k_ground decade sweep: is gradient-horizon divergence chaos or spring?

For each k_ground over a decade, measures:
  * Benettin lambda (pair-divergence, fp64)      -- dynamics-level
  * fp32/fp64 divergence rate                    -- precision-level
Verdict rule:
  lambda invariant under k        -> genuine dynamics property
  |lambda| grows ~ sqrt(k)        -> contact spring artifact (design param)
"""
import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
torch.set_num_threads(8)

from diffsim.lyapunov import benettin_lambda, fp32_fp64_divergence   # noqa


def main():
    ks = [1.5e3, 5e3, 1.5e4, 5e4, 1.5e5]
    print(f"{'k':>8} {'lam_benettin(/s)':>18} {'lam_fp32fp64(/s)':>18}")
    rows = []
    for k in ks:
        lam_b, hist = benettin_lambda(k_ground=k, E=16, settle=40, steps=150,
                                      renorm=5, seed=7)
        curve, lam_f = fp32_fp64_divergence(k_ground=k, E=16, settle=40,
                                            steps=100, seed=8)
        rows.append((k, lam_b, lam_f))
        print(f"{k:>8.0e} {lam_b:>18.1f} "
              f"{('%+.1f' % lam_f) if lam_f is not None else 'n/a':>18}",
              flush=True)

    print("\nverdict:")
    lams = [abs(r[1]) for r in rows if r[1] == r[1]]
    spread = max(lams) / max(min(lams), 1e-12)
    print(f"  |lambda| range across decade: {min(lams):.1f} .. {max(lams):.1f} "
          f"(spread x{spread:.2f})")
    if spread < 3:
        print("  INVARIANT under k -> not a contact-spring artifact.")
        if all(r[1] < 0 for r in rows):
            print("  All exponents negative -> standing is asymptotically")
            print("  stable; no chaos in this task. Gait questions move to")
            print("  the walker testbed.")
    else:
        print("  Varies with sqrt(k)-like scaling -> contact spring.")


if __name__ == "__main__":
    main()
