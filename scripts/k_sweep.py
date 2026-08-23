import os, sys, torch
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
torch.set_num_threads(8)

from diffsim.lyapunov import make_stand_sim, standing_step_fn_factory, benettin_generic, DT

def main():
    ks = [2.5e3, 2.5e4, 2.5e5]
    print(f"{'k_ground':>10} {'lambda(/s)':>12}")
    for k in ks:
        model, sim = make_stand_sim(k_ground=k)
        dt_per_call = sim.cfg.dt * sim.cfg.n_substeps
        step = standing_step_fn_factory(sim, dt_per_call)
        nq = sim.art.nq
        q = torch.zeros(1, nq, dtype=DT)
        q[0, model.q_free_start] = 1.0
        q[0, model.q_free_start + 6] = 0.883
        w = torch.zeros(1, model.v_dim, dtype=DT)
        x1 = torch.cat([q.reshape(-1), w.reshape(-1)])
        kick = torch.zeros_like(x1); kick[nq] = 1e-9   # velocity kick on torso
        x2 = x1 + kick
        lam, _ = benettin_generic(step, x1, x2,
                                  dt_substep=dt_per_call,
                                  steps=100, renorm=25, delta0=1e-9)
        print(f"{k:>10.0e} {lam:>12.1f}", flush=True)

if __name__ == "__main__":
    main()
