import os
"""Train push-recovery standing on the SOMA-class humanoid with SHAC-lite.

Short-horizon BPTT (H steps, measured clean-gradient window) + value
bootstrap beyond.  Initial states include random pushes so the policy must
learn active recovery beyond the fixed PD baseline.
"""
import sys
import time

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from diffsim.humanoid import make_soma_humanoid, initial_pose   # noqa: E402
from diffsim import build_geoms_compat                           # noqa: E402
from diffsim.sim import DiffSim, SimConfig, ContactConfig        # noqa: E402
from diffsim.algo.shac import ActorCritic, ShacTrainer           # noqa: E402

DT = torch.float64

# ---- configuration ----------------------------------------------------
CFG = dict(
    E=32, H=16, iters=6,            # smoke settings (overridden below)
    K=48,                            # value-rollout horizon
    kp=80.0, kd=10.0,
    fall_z=0.45, fall_up=-0.2,
    eval_steps=120,
)


def build():
    model, gspec, feet = make_soma_humanoid()
    cc = ContactConfig(k_ground=1.5e4, k_pair=8e3, damping=200.0)
    sim = DiffSim(model, build_geoms_compat(gspec),
                  SimConfig(dt=5e-4, n_substeps=8, contact=cc), dtype=DT)
    return model, sim


def sample_init(model, E, scale=1.0):
    """Standing pose with limit-respecting noise + random lateral pushes.

    `scale` is the curriculum multiplier (0..1): all perturbation
    magnitudes grow with policy competence.
    """
    q, w = initial_pose(model, E, dtype=DT)
    s = max(min(scale, 1.0), 0.0)
    noise = 0.06 * s * torch.randn(E, 15, dtype=DT)
    jp = (q[:, 7:] + noise).clamp(-0.3, 0.3)
    knee_pos = [7, 12]                                    # within q[:,7:]
    for k in knee_pos:
        jp[:, k] = jp[:, k].clamp(max=0.04)
    q[:, 7:] = jp
    q[:, mh_qz(model)] += s * ((-0.03) + 0.04 * torch.rand(E, dtype=DT))
    w[:, :3] += 0.3 * s * torch.randn(E, 3, dtype=DT)      # base ang vel
    w[:, 6:] += 0.3 * s * torch.randn(E, 15, dtype=DT)     # joint vel
    push = 0.4 * s * torch.randn(E, 2, dtype=DT)
    w[:, 3:5] += push                                      # x/y linear push
    return q, w


def mh_qz(model):
    return model.q_free_start + 6


def obs_of(R_w, q, w):
    up_z = R_w[:, 1][:, 2, 2].unsqueeze(-1)
    return torch.cat([up_z, w[:, 3:6], q[:, 7:], w[:, 6:]], dim=-1)


def reward_of(r):
    R_torso = r.R_w[:, 1]
    upright = R_torso[:, 2, 2].clamp(-1., 1.)
    com_err = (r.com_z - 0.8557).clamp(-0.5, 0.5)
    fell = (r.com_z < CFG["fall_z"]) | (upright < CFG["fall_up"])
    rw = 2.0 * upright + 0.25 - 0.002 * (r.qd ** 2).sum(-1) \
        - 0.05 * com_err ** 2 - 2.0 * fell.to(r.q.dtype)
    return rw, fell


@torch.no_grad()
def evaluate(ac, sim, model, E=32, seed=999, scale=1.0):
    g = torch.get_rng_state()
    torch.manual_seed(seed)
    q, w = sample_init(model, E, scale)
    torch.set_rng_state(g)
    total = torch.zeros(E, dtype=DT)
    disc = torch.ones(E, dtype=DT)
    alive = torch.ones(E, dtype=torch.bool)
    for _ in range(CFG["eval_steps"]):
        R_w, _ = sim.art.kinematics(q)
        a = ac.act(obs_of(R_w, q, w)).clamp(-0.5, 0.5)
        tau = sim.pd_torques(q, w, torch.zeros(E, 15, dtype=DT) + a,
                             kp=CFG["kp"], kd=CFG["kd"])
        r = sim.step(q, w, tau_ext=tau)
        q, w = r.q, r.qd
        rw, fell = reward_of(r)
        total += disc * torch.where(fell, -2.0 * torch.ones_like(rw), rw)
        disc *= 0.995
        alive &= ~fell.cpu()
    return float(total.mean()), float(alive.float().mean())


def main(smoke=False):
    cfg = dict(CFG)
    if not smoke:
        cfg.update(E=64, H=32, iters=60, K=96)

    model, sim = build()
    nv, nq = model.v_dim, model.q_dim
    obs_dim = 1 + 3 + 15 + 15
    ac = ActorCritic(obs_dim, 15, hidden=128)
    trainer = ShacTrainer(ac, gamma=0.995, lam=0.95)

    start_iter = 0
    import os
    ckpt_path = os.path.join(_ROOT, "models", "shac_balance.pt")
    if "--resume" in sys.argv and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        ac.load_state_dict(ck["ac"])
        trainer.opt_actor.load_state_dict(ck["opt_actor"])
        trainer.opt_critic.load_state_dict(ck["opt_critic"])
        start_iter = ck["iter"] + 1
        print(f"resumed from iter {start_iter}")

    print(f"== SHAC-lite balance | {'SMOKE' if smoke else 'FULL'} | "
          f"E={cfg['E']} H={cfg['H']} iters={cfg['iters']} ==")
    ev_ret, ev_alive = evaluate(ac, sim, model)
    print(f"iter  -1  eval_return={ev_ret:9.3f} survival={ev_alive:.2f}")

    for it in range(start_iter, cfg["iters"]):
        t0 = time.time()
        cur = min(1.0, 0.15 + 0.05 * it)                   # perturbation curriculum

        # ---- short-horizon differentiable rollout --------------------
        q, w = sample_init(model, cfg["E"], cur)
        rewards = []
        fall_count = 0
        for _ in range(cfg["H"]):
            R_w, _ = sim.art.kinematics(q)
            a = (ac.act(obs_of(R_w, q, w))
                 + trainer.explore_sigma * torch.randn_like(a)).clamp(-0.5, 0.5)
            tau = sim.pd_torques(q, w, a.clone(), kp=CFG["kp"], kd=CFG["kd"])
            r = sim.step(q, w, tau_ext=tau, train_mode=True)
            q, w = r.q, r.qd
            rw, fell = reward_of(r)
            fall_count += float(fell.float().mean())
            rewards.append(rw)
        rewards = torch.stack(rewards, dim=1)              # [E,H]
        R_w, _ = sim.art.kinematics(q)
        v_boot = ac.value(obs_of(R_w, q, w)).detach()

        j_mean, gnorm = trainer.policy_step(rewards, v_boot)
        fall_rate = fall_count / (cfg["E"] * cfg["H"])

        # ---- value data (no grad) -------------------------------------
        with torch.no_grad():
            q, w = sample_init(model, cfg["E"])
            obs_list, rew_list, fall_list = [], [], []
            for _ in range(cfg["K"]):
                R_w, _ = sim.art.kinematics(q)
                o = obs_of(R_w, q, w)
                a = ac.act(o).clamp(-0.5, 0.5)
                tau = sim.pd_torques(q, w, a, kp=CFG["kp"], kd=CFG["kd"])
                r = sim.step(q, w, tau_ext=tau)
                q, w = r.q, r.qd
                rw, fell = reward_of(r)
                obs_list.append(o)
                rew_list.append(rw)
                fall_list.append(fell)
            obs_seq = torch.stack(obs_list, dim=1)          # [E,K,obs]
            rew_seq = torch.stack(rew_list, dim=1)
            fell_seq = torch.stack(fall_list, dim=1)
            v_next = ac.value(obs_seq.reshape(-1, obs_dim)).reshape(cfg["E"], cfg["K"])
            G = trainer.td_lambda_targets(rew_seq, v_next, fell_seq)

            vl = 0.0
            for ep in range(2):
                perm = torch.randperm(cfg["E"] * cfg["K"])
                o_flat = obs_seq.reshape(-1, obs_dim)[perm]
                g_flat = G.reshape(-1)[perm]
                for s in range(0, len(o_flat), 2048):
                    vl = trainer.value_step(o_flat[s:s + 2048],
                                            g_flat[s:s + 2048])

        msg = f"iter {it:3d}  J={j_mean:9.3f} |g|={gnorm:9.4f} clip={int(trainer.last_clip_hit)} vloss={vl:9.4f} " \
              f"fall={fall_rate:.2f} cur={cur:.2f} ({time.time()-t0:.1f}s)"
        if (it + 1) % 10 == 0 or it == cfg["iters"] - 1 or smoke:
            ev_ret, ev_alive = evaluate(ac, sim, model, scale=cur)
            msg += f"  eval={ev_ret:9.3f} surv={ev_alive:.2f}"
        print(msg, flush=True)

        import os
        torch.save({"ac": ac.state_dict(),
                    "opt_actor": trainer.opt_actor.state_dict(),
                    "opt_critic": trainer.opt_critic.state_dict(),
                    "iter": it},
                   os.path.join(_ROOT, "models", "shac_balance.pt"))


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
