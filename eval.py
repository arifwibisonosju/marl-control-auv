# eval.py
import argparse
import time
from pathlib import Path
import numpy as np
import torch

from multi_env import MultiUAVEnv, N_AGENTS, OBS_DIM, ACT_DIM

# =======================
# Model definitions
# =======================
class ActorNet(torch.nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, act_dim), torch.nn.Tanh()
        )
    def forward(self, x):
        return self.net(x)

class MultiAgentPolicy:
    """
    Memuat actor per agen dari folder model_path:
      agent0_actor.pth, agent1_actor.pth, agent2_actor.pth
    """
    def __init__(self, model_path: str, n_agents: int, obs_dim: int, act_dim: int, device="cpu"):
        self.device = torch.device(device)
        self.actors = []
        base = Path(model_path)
        for i in range(n_agents):
            actor = ActorNet(obs_dim, act_dim).to(self.device)
            ckpt = base / f"agent{i}_actor.pth"
            if not ckpt.exists():
                raise FileNotFoundError(f"Model tidak ditemukan: {ckpt}")
            state = torch.load(ckpt, map_location=self.device)
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            actor.load_state_dict(state)
            actor.eval()
            self.actors.append(actor)

    @torch.no_grad()
    def act(self, obs_list):
        actions = []
        for i, o in enumerate(obs_list):
            o_t = torch.as_tensor(o, dtype=torch.float32, device=self.device).unsqueeze(0)
            a = self.actors[i](o_t).squeeze(0).cpu().numpy()
            actions.append(a)
        return actions

# =======================
# Evaluator
# =======================
def evaluate(algo: str, model_path: str, episodes: int, render: bool, device: str):
    env = MultiUAVEnv(render=render)

    # inisialisasi pertama untuk cek dimensi
    obs_list = env.reset()
    assert len(obs_list) == N_AGENTS, "Jumlah obs tidak cocok dengan N_AGENTS"
    for o in obs_list:
        assert o.shape[-1] == OBS_DIM, f"OBS_DIM env ({OBS_DIM}) != obs.shape ({o.shape})"

    # load policy (MAPPO & MADDPG sama-sama per-agent di setup ini)
    policy = MultiAgentPolicy(model_path, N_AGENTS, OBS_DIM, ACT_DIM, device=device)

    all_eps = []
    t0 = time.time()

    for ep in range(1, episodes + 1):
        obs_list = env.reset()
        ep_ret = 0.0
        ep_overflow_sum = 0.0
        ep_fx_sum = 0.0
        ep_energy_sum = 0.0
        steps = 0

        while True:
            actions = policy.act(obs_list)
            next_obs, rewards, done, info = env.step(actions)

            ep_ret += float(np.mean(rewards))
            ep_overflow_sum += float(info.get("overflow", 0.0))
            ep_fx_sum += float(info.get("FX", 0.0))
            ep_energy_sum += float(info.get("energy", 0.0))
            steps += 1
            obs_list = next_obs

            if steps >= 500:  # batas langkah evaluasi
                break

        avg_overflow = ep_overflow_sum / steps
        avg_fx = ep_fx_sum / steps
        avg_energy = ep_energy_sum / steps

        all_eps.append((ep_ret, avg_overflow, avg_fx, avg_energy))
        print(f"[{algo.upper()}] Ep {ep}/{episodes} | Return={ep_ret:.2f} "
              f"| Overflow(avg)={avg_overflow:.2f} | FX(avg)={avg_fx:.2f} | Energy(avg)={avg_energy:.2f}")

    dur = time.time() - t0
    avg = np.mean(np.array(all_eps), axis=0)
    print("\n=== EVAL SUMMARY ===")
    print(f"Return={avg[0]:.3f} | Overflow(avg)={avg[1]:.3f} | FX(avg)={avg[2]:.3f} | Energy(avg)={avg[3]:.3f}")
    print(f"Total time={dur:.2f}s | per-ep={dur/max(1,episodes):.2f}s")

    # simpan CSV ke folder model_path
    out_dir = Path(model_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"eval_{algo}_{int(time.time())}.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("episode,return,overflow_avg,fx_avg,energy_avg\n")
        for i, (ret, ov, fx, en) in enumerate(all_eps, 1):
            f.write(f"{i},{ret},{ov},{fx},{en}\n")
    print(f"Saved CSV: {csv_path}")
    
# =======================
# CLI
# =======================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", type=str, required=True, choices=["maddpg", "mappo"])
    ap.add_argument("--model_path", type=str, required=True,
                    help="Path folder model: saved_models/maddpg atau saved_models/mappo")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    evaluate(args.algo, args.model_path, args.episodes, args.render, device)
