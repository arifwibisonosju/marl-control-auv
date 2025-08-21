import argparse
import time
from pathlib import Path
import numpy as np
import torch

from multi_env import MultiUAVEnv  # <<< hanya env, tanpa konstanta

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

# <<< helper untuk adaptasi 12D -> 6D (model lama 2D)
def obs12_to_obs6(o_np: np.ndarray) -> np.ndarray:  # <<<
    """
    Ambil komponen 2D dari observasi 12D ter-normalisasi:
    [x,y,z, vx,vy,vz, dx,dy,dz, cx,cy,cz] -> [x,y, vx,vy, dx,dy]
    Jika panjang tidak 12, fallback ambil 6 elemen pertama.
    """
    if o_np.shape[-1] >= 12:
        return np.asarray([o_np[0], o_np[1], o_np[3], o_np[4], o_np[6], o_np[7]], dtype=np.float32)
    return np.asarray(o_np[:6], dtype=np.float32)

class MultiAgentPolicy:
    """
    Memuat actor per agen dari folder model_path:
      agent0_actor.pth, agent1_actor.pth, agent2_actor.pth

    Mendukung dua mode:
      - Native 3D: obs_dim=12, act_dim=3
      - Kompat 2D: memuat bobot (6->2), map obs12->obs6 dan aksi2->aksi_env
    """
    def __init__(self, model_path: str, n_agents: int, obs_dim: int, act_dim_env: int, device="cpu"):
        self.device = torch.device(device)
        self.n_agents = n_agents
        self.act_dim_env = act_dim_env  # <<< simpan dim aksi env
        self.used_2d_compat = False     # <<< flag jika fallback 2D dipakai

        self.actor_fns = []  # <<< simpan callable per-agen: obs(np)->action(np)

        base = Path(model_path)
        for i in range(n_agents):
            ckpt = base / f"agent{i}_actor.pth"
            if not ckpt.exists():
                raise FileNotFoundError(f"Model tidak ditemukan: {ckpt}")

            # --- coba load sebagai model 3D (obs_dim, act_dim_env) ---
            native_actor = ActorNet(obs_dim, act_dim_env).to(self.device)  # <<<
            state = torch.load(ckpt, map_location=self.device)
            if isinstance(state, dict) and "model" in state:
                state = state["model"]

            try:
                native_actor.load_state_dict(state)  # coba native
                native_actor.eval()
                # buat fn yang langsung pakai obs 3D
                def make_native_fn(actor):
                    @torch.no_grad()
                    def _fn(o_np: np.ndarray):
                        o_t = torch.as_tensor(o_np, dtype=torch.float32, device=self.device).unsqueeze(0)
                        a = actor(o_t).squeeze(0).cpu().numpy()
                        return a
                    return _fn
                self.actor_fns.append(make_native_fn(native_actor))  # <<<
            except RuntimeError:
                # --- fallback: model lama 2D (6 -> 2) ---
                self.used_2d_compat = True  # <<<
                compat_actor = ActorNet(6, 2).to(self.device)  # <<<
                compat_actor.load_state_dict(state)
                compat_actor.eval()

                # fn yang adaptasikan obs12->obs6 dan aksi2->aksi_env
                def make_compat_fn(actor, act_dim_env):
                    @torch.no_grad()
                    def _fn(o_np: np.ndarray):
                        o6 = obs12_to_obs6(np.asarray(o_np, dtype=np.float32))
                        o_t = torch.as_tensor(o6, dtype=torch.float32, device=self.device).unsqueeze(0)
                        a2 = actor(o_t).squeeze(0).cpu().numpy()
                        # map ke dim aksi env
                        if act_dim_env <= 2:
                            return a2[:act_dim_env]
                        a_env = np.zeros(act_dim_env, dtype=np.float32)
                        a_env[:2] = a2
                        # komponen z = 0 (dikunci)
                        return a_env
                    return _fn
                self.actor_fns.append(make_compat_fn(compat_actor, self.act_dim_env))  # <<<

    @torch.no_grad()
    def act(self, obs_list):
        actions = []
        for i, o in enumerate(obs_list):
            a = self.actor_fns[i](np.asarray(o, dtype=np.float32))  # <<<
            actions.append(a)
        return actions

# =======================
# Evaluator
# =======================
def evaluate(algo: str, model_path: str, episodes: int, render: bool, device: str):
    # --- Buat env untuk evaluasi; coba opsi yang stabil saat eval ---  # <<<
    try:
        env = MultiUAVEnv(render=render, static_nodes=True, curriculum=False)  # <<<
    except TypeError:
        env = MultiUAVEnv(render=render)  # fallback signature lama               # <<<

    # --- Deteksi dimensi dari env (tanpa impor konstanta) ------------- # <<<
    obs_list = env.reset()
    N_AGENTS = len(obs_list)
    OBS_DIM  = int(np.asarray(obs_list[0]).shape[-1])
    ACT_DIM  = 2 if getattr(env, "lock_z", False) else 3
    print(f"[EVAL] Detected dims -> n_agents={N_AGENTS}, obs_dim={OBS_DIM}, act_dim={ACT_DIM}")

    # load policy (dengan fallback 2D jika perlu)
    policy = MultiAgentPolicy(model_path, N_AGENTS, OBS_DIM, ACT_DIM, device=device)  # <<<

    # Jika memakai model 2D, kunci sumbu Z agar konsisten
    if getattr(policy, "used_2d_compat", False):  # <<<
        if hasattr(env, "lock_z"):
            env.lock_z = True
            print("[EVAL] 2D checkpoint terdeteksi -> lock_z diaktifkan & aksi Z=0.")  # <<<
        else:
            print("[EVAL] 2D checkpoint terdeteksi; aksi Z akan dipaksa 0 di sisi policy.")  # <<<

        # Jika ACT_DIM awal 3 tapi Z dikunci, tidak masalah: policy sudah padding 0.  # <<<

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
