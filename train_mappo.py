import numpy as np
import torch
from multi_env import MultiUAVEnv
from mappo_agent import MAPPOAgent
import random
from collections import deque
import matplotlib.pyplot as plt
import os

# === Hyperparameters ===
N_AGENTS = 3
OBS_DIM = 6
ACT_DIM = 2
MAX_EPISODES = 1000
MAX_STEPS = 100
BATCH_SIZE = 128
BUFFER_SIZE = 100000
EVAL_INTERVAL = 50

# === Replay Buffer ===
class ReplayBuffer:
    def __init__(self):
        self.buffer = deque(maxlen=BUFFER_SIZE)

    def add(self, obs, act, rew, next_obs):
        self.buffer.append((obs, act, rew, next_obs))

    def sample(self):
        samples = random.sample(self.buffer, BATCH_SIZE)
        obs_n = [[] for _ in range(N_AGENTS)]
        act_n = [[] for _ in range(N_AGENTS)]
        rew_n = [[] for _ in range(N_AGENTS)]
        next_obs_n = [[] for _ in range(N_AGENTS)]

        for obs, act, rew, next_obs in samples:
            for i in range(N_AGENTS):
                obs_n[i].append(obs[i])
                act_n[i].append(act[i])
                rew_n[i].append([rew[i]])
                next_obs_n[i].append(next_obs[i])

        obs_n = [torch.FloatTensor(o) for o in obs_n]
        act_n = [torch.FloatTensor(a) for a in act_n]
        reward_n = [torch.FloatTensor(r) for r in rew_n]
        next_obs_n = [torch.FloatTensor(no) for no in next_obs_n]

        return obs_n, act_n, reward_n, next_obs_n

    def __len__(self):
        return len(self.buffer)

# === Initialize ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = MultiUAVEnv(render=True)
agents = [MAPPOAgent(i, OBS_DIM, ACT_DIM, N_AGENTS, {
    'device': device
}) for i in range(N_AGENTS)]
buffer = ReplayBuffer()

reward_history = []
energy_history = []
overflow_history = []
fx_history = []

# === Training Loop ===
for ep in range(MAX_EPISODES):
    obs_n = env.reset()
    episode_reward = np.zeros(N_AGENTS)

    for step in range(MAX_STEPS):
        act_n = [agents[i].select_action(obs_n[i]) for i in range(N_AGENTS)]
        next_obs_n, rew_n, done_n, info = env.step(act_n)

        buffer.add(obs_n, act_n, rew_n, next_obs_n)
        obs_n = next_obs_n
        episode_reward += np.array(rew_n)

        if len(buffer) >= BATCH_SIZE:
            sample = buffer.sample()
            for agent in agents:
                if hasattr(agent, "update"):
                    agent.update(sample, agents)

    avg_reward = np.mean(episode_reward)
    reward_history.append(avg_reward)

    if (ep + 1) % EVAL_INTERVAL == 0:
        print(f"[Eval] Episode {ep+1} | Avg Reward: {avg_reward:.2f} | Overflow: {info['overflow']} | FX: {info['FX']} | Avg Energy: {info['energy']:.2f}")
        energy_history.append(info['energy'])
        overflow_history.append(info['overflow'])
        fx_history.append(info['FX'])

    print("Episode %d | Rewards: %s" % (ep + 1, [round(r, 2) for r in episode_reward]))

# # === Plot Reward ===
# plt.plot(reward_history)
# plt.xlabel("Episode")
# plt.ylabel("Average Reward")
# plt.title("Training Reward per Episode (MAPPO)")
# plt.grid(True)
# plt.savefig("reward_plot_mappo.png")
# plt.show()

# # === Save Trained Actor Models ===
# os.makedirs("saved_models/mappo", exist_ok=True)

# for i, agent in enumerate(agents):
#     model_path = f"saved_models/mappo/agent{i}_actor.pth"
#     torch.save(agent.actor.state_dict(), model_path)
#     print(f"✔️ Saved Agent {i} actor to {model_path}")

# ==== helper: moving average & std ====
def smooth_and_band(y, window=15):
    y = np.asarray(y, dtype=float)
    if len(y) < window:  # fallback kalau episode masih sedikit
        window = max(3, len(y)//2 or 1)
    ma = np.convolve(y, np.ones(window)/window, mode='valid')
    std = np.array([
        y[max(0, i-window+1):i+1].std()
        for i in range(len(y))
    ])[window-1:]
    x  = np.arange(1, len(y)+1)[window-1:]
    return x, ma, std

# ==== palet warna halus ====
COLORS = {
    "blue":   "#3B82F6",  # biru
    "sky":    "#60A5FA",  # biru langit
    "green":  "#22C55E",  # hijau
    "orange": "#F59E0B",  # oranye
}

def plot_with_fill(ax, y, color, label, window=15):
    x, ma, std = smooth_and_band(y, window)
    # garis halus
    ax.plot(x, ma, lw=2.5, color=color, label=label)
    # band variasi (±1 std)
    ax.fill_between(x, ma-std, ma+std, alpha=0.18, color=color)

# ====== contoh pemakaian ======
# Anda sudah punya reward_history (satu kurva).
# Kalau hanya satu:
fig, ax = plt.subplots(figsize=(8,5))
plot_with_fill(ax, reward_history, COLORS["blue"], "MAPPO", window=15)
ax.set_xlabel("Episode")
ax.set_ylabel("Average Reward")
ax.set_title("Training Reward per Episode (MAPPO)")
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig("reward_plot_mappo.png", dpi=200)
plt.show()




