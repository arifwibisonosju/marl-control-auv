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
OBS_DIM = 12            # was 6; sekarang 3D: [p(3), v(3), rel(3), cur(3)]
ACT_DIM = 3             # was 2; sekarang aksi 3D: [ax, ay, az]
MAX_EPISODES = 1000
MAX_STEPS = 200         # bisa diperpanjang untuk 3D
BATCH_SIZE = 128
BUFFER_SIZE = 200000    # optional: lebih besar
NOISE = 0.1             # tambahan
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
# ---------- Smoothing & band (lebih robust) ----------
def smooth_and_band(y, window=None, method="ema", band="percentile", q=(10, 90), alpha=0.2):
    """
    method: 'ema' (exponential moving average) atau 'ma' (moving average)
    band:   'percentile' (robust) atau 'std' (±1σ)
    q:      persentil bawah-atas untuk band jika band='percentile'
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if window is None:
        window = max(15, n // 50)        # ~2% panjang seri (lebih halus)
    window = max(3, min(window, n))

    # smoothing
    if method == "ema":
        beta = 2 / (window + 1)
        ma = np.empty(n, float)
        ma[0] = y[0]
        for i in range(1, n):
            ma[i] = beta * y[i] + (1 - beta) * ma[i-1]
    else:  # 'ma'
        kernel = np.ones(window) / window
        ma = np.convolve(y, kernel, mode="same")

    # band (robust percentiles default)
    if band == "percentile":
        lo, hi = np.empty(n), np.empty(n)
        half = window // 2
        for i in range(n):
            a = max(0, i - half)
            b = min(n, i + half + 1)
            seg = y[a:b]
            lo[i] = np.percentile(seg, q[0])
            hi[i] = np.percentile(seg, q[1])
    else:  # 'std'
        std = np.empty(n)
        half = window // 2
        for i in range(n):
            a = max(0, i - half)
            b = min(n, i + half + 1)
            std[i] = y[a:b].std()
        lo, hi = ma - std, ma + std

    x = np.arange(1, n + 1)
    return x, ma, lo, hi, alpha

# ---------- Palet opsional ----------
COLORS = {
    "blue":   "#2563EB",  # lebih kontras
    "sky":    "#60A5FA",
    "green":  "#16A34A",
    "orange": "#F59E0B",
}

def plot_with_fill(ax, y, color, label, **kw):
    x, ma, lo, hi, alpha = smooth_and_band(y, **kw)
    ax.fill_between(x, lo, hi, alpha=0.22, color=color, linewidth=0, zorder=1)
    ax.plot(x, ma, lw=3.0, color=color, label=label, zorder=2)
    return ax

# ---------- Template plotting yang “tajam” ----------
plt.rcParams.update({
    "figure.figsize": (12, 7),
    "figure.dpi": 160,
    "axes.titlesize": 20,
    "axes.labelsize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "lines.antialiased": True,
})

# ====== contoh pakai ======
fig, ax = plt.subplots(constrained_layout=True)
plot_with_fill(
    ax,
    reward_history,             # <- data Anda
    COLORS["blue"],
    "MAPPO",
    window=25,                  # lebih halus & stabil
    method="ema",               # smoothing eksponensial
    band="percentile",          # band robust p10–p90
    q=(10, 90),
)

ax.set_title("Training Reward per Episode (MAPPO)")
ax.set_xlabel("Episode")
ax.set_ylabel("Average Reward")
ax.legend(loc="upper right", frameon=True)
ax.margins(x=0.01)

# Simpan high-res
fig.savefig("training_mappo_hd.png", dpi=300, bbox_inches="tight")


