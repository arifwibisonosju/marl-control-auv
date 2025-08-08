import numpy as np
import copy
import math
import time
import tkinter as tk

# === PARAMETER GLOBAL ===
N_AGENTS = 3
N_NODES = 100
MAP_SIZE = 400
RADIUS = 30
MAX_SPEED = 6.0  # ditingkatkan agar gerakan lebih dinamis
OBS_DIM = 6  # x, y, vx, vy, buffer, distance to node
ACT_DIM = 2  # delta_x, delta_y
BUFFER_MAX = 500
COLLECT_RADIUS = 35
NODE_RADIUS = 5  # ukuran node lebih besar

class MultiUAVEnv:
    def __init__(self, render=True):
        self.n_agents = N_AGENTS
        self.n_nodes = N_NODES
        self.size = MAP_SIZE
        self.radius = RADIUS
        self.max_speed = MAX_SPEED
        self.render_on = render

        self.agent_pos = np.random.uniform(50, 350, size=(self.n_agents, 2))
        self.agent_vel = np.zeros((self.n_agents, 2))
        self.node_pos = np.random.uniform(50, 350, size=(self.n_nodes, 2))
        self.node_buffer = np.random.randint(450, 550, size=(self.n_nodes)).astype(float)
        self.BUFFER_MAX = BUFFER_MAX

        self.max_energy = 100.0
        self.agent_energy = np.ones(self.n_agents) * self.max_energy

        if self.render_on:
            self.window = tk.Tk()
            self.canvas = tk.Canvas(self.window, width=self.size, height=self.size, bg="white")
            self.canvas.pack()
            self.gui_agents = [
                self.canvas.create_oval(0, 0, 0, 0, fill=color)
                for color in ["red", "blue", "green"]
            ]
            import threading
            threading.Thread(target=self.window.mainloop, daemon=True).start()

    def reset(self):
        self.agent_pos = np.random.uniform(50, 350, size=(self.n_agents, 2))
        self.agent_vel = np.zeros((self.n_agents, 2))
        self.node_pos = np.random.uniform(50, 350, size=(self.n_nodes, 2))
        self.node_buffer = np.random.randint(450, 550, size=(self.n_nodes)).astype(float)

        self.agent_energy = []
        for _ in range(self.n_agents):
            if np.random.rand() < 0.3:
                e = np.random.uniform(0.1, 0.2) * self.max_energy
            else:
                e = self.max_energy
            self.agent_energy.append(e)
        self.agent_energy = np.array(self.agent_energy)

        return [self._get_obs(i) for i in range(self.n_agents)]

    def step(self, actions):
        rewards = []
        next_obs = []

        self.node_buffer += np.random.randint(0, 10, size=(self.n_nodes))
        self.node_buffer = np.clip(self.node_buffer, 0, BUFFER_MAX * 2)

        for i in range(self.n_agents):
            dx, dy = actions[i]
            dx = np.clip(dx, -1, 1) * self.max_speed
            dy = np.clip(dy, -1, 1) * self.max_speed
            self.agent_vel[i] = np.array([dx, dy])
            self.agent_pos[i] += self.agent_vel[i]
            self.agent_pos[i] = np.clip(self.agent_pos[i], 0, self.size)

            energy_used = np.linalg.norm(self.agent_vel[i]) * 0.5
            self.agent_energy[i] = max(0.0, self.agent_energy[i] - energy_used)

            reward = 0.0
            for j in range(self.n_nodes):
                dist = np.linalg.norm(self.agent_pos[i] - self.node_pos[j])
                if dist <= COLLECT_RADIUS:
                    if self.node_buffer[j] > BUFFER_MAX:
                        reward += 2.0
                    collected = min(10, self.node_buffer[j])
                    self.node_buffer[j] -= collected
                    reward += collected / 10.0
                    if self.node_buffer[j] < 50:
                        self.agent_energy[i] = min(self.max_energy, self.agent_energy[i] + 5.0)
                        reward += 1.0

            if self.agent_energy[i] < 0.05 * self.max_energy:
                reward -= 10.0

            reward -= 0.01 * np.linalg.norm(self.agent_vel[i])
            rewards.append(reward)
            next_obs.append(self._get_obs(i))

        done = [False for _ in range(self.n_agents)]
        if self.render_on:
            self._render()

        info = {
            'overflow': int(np.sum(self.node_buffer > self.BUFFER_MAX)),
            'FX': int(np.sum(np.any((self.agent_pos < 0) | (self.agent_pos > self.size), axis=1))),
            'energy': float(np.mean(self.agent_energy))
        }

        return next_obs, rewards, done, info

    def _get_obs(self, i):
        closest_dist = 1e6
        closest_buf = 0
        for j in range(self.n_nodes):
            d = np.linalg.norm(self.agent_pos[i] - self.node_pos[j])
            if d < closest_dist:
                closest_dist = d
                closest_buf = min(self.node_buffer[j], BUFFER_MAX) / BUFFER_MAX
        pos = self.agent_pos[i] / self.size
        vel = self.agent_vel[i] / self.max_speed
        dist = np.clip(closest_dist / self.size, 0, 1)
        return np.concatenate([pos, vel, [closest_buf], [dist]])

    def _render(self):
        self.canvas.delete("all")
        for j in range(self.n_nodes):
            x, y = self.node_pos[j]
            color = "black" if self.node_buffer[j] <= BUFFER_MAX else "orange"
            self.canvas.create_oval(
                x - NODE_RADIUS, y - NODE_RADIUS,
                x + NODE_RADIUS, y + NODE_RADIUS,
                fill=color
            )
        for i in range(self.n_agents):
            x, y = self.agent_pos[i]
            self.canvas.create_oval(
                x - self.radius, y - self.radius,
                x + self.radius, y + self.radius,
                fill=["red", "blue", "green"][i % 3]
            )
        self.window.update_idletasks()
        self.window.update()

    def sample_action(self):
        return [np.random.uniform(-1, 1, ACT_DIM) for _ in range(self.n_agents)]

if __name__ == '__main__':
    env = MultiUAVEnv(render=True)
    obs = env.reset()
    for _ in range(200):
        env.step(env.sample_action())
        time.sleep(0.05)
