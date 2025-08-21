import numpy as np
import time

# --- Render (ubah backend jika perlu) ---
import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


class MultiUAVEnv:
    """
    3D continuous env (API kompatibel):
      reset() -> list obs_n (len = n_agents)
      step(action_n) -> (next_obs_n, rew_n, done_n, info)

    Observation (per agent): [x,y,z, vx,vy,vz, dx,dy,dz, cx,cy,cz]  (dim=12), ter-NORMALISASI
      - rel (dx,dy,dz) = posisi relatif ke node terdekat
      - koordinat & kecepatan diskalakan ke [-1,1] berbasis domain & vmax
    Action (per agent): [ax, ay, az] in [-1,1] -> diskalakan oleh a_scale
    """

    def __init__(self,
                 n_agents=3, dt=0.2,
                 domain=(1000, 1000, 300),
                 vmax=3.0, seed=0,
                 render=False, render_every=2,
                 n_nodes=100, node_capacity=1000.0,
                 blink_every=3, node_size=10, node_alpha=0.6, auv_size=30,
                 # ---- fitur baru ----
                 a_scale=1.0,                  # skala aksi (<=1 disarankan)
                 lock_z=False,                 # kunci sumbu Z (latihan awal 2.5D)
                 curriculum=False,             # aktifkan pembesaran domain bertahap
                 curri_k=0.0005,               # laju pertumbuhan domain
                 start_domain=(200, 200, 80),  # domain awal saat curriculum
                 bonus_radius=30.0,            # radius bonus kedekatan node
                 static_nodes=True             # >>> node tetap antar-episode
                 ):
        self.rng = np.random.default_rng(seed)
        self.n = int(n_agents)
        self.dt = float(dt)

        self._domain_full = np.array(domain, dtype=np.float32)
        self._domain_cur = np.array(start_domain if curriculum else domain, dtype=np.float32)
        self.curriculum = bool(curriculum)
        self.curri_k = float(curri_k)

        self.vmax = float(vmax)
        self.a_scale = float(a_scale)
        self.lock_z = bool(lock_z)

        # ---- Nodes (z <= 100) ----
        self.n_nodes = int(n_nodes)
        self.node_capacity = float(node_capacity)
        self.static_nodes = bool(static_nodes)

        # generate nodes SEKALI berdasarkan domain penuh (tetap antar-episode)
        self.nodes = self.rng.uniform(
            [0, 0, 0],
            [self._domain_full[0], self._domain_full[1], min(100.0, self._domain_full[2])],
            size=(self.n_nodes, 3)
        ).astype(np.float32)

        # buffer & flags
        self.node_buf = np.zeros(self.n_nodes, dtype=np.float32)
        self.node_overflow_flags = np.zeros(self.n_nodes, dtype=np.float32)

        # ---- Render configs ----
        self.render = render
        self.render_every = int(max(1, render_every))
        self._fig = None; self._ax = None
        self._scat = None; self._quiver = None; self._nodes_artist = None
        self._last_render_time = 0; self._target_fps = 30

        # style
        self._node_size = int(node_size)
        self._node_alpha = float(node_alpha)
        self._auv_size = int(auv_size)
        self._blink_every = int(max(1, blink_every))

        # hyper kecil lain
        self.bonus_radius = float(bonus_radius)

        # cache normalisasi
        self._eps = 1e-8

        self.reset()
        if self.render:
            self._setup_render()

    # =================== Util & Dinamika ===================
    @property
    def Xmax(self): return float(self._domain_cur[0])
    @property
    def Ymax(self): return float(self._domain_cur[1])
    @property
    def Zmax(self): return float(self._domain_cur[2])

    def _grow_domain(self):
        if not self.curriculum:
            return
        # tumbuhkan domain menuju domain penuh secara eksponensial halus
        self._domain_cur = self._domain_full - (self._domain_full - self._domain_cur) * np.exp(-self.curri_k * self.t)
        if self.lock_z:
            self._domain_cur[2] = max(80.0, self._domain_cur[2])

    def _current_field(self, p, t):
        x, y, z = p
        vx = 0.3*np.sin(2*np.pi*y/500 + 0.01*t)
        vy = 0.3*np.cos(2*np.pi*x/500 + 0.01*t)
        vz = 0.05*np.sin(2*np.pi*z/200 + 0.02*t)
        if self.lock_z:
            vz = 0.0
        return np.array([vx, vy, vz], dtype=np.float32)

    def _nearest_node_rel(self):
        diffs = self.nodes[None, :, :] - self.p[:, None, :]   # (n_agents, n_nodes, 3)
        dists = np.linalg.norm(diffs, axis=2)                 # (n_agents, n_nodes)
        idx = np.argmin(dists, axis=1)                        # (n_agents,)
        rel = diffs[np.arange(self.n), idx, :]                # (n_agents, 3)
        return rel, idx, dists

    # ---------- Normalisasi ----------
    def _norm_pos(self, P):
        s = np.array([self.Xmax, self.Ymax, self.Zmax], dtype=np.float32)
        return (2.0 * (P / (s + self._eps)) - 1.0).astype(np.float32)

    def _norm_vel(self, V):
        return np.clip(V / (self.vmax + self._eps), -1.0, 1.0).astype(np.float32)

    def _norm_rel(self, R):
        d = np.linalg.norm([self.Xmax, self.Ymax, self.Zmax])
        return np.clip(R / (d + self._eps), -1.0, 1.0).astype(np.float32)

    def _obs(self):
        rel, _, _ = self._nearest_node_rel()
        cur = np.array([self._current_field(pi, self.t) for pi in self.p], dtype=np.float32)
        # normalisasi
        p_n = self._norm_pos(self.p)
        v_n = self._norm_vel(self.v)
        rel_n = self._norm_rel(rel)
        cur_n = self._norm_vel(cur)
        obs_n = np.concatenate([p_n, v_n, rel_n, cur_n], axis=1).astype(np.float32)
        return [obs_n[i] for i in range(self.n)]

    # =================== Reset & Step ===================
    def reset(self):
        self.t = 0

        # Hanya acak ulang node saat static_nodes=False
        if not self.static_nodes:
            self.nodes = self.rng.uniform(
                [0, 0, 0],
                [self.Xmax, self.Ymax, min(100.0, self.Zmax)],
                size=(self.n_nodes, 3)
            ).astype(np.float32)

        # posisi awal agent (setengah atas kedalaman)
        self.p = self.rng.uniform([0, 0, 0],
                                  [self.Xmax, self.Ymax, self.Zmax/2],
                                  size=(self.n, 3)).astype(np.float32)
        if self.lock_z:
            self.p[:, 2] = self.Zmax * 0.3

        self.v = np.zeros((self.n, 3), dtype=np.float32)
        self.buff = np.zeros(self.n, dtype=np.float32)

        self.node_buf[:] = 0.0
        self.node_overflow_flags[:] = 0.0

        self.episode_energy = 0.0
        self.episode_fx = 0.0
        self.episode_steps = 0

        # inisialisasi jarak sebelumnya untuk sinyal 'progress'
        rel0, idx_nn0, dists0 = self._nearest_node_rel()
        self._prev_min_dists = dists0[np.arange(self.n), idx_nn0].copy().astype(np.float32)

        self.traj = [[self.p[i].copy()] for i in range(self.n)]
        return self._obs()

    def step(self, action_n):
        a = np.array(action_n, dtype=np.float32)
        a = np.clip(a, -1.0, 1.0) * self.a_scale
        if self.lock_z:
            a[:, 2] = 0.0

        cur = np.array([self._current_field(pi, self.t) for pi in self.p], dtype=np.float32)

        v_prev = self.v.copy()
        self.v = np.clip(self.v + a + cur, -self.vmax, self.vmax)
        self.p = self.p + self.dt * self.v

        # FX: keluar domain
        fx = ((self.p[:, 0] < 0) | (self.p[:, 0] > self.Xmax) |
              (self.p[:, 1] < 0) | (self.p[:, 1] > self.Ymax) |
              (self.p[:, 2] < 0) | (self.p[:, 2] > self.Zmax)).astype(np.float32)
        self.p[:, 0] = np.clip(self.p[:, 0], 0, self.Xmax)
        self.p[:, 1] = np.clip(self.p[:, 1], 0, self.Ymax)
        self.p[:, 2] = np.clip(self.p[:, 2], 0, self.Zmax)

        # data gain dari node terdekat (maks kedekatan)
        rel, idx_nn, dists = self._nearest_node_rel()
        min_dists = dists[np.arange(self.n), idx_nn]
        data_gain = np.exp(-min_dists / 200.0).astype(np.float32)

        # ----- REWARD SHAPING -----
        dom_diag = np.linalg.norm([self.Xmax, self.Ymax, self.Zmax]) + self._eps
        dist_norm = (min_dists / dom_diag)
        r_dist = -dist_norm

        progress = (self._prev_min_dists - min_dists) / (dom_diag + self._eps)
        self._prev_min_dists = min_dists.copy()

        r_bonus = (min_dists < self.bonus_radius).astype(np.float32) * 0.5

        # buffer agent
        self.buff += data_gain

        # tambahkan traffic ke node terdekat
        node_load = np.zeros(self.n_nodes, dtype=np.float32)
        for i in range(self.n):
            node_load[idx_nn[i]] += data_gain[i]
        self.node_buf += node_load

        # overflow node
        node_overflow_flags = (self.node_buf > self.node_capacity).astype(np.float32)
        node_overflow_rate = float(node_overflow_flags.mean())
        node_overflow_cnt = int(node_overflow_flags.sum())
        self.node_overflow_flags = node_overflow_flags

        # penalti utk agent yang melayani node overflow
        node_is_over = node_overflow_flags[idx_nn]
        overflow_pen_agent = 1.0 * node_is_over

        # energi (biaya gerak) + jerk penalti agar smooth
        move_cost = np.sum(a**2, axis=1)
        jerk = np.sum((self.v - v_prev)**2, axis=1)
        energy = (0.01*move_cost + 0.002*jerk).astype(np.float32)

        # ---- komposisi reward (tweakable) ----
        w_data, w_prog, w_dist = 1.0, 0.6, 0.4
        w_energy, w_fx, w_over = 0.5, 2.0, 0.5
        reward = ( w_data*data_gain
                   + w_prog*progress
                   + w_dist*r_dist
                   + r_bonus
                   - w_energy*energy
                   - w_fx*fx
                   - w_over*overflow_pen_agent ).astype(np.float32)

        # waktu & akumulasi
        self.t += 1
        self.episode_steps += 1
        self.episode_energy += float(np.mean(energy))
        self.episode_fx += float(np.mean(fx))

        # simpan traj
        for i in range(self.n):
            self.traj[i].append(self.p[i].copy())

        # curriculum: perbesar domain perlahan
        self._grow_domain()

        # render
        if self.render and (self.t % self.render_every == 0):
            self._render(cur)

        obs_n = self._obs()
        rew_n = [float(r) for r in reward]
        done_n = [False]*self.n
        info = {
            "overflow": node_overflow_rate,
            "overflow_count": node_overflow_cnt,
            "FX": float(np.mean(fx)),
            "energy": float(np.mean(energy)),
            "avg_dist": float(np.mean(min_dists))
        }
        return obs_n, rew_n, done_n, info

    # =================== Rendering ===================

    def _visible_nodes_mask(self):
        """Mask node yang berada dalam domain aktif saat ini (untuk curriculum)."""
        return ((self.nodes[:, 0] <= self.Xmax) &
                (self.nodes[:, 1] <= self.Ymax) &
                (self.nodes[:, 2] <= self.Zmax))

    def _setup_render(self):
        plt.ion()
        self._fig = plt.figure(figsize=(7, 6))
        self._ax = self._fig.add_subplot(111, projection='3d')
        self._ax.set_xlim(0, self.Xmax); self._ax.set_ylim(0, self.Ymax); self._ax.set_zlim(0, self.Zmax)
        self._ax.set_xlabel('X'); self._ax.set_ylabel('Y'); self._ax.set_zlabel('Z (depth)')
        self._ax.set_title('AUV3D-Env (live)')

        self._scat = self._ax.scatter(self.p[:,0], self.p[:,1], self.p[:,2],
                                      s=self._auv_size, c='blue', label='AUV')

        mask = self._visible_nodes_mask()
        nodes_to_draw = self.nodes[mask]
        self._nodes_artist = self._ax.scatter(nodes_to_draw[:,0], nodes_to_draw[:,1], nodes_to_draw[:,2],
                                              s=self._node_size, marker='o', c='red', alpha=self._node_alpha, label='Node')

        cur = np.array([self._current_field(pi, self.t) for pi in self.p], dtype=np.float32)
        self._quiver = self._ax.quiver(self.p[:,0], self.p[:,1], self.p[:,2],
                                       cur[:,0], cur[:,1], cur[:,2], length=10.0, linewidth=1.0)

        self._fig.canvas.draw(); self._fig.canvas.flush_events()

    def _render(self, cur):
        now = time.time()
        if now - self._last_render_time < (1.0 / self._target_fps):
            return
        self._last_render_time = now

        # update batas sumbu jika curriculum berubah
        self._ax.set_xlim(0, self.Xmax); self._ax.set_ylim(0, self.Ymax); self._ax.set_zlim(0, self.Zmax)

        # update AUV
        self._scat._offsets3d = (self.p[:,0], self.p[:,1], self.p[:,2])

        # update arus
        try: self._quiver.remove()
        except Exception: pass
        self._quiver = self._ax.quiver(self.p[:,0], self.p[:,1], self.p[:,2],
                                       cur[:,0], cur[:,1], cur[:,2], length=10.0, linewidth=1.0)

        # BLINK overflow + mask sesuai domain aktif
        blink_on = ((self.t // self._blink_every) % 2) == 0
        base_colors = np.where(self.node_overflow_flags > 0.5,
                               np.where(blink_on, 'red', 'white'),
                               'red')

        mask = self._visible_nodes_mask()
        nodes_to_draw = self.nodes[mask]
        colors = base_colors[mask]

        try: self._nodes_artist.remove()
        except Exception: pass
        if nodes_to_draw.shape[0] > 0:
            self._nodes_artist = self._ax.scatter(nodes_to_draw[:,0], nodes_to_draw[:,1], nodes_to_draw[:,2],
                                                  s=self._node_size, marker='o', c=colors.tolist(), alpha=self._node_alpha)
        else:
            # jika domain terlalu kecil & tidak ada node terlihat
            self._nodes_artist = self._ax.scatter([], [], [], s=self._node_size, marker='o')

        self._ax.set_title(f'AUV3D-Env (t={self.t})')
        self._fig.canvas.draw_idle()
        plt.pause(0.001)

    def close(self):
        if self._fig is not None:
            plt.ioff()
            plt.close(self._fig)
            self._fig = None





