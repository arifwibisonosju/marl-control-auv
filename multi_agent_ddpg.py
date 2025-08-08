# multi_agent_ddpg.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from copy import deepcopy

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, act_dim), nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

class Critic(nn.Module):
    def __init__(self, total_obs_dim, total_act_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(total_obs_dim + total_act_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs_all, act_all):
        x = torch.cat([obs_all, act_all], dim=1)
        return self.net(x)

class DDPGAgent:
    def __init__(self, agent_id, obs_dim, act_dim, n_agents, args):
        self.id = agent_id
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = args['device']

        self.actor = Actor(obs_dim, act_dim).to(self.device)
        self.target_actor = deepcopy(self.actor)
        self.critic = Critic(obs_dim * n_agents, act_dim * n_agents).to(self.device)
        self.target_critic = deepcopy(self.critic)

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=args['actor_lr'])
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=args['critic_lr'])

        self.gamma = args['gamma']
        self.tau = args['tau']
        self.loss_fn = nn.MSELoss()

    def select_action(self, obs, noise=0.0):
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        action = self.actor(obs)
        if noise > 0.0:
            action += torch.normal(0, noise, size=action.shape).to(self.device)
        return action.clamp(-1, 1).cpu().detach().numpy()[0]

    def update(self, sample, agents):
        obs, act, rew, next_obs = sample

        obs_all = torch.cat(obs, dim=1).to(self.device)
        act_all = torch.cat(act, dim=1).to(self.device)
        next_obs_all = torch.cat(next_obs, dim=1).to(self.device)

        with torch.no_grad():
            next_act_all = torch.cat([
                agents[i].target_actor(next_obs[i].to(self.device))
                for i in range(self.n_agents)
            ], dim=1)
            target_q = self.target_critic(next_obs_all, next_act_all)
            y = rew[self.id].to(self.device) + self.gamma * target_q

        q_val = self.critic(obs_all, act_all)
        critic_loss = self.loss_fn(q_val, y.detach())

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        curr_act_all = torch.cat([
            self.actor(obs[self.id].to(self.device)) if i == self.id else act[i].to(self.device).detach()
            for i in range(self.n_agents)
        ], dim=1)
        actor_loss = -self.critic(obs_all, curr_act_all).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self.soft_update(self.target_actor, self.actor)
        self.soft_update(self.target_critic, self.critic)

        return critic_loss.item(), actor_loss.item()  # tambahkan ini

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
