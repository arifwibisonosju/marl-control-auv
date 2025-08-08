import torch
import torch.nn as nn
import numpy as np

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

class MAPPOAgent:
    def __init__(self, agent_id, obs_dim, act_dim, n_agents, args):
        self.device = args['device']
        self.agent_id = agent_id
        self.actor = Actor(obs_dim, act_dim).to(self.device)

    def select_action(self, obs):
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        action = self.actor(obs)
        return action.clamp(-1, 1).cpu().detach().numpy()[0]

    def load_model(self, path):
        self.actor.load_state_dict(torch.load(path, map_location=self.device))
        self.actor.eval()
        print(f"✔️ Loaded MAPPO model from {path}")
