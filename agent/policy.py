from copy import deepcopy
import math
import time
from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, row_col, translate
from kaggle_environments.helpers import histogram
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

GAMMA = 0.99
MAX_STEPS = 150
NUM_EPISODES = 1000
NUM_PROCESSES = 16
NUM_ADVANCED_STEP = 5

value_loss_coef = 0.5
entropy_coef = 0.01
max_grad_norm = 0.5

# Memory class
class RolloutStrage(object):
    def __init__(self, num_steps):
        self.masks = torch.ones(num_steps + 1, 1)
        self.rewards = torch.zeros(num_steps, 1)
        self.returns = torch.zeros(num_steps + 1, 1)
        self.index = 0
        self.observations = []
        self.last_observations = []
        self.actions = []

    def insert(self, current_obs, last_obs, action, reward, mask):
        self.observations.append(current_obs)
        self.last_observations.append(last_obs)
        self.masks[self.index + 1].copy_(mask)
        self.rewards[self.index].copy_(reward)
        self.actions.append(action)
        self.index = (self.index + 1) % NUM_ADVANCED_STEP

    def after_update(self):
        self.observations = []
        self.last_observations = []
        self.actions = []
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value):
        self.returns[-1] = next_value
        for ad_step in reversed(range(self.rewards.size(0))):
            self.returns[ad_step] = self.returns[ad_step + 1] * GAMMA * self.masks[ad_step + 1] + self.rewards[ad_step]

# TorusConv class
class TorusConv2d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, bn):
        super().__init__()
        self.edge_size = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size)
        self.bn = nn.BatchNorm2d(output_dim) if bn else None # BatchNormalize

    def forward(self, x):
        h = torch.cat([x[:,:,:,-self.edge_size[1]:], x, x[:,:,:,:self.edge_size[1]]], dim=3)
        h = torch.cat([h[:,:,-self.edge_size[0]:], h, h[:,:,:self.edge_size[0]]], dim=2)
        h = self.conv(h)
        h = self.bn(h) if self.bn is not None else h
        return h

# Geese Net class
class GNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers, filters = 12, 32
        self.conv0 = TorusConv2d(17, filters, (3, 3), True)
        self.blocks = nn.ModuleList([TorusConv2d(filters, filters, (3, 3), True) for _ in range(layers)])
        self.head_p = nn.Linear(filters, 4, bias=False)
        self.head_v = nn.Linear(filters * 2, 1, bias=False)
        # self.actions = [Action.NORTH, Action.SOUTH, Action.WEST, Action.EAST]
        self.actions = ['NORTH', 'SOUTH', 'WEST', 'EAST']

    def forward(self, x):
        '''Define the forward of Network'''
        h = F.relu_(self.conv0(x))
        for block in self.blocks:
            h = F.relu_(h + block(h))
        h_head = (h * x[:,:1]).view(h.size(0), h.size(1), -1).sum(-1)
        h_avg = h.view(h.size(0), h.size(1), -1).mean(-1)
        p = torch.softmax(self.head_p(h_head), 1)
        v = torch.tanh(self.head_v(torch.cat([h_head, h_avg], 1)))

        return p, v

    def _make_input(self, obs, last_obs, index):
        '''Make input for Neural Network'''
        b = np.zeros((17, 7*11), dtype=np.float32)

        for p, pos_list in enumerate(obs.geese):
            # head position
            for pos in pos_list[:1]:
                b[0 + (p - index) % 4, pos] = 1
            # tip position
            for pos in pos_list[-1:]:
                b[4 + (p - index) % 4, pos] = 1
            # whole position
            for pos in pos_list:
                b[8 + (p - index) % 4, pos] = 1

        if last_obs is not None:
            for p, pos_list in enumerate(last_obs.geese):
                for pos in pos_list[:1]:
                    b[12 + (p-index) % 4, pos] = 1

        # food
        for pos in obs.food:
            b[16, pos] = 1

        return b.reshape(-1, 7, 11)
    
    def predict(self, obs, last_obs, index):
        x = self._make_input(obs, last_obs, index)
        with torch.no_grad():
            xt = torch.from_numpy(x).unsqueeze(0)
            p, v = self.forward(xt)

        return p.squeeze(0).detach().numpy(), v.item()

    def predict_for_grad(self, obs, last_obs, index):
        x = self._make_input(obs, last_obs, index)
        xt = np.expand_dims(x, 0)
        xt = torch.tensor(xt, requires_grad=True)
        p, v = self.forward(xt)

        return p.squeeze(0), v

    # Get Action
    def get_action(self, obs, last_obs, index):
        p, _ = self.predict(obs, last_obs, index)
        action = self.actions[np.argmax(p)]

        return action

    # Get Values
    def get_values(self, obs, last_obs, index):
        _, v = self.predict(obs, last_obs, index)

        return v

    # Evaluate actions
    def evaluate_actions(self, obs, last_obs, index, actions):
        num_sample = len(obs)

        p = torch.zeros(num_sample, 4)
        v = torch.zeros(num_sample, 1)
        x_array = []
        
        for i in range(num_sample):
            p[i], v[i] = self.predict_for_grad(obs[i], last_obs[i], index)

        log_probs = torch.log(p)
        actions_ = [np.where(np.array(self.actions) == actions[i])[0].tolist() for i in range(len(actions))] 
        actions_ = torch.tensor(actions_)
        action_log_probs = log_probs.gather(1, actions_)
        entropy = - (log_probs * p).sum(-1).mean()

        return v, action_log_probs, entropy

# Brain class
class Brain():
    def __init__(self, actor_critic):
        self.actor_critic = actor_critic # GNet actor-critic instance
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=0.01) # Adam optimizer

    def update(self, rollouts):
        num_steps = NUM_ADVANCED_STEP
        index = 0

        # values, action_log_probs, entropy = self.actor_critic.evaluate_actions(rollouts.observations[:-1], rollouts.last_observations[:-1], index, rollouts.actions[:-1])
        values, action_log_probs, entropy = self.actor_critic.evaluate_actions(rollouts.observations, rollouts.last_observations, index, rollouts.actions)

        values = values.view(num_steps, 1) # torch.Size([5, 1])
        action_log_probs = action_log_probs.view(num_steps, 1)
        
        # Advantage
        advantages = rollouts.returns[:-1] - values # torch.Size([5, 1])
        
        # Critic loss
        value_loss = advantages.pow(2).mean()
        
        # Action gain
        action_gain = (action_log_probs * advantages.detach()).mean()
        
        # Surrogate loss
        total_loss = (value_loss * value_loss_coef - action_gain - entropy*entropy_coef)

        # Update parameters
        self.actor_critic.train()
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_grad_norm)
        self.optimizer.step()