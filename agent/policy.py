import numpy as np
import time
from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, row_col, translate
from kaggle_environments.helpers import histogram
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from copy import deepcopy

# Neural Network for Hungry Geese
class TorusConv2d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, bn):
        super().__init__()
        self.edge_size = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size)
        self.bn = nn.BatchNorm2d(output_dim) if bn else None

    def forward(self, x):
        h = torch.cat([x[:,:,:,-self.edge_size[1]:], x, x[:,:,:,:self.edge_size[1]]], dim=3)
        h = torch.cat([h[:,:,-self.edge_size[0]:], h, h[:,:,:self.edge_size[0]]], dim=2)
        h = self.conv(h)
        h = self.bn(h) if self.bn is not None else h
        return h

class GeeseNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers, filters = 12, 32
        self.conv0 = TorusConv2d(17, filters, (3, 3), True)
        self.blocks = nn.ModuleList([TorusConv2d(filters, filters, (3, 3), True) for _ in range(layers)])
        self.head_p = nn.Linear(filters, 4, bias=False)
        self.head_v = nn.Linear(filters * 2, 1, bias=False)

    def forward(self, x):
        h = F.relu_(self.conv0(x))
        for block in self.blocks:
            h = F.relu_(h + block(h))
        h_head = (h * x[:,:1]).view(h.size(0), h.size(1), -1).sum(-1)
        h_avg = h.view(h.size(0), h.size(1), -1).mean(-1)
        p = torch.softmax(self.head_p(h_head), 1)
        v = torch.tanh(self.head_v(torch.cat([h_head, h_avg], 1)))

        return p, v

class NNAgent():
    def __init__(self, state_dict=None):
        self.model = GeeseNet()
        if state_dict is not None:
            self.model.load_state_dict(state_dict)
            self.model.eval()
        
    def predict(self, obs, last_obs, index):
        x = self._make_input(obs, last_obs, index)
        with torch.no_grad():
            xt = torch.from_numpy(x).unsqueeze(0)
            p, v = self.model(xt)
            
        return p.squeeze(0).detach().numpy(), v.item()
        
    # Input for Neural Network
    def _make_input(self, obs, last_obs, index):
        b = np.zeros((17, 7 * 11), dtype=np.float32)
        
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

        # previous head position
        if last_obs is not None:
            for p, pos_list in enumerate(last_obs.geese):
                for pos in pos_list[:1]:
                    b[12 + (p - index) % 4, pos] = 1

        # food
        for pos in obs.food:
            b[16, pos] = 1

        return b.reshape(-1, 7, 11)

# HugryGeese game 
class HungryGeese(object):
    def __init__(self,
                 rows=7,
                 columns=11,
                 actions=[Action.NORTH, Action.SOUTH, Action.WEST, Action.EAST],
                 hunger_rate=40):
        self.rows = rows
        self.columns = columns
        self.actions = actions
        self.hunger_rate = hunger_rate

    def getActionSize(self):
        return len(self.actions)

    def getNextState(self, obs, last_obs, directions):
        next_obs = deepcopy(obs)
        next_obs.step += 1
        geese = next_obs.geese
        food = next_obs.food
        
        # for i in range(4):
        for i in range(1):
            goose = geese[i]
            
            if len(goose) == 0: 
                continue
            
            head = translate(goose[0], directions[i], self.columns, self.rows)
            
            # Check action direction
            if last_obs is not None and head == last_obs.geese[i][0]:
                geese[i] = []
                continue

            # Consume food or drop a tail piece.
            if head in food:
                food.remove(head)
            else:
                goose.pop()
            
            # Add New Head to the Goose.
            goose.insert(0, head)

            # If hunger strikes remove from the tail.
            if next_obs.step % self.hunger_rate == 0:
                if len(goose) > 0:
                    goose.pop()

        goose_positions = histogram(
            position
            for goose in geese
            for position in goose
        )

        # Check for collisions.
        # for i in range(4):
        for i in range(1):
            if len(geese[i]) > 0:
                head = geese[i][0]
                if goose_positions[head] > 1:
                    geese[i] = []
        
        return next_obs

    def getValidMoves(self, obs, last_obs, index):   
        geese = obs.geese
        pos = geese[index][0]
        obstacles = {position for goose in geese for position in goose[:-1]}
        if last_obs is not None: obstacles.add(last_obs.geese[index][0])
        
        valid_moves = [
            translate(pos, action, self.columns, self.rows) not in obstacles
            for action in self.actions
        ]
    
        return valid_moves

    def stringRepresentation(self, obs):
        # print("geese", obs.geese)   
        return str(obs.geese + obs.food)

class MCTS():
    def __init__(self, game, nn_agent, eps=1e-8, cpuct=1.0):
        self.game = game
        self.nn_agent = nn_agent
        self.eps = eps
        self.cpuct = cpuct
        
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Vs = {}  # stores game.getValidMoves for board s
        
        self.last_obs = None

    def getActionProb(self, obs, timelimit=1.0):
        start_time = time.time()
        # print(obs)
        # print(self.last_obs)
        while time.time() - start_time < timelimit:
            self.search(obs, self.last_obs)

        s = self.game.stringRepresentation(obs)
        i = obs.index
        counts = [
            self.Nsa[(s, i, a)] if (s, i, a) in self.Nsa else 0
            for a in range(self.game.getActionSize())
        ]
        prob = counts / np.sum(counts)
        self.last_obs = obs
        return prob

    def search(self, obs, last_obs):
        s = self.game.stringRepresentation(obs)
        # print("start search: ", s, self.Ns)

        if s not in self.Ns:
            # print("s not in Ns")
            # values = [-10] * 4
            values = [-10]
            # for i in range(4):
            for i in range(1):
                if len(obs.geese[i]) == 0:
                    continue
                    
                # leaf node
                self.Ps[(s, i)], values[i] = self.nn_agent.predict(obs, last_obs, i)
                    
                valids = self.game.getValidMoves(obs, last_obs, i)    
                self.Ps[(s, i)] = self.Ps[(s, i)] * valids  # masking invalid moves
                sum_Ps_s = np.sum(self.Ps[(s, i)])
                if sum_Ps_s > 0:
                    self.Ps[(s, i)] /= sum_Ps_s  # renormalize

                self.Vs[(s, i)] = valids
                self.Ns[s] = 0
            # print("values from if ", values)
            return values

        # best_acts = [None] * 4
        best_acts = [None]
        # for i in range(4):
        for i in range(1):
            if len(obs.geese[i]) == 0:
                continue
            
            valids = self.Vs[(s, i)]
            cur_best = -float('inf')
            best_act = self.game.actions[-1]

            # pick the action with the highest upper confidence bound
            for a in range(self.game.getActionSize()):
                if valids[a]:
                    if (s, i, a) in self.Qsa:
                        u = self.Qsa[(s, i, a)] + self.cpuct * self.Ps[(s, i)][a] * math.sqrt(
                                self.Ns[s]) / (1 + self.Nsa[(s, i, a)])
                    else:
                        u = self.cpuct * self.Ps[(s, i)][a] * math.sqrt(
                            self.Ns[s] + self.eps)  # Q = 0 ?

                    if u > cur_best:
                        cur_best = u
                        best_act = self.game.actions[a]
                        
            best_acts[i] = best_act
        
        next_obs = self.game.getNextState(obs, last_obs, best_acts)
        values = self.search(next_obs, obs)

        # for i in range(4):
        for i in range(1):
            if len(obs.geese[i]) == 0:
                continue
                
            a = self.game.actions.index(best_acts[i])
            v = values[i]

            # print(s, i, a)
            # print("values origin: ", values)
            # print("values: ", v)
            # print("---")

            if (s, i, a) in self.Qsa:
                self.Qsa[(s, i, a)] = (self.Nsa[(s, i, a)] * self.Qsa[(s, i, a)] + v) / (self.Nsa[(s, i, a)] + 1)
                self.Nsa[(s, i, a)] += 1

            else:
                self.Qsa[(s, i, a)] = v
                self.Nsa[(s, i, a)] = 1

        self.Ns[s] += 1
        return values