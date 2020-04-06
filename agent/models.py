import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def fan_in_uniform_init(tensor, fan_in=None):
    """Utility function for initializing actor and critic"""
    if fan_in is None:
        fan_in = tensor.size(-1)

    w = 1. / np.sqrt(fan_in)
    nn.init.uniform_(tensor, -w, w)


class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.layernorm1 = nn.LayerNorm(400)
        self.layernorm2 = nn.LayerNorm(300)
        self.linear1 = nn.Linear(state_size, 400)
        self.linear2 = nn.Linear(400 + action_size, 300)
        self.linear3 = nn.Linear(300, 1)

        self.init_layers()

    def forward(self, state, action):
        x = self.linear1(state)
        x = self.layernorm1(x)
        x = F.relu(x)

        x = torch.cat([x, action], 1)
        x = self.linear2(x)
        x = self.layernorm2(x)
        x = F.relu(x)

        x = self.linear3(x)

        return x

    def init_layers(self):
        for i in (self.linear1, self.linear2):
            fan_in_uniform_init(i.weight)
            fan_in_uniform_init(i.bias)
        nn.init.uniform_(self.linear3.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.linear3.bias, -3e-4, 3e-4)


class Actor(nn.Module):
    def __init__(self, state_size, action_size, action_range=None):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.action_range = action_range

        self.layernorm1 = nn.LayerNorm(400)
        self.layernorm2 = nn.LayerNorm(300)
        self.linear1 = nn.Linear(state_size, 400)
        self.linear2 = nn.Linear(400, 300)
        self.linear3 = nn.Linear(300, action_size)

        self.init_layers()

    def forward(self, state):
        x = self.linear1(state)
        x = self.layernorm1(x)
        x = F.relu(x)

        x = self.linear2(x)
        x = self.layernorm2(x)
        x = F.relu(x)

        x = self.linear3(x)
        return x

    def init_layers(self):
        for i in (self.linear1, self.linear2):
            fan_in_uniform_init(i.weight)
            fan_in_uniform_init(i.bias)
        nn.init.uniform_(self.linear3.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.linear3.bias, -3e-4, 3e-4)
        
