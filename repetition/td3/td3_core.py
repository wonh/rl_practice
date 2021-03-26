from torch import nn
import torch
import numpy as np
from gym.spaces import Discrete

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for i in range(len(sizes)-1):
        act = activation if i < len(sizes)-2 else output_activation
        # layers.append(nn.Linear(sizes[i], sizes[i+1]))
        # layers.append(act())
        layers += [nn.Linear(sizes[i], sizes[i+1]), act()]
    return nn.Sequential(*layers)

class Actor(nn.Module):

    def __init__(self, obs_dim, act_dim, limit, hid_size, activation):
        super().__init__()
        self.model = mlp([obs_dim]+list(hid_size)+[act_dim], activation, nn.Tanh)
        self.act_limit = limit

    def forward(self, obs):
        return self.model(obs) * self.act_limit

class Critic(nn.Module):

    def __init__(self, obs_dim, act_dim, hid_size, activation):
        super().__init__()
        self.model = mlp([obs_dim + act_dim] + list(hid_size) + [1], activation)

    def forward(self, obs, act):
        return self.model(torch.cat((obs, act), -1)).squeeze(-1)

class Actor_Critic(nn.Module):

    def __init__(self, obs_space, act_space, hid_size=(256,256), activation=nn.ReLU):
        super().__init__()
        if isinstance(act_space, Discrete):
            raise TypeError
        else:
            obs_dim = obs_space.shape[0]
            act_dim = act_space.shape[0]
        act_limit = act_space.high[0]
        self.q = Critic(obs_dim, act_dim, hid_size, activation)
        self.q2 = Critic(obs_dim, act_dim, hid_size, activation)
        self.pi = Actor(obs_dim, act_dim, act_limit, hid_size, activation)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).numpy()