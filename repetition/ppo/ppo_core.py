import torch
from torch import nn
import numpy as np
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from gym.spaces import Box, Discrete
import scipy

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def mlp(sizes, activation, output_activation=nn.Identity):
    model = []
    for i,num in enumerate(sizes[:-1]):
        # if i == len(sizes) - 1:
            # model.add_module('layer_{}'.format(i), nn.Softmax(num))
        act = activation if i < len(sizes) -2 else output_activation
        # model.add_module('layer_{}'.format(i), nn.Linear(num, layers[i+1]))
        model += [nn.Linear(sizes[i], sizes[i+1]), act()]
    return nn.Sequential(*model)

class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a

class MLPCategoryActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.model = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.model(obs)
        # print(logits.shape)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        # print(act.shape)
        return pi.log_prob(act)

class MLPGaussionActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_size, activation):
        super().__init__()
        self.mu_net = mlp([obs_dim] + list(hidden_size) + [act_dim], activation)
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_size, activation):
        super().__init__()
        self.val_net = mlp([obs_dim] + list(hidden_size) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.val_net(obs), -1)

class ActorCritic(nn.Module):

    def __init__(self, observation_space,action_space,hidden_size=(64,64),activation=nn.Tanh, **kwargs):
        super().__init__()
        obs_dim = observation_space.shape[0]
        if isinstance(action_space, Discrete):
            self.pi = MLPCategoryActor(obs_dim, action_space.n, hidden_size, activation)
        else:
            self.pi = MLPGaussionActor(obs_dim, action_space.shape[0], hidden_size, activation)

        self.v = MLPCritic(obs_dim, hidden_size, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            act = pi.sample()
            prob = self.pi._log_prob_from_distribution(pi, act)
            v = self.v(obs)
        return act.numpy(), v.numpy(), prob.numpy()

    def act(self, obs):
        return self.step(obs)[0]
