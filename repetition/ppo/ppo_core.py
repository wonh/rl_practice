import torch
from torch import nn
import numpy as np
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from gym.spaces import Box, Discrete
import scipy
import ray

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

class Model(object):
    def __init__(self, args):
        self.ac = ActorCritic(args.observation_space, args.action_space, args.hidden_size, args.activation)
        self.pi_optimizer = torch.optim.Adam(self.ac.pi.parameters(), args.pi_lr)
        self.v_optimizer = torch.optim.Adam(self.ac.v.parameters(), args.vf_lr)

    def step(self, obs):
        return self.ac.step(obs)

    def act(self, obs):
        return self.ac.act(obs)

    def get_weights(self):
        values = []
        keys = []
        for k, v in self.ac.named_parameters():
            keys.append(k)
            values.append(v)
        return keys, values

    def set_weights(self, keys, values):
        for v, para in zip(values, self.ac.parameters()):
            # keys.append(k)
            # values.append(v)
            para.data = v

    def train(self, replay_buffer, args):
        # compute loss pi
        def compute_loss_pi(data, clip_ratio=args.clip_ratio):
            obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
            # print(obs.shape,act.shape)
            pi, logp = self.ac.pi(obs, act)
            ratio = torch.exp(logp - logp_old)
            clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
            # print(adv.mean())
            loss_pi = - (torch.min(ratio * adv, clip_adv)).mean()

            kl_divergence = (logp_old - logp).mean().item()
            cliped = ratio.gt(1 - clip_ratio) | ratio.gt(1 + clip_ratio)
            pi_info = dict(kl=kl_divergence, clip=cliped)
            return loss_pi, pi_info

        # compute loss v
        def compute_loss_v(data):
            obs, ret = data['obs'], data['ret']
            return ((self.ac.v(obs) - ret) ** 2).mean()

        # update
        def update():
            data = ray.get(replay_buffer.get.remote())
            pi_l_old, pi_info_old = compute_loss_pi(data)
            pi_l_old = pi_l_old.item()
            v_l_old = compute_loss_v(data).item()

            # train policy with multistep of gradient desent
            for i in range(args.train_pi_iters):
                self.pi_optimizer.zero_grad()
                loss_pi, pi_info = compute_loss_pi(data)
                kl = pi_info['kl']
                if kl > 1.5 * args.target_kl:
                    print('early stopping due to reaching max kl')
                    break
                loss_pi.backward()
                self.pi_optimizer.step()
            # value function learning
            for i in range(args.train_v_iters):
                self.v_optimizer.zero_grad()
                loss_v = compute_loss_v(data)
                loss_v.backward()
                self.v_optimizer.step()

            print('losspi{0}, loss_v{1}'.format(loss_pi.item(), loss_v.item()))
            l_pi = loss_pi.item() - pi_l_old
            l_v = loss_v.item() - v_l_old
            print('delta_pi{0}, delta_v{1}, kl{2}'.format(l_pi, l_v, kl))
        update()

    def test_agent(self, test_env, args, n=10):
        test_ret = []
        for j in range(n):
            o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            while not (d or ep_len==args.max_ep_len):
                o, r, d, _ = test_env.step(self.act(torch.as_tensor(o, dtype=torch.float32)))
                ep_len += 1
                ep_ret += r
            test_ret.append(ep_ret)
        return sum(test_ret)/n
