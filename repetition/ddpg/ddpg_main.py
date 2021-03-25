import torch
import numpy as np
import argparse
from gym.spaces import Discrete, Box
from torch import nn
from ppo_core import mlp
from torch.distributions import Normal
import gym
import copy

class q_net(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.model = mlp([obs_dim + act_dim] + [256, 256, 1], activation= nn.Tanh)
    def forward(self, obs, act):
        res = self.model(torch.cat((obs, act), -1))
        return torch.squeeze(res,-1)

class pi_net(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit):
        super().__init__()
        self.mu_net = mlp([obs_dim] + [256, 256] + [act_dim], activation=nn.Tanh)
        self.limit = act_limit
        # log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        # self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

    def forward(self, obs):
        # mu = self.mu_net(obs)
        # std = torch.exp(self.log_std)
        # logits = self.model(obs)
        # return Normal(mu, std).sample()
        return self.limit * self.mu_net(obs)

    def act(self, obs):
        with torch.no_grad():
            res = self.mu_net(obs) * self.limit
        return res.numpy()

class Buffer(object):
    def __init__(self, obs_dim, act_dim, size):
        self.size = size
        self.max_size = size
        self.buf_rew = np.zeros((size,), dtype=np.float32)
        self.buf_obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.buf_act = np.zeros((size, act_dim), dtype=np.float32)
        self.buf_obs2 = np.zeros((size, obs_dim), dtype=np.float32)
        self.done = np.zeros((size,), dtype=np.float32)
        self.start_ptr, self.ptr = 0, 0

    def store(self, obs, act, obs2, rew, done):
        assert self.ptr < self.size
        self.buf_obs[self.ptr] = obs
        self.buf_obs2[self.ptr] = obs2
        self.buf_act[self.ptr] = act
        self.buf_rew[self.ptr] = rew
        self.done[self.ptr] = done
        # self.ptr += 1
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    # def finish_path(self):

    def get(self):
        assert self.ptr == self.size
        self.start_ptr, self.ptr = 0, 0
        data = {'s':self.buf_obs, 'a':self.buf_act, 'r':self.buf_rew,
                's2':self.buf_obs2, 'd':self.done}
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.buf_obs[idxs],
                     obs2=self.buf_obs2[idxs],
                     act=self.buf_act[idxs],
                     rew=self.buf_rew[idxs],
                     done=self.done[idxs])
        # print(self.buf_act[idxs].shape)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}


def DDPG(env_fn, start_steps, total_steps, q_net, pi_net,q_lr, pi_lr,
         noise_scale=0.1, seed=0, polyak=0.995, gamma=0.99,update_after=1000, update_every=50, max_ep_len=1000,batch_size=100,
         buffer_size=int(1e6)):

    torch.manual_seed(seed)
    np.random.seed(seed)
    env = env_fn()
    print('creat env')
    if isinstance(env.action_space, Discrete):
        raise TypeError
    else:
        act_dim = env.action_space.shape[0]
    obs_dim = env.observation_space.shape[0]

    act_limit = env.action_space.high[0]
    print(obs_dim, act_dim)
    q_model = q_net(obs_dim, act_dim)
    q_target = copy.deepcopy(q_model)
    pi_model = pi_net(obs_dim, act_dim, act_limit)
    pi_target= copy.deepcopy(pi_model)

    for (q, pi) in zip(q_target.parameters(), pi_target.parameters()):
        q.requires_grad = False
        pi.requires_grad = False

    q_optimizer = torch.optim.Adam(q_model.parameters(), lr=q_lr)
    pi_optimizer = torch.optim.Adam(pi_model.parameters(), lr=pi_lr)
    buffer = Buffer(obs_dim, act_dim, buffer_size)

    def compute_q_loss(data):
        s, a, r, s2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        # loss = ((q_model(s,a) - (r + gamma * (1-d)* q_target(s2, pi_target(s2))))**2).mean()
        # print(s.size(), a.size())
        q = q_model(s,a)
        with torch.no_grad():
            q_pi_targ = q_target(s2, pi_target(s2))
            backup = r + gamma * (1 - d) * q_pi_targ

            # MSE loss against Bellman backup
        loss_q = ((q - backup) ** 2).mean()
        return loss_q

    def compute_pi_loss(data):
        s, a, r, s2, d = data.values()
        loss = -q_model(s, pi_model(s)).mean()
        # print('loss_pi', loss)
        return loss

    def update(data):
        # for i in range(train_q_steps):
        q_optimizer.zero_grad()
        loss_q = compute_q_loss(data)
        loss_q.backward()
        q_optimizer.step()
        # froozen q
        for q in q_model.parameters():
            q.requires_grad = False

        # for i in range(train_pi_steps):
        pi_optimizer.zero_grad()
        loss_pi = compute_pi_loss(data)
        # print(loss_pi)
        loss_pi.backward()
        pi_optimizer.step()
        #unfroozen q
        for q in q_model.parameters():
            q.requires_grad = True

        #update target model
        with torch.no_grad():
            for (q, q_tar, pi, pi_tar) in zip(q_model.parameters(), q_target.parameters(), pi_model.parameters(), pi_target.parameters()):
                q_tar.data.mul_(polyak)
                q_tar.data.add_((1-polyak)*q.data)
                pi_tar.data.mul_(polyak)
                pi_tar.data.add_((1-polyak)*pi.data)

        # print("loss_pi, loss_q",loss_pi, loss_q)

    def get_action(o, noise_scale):
        a = pi_model.act(torch.as_tensor(o, dtype=torch.float32))
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)

    obs = env.reset()
    for step in range(total_steps):
        ep_len, ep_ret = 0, 0
        # print('replay')
        if step < start_steps:
            act = env.action_space.sample()
        else:
            act = get_action(obs, noise_scale)
        # print(act.shape)
        obs2, rew, done, _ = env.step(act)
        buffer.store(obs, act, obs2, rew, done)
        obs = obs2
        ep_len += 1
        ep_ret += rew

        epoch_done = ep_len == max_ep_len

        if done or epoch_done:
            print(step, ep_len, ep_ret)
            obs, ep_ret, ep_len = env.reset(), 0, 0

        if step > update_after and step % update_every ==0:
            for _ in range(update_every):
                batch = buffer.sample_batch(batch_size)
                update(data=batch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    # parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--total_steps', type=int, default=4e6)
    parser.add_argument('--exp_name', type=str, default='ppo')
    args = parser.parse_args()

    # mpi_fork(args.cpu)  # run parallel code with mpi

    # from spinup.utils.run_utils import setup_logger_kwargs

    # logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    DDPG(lambda: gym.make(args.env), 10000, int(args.total_steps), q_net, pi_net, 1e-3, 1e-3,)
