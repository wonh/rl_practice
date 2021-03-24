import torch
import numpy as np
from .td3_core import Actor_Critic
from copy import deepcopy
from gym.spaces import Discrete

class Buffer(object):

    def __init__(self, obs_dim, act_dim, size):
        #sarsd
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros((size,), dtype=np.float32)
        self.obs2_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.done_buf = np.zeros((size, ), dtype=np.float32)
        self.ptr, self.size = 0, size

    def store(self, obs, act, rew, obs2, done):
        assert self.ptr < self.size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.obs2_buf[self.ptr] = obs2
        self.done_buf[self.ptr] = done
        self.ptr += 1

    def sample_batch(self, batch_size=32):
        idx = np.random.randint(0, self.size, batch_size)
        data = dict(s1=self.obs_buf[idx], a=self.act_buf[idx], r=self.rew_buf[idx], s2=self.obs2_buf[idx], d=self.done_buf[idx])

        return {k:torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


def td3(env_fn, epochs, steps_per_epoch, max_ep_len, start_steps, buffer_size, batch_size, q_lr, pi_lr, update_every, update_after, polyad=0.995,
        gamma=0.99, act_noise=0.1, sigma=0.1, policy_delay=2):

    env = env_fn()
    obs_dim, act_dim = env.observe_space.shape[0], env.act_space.shape[0]
    limit = env.act_space.high[0]
    clip = 0.5

    ac = Actor_Critic(env.observe_space, env.act_space)
    ac_target = deepcopy(ac)

    buffer = Buffer(obs_dim, act_dim, buffer_size)

    q1_optimizer = torch.optim.Adam(ac.q.parameters(), lr=q_lr)
    q2_optimizer = torch.optim.Adam(ac.q2.parameters(), lr=q_lr)
    pi_optimizer = torch.optim.Adam(ac.pi.parameters(), lr=pi_lr)

    def compute_q_loss(data):
        s, a, r, s2, d = done['s'], done['a'], data['r'], data['s2'], data['d']
        eplison = np.random.normal(0, sigma)
        a2 = np.clip(ac_target.act(s2) + np.clip(eplison, -clip, clip), -limit, limit)
        y = r + gamma*(1-d) * min(ac_target.q(s2, a2), ac_target.q2(s2, a2))
        q_loss_1 = -(ac.q(s,a) -y).mean()
        q_loss_2 = -(ac.q2(s,a) -y).mean()
        return q_loss_1, q_loss_2

    def compute_pi_loss(data):
        s, a, r, s2, d = done['s'], done['a'], data['r'], data['s2'], data['d']
        for params in ac.q.parameters():
            params.require_grad = False
        loss = (ac.q(s, ac.pi(s))).mean()
        return loss

    def update(data):
        q_loss, q2_loss = compute_q_loss(data)
        q1_optimizer.zero_grad()
        q2_optimizer.zero_grad()
        q_loss.backward()
        q2_loss.backward()




    def get_action(obs, act_noise):
        act = ac.act(torch.as_tensor(obs))
        act += act_noise * np.random.randn(act_dim)
        return np.clip(act, -limit, limit)

    total_steps = epochs * steps_per_epoch

    obs, ep_len, ep_ret = env.reset(), 0, 0
    for t in range(total_steps):
        # replay
        if t > start_steps:
            act = get_action(obs, act_noise)
        else:
            act = env.action_space.sample()

        obs2, rew, done, _ = env.step(act)
        buffer.store(obs, act, rew, obs2, done)
        ep_len += 1
        ep_ret += rew
        obs = obs2

        epoch_done = done or ep_len==max_ep_len
        if epoch_done:
            print(t, ep_len, ep_ret)
            obs, ep_len, ep_ret = env.reset(), 0, 0

        if t > update_after and t % update_every==0:
            data = buffer.sample_batch(batch_size)
            update(data)
            if t//policy_delay ==0:
                pi_optimizer.zero_grad()
                loss_pi = compute_pi_loss(data)
                loss_pi.backward()
                pi_optimizer.step()

