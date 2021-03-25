import torch
import numpy as np
from td3_core import Actor_Critic
# from .td3_core
from copy import deepcopy
from gym.spaces import Discrete
import argparse
import gym

class Buffer(object):

    def __init__(self, obs_dim, act_dim, size):
        #sarsd
        self.max_size=size
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
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idx = np.random.randint(0, self.size, batch_size)
        data = dict(s=self.obs_buf[idx], a=self.act_buf[idx], r=self.rew_buf[idx], s2=self.obs2_buf[idx], d=self.done_buf[idx])

        return {k:torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


def td3(env_fn, epochs, steps_per_epoch, max_ep_len, start_steps, buffer_size, batch_size, q_lr, pi_lr,
        update_every, update_after, polyak=0.995, gamma=0.99, act_noise=0.1, sigma=0.1):

    env = env_fn()
    if isinstance(env.action_space, Discrete):
        raise TypeError
    else:
        act_dim = env.action_space.shape[0]
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    limit = env.action_space.high[0]
    clip = 0.5

    ac = Actor_Critic(env.observation_space, env.action_space)
    ac_target = deepcopy(ac)
    for para in ac_target.parameters():
        para.require_grad = False

    buffer = Buffer(obs_dim, act_dim, buffer_size)

    q1_optimizer = torch.optim.Adam(ac.q.parameters(), lr=q_lr)
    q2_optimizer = torch.optim.Adam(ac.q2.parameters(), lr=q_lr)
    pi_optimizer = torch.optim.Adam(ac.pi.parameters(), lr=pi_lr)

    # def compute_q_loss(data):
    #
    #     return q_loss_1, q_loss_2

    def compute_pi_loss(data):
        s, a, r, s2, d = data['s'], data['a'], data['r'], data['s2'], data['d']
        for params in ac.q.parameters():
            params.require_grad = False
        loss = (ac.q(s, ac.pi(s))).mean()
        return loss

    def update_q(data):
        s, a, r, s2, d = data['s'], data['a'], data['r'], data['s2'], data['d']
        eplison = np.random.normal(0, sigma)
        a2 = np.clip(ac_target.act(s2) + np.clip(eplison, -clip, clip), -limit, limit)
        a2 = torch.as_tensor(a2, dtype=torch.float32)
        # print(ac_target.q(s2, a2).shape)
        with torch.no_grad():
            y = r + gamma * (1 - d) * torch.min(ac_target.q(s2, a2), ac_target.q2(s2, a2))
        q1_optimizer.zero_grad()
        q_loss = -(ac.q(s, a) - y).mean()
        q_loss.backward()
        q2_optimizer.zero_grad()
        q2_loss = -(ac.q2(s, a) - y).mean()
        q2_loss.backward()

    def update_target(model, tar_model):
        for para, tar_para in zip(model.parameters(), tar_model.parameters()):
            tar_para.data.mul_(polyak)
            tar_para.data.add_((1-polyak)*para.data)

    def update_pi_tar(data):
        # update policy
        pi_optimizer.zero_grad()
        loss_pi = compute_pi_loss(data)
        loss_pi.backward()
        pi_optimizer.step()
        # update target
        update_target(ac.q, ac_target.q)
        update_target(ac.q2, ac_target.q2)
        update_target(ac.pi, ac_target.pi)

    def get_action(obs, act_noise):
        act = ac.act(torch.as_tensor(obs, dtype=torch.float32))
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
            update_q(data)
            if t % update_every*2 == 0:
                update_pi_tar(data)

# if __name__ == '__main__':
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default='CartPole-v0')
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--steps_per_epoch", type=int, default=4000)
    parser.add_argument("--start_steps", type=int, default=10000)
    parser.add_argument("--max_ep_len", type=int, default=1000)
    parser.add_argument("--buffer_size", type=int, default=10000)
    parser.add_argument("--update_after", type=int, default=1000)
    parser.add_argument("--update_every", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=10000)
    args = parser.parse_args()

    td3(lambda: gym.envs.make(args.env), args.epochs, args.steps_per_epoch, args.max_ep_len, args.start_steps, args.buffer_size, args.batch_size, 1e-3, 1e-3,
            args.update_every, args.update_after, polyak=0.995, gamma=0.99, act_noise=0.1, sigma=0.1)