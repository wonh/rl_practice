import numpy as np
from .ppo_core import discount_cumsum
import torch
import gym
from gym.spaces import Discrete,Box
from .ppo_core import ActorCritic
from torch.optim import Adam

def combine_shape(length, shape=None):
    if shape==None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)
class ReplayBuffer(object):

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.ob = np.zeros(combine_shape(size, obs_dim), dtype=np.float32)
        self.act = np.zeros(combine_shape(size, act_dim), dtype=np.float32)
        self.val = np.zeros(size, dtype=np.float32)
        self.rew = np.zeros(size, dtype=np.float32)
        self.adv = np.zeros(size, dtype=np.float32)
        self.logp = np.zeros(size, dtype=np.float32)
        self.ret= np.zeros(size, dtype=np.float32)
        self.start_ptr, self.ptr, self.size = 0, 0, size
        self.gamma, self.lam = gamma, lam

    def store(self, ob, act, val, rew, logp):
        # print(self.ptr, self.size)
        assert self.ptr < self.size
        self.ob[self.ptr] = ob
        self.act[self.ptr] = act
        self.val[self.ptr] = val
        self.rew[self.ptr] = rew
        self.logp[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        path_index = slice(self.start_ptr, self.ptr)
        # print(self.ob)
        # t1 = self.ob[path_index][1:] - self.ob[path_index][:-1]
        # t1[t1<0] =0
        # t1[t1>0] = 0.2
        # # print(self.rew[path_index])
        # self.rew[path_index]+= self.ob[path_index][:,0]
        # print(self.rew[path_index])
        rews = np.append(self.rew[path_index], last_val)
        vals = np.append(self.val[path_index], last_val)
        # delta = rews[:-1] + vals[:-1] - vals[1:]*self.gamma
        delta = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv[path_index] = discount_cumsum(delta, self.gamma*self.lam)
        self.ret[path_index] = discount_cumsum(rews, self.gamma)[:-1]
        self.start_ptr = self.ptr

    def get(self):
        assert self.ptr == self.size
        self.start_ptr, self.ptr = 0, 0
        adv_mu, adv_std = np.mean(self.adv), np.std(self.adv)
        # print(self.adv.mean())
        self.adv = (self.adv - adv_mu) / adv_std
        # print(adv_mu, adv_std, self.adv.mean())
        data = {"obs":self.ob, "act":self.act, "adv":self.adv, "logp":self.logp, "ret":self.ret}
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}

def ppo(env_fn, actor_critic=ActorCritic, ac_kwargs=dict(), seed=2,
        steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=320, train_v_iters=320, lam=0.97, max_ep_len=1000,
        target_kl=0.01, logger_kwargs=dict(), save_freq=10):
    #set seed
    # seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)
    #create env
    env = env_fn()
    #create actor_crictic
    ac = actor_critic(env.observation_space, env.action_space)
    #set up optimizer
    pi_optimizer = Adam(ac.pi.parameters(),pi_lr)
    v_optimizer = Adam(ac.v.parameters(),vf_lr)
    #get buffer
    # if isinstance(env.action_space, Discrete):
    #     act_dim = env.action_space.n
    # elif isinstance(env.action_space, Box):
    #     act_dim = env.action_space.shape[0]
    # print(env.observation_space.shape[0], act_dim)
    buffer = ReplayBuffer(env.observation_space.shape, env.action_space.shape, steps_per_epoch,
                          gamma, lam)
    #compute loss pi
    def compute_loss_pi(data,clip_ratio=clip_ratio):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        # print(obs.shape,act.shape)
        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        # print(adv.mean())
        loss_pi = - (torch.min(ratio * adv, clip_adv)).mean()

        kl_divergence = (logp_old - logp).mean().item()
        cliped = ratio.gt(1-clip_ratio) | ratio.gt(1+clip_ratio)
        pi_info = dict(kl=kl_divergence, clip=cliped)
        return loss_pi, pi_info

    #compute loss v
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((ac.v(obs) - ret)**2).mean()
    #update
    def update():
        data = buffer.get()
        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        #train policy with multistep of gradient desent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = pi_info['kl']
            if kl >1.5* target_kl:
                print('early stopping due to reaching max kl')
                break
            loss_pi.backward()
            pi_optimizer.step()
        #value function learning
        for i in range(train_v_iters):
            v_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            v_optimizer.step()
        print('losspi{0}, loss_v{1}'.format(loss_pi.item(), loss_v.item()))
        l_pi = loss_pi.item() - pi_l_old
        l_v = loss_v.item() - v_l_old
        print('delta_pi{0}, delta_v{1}, kl{2}'.format(l_pi, l_v, kl))
    #main loop
    obs, ep_ret, ep_len, real_ret = env.reset(), 0, 0, 0
    for epoch in range(epochs):
        for step in range(steps_per_epoch):
            act, val, logp = ac.step(torch.as_tensor(obs, dtype=torch.float32))
            next_obs, rew, done, info = env.step(act)
            real_ret += rew
            # rew += obs[0] + 1.2
            ep_ret += rew
            ep_len += 1
            buffer.store(obs, act, val, rew, logp)
            #update obs
            obs = next_obs
            timeout = ep_len==max_ep_len
            terminal = done or timeout
            epoch_ended = step==steps_per_epoch-1
            if terminal or epoch_ended:
                if epoch_ended and not terminal:
                    print("epoch ended")
                if timeout or epoch_ended:
                    _, v, _ = ac.step(torch.as_tensor(obs, dtype=torch.float32))
                else:
                    v = 0
                buffer.finish_path(v)
                print('ep_ret: ',real_ret)
                obs, ep_ret, ep_len,real_ret = env.reset(), 0, 0,0

        update()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='ppo')
    args = parser.parse_args()

    # mpi_fork(args.cpu)  # run parallel code with mpi

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    ppo(lambda : gym.make(args.env), actor_critic=ActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma,
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        logger_kwargs=logger_kwargs)