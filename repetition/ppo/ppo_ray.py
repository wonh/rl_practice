import ray
import torch
from absl import flags
import argparse
import numpy as np
import scipy
import pickle as pk
from ppo_core import ActorCritic, Model
from spinup.utils.logx import EpochLogger
import gym
import time
from torch import nn

def advantage_estimate(list_val, rew, lam, gamma):
    delta = 0
    for i in range(len(list_val)):
        delta += (lam * gamma)**i * (rew[i] - list_val[i] + gamma * list_val[i+1])
    return delta

def reward_to_go(rew, gamma):
    ret = 0
    for i in range(len(rew)):
        ret += rew[i] * gamma**i
    return ret

def combine_shape(length, shape=None):
    if shape==None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

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

@ray.remote
class ReplayBuffer(object):

    def __init__(self, obs_dim, act_dim, size, workers, lam=0.05, gamma=0.99):
        self.ob = np.zeros(combine_shape(size*workers, obs_dim), dtype=np.float32)
        self.act = np.zeros(combine_shape(size*workers, act_dim), dtype=np.float32)
        self.val = np.zeros(size*workers, dtype=np.float32)
        self.rew = np.zeros(size*workers, dtype=np.float32)
        self.adv = np.zeros(size*workers, dtype=np.float32)
        self.logp = np.zeros(size*workers, dtype=np.float32)
        self.ret = np.zeros(size*workers, dtype=np.float32)
        self.start_ptr, self.ptr, self.size = [0] * workers, [0] * workers, size
        self.gamma, self. lam = gamma, lam
        self.wokers = workers

    def store(self, ob, act, val, rew, logp, worker_index):
        # print(self.ptr[worker_index] , self.size)
        assert self.ptr[worker_index] < self.size
        self.ob[self.ptr[worker_index] + worker_index*self.size] = ob
        self.act[self.ptr[worker_index] + worker_index*self.size] = act
        self.val[self.ptr[worker_index] + worker_index*self.size] = val
        self.rew[self.ptr[worker_index] + worker_index*self.size] = rew
        self.logp[self.ptr[worker_index] + worker_index*self.size] = logp
        self.ptr[worker_index] += 1

    def finish_path(self,last_val=0, worker_index=0):
        path_index = slice(self.start_ptr[worker_index], self.ptr[worker_index])
        rews = np.append(self.rew[path_index], last_val)
        vals = np.append(self.val[path_index], last_val)
        # self.adv[path_index] = advantage_estimate(self.val[path_index], self.rew[path_index], self.lam, self.gamma)
        delta = rews[:-1] + vals[:-1] - vals[1:]*self.gamma
        self.adv[path_index] = discount_cumsum(delta, self.gamma*self.lam)
        # self.ret[path_index] = reward_to_go(self.rew, gamma=self.gamma)
        self.ret[path_index] = discount_cumsum(rews, self.gamma)[:-1]
        self.start_ptr[worker_index] = self.ptr[worker_index]

    def get(self):
        assert sum(self.ptr) == self.size * self.wokers
        self.start_ptr, self.ptr = [0]*self.wokers, [0]*self.wokers
        #正规化advtage
        adv_mu, adv_std = np.mean(self.adv), np.std(self.adv)
        self.adv = (self.adv - adv_mu) / adv_std
        data = {"obs":self.ob, "act":self.act,
                "adv":self.adv, "logp":self.logp,
                "ret":self.ret}
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}

@ray.remote
class ParameterServer(object):
    # keep the newest network weights here
    # could pull and push the weights
    # also could save the weights to local
    def __init__(self, keys, values):
        values = [value.detach() for value in values]
        self.params = dict(zip(keys, values))
        # print(self.params)

    def push(self, keys, values):
        for (key, value) in zip(keys, values):
            self.params[key] = value

    def pull(self, keys):
        return [self.params[key] for key in keys]

    def save(self, name):
        with open(name + "_weights.pkl", 'wb') as f:
            pk.dump(self.params, f)

@ray.remote
def worker_train(ps, replay_buffer, args):
    #build learner network
    #pull weights from ps
    #for loop
      #compute grad
      #update ps parameters
    agent = Model(args)
    keys = agent.get_weights()[0]
    weights = ray.get(ps.pull.remote(keys))
    agent.set_weights(keys, weights)

    cnt = 1
    while True:
        agent.train(replay_buffer, args)

        if cnt % 300 ==0:
            keys, values = agent.get_weights()
            ps.push.remote(keys, values)
        cnt += 1

@ray.remote
def worker_rollout(ps, replay_buffer, args, worker_index):
    #build rollout network
    #pull weights from ps
    #for loop
        #interact with env and
        #store in replay_buffer
    # model =
    agent = Model(args)
    keys = agent.get_weights()[0]
    weights = ray.get(ps.pull.remote(keys))
    agent.set_weights(keys, weights)

    env = args.env_fn()
    o, r, done, ep_len, ep_ret = env.reset(), 0, False, 0, 0

    for t in range(args.steps_per_epoch):
        a, v, logp = agent.step(torch.as_tensor(o, dtype=torch.float32))
        o2, r, done, info = env.step(a)
        ep_ret += r
        ep_len += 1
        replay_buffer.store.remote(o, a, v, r, logp, worker_index)
        o = o2

        time_out = (ep_len==args.max_ep_len)
        terminal = time_out or done
        epoch_done = t % args.steps_per_epoch == 0 and t > 0
        if terminal or epoch_done:
            if time_out or epoch_done:
                _, v, _ = agent.step(torch.as_tensor(o, dtype=torch.float32))
            else:
                v = 0
            replay_buffer.finish_path.remote(v, worker_index)
            o, ep_ret, ep_len, done = env.reset(), 0, 0, False

            weights = ray.get(ps.pull.remote(keys))
            agent.set_weights(keys, weights)

@ray.remote
def worker_test(ps, replay_buffer, args, worker_index=0):
    #build test network
    #pull weights prom ps and do test
    #save model optionally
    from spinup.utils.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    logger = EpochLogger(**logger_kwargs)
    config = locals()
    del config['ps']

    agent = Model(args)
    keys = agent.get_weights()[0]
    weights = ray.get(ps.pull.remote(keys))
    agent.set_weights(keys, weights)

    test_env = args.env_fn()
    while True:
        avg_ret = agent.test_agent(test_env, args)
        logger.log_tabular('AverageTestEpRet', avg_ret)
        logger.log_tabular('Time', time.time() - start_time)
        logger.dump_tabular()
        weights = ray.get(ps.pull.remote(keys))
        agent.set_weights(keys, weights)
        print(avg_ret)
if __name__ == '__main__':

    parser = argparse.ArgumentParser("ppo paralization")
    parser.add_argument('--env', type=str, default='LunarLanderContinuous-v2')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.97)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--steps_per_epoch', type=int, default=4000)
    parser.add_argument('--exp_name', type=str, default='dppo_6worker')
    args = parser.parse_args()

    # ac_kwargs = dict()
    args.seed = 0
    args.env_fn = lambda: gym.make(args.env)
    args.total_steps = args.epochs * args.steps_per_epoch
    args.clip_ratio = 0.2
    args.pi_lr = 3e-4
    args.vf_lr = 1e-3
    args.train_pi_iters = 80
    args.train_v_iters = 80
    args.max_ep_len = 1000
    args.target_kl = 0.15

    env = args.env_fn()
    if isinstance(env.observation_space, gym.spaces.Discrete):
        args.act_dim = env.action_space.n
    else:
        args.act_dim = env.action_space.shape

    args.obs_dim = env.observation_space.shape[0]
    args.observation_space = env.observation_space
    args.action_space = env.action_space
    args.hidden_size = [args.hid] * args.l
    args.activation = nn.Tanh

    args.num_workers = 6
    args.num_learners = 1

    ray.init()

    net = Model(args)
    all_keys, all_values = net.get_weights()
    ps = ParameterServer.remote(all_keys, all_values)

    # replay_buffer = ReplayBuffer.remote(args.obs_dim, args.act_dim, args.replay_size)
    replay_buffer = ReplayBuffer.remote(env.observation_space.shape, env.action_space.shape, args.steps_per_epoch, args.num_workers,
                          args.gamma, args.lam)
    start_time = time.time()

    # Start some training tasks.
    task_rollout = [worker_rollout.remote(ps, replay_buffer, args, i) for i in range(args.num_workers)]  #同步

    time.sleep(10)

    task_train = [worker_train.remote(ps, replay_buffer, args) for i in range(args.num_learners)]

    time.sleep(10)

    task_test = worker_test.remote(ps, start_time, args)
    ray.wait(task_rollout)