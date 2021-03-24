import ray
import torch
from absl import flags
import argparse
import numpy as np
import scipy

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

    def __init__(self, obs_dim, act_dim, size, lam=0.05, gamma=0.99):
        self.ob = np.zeros(combine_shape(size, obs_dim), dtype=np.float32)
        self.act = np.zeros(combine_shape(size, act_dim), dtype=np.float32)
        self.val = np.zeros(size, dtype=np.float32)
        self.rew = np.zeros(size, dtype=np.float32)
        self.logp = np.zeros(size, dtype=np.float32)
        self.ret = np.zeros(size, dtype=np.float32)
        self.start_ptr, self.ptr, self.size = 0, 0, size
        self.gamma, self. lam = gamma, lam

    def store(self, ob, act, val, rew, logp):
        assert self.ptr < self.size
        self.ob[self.ptr] = ob
        self.act[self.ptr] = act
        self.val[self.ptr] = val
        self.rew[self.ptr] = rew
        self.logp[self.ptr] = logp
        self.ptr+=1

    def finish_path(self,last_val=0):
        path_index = slice(self.start_ptr, self.ptr)
        rews = np.append(self.rew[path_index], last_val)
        vals = np.append(self.val[path_index], last_val)
        # self.adv[path_index] = advantage_estimate(self.val[path_index], self.rew[path_index], self.lam, self.gamma)
        delta = rews[:-1] + vals[:-1] - vals[1:]*self.gamma
        self.adv[path_index] = discount_cumsum(delta, self.gamma*self.lam)
        # self.ret[path_index] = reward_to_go(self.rew, gamma=self.gamma)
        self.ret[path_index] = discount_cumsum(rews, self.gamma)[:-1]
        self.start_ptr = self.ptr

    def get(self):
        assert self.ptr == self.size
        self.start_ptr, self.ptr = 0, 0
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
    def __init__(self, parameters):
        self.params = parameters

    def update_param(self, grad):
        self.params += grad

    def get_param(self):
        return self.params

@ray.remote(num_gpus=1, max_call=1)
def worker_train(ps, replay_buffer, opt, leaner_index):
    #build learner network
    #pull weights from ps
    #for loop
      #compute grad
      #update ps parameters
    pass

@ray.remote
def worker_rollout(ps, replay_buffer, opt, worker_index):
    #build rollout network
    #pull weights from ps
    #for loop
        #interact with env and
        #store in replay_buffer
    # model =
    pass

@ray.remote
def worker_test(ps, replay_buffer, opt, worker_index=0):
    #build test network
    #pull weights prom ps and do test
    #save model optionally
    pass

if __name__ == '__main__':

    parser = argparse.ArgumentParser("ppo paralization")
    parser.add_argument()
    # create the parameter server
    if FLAGS.is_restore == "True":
        ps = ParameterServer.remote([], [], is_restore=True)
    else:
        net = Learner(opt, job="main")
        all_keys, all_values = net.get_weights()
        ps = ParameterServer.remote(all_keys, all_values)

    # create replay buffer
    replay_buffer = ReplayBuffer.remote(obs_dim=opt.obs_dim, act_dim=opt.act_dim, size=opt.replay_size)

    # Start some rollout tasks.
    task_rollout = [worker_rollout.remote(ps, replay_buffer, opt, i) for i in range(FLAGS.num_workers)]

    time.sleep(5)

    # start training tasks
    task_train = [worker_train.remote(ps, replay_buffer, opt, i) for i in range(FLAGS.num_learners)]

    # start testing
    task_test = worker_test.remote(ps, replay_buffer, opt)

    # wait util task test end
    # Keep the main process running. Otherwise everything will shut down when main process finished.
    ray.wait([task_test, ])