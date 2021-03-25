# rl_practice
## repetition
### ppo
#### problem1: how to compute advantage function
#### problem2: how to build pi actor (backprogation through Nomal and Categorical)
#### problem3: change value when episode is done
### ddpg
#### problem1: how to create and update target q and target pi(copy.deepcopy and model.parameters.data.add_(mul_) )
#### problem2: sample strategy 
#### problem3: update strategy
### td3
#### problem1: different activation nn.Relu and nn.Tanh, (leads to act explode)
#### problem2: how to compose different parameters(itertool.chain)
#### problem3: clip, np.clip, torch.clamp()
#### problem4: min, torch.min(tensor1, tensor2)
#### problem5: sample, normal: np.random.randn(shape), torch.rand_like(shape)
#### problem6: update strategy, every 50 steps for q, every 100 steps for pi and target
#### problem7: target policy smoothing, cilp  when compute backup(y), add gaussion nosie to act