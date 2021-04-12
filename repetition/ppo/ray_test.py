import ray

@ray.remote
class Counter(object):
    def __init__(self, size):
        self.array = [0] * size

    def plus(self, woker_index):
        self.array[woker_index] += 1

    def get(self):
        return self.array

import time
@ray.remote
def f(k, index):
    for i in range(100):
        k.plus.remote(index)
    return
if __name__ == '__main__':
    ray.init()
    k = Counter.remote(10)
    task = [f.remote(k, i) for i in range(10)]
    # time.sleep(0.0001)
    # ray.wait(task)
    ray.get(task)
    print(ray.get(k.get.remote()))
    # results = ray.get([f.remote() for i in range(4)])
    # print(results)