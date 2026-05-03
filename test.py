from gymnasium.spaces import Discrete
import numpy as np

s = Discrete(5, seed=42)
res = []

for i in range(1000):
    res.append(s.sample(probability=np.array([0,0,0.5,0.5,0])))

print(np.average(res))

from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv
from gymnasium.spaces.utils import flatdim
env = FrozenLakeEnv()
print(flatdim(env.action_space))