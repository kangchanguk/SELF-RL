'''
김성훈 교수님 코드 참조
learning rate 사용
'''

import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register
import random as pr

register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4',
            'is_slippery':True}
)
env=gym.make('FrozenLake-v3')

Q= np.zeros([env.observation_space.n,env.action_space.n])#q테이블 생성
num_episodes = 2000
dis=0.99
a=0.1
rList = []
for i in range(num_episodes):
    state = env.reset()
    rAll = 0
    done = False
    e=1./((i/100)+1)
    while not done:
        if np.random.rand(1)<e:
            action = env.action_space.sample()
        else:
            action=np.argmax(Q[state, :])
        new_state, reward, done,_= env.step(action)
        Q[state,action] = (1-a)*Q[state,action]+a*(reward + dis * np.max(Q[new_state,:]))#learning rate=a, q를 a만큼만 신뢰한다.
        rAll +=reward
        state = new_state
    rList.append(rAll)

print("Success rate: " + str(sum(rList)/num_episodes))
print("Final q-table values")
print("left down right up")
print(Q)
plt.bar(range(len(rList)),rList, color="blue")
plt.show()
