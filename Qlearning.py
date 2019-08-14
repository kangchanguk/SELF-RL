'''
q learning 에서는 한 state에서 다음 action을 선택할 때 1이 있다면 무조건
1을 따라가는 방식을 택하는 데 이러한 방식은 무조건 아는 길로만 간다는 단점이 존재한다.
따라서 exploit과 exploration 방식을 적용
일정한 방식으로 랜덤하게 action 선택, noise를 섞어서 선택
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

Q= np.zeros([env.observation_space.n,env.action_space.n])
num_episodes = 2000
dis = .99

rList = []
for i in range(num_episodes):
    state = env.reset()
    rAll = 0
    done = False

    while not done:
        action = np.argmax(Q[state, :]+np.random.randn(1, env.action_space.n)/(i+1)) #noise를 섞는다.
        new_state, reward, done,_= env.step(action)
        Q[state,action] = reward + dis * np.max(Q[new_state,:])
        rAll +=reward
        state = new_state
    rList.append(rAll)

print("Success rate: " + str(sum(rList)/num_episodes))
print("Final q-table values")
print("left down right up")
print(Q)
plt.bar(range(len(rList)),rList, color="blue")
plt.show()
