import gym
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import random

learning_rate = 1e-1
dis = 0.99
wholeepisode=10000

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.n1 = nn.Linear(4,256)
        self.n2 = nn.Linear(256,2)
    
    def forward(self,x):
        x=F.relu(self.n1(x))
        x=self.n2(x)
        return x
    
    def sample_action(self,obs,epsilon):
        out = self.forward(obs)
        coin =random.random()
        if coin < epison:
            return random.randint(0,1)
        else:
            return out.argmax.item()

def main():
    env=gym.make('CartPole-v0')
    q=Qnet()
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)
    for i in range(wholeepisode):
        epsilon= e=1./((i/50)+10)
        s=env.reset()
        Rall=0
        done=False
        local_loss=[]
        while not done:
            a=q.sample_action(torch.from_numpy(s).float,epsilon)
            s,r,done,info=env.step(a)
            q_out = q(s)
            q_a = q_out.gather(1,a)
            max_q = q(s).max(1)[0].unsqueeze(1)
            target = r+dis*max_q
            loss = F.smooth_l1_loss(q_a,target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            Rall+=r
        
        env.render()
        print("episode:{}, avg score:{}".i,rall)
        
    env.close()

if __name__ == '__main__':
    main()
