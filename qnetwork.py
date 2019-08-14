'''
80x80의 흑백 맵 -> 2의 80*80제곱의 q테이블 크기 (너무 방대하다!!!)
table을 대체할 요소 network

state -> input layer -> hidden layer -> hidden layer -> output layer
-> quality(reward) for all actions

네트워크는 오직 state만 입력받는다.!!!

q network training(linear regression)

s를 입력하고 ws가 나온다면 cost = 1/m * (w-y)제곱 을 최소화 한다.
y=reward + dis*Q(S,A) 과 q prediction간의 차이를 최소화!!!
q preiction(s,a,O)-O는 weight,network
min(q(s,a,O)-(r+rmax(q(s,a,O)))제곱)

set y=  r (terminal)
        r+dis*maxQ(S,a) (non terminal)

neural network 에서는 deterministic 방식을 택하는 이유
(y-예상값)의 제곱를 minimize하는 것이 stachastic 방식을 택하는 것과 같은 방식!!!!!

(because)
correlations between samples
non-stationary targets
학습이 힘들다.!!!!!!

<구현>
입력과 출력을 먼저 설계한다.
state(입력 시 one hot 인코딩 방식을 사용한다!!!)
단어 집합의 크기를 벡터의 차원으로 하고 표현하고 싶은 단어의 인덱스에 1값을 부여하고 
다른 인덱스에는 0을 부여하는 벡터 표현 방식-one hot 인코딩방식

state: np.identity(16)[s1:s1+1]
identity함수는 대각선이 1인 16x16의 행렬을 만들어 준다.

input_size = env.observation_space.n
output_size = env.action_space.n

<net work>
x = tf.placeholder(shape=[1,input_size]-크기 16의 행렬로 주겠다,dtype=tf.float32) state input
w = tf.variable(tf.random_uniform([input_size, output_size],0,0.01))<-학습이 가능한 값
Qpred = tf.matmul(X,W)
y=tf.placeholder(shape=[1,ouput_size],dtype=tf.float32 )
loss = tf.reduce_sum(tf.square(Y-Qpred))
train=tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
(그래프에서 최솟값을 찾을 때 기울기를 보고 얼만큼을 가서 최솟값을 찾을 것을가를 표현한 것이 learning_rate)


'''
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def one_hot(x):
    return np.identity(16)[x:x+1]

env=gym.make('FrozenLake-v0')

input_size = env.observation_space.n
output_size = env.action_space.n
learning_rate = 0.1

X= tf.placeholder(shape=[1,input_size],dtype=tf.float32) #입력을 받는 x
W= tf.Variable(tf.random_uniform([input_size, output_size],0,0.01))# 네트워크 계층, x와 행렬곱 연산을 하면 Qpred이 나온다.

Qpred = tf.matmul(X,W)
Y = tf.placeholder(shape=[1,output_size],dtype=tf.float32)#출력값

loss = tf.reduce_sum(tf.square(Y-Qpred))
train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)#최소값을 찾아 나가는 연산 

dis = 0.99
num_episodes = 2000

rList = []
init = tf.global_variables_initializer()

with tf.Session() as sess: #세션 객체는 자신만의 물리적자원을 사용하기 때문에 블럭이 끝나면 자동으로 연결을 해제해주는 with 블럭내에서 해결하는 것이 좋다.
    sess.run(init)
    for i in range(num_episodes):
        s=env.reset()
        e=1./((i/50)+10)
        rAll=0
        done=False
        local_loss=[]

        while not done:
            Qs = sess.run(Qpred,feed_dict={X: one_hot(s)})
            if np.random.rand(1)<e:
                a= env.action_space.sample()
            else:
                a=np.argmax(Qs)
            
            s1, reward, done, _=env.step(a)    
            if done:
                Qs[0, a]=reward

            else:
                # 네트워크를 통해 S1의 입력값에 대한 Qprediction을 구한다
                Qs1=sess.run(Qpred, feed_dict={X: one_hot(s1)})
                # 내가 한 ACTION에 대한 Q값만을 업데이트
                Qs[0, a] = reward + dis * np.max(Qs1)
                # Qs는 첫번재 성분-2차원 1x4 array이기 때문!!
            sess.run(train, feed_dict={X: one_hot(s), Y: Qs})#feed_dict는 입력값과 출력값을 정해주는 함수
            rAll +=reward
            s = s1
        rList.append(rAll)
    
print("Percent of successful episodes:" +str(sum(rList)/num_episodes) + "%")
plt.bar(range(len(rList)), rList, color="blue")
plt.show()



 