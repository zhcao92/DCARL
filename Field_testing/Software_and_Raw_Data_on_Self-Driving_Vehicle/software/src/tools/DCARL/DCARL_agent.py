import gym
import gym_routing

import numpy as np
import sys

if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

if '/opt/ros/melodic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')

from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN



zzz_env = gym.make('zzz_lane-v0')


try:
    model = DQN.load("cz_deepq_0318",env=zzz_env)
    print("load saved model")
except:
    model = DQN(MlpPolicy, env=zzz_env) #, verbose=1
    print("build new model")

model.learn(total_timesteps=1000000)
model.save("cz_deepq_0318")

del model # remove to demonstrate saving and loading


print("load model to test")
model = DQN.load("cz_deepq_0318")

obs = zzz_env.reset()


while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = zzz_env.step(action)
    #zzz_env.render()
