import os
import sys
import numpy as np
from nest import set_verbosity

from agent.ddpg import DDPGAgent
from env.neur_env import Neuron_env
from utils.utils import Recorder
from utils.utils import take_checkpoint, load_checkpoint

fixed = []
if len(sys.argv) > 1:
    for i in sys.argv[1:]:
        fixed.append(i)

set_verbosity(30)

episodes = 50000
batch_size = 64



tau = 1e-3
buffer_maxlen = 100000
critic_lr = 1e-3
actor_lr = 1e-4

env = Neuron_env(fixed)
agent = DDPGAgent(env, tau, buffer_maxlen, critic_lr, actor_lr)
recorder = Recorder(env)
start = load_checkpoint(agent, recorder)

for episode in range(start, episodes + start):

    state = env.reset()
    action = agent.get_action(state)
    reward, next_state = env.step(action)
    
    agent.replay_buffer.push(state, action, reward)
    q_loss, policy_loss = agent.update(batch_size)

    recorder.push(episode, float(reward), *next_state, q_loss, policy_loss, *action)

    print('Episode ' + str(episode), end='\r')

    if episode % 490 == 0:
        agent.test = True
    if episode % 500 == 0:
        agent.test = False
        take_checkpoint(agent, recorder, episode)

