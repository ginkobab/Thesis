import numpy as np
from nest import set_verbosity

from agent.ddpg import DDPGAgent
from env.neur_env import Neuron_env
from utils.utils import Recorder
from utils.utils import take_checkpoint, load_checkpoint

fixed = ['neuron', 'structure']

set_verbosity(30)

episodes = 50000
batch_size = 248



tau = 1e-3
buffer_maxlen = 100000
critic_lr = 1e-3
actor_lr = 1e-4

env = Neuron_env(fixed)
agent = DDPGAgent(env, tau, buffer_maxlen, critic_lr, actor_lr)
recorder = Recorder(env)
start = load_checkpoint(agent, recorder)

for i in range(100000):
    agent.update(batch_size)
    print(i, end='\r')

for i in range(10):
    state = env.reset()
    action = agent.get_action(state)
    reward, next_state = env.step(action)
    print('\n', i)
    print(reward)
    print(next_state)


#take_checkpoint(agent, recorder, episode)
print('done')




