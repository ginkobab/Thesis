from agent.ddpg import DDPGAgent
from env.neur_env import Neuron_env
from utils.utils import Recorder
from utils.utils import take_checkpoint, load_checkpoint

fixed = ['neuron', 'synapse']

buffer_maxlen = 100000
critic_lr = 1e-3
actor_lr = 1e-4

env = Neuron_env(fixed, plot=True)
agent = DDPGAgent(env, buffer_maxlen, critic_lr, actor_lr)
recorder = Recorder(env)
start = load_checkpoint(agent, recorder)
agent.test = True


state = env.reset()
action = agent.get_action(state)
reward, next_state = env.step(action)
print('\n')
print(reward)
print(next_state)
