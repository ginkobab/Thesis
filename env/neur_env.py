from os import listdir
import nest
import numpy as np
from env.example import run_simulation
from env.network_params import net_dict, net_states

class Space(object):
    def __init__(self, shape):
        self.shape = shape


class Neuron_env(object): 

    def __init__(self): 
        self.action_space = Space((len(net_states),))
        self.observation_space = Space((1,))

        self.backup_parameters = net_states.copy()
        self.full_parameters = net_dict.copy()
        self.mutable_parameters = net_states.copy() 

        self.target_values = np.array([0.3, 1.4, 2.5, 0.5])
    
    def reset(self):
        self.mutable_parameters = self.backup_parameters.copy()
        self.full_parameters.update(self.backup_parameters)

        return np.asarray(1.0, dtype='f').reshape([-1,])

    def step(self, action):
        self.reset()
        self._act(action)
        try: 
            fire_rates = self._simulate()
            reward = self._get_reward(fire_rates)
        except IndexError:
            fire_rates = np.zeros([4])
            reward = np.array(-20, dtype='f').reshape([1,])

        return reward, fire_rates 

    def _act(self, action):
        for index, (key, value) in enumerate(self.mutable_parameters.items()):
            self.mutable_parameters[key] = value + action[index] * value / 10
        
        self.full_parameters.update(self.mutable_parameters)

    def _get_reward(self, next_state):
        reward = -np.sum(np.abs(self.target_values - next_state))
        reward = np.asarray(reward, dtype='f').reshape([1,])

        return reward

    def _simulate(self):
        fire_rates = run_simulation(self.full_parameters)

        return fire_rates

