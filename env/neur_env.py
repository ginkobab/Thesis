from os import listdir
import nest
import numpy as np
from env.example import run_simulation
from env.network_params import Net_dict
from env.helpers import Space


class Neuron_env(object): 

    def __init__(self, fixed=[], plot=False): 
        self.network_dict = Net_dict(fixed=fixed)
        self.mutable_params = self.network_dict.get_initial_values().copy()

        self.action_space = Space((len(self.mutable_params),))
        self.observation_space = Space((1,))

        self.target_values = np.array([0.3, 1.4, 2.5, 0.5])
        self.plot = plot
    
    def reset(self):
        self.network_dict.reset()
        self.mutable_params = self.network_dict.get_initial_values().copy()

        return np.asarray(1.0, dtype='f').reshape([-1,])

    def step(self, action):
        self.reset()
        self._act(action)
        reward, fire_rates = self._try_simulation()
        return reward, fire_rates

    def _act(self, action):
        for i in range(len(action)):
            self.mutable_params[i] += (action[i] * self.mutable_params[i] / 5)
        
        self.network_dict.set_values(self.mutable_params)

    def _try_simulation(self):
        try: 
            fire_rates = self._simulate()
        except Exception:
            fire_rates = np.zeros([4])

        reward = self._get_reward(fire_rates)

        return reward, fire_rates 

    def _get_reward(self, next_state):
        reward = -np.sum(np.abs(self.target_values - next_state))
        reward = np.asarray(reward, dtype='f').reshape([1,])

        return reward

    def _simulate(self):
        fire_rates = run_simulation(self.network_dict.get_dict(), plot=self.plot)

        return fire_rates
