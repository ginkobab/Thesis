import numpy as np
import env.network as network
from env.sim_params import sim_dict
from env.stimulus_params import stim_dict

def run_simulation(net_dict, plot=False):
    net = network.Network(sim_dict, net_dict, stim_dict)
    net.setup()
    net.simulate()

    raster_plot_time_idx = np.array(
        [stim_dict['th_start'] - 100.0, stim_dict['th_start'] + 100.0]
        )
    fire_rate_time_idx = np.array([500.0, sim_dict['t_sim']])
    net.evaluate(raster_plot_time_idx, fire_rate_time_idx, plot)
    rates = []
    for i in range(0,7,2):
        rates.append(np.mean(np.load('data/rate{}.npy'.format(i))))

    return np.array(rates)
