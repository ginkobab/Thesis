import numpy as np


class Net_dict:

    def __init__(self, fixed=[]):
        self.load_parameters()
        self.fixed = fixed

        self.flat = {}
        self.nested = {}

        self.reset()

    def get_dict(self):
        self.build_nested()
        return self.nested

    def get_initial_values(self):
        free_values = []

        if 'synapse' not in self.fixed:
            free_values.extend(self.initial_values[:6])
        if 'neuron'  not in self.fixed:
            free_values.extend(self.initial_values[6:31])
        if 'structure' not in self.fixed:
            free_values.extend(self.initial_values[31:])

        return free_values

    def set_values(self, params):
        self.build_flat(params)

    def reset(self):
        self.build_flat(self.initial_values, resetting=True)
        self.build_nested()

    def build_nested(self):
        self.copy_synapse_params()
        self.copy_neuron_params()
        self.copy_connection_params()

        self.generate_mean_values()
        self.update_with_mean_values()

    def build_flat(self, parameters, resetting=False):
        params = parameters.copy()
        if 'synapse' not in self.fixed or resetting:
            self.set_synapse_params(params[0:6])
            del params[0:6]

        if 'neuron' not in self.fixed or resetting:
            self.set_neuron_params(params[0:9])
            self.set_voltage_mean(params[9:17])
            self.set_sd_mean(params[17:25])
            del params[0:25]

        if 'structure' not in self.fixed or resetting:
            self.set_connection_matrix(params[0:64])

    def set_synapse_params(self, params):
        for index, param_name in enumerate(self.synapse_params):
            self.flat[param_name] = params[index]

    def set_neuron_params(self, params):
        for index, param_name in enumerate(self.neuron_params):
            self.flat[param_name] = params[index]

    def set_voltage_mean(self, params):
        for i in range(8):
            self.flat['V0_mean_{}'.format(i)] = params[i]

    def set_sd_mean(self, params):
        for i in range(8):
            self.flat['V0_sd_{}'.format(i)] = params[i]

    def set_connection_matrix(self, params):
        i = 0
        for row in range(8):
            for col in range(8):
                self.flat['conn_probs_{}_{}'.format(row, col)] = params[i]
                i += 1

    def copy_synapse_params(self):
        for param in self.synapse_params:
            self.nested[param] = self.flat[param]

    def copy_neuron_params(self):
        self.nested['neuron_params'] = {neur_param: self.flat[neur_param] 
                                        for neur_param in self.neuron_params}

        voltage_mean = self.get_voltage_mean()
        voltage_sd = self.get_voltage_sd()

        self.nested['neuron_params']['V0_mean'] = voltage_mean
        self.nested['neuron_params']['V0_sd'] = voltage_sd

    def copy_connection_params(self):
        conn_values = np.array([self.flat['conn_probs_{}_{}'.format(row, col)]
                      for row in range(8) for col in range(8)]).reshape([8,8])

        self.nested['conn_probs'] = conn_values

    def get_voltage_mean(self):
        return {'original' :58.0, 
                'optimized': [self.flat['V0_mean_{}'.format(i)]
                              for i in range(8)]}
    def get_voltage_sd(self):
        return {'original' : 10.0, 
                'optimized': [self.flat['V0_sd_{}'.format(i)] 
                              for i in range(8)]}

    def generate_mean_values(self):

        self.nested.update(self.immutable)

        self.mean_params = {
        'PSP_mean_matrix': get_mean_PSP_matrix(
        self.nested['PSP_e'], self.nested['g'], len(self.nested['populations'])
        ),
        'PSP_std_matrix': get_std_PSP_matrix(
        self.nested['PSP_sd'], len(self.nested['populations'])
        ),
        'mean_delay_matrix': get_mean_delays(
        self.nested['mean_delay_exc'], self.nested['mean_delay_inh'],
        len(self.nested['populations'])
        ),
        'std_delay_matrix': get_std_delays(
        self.nested['mean_delay_exc'] * self.nested['rel_std_delay'],
        self.nested['mean_delay_inh'] * self.nested['rel_std_delay'],
        len(self.nested['populations'])
        )}

    def update_with_mean_values(self):
        self.nested.update(self.mean_params)

    def load_parameters(self):
        self.immutable = {
        'K_scaling': 0.1,
        'N_scaling': 0.1,
        'bg_rate': 8.,
        'poisson_input': True,
        'poisson_delay': 1.5,
        'V0_type': 'optimized',
        'neuron_model': 'iaf_psc_exp',
        'rec_dev': ['spike_detector'],
        'populations': 
        ['L23E', 'L23I', 'L4E', 'L4I', 'L5E', 'L5I', 'L6E', 'L6I'],
        'N_full': np.array([20683, 5834, 21915, 5479, 4850, 1065, 14395, 2948]),
        'K_ext': np.array([1600, 1500, 2100, 1900, 2000, 1900, 2900, 2100]),
        'full_mean_rates':
        np.array([0.971, 2.868, 4.746, 5.396, 8.142, 9.078, 0.991, 7.523])}

        self.initial_values = [0.15, 0.1, -4, 1.5, 0.75, 0.5, -65.0, -50.0,
                -65.0, 250.0, 10.0, 0.5, 0.5, 0.5, 2.0, -68.28, -63.16, -63.33,
                -63.4, -63.11, -61.66, -66.72, -61.43, 5.36, 4.57, 4.74, 4.94,
                4.94, 4.55, 5.46, 4.48, 0.1009, 0.1689, 0.0437, 0.0818, 0.0323,
                0., 0.0076, 0., 0.1346, 0.1371, 0.0316, 0.0515, 0.0755, 0.,
                0.0042, 0., 0.0077, 0.0059, 0.0497, 0.135, 0.0067, 0.0003,
                0.0453, 0., 0.0691, 0.0029, 0.0794, 0.1597, 0.0033, 0.,
                0.1057, 0., 0.1004, 0.0622, 0.0505, 0.0057, 0.0831, 0.3726,
                0.0204, 0., 0.0548, 0.0269, 0.0257, 0.0022, 0.06, 0.3158,
                0.0086, 0., 0.0156, 0.0066, 0.0211, 0.0166, 0.0572, 0.0197,
                0.0396, 0.2252, 0.0364, 0.001, 0.0034, 0.0005, 0.0277, 0.008,
                0.0658, 0.1443]

        self.synapse_params = ['PSP_e', 'PSP_sd', 'g', 'mean_delay_exc',
                            'mean_delay_inh', 'rel_std_delay']

        self.neuron_params = ['E_L', 'V_th', 'V_reset', 'C_m', 'tau_m', 
                              'tau_syn_ex', 'tau_syn_in', 'tau_syn_E', 't_ref']


def get_mean_delays(mean_delay_exc, mean_delay_inh, number_of_pop):
    dim = number_of_pop
    mean_delays = np.zeros((dim, dim))
    mean_delays[:, 0:dim:2] = mean_delay_exc
    mean_delays[:, 1:dim:2] = mean_delay_inh
    return mean_delays


def get_std_delays(std_delay_exc, std_delay_inh, number_of_pop):
    dim = number_of_pop
    std_delays = np.zeros((dim, dim))
    std_delays[:, 0:dim:2] = std_delay_exc
    std_delays[:, 1:dim:2] = std_delay_inh
    return std_delays


def get_mean_PSP_matrix(PSP_e, g, number_of_pop):
    dim = number_of_pop
    weights = np.zeros((dim, dim))
    exc = PSP_e
    inh = PSP_e * g
    weights[:, 0:dim:2] = exc
    weights[:, 1:dim:2] = inh
    weights[0, 2] = exc * 2
    return weights


def get_std_PSP_matrix(PSP_rel, number_of_pop):
    dim = number_of_pop
    std_mat = np.zeros((dim, dim))
    std_mat[:, :] = PSP_rel
    return std_mat

