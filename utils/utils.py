import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import namedtuple


class Recorder:
    def __init__(self, arguments):
        self.recorder = []
        self.arguments = ['episode', 'reward', 'layer_23', 'layer_4', 'layer_5', 'layer_6', 'q_loss', 
                          'policy_loss', *arguments]
        self.namedtuple = namedtuple('State', self.arguments)
        

    def push(self, *args):
        self.recorder.append(self.namedtuple(*args))

    def export(self, path):
        full_data = self.namedtuple(*zip(*self.recorder))
        self.df = pd.DataFrame(columns=self.arguments)
        for n, i in enumerate(self.arguments):
            self.df[i] = full_data[n]
        # self.df = self.df.dropna()
        self.df = self.df.fillna(0.0)
        self.df.reward = self.df.loc[:, 'reward'].apply(lambda x: x[0])

        ax_index, rows, cols = get_ax_index(len(self.arguments[1:8]))
        fig, ax = plt.subplots(rows, cols) 
        for n, arg in enumerate(self.arguments[1:8]):
            sns.scatterplot(x='episode', y=arg, data=self.df, ax=ax[ax_index[n]])

        self.df.to_csv(path + 'recorded_data.csv', index=False)
        fig.savefig(path + 'plots.png')

        plt.close()
        del self.df



def take_checkpoint(agent, _recorder, path, episode):
    _recorder.export(path)
    agent.save_checkpoint(path, episode)


def get_ax_index(total_elements):
    root = int(round(np.sqrt(total_elements))) # this will round down
    if root**2 < total_elements:
        x, y = root + 1, root
    else:
        x, y = root, root


    ax_index = []
    for row in range(x):
        for col in range(y):
            ax_index.append((row, col))
            
    return ax_index, x, y



