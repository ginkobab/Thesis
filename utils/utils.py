import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class Recorder:
    def __init__(self, env):
        action_list = get_action_list(env)
        self.arguments = ['episode', 'reward', 'layer_23', 'layer_4',
                          'layer_5', 'layer_6', 'q_loss', 'policy_loss',
                          *action_list]

        self.df = pd.DataFrame(columns=self.arguments)

    def push(self, *values):
        index = len(self.df.index) + 1
        self.df.loc[index] = values

    def export(self):
        self.clean_df()
        self.save_plot()
        self.df.to_csv('checkpoints/recorded_data.csv', index=False)

    def save_plot(self):
        fig, axis = plt.subplots(2,4) 
        idx = get_axis_index()
        for n, arg in enumerate(self.arguments[1:8]):
            sns.scatterplot(x='episode', y=arg, data=self.df, ax=axis[idx[n]])

        fig.savefig('checkpoints/plots.png')
        plt.close()

    def clean_df(self):
        self.df = self.df.fillna(0.0)
        self.df.reward = self.df.loc[:, 'reward'].apply(lambda x: x[0])

    def load_df(self, path):
        self.df = pd.read_csv('checkpoints/recorded_data.csv')


def take_checkpoint(agent, recorder, episode):
    recorder.export()
    agent.save_checkpoint(episode)

def load_checkpoint(agent, recorder):
    episode = 1
    if os.path.isfile('checkpoints/model.pth.tar'):
        episode += agent.load_checkpoint('checkpoints/model.pth.tar')
        recorder.load_df


    return episode

def get_axis_index():
    index = []
    for r in range(2):
        for c in range(4):
            index.append((r,c))
    return index

def get_action_list(env):
    return ['action_{}'.format(n) for n in range(len(env.mutable_params))]
