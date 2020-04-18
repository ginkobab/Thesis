import os
import gc
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple
from agent.models import Critic, Actor
import random

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward'))


class Buffer(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        if any(i is None for i in args):
            return
        transit = Transition(*(torch.from_numpy(i).to('cuda') for i in args))
        self.memory[self.position] = transit
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return (len(self.memory))


class DDPGAgent:

    def __init__(self, env, buffer_maxlen,
                 critic_learning_rate, actor_learning_rate):
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")

        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        self.env = env
        self.test = False
        self.noise = 1.0
        self.noise_decay = 0.9999

        self.critic = Critic(self.obs_dim, self.action_dim).to(self.device)
        self.actor = Actor(self.obs_dim, self.action_dim).to(self.device)

        self.critic_optimizer = optim.Adam(self.critic.parameters(),
                                           lr=critic_learning_rate,
                                           weight_decay=0.01)
        self.actor_optimizer = optim.Adam(self.actor.parameters(),
                                          lr=actor_learning_rate,
                                          weight_decay=0.01)

        self.replay_buffer = Buffer(buffer_maxlen)

    def get_action(self, obs):
        state = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        action = self.actor(state)
        if not self.test:
            action = action + (torch.randn_like(action) * self.noise)
        action = action.squeeze(0).cpu().detach().numpy()

        return action

    def update(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return 0.0, 0.0

        transitions = self.replay_buffer.sample(batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state).view(-1,
                                                  self.obs_dim).to(self.device)
        action_batch = torch.cat(batch.action).view(-1,
                                                    self.action_dim).to(
                                                                self.device)
        reward_batch = torch.cat(batch.reward).view(-1, 1).to(self.device)

        q_values = self.critic(state_batch, action_batch)
        q_targets = reward_batch

        q_loss = F.mse_loss(q_values, q_targets.detach())
        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()

        policy_loss = -self.critic(state_batch, self.actor(state_batch)).mean()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.noise *= self.noise_decay

        return q_loss.item(), policy_loss.item()

    def save_checkpoint(self, last_timestep):
        checkpoint_name = 'checkpoints/model.pth.tar'
        checkpoint = {
            'last_timestep': last_timestep,
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'replay_buffer': self.replay_buffer,
        }
        torch.save(checkpoint, checkpoint_name)
        gc.collect()

    def load_checkpoint(self, checkpoint_path):

        if os.path.isfile(checkpoint_path):
            key = 'cuda' if torch.cuda.is_available() else 'cpu'
            checkpoint = torch.load(checkpoint_path, map_location=key)
            start_timestep = checkpoint['last_timestep']
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic_optimizer.load_state_dict(
                                  checkpoint['critic_optimizer'])
            self.replay_buffer = checkpoint['replay_buffer']
            self.noise = self.noise_decay ** start_timestep

            gc.collect()
            self.actor.eval()
            self.critic.eval()

        else:
            raise OSError('Checkpoint not found')
        return start_timestep
