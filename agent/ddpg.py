import os
import gc
import torch
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from collections import namedtuple
from agent.models import Critic, Actor
import random

Transition = namedtuple('Transition',
                       ('state', 'action','reward'))

class Buffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*(torch.from_numpy(i) for i in args))
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return (len(self.memory))



class DDPGAgent:
    
    def __init__(self, env, tau, buffer_maxlen, critic_learning_rate, actor_learning_rate):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        
        # hyperparameters
        self.env = env
        self.tau = tau
        self.test = False
        self.noise = 1.0
        self.noise_decay = 0.9999
        
        # initialize actor and critic networks
        self.critic = Critic(self.obs_dim, self.action_dim).to(self.device)
        self.critic_target = Critic(self.obs_dim, self.action_dim).to(self.device)
        
        self.actor = Actor(self.obs_dim, self.action_dim).to(self.device)
        self.actor_target = Actor(self.obs_dim, self.action_dim).to(self.device)
    
        # Copy critic target parameters
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        
        # optimizers
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate, weight_decay=0.01)
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_learning_rate, weight_decay=0.01)
    
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
            return None, None

        transitions = self.replay_buffer.sample(batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state).view(-1, self.obs_dim).to(self.device)
        action_batch = torch.cat(batch.action).view(-1, self.action_dim).to(self.device)
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

        # update target networks 
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
       
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        self.noise *= self.noise_decay


        return q_loss.item(), policy_loss.item()


    def save_checkpoint(self, checkpoint_dir):
        checkpoint_name = checkpoint_dir + 'model.pth.tar'
        checkpoint = {
            # 'last_timestep': last_timestep,
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'replay_buffer': self.replay_buffer,
        }
        torch.save(checkpoint, checkpoint_name)
        gc.collect()


    def load_checkpoint(self, checkpoint_path=None):

        if os.path.isfile(checkpoint_path):
            key = 'cuda' if torch.cuda.is_available() else 'cpu'

            checkpoint = torch.load(checkpoint_path, map_location=key)
            # start_timestep = checkpoint['last_timestep'] + 1
            self.actor.load_state_dict(checkpoint['actor']).to(self.device)
            self.critic.load_state_dict(checkpoint['critic']).to(self.device)
            self.actor_target.load_state_dict(checkpoint['actor_target']).to(self.device)
            self.critic_target.load_state_dict(checkpoint['critic_target']).to(self.device)
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
            self.replay_buffer = checkpoint['replay_buffer']

            gc.collect()
        else:
                raise OSError('Checkpoint not found')
