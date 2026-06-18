# Spring 2025, 535514 Reinforcement Learning
# HW2: DDPG

import sys
import gym
import numpy as np
import os
import time
import random
from collections import namedtuple
import torch
import torch.nn as nn
import wandb
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange 

# Configure a wandb log
# #wandb.login()
#run = wandb.init(
#    project="my-ddpg-project",  # Specify your project
#    config={                    # Track hyperparameters and metadata
#        "learning_rate": 0.01,
#    },
#)

# Define a tensorboard writer
#writer = SummaryWriter("./tb_record_3")

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

Transition = namedtuple(
    'Transition', ('state', 'action', 'mask', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class OUNoise:

    def __init__(self, dim, scale=0.2):           
        self.action_dimension = dim           
        self.scale = scale
        self.mu, self.theta, self.sigma = 0., 0.15, 0.2
        self.state = np.ones(self.action_dimension) * self.mu

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu
    def noise(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dimension)
        self.state += dx
        return self.state * self.scale  

class Actor(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Actor, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        ########## YOUR CODE HERE (5~10 lines) ##########
        # Construct your own actor network
        self.fc1 = nn.Linear(num_inputs, hidden_size)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_outputs)

        
        ########## END OF YOUR CODE ##########
        
    def forward(self, inputs):
        
        ########## YOUR CODE HERE (5~10 lines) ##########
        # Define the forward pass your actor network
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x))
        return action
        
        
        ########## END OF YOUR CODE ##########

class Critic(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Critic, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        ########## YOUR CODE HERE (5~10 lines) ##########
        # Construct your own critic network
        self.fc1 = nn.Linear(num_inputs + num_outputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)



        ########## END OF YOUR CODE ##########

    def forward(self, inputs, actions):
        
        ########## YOUR CODE HERE (5~10 lines) ##########
        # Define the forward pass your critic network
        x = torch.cat([inputs, actions], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value
        
        
        ########## END OF YOUR CODE ##########        
        

class DDPG(object):
    def to(self, device):
        self.actor.to(device)
        self.actor_target.to(device)
        self.critic.to(device)
        self.critic_target.to(device)
        return self
    def __init__(self, num_inputs, action_space, gamma=0.995, tau=0.002, hidden_size=256, lr_a=2e-4, lr_c=1e-4):

        self.num_inputs = num_inputs
        self.action_space = action_space

        self.actor = Actor(hidden_size, self.num_inputs, self.action_space)
        self.actor_target = Actor(hidden_size, self.num_inputs, self.action_space)
        self.actor_perturbed = Actor(hidden_size, self.num_inputs, self.action_space)
        self.actor_optim = Adam(self.actor.parameters(), lr=lr_a)

        self.critic = Critic(hidden_size, self.num_inputs, self.action_space)
        self.critic_target = Critic(hidden_size, self.num_inputs, self.action_space)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr_c)

        self.gamma = gamma
        self.tau = tau

        hard_update(self.actor_target, self.actor) 
        hard_update(self.critic_target, self.critic)


    def select_action(self, state, action_noise=None):
        self.actor.eval()
        mu = self.actor((Variable(state)))
        mu = mu.data
        
        ########## YOUR CODE HERE (3~5 lines) ##########
        # Add noise to your action for exploration
        # Clipping might be needed 
        action = mu.cpu().numpy()[0]
        if action_noise is not None:
            action += action_noise.noise()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return action
        ########## END OF YOUR CODE ##########


    def update_parameters(self, batch):
        state_batch  = torch.cat([t.state      for t in batch]).to(next(self.actor.parameters()).device)
        action_batch = torch.cat([t.action     for t in batch]).to(state_batch.device)
        reward_batch = torch.cat([t.reward     for t in batch]).unsqueeze(1).to(state_batch.device)
        mask_batch   = torch.cat([t.mask       for t in batch]).unsqueeze(1).to(state_batch.device)
        next_state_batch = torch.cat([t.next_state for t in batch]).to(state_batch.device)
        
        ########## YOUR CODE HERE (10~20 lines) ##########
        # Calculate policy loss and value loss
        # Update the actor and the critic
        with torch.no_grad():
            next_q = self.critic_target(next_state_batch, self.actor_target(next_state_batch))
            target_q = reward_batch + self.gamma * mask_batch * next_q
        current_q = self.critic(state_batch, action_batch)
        value_loss = F.mse_loss(current_q, target_q)
        self.critic_optim.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)  # 只在 TODO 內新增 grad clip
        self.critic_optim.step()

        actor_loss = -self.critic(state_batch, self.actor(state_batch)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)   # 同上
        self.actor_optim.step()
        ########## END OF YOUR CODE ########## 

        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return value_loss.item(), actor_loss.item()


    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        local_time = time.localtime()
        timestamp = time.strftime("%m%d%Y_%H%M%S", local_time)
        if not os.path.exists('preTrained/'):
            os.makedirs('preTrained/')

        if actor_path is None:
            actor_path = "preTrained/ddpg_cheetah_actor_{}_{}_{}".format(env_name, timestamp, suffix) 
        if critic_path is None:
            critic_path = "preTrained/ddpg_cheetah_critic_{}_{}_{}".format(env_name, timestamp, suffix) 
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path))
        if critic_path is not None: 
            self.critic.load_state_dict(torch.load(critic_path))

def train():    
    
    hidden_size = 256
    replay_size = 1000000
    batch_size = 256
    total_numsteps = 500000
    start_steps = 15000
    env_name = 'HalfCheetah-v4'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = gym.make(env_name)
    buf = ReplayMemory(replay_size)
    agent = DDPG(env.observation_space.shape[0], env.action_space,
             hidden_size=hidden_size).to(device)  
    noise = OUNoise(env.action_space.shape[0])
    writer = SummaryWriter('./runs/ddpg_cheetah')
    wandb.init(project='ddpg-cheetah', config={'total_steps':total_numsteps})

    s_np, _ = env.reset(seed=10)
    state = torch.tensor(s_np, dtype=torch.float32, device=device).unsqueeze(0) 
    ep_ret, ep, pbar = 0.0, 0, trange(1, total_numsteps + 1, desc='DDPG') 
    ewma_reward = 0.0
    ewma_alpha = 0.05 
    for step in pbar: 
        if step < start_steps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                a_net = agent.actor(state).cpu().numpy()[0]
            a_noise = noise.noise() if noise is not None else 0.0
            action  = np.clip(a_net + a_noise, env.action_space.low, env.action_space.high)

        ns_np, reward, done, trunc, _ = env.step(action)
        done = done or trunc
        next_state = torch.tensor(ns_np, dtype=torch.float32, device=device).unsqueeze(0)
        mask = torch.tensor([0.0], device=device) if done else torch.tensor([1.0], device=device)
        buf.push(state,
             torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0), 
             mask, next_state,
             torch.tensor([reward], dtype=torch.float32, device=device))  
    
        state = next_state
        ep_ret += reward
        if done:
            writer.add_scalar('return', ep_ret, ep)
            wandb.log({'reward': ep_ret, 'episode': ep})
            ewma_reward = ewma_alpha * ep_ret + (1 - ewma_alpha) * ewma_reward
            writer.add_scalar('return/ewma', ewma_reward, ep)
            wandb.log({'ewma_return': ewma_reward, 'episode': ep})
            ep += 1
            s_np, _ = env.reset()
            state = torch.tensor(s_np, dtype=torch.float32, device=device).unsqueeze(0)
            ep_ret, ep_len = 0.0, 0
            noise.reset()

        if step >= start_steps and len(buf) >= batch_size:
            c_loss, a_loss = agent.update_parameters(buf.sample(batch_size))
            pbar.set_postfix(ep_return=ep_ret, critic=f'{c_loss:.3f}', actor=f'{a_loss:.3f}') 
            writer.add_scalar('loss/critic', c_loss, step)
            writer.add_scalar('loss/actor',  a_loss, step)
            wandb.log({'critic_loss': c_loss, 'actor_loss': a_loss, 'step': step})

        if step % 10_000 == 0:
            print(f'step {step}/{total_numsteps}')
    
    agent.save_model(env_name, '.pth')   


if __name__ == '__main__':
    random_seed = 10  
    env = gym.make('HalfCheetah-v4')
    torch.manual_seed(random_seed) 
    train()


