# Spring 2025, 535514 Reinforcement Learning
# HW2: DDPG

import sys
import gym
import numpy as np
import os
import time
import random
from pathlib import Path
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

HW_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = HW_DIR / "checkpoints" / "ddpg_cdq_cheetah"
RUN_DIR = HW_DIR / "runs" / "ddpg_cdq_cheetah"

# Define a tensorboard writer
#writer = SummaryWriter(str(RUN_DIR))

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
        self.fc1 = nn.Linear(num_inputs, 400)
        self.ln1 = nn.LayerNorm(400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, num_outputs)

        
        ########## END OF YOUR CODE ##########
        
    def forward(self, inputs):
        
        ########## YOUR CODE HERE (5~10 lines) ##########
        # Define the forward pass your actor network
        x = F.relu(self.ln1(self.fc1(inputs)))
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
        self.fc1 = nn.Linear(num_inputs + num_outputs, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)


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
        for net in (self.actor, self.actor_t, self.Q1, self.Q1_t, self.Q2, self.Q2_t):
            net.to(device)
        self.low = self.low.to(device); self.high = self.high.to(device)
        return self
    
    def __init__(self, num_inputs, action_space, gamma=0.99, tau=0.005, hidden_size=512, lr_a=1e-4, lr_c=1e-3):

        self.low, self.high = torch.tensor(action_space.low), torch.tensor(action_space.high)

        self.actor     = Actor(hidden_size, num_inputs, action_space)
        self.actor_t   = Actor(hidden_size, num_inputs, action_space)
        self.actor_o   = Adam(self.actor.parameters(), lr=lr_a)

        

        self.Q1 = Critic(hidden_size, num_inputs, action_space)
        self.Q2 = Critic(hidden_size, num_inputs, action_space)
        self.Q1_t = Critic(hidden_size, num_inputs, action_space)
        self.Q2_t = Critic(hidden_size, num_inputs, action_space)
        self.Q1_o = Adam(self.Q1.parameters(), lr=lr_c)
        self.Q2_o = Adam(self.Q2.parameters(), lr=lr_c)

        self.g, self.tau, self.it = gamma, tau, 0
        soft_update(self.actor_t, self.actor, 1.0)
        soft_update(self.Q1_t, self.Q1, 1.0)
        soft_update(self.Q2_t, self.Q2, 1.0)


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


    def update_parameters(self, batch, pol_noise=0.2, noise_clip=0.5):
        state_batch  = torch.cat([t.state      for t in batch])
        action_batch = torch.cat([t.action     for t in batch])
        reward_batch = torch.cat([t.reward     for t in batch]).unsqueeze(1) * 0.1  
        mask_batch   = torch.cat([t.mask       for t in batch]).unsqueeze(1)
        next_state_batch = torch.cat([t.next_state for t in batch])
        ########## YOUR CODE HERE (10~20 lines) ##########
        # Calculate policy loss and value loss
        # Update the actor and the critic
        with torch.no_grad():
            # ---- target policy smoothing ---- #
            eps = (pol_noise*torch.randn_like(action_batch)).clamp(-noise_clip,noise_clip)
            na  = (self.actor_t(next_state_batch)+eps).clamp(self.low, self.high) 
            q1_t = self.Q1_t(next_state_batch,na)
            q2_t = self.Q2_t(next_state_batch,na)
            y = reward_batch + self.g*mask_batch*torch.min(q1_t,q2_t)   # CDQ target
        q1 = self.Q1(state_batch,action_batch); q2 = self.Q2(state_batch,action_batch)
        l1 = F.mse_loss(q1,y); l2 = F.mse_loss(q2,y)
        self.Q1_o.zero_grad(); l1.backward(); self.Q1_o.step()
        self.Q2_o.zero_grad(); l2.backward(); self.Q2_o.step()
        if self.it%2==0:
            a_loss = -self.Q1(state_batch, self.actor(state_batch)).mean()
            self.actor_o.zero_grad(); a_loss.backward(); self.actor_o.step()
        else:
            a_loss = torch.tensor(0.)
        ########## END OF YOUR CODE ########## 

        soft_update(self.actor_t, self.actor, self.tau)
        soft_update(self.Q1_t, self.Q1, self.tau)
        soft_update(self.Q2_t, self.Q2, self.tau)
        self.it+=1
        return (l1.item()+l2.item())*0.5, a_loss.item()


    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        local_time = time.localtime()
        timestamp = time.strftime("%m%d%Y_%H%M%S", local_time)
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

        if actor_path is None:
            actor_path = CHECKPOINT_DIR / "ddpg_cdq_cheetah_actor_{}_{}_{}".format(env_name, timestamp, suffix)
        if critic_path is None:
            critic_path = CHECKPOINT_DIR / "ddpg_cdq_cheetah_critic_{}_{}_{}".format(env_name, timestamp, suffix)
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

    replay_size = 1000000
    batch_size = 256
    total_numsteps = 500000
    start_steps = 10000
    env_name = 'HalfCheetah-v4'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = gym.make(env_name)
    buf = ReplayMemory(replay_size)
    agent = DDPG(env.observation_space.shape[0], env.action_space, hidden_size=256).to(device)
    noise = OUNoise(env.action_space.shape[0])
    writer = SummaryWriter(str(RUN_DIR))
    wandb.init(project='ddpg-cdq-cheetah', config={'total_steps':total_numsteps})

    s_np, _ = env.reset(seed=10)
    state = torch.tensor(s_np, dtype=torch.float32, device=device).unsqueeze(0) 
    ep_ret, ep, pbar = 0.0, 0, trange(1, total_numsteps + 1, desc='CDQ') 
    ewma_reward = 0.0
    ewma_alpha = 0.05  
    for step in pbar: 
        # ---------- 取得動作 ---------- #
        if step < start_steps:
            action = env.action_space.sample()
        else:
            # === 內嵌 act() 邏輯 (原先 agent.act) === # MOD
            with torch.no_grad():
                a_net = agent.actor(state).cpu().numpy()[0]
            a_noise = noise.noise() if noise is not None else 0.0
            action  = np.clip(a_net + a_noise, env.action_space.low, env.action_space.high)

        # ---------- 與環境互動 ---------- #
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
            wandb.log({'episode_return': ep_ret, 'episode': ep})
            ewma_reward = ewma_alpha * ep_ret + (1 - ewma_alpha) * ewma_reward
            writer.add_scalar('return/ewma', ewma_reward, ep)
            wandb.log({'ewma': ewma_reward, 'episode': ep})
            ep += 1; ep_ret = 0.0
            s_np, _ = env.reset()

        if step >= start_steps and len(buf) >= batch_size:
            c_loss, a_loss = agent.update_parameters(buf.sample(batch_size))
            writer.add_scalar('loss/critic', c_loss, step)
            writer.add_scalar('loss/actor',  a_loss, step)
            wandb.log({'critic_loss': c_loss, 'actor_loss': a_loss, 'step': step})

        if step % 10_000 == 0:
            print(f'step {step}/{total_numsteps}')
        
    agent.save_model(env_name, '.pth')   



if __name__ == '__main__':
    # For reproducibility, fix the random seed
    random_seed = 10  
    env = gym.make('HalfCheetah-v4')
    torch.manual_seed(random_seed) 
    train()


