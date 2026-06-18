# Spring 2025, 535514 Reinforcement Learning
# HW1: REINFORCE with baseline and GAE

import os
from pathlib import Path
import gym
from itertools import count
from collections import namedtuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.optim.lr_scheduler as Scheduler
from torch.utils.tensorboard import SummaryWriter

torch.set_default_dtype(torch.float64)

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
HW_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = HW_DIR / "checkpoints" / "reinforce_baseline"
RUN_DIR = HW_DIR / "runs" / "reinforce_baseline"
writer = SummaryWriter(str(RUN_DIR))

def gym_reset(env, seed=None):
    try:
        result = env.reset(seed=seed) if seed is not None else env.reset()
        return result[0] if isinstance(result, tuple) else result
    except TypeError:
        return env.reset()

def gym_step(env, action):
    result = env.step(action)
    if len(result) == 5:
        next_state, reward, terminated, truncated, info = result
        done = terminated or truncated
    else:
        next_state, reward, done, info = result
    return next_state, reward, done, info

class Policy(nn.Module):
    """
        Implement both policy network and the value network in one model
        - Note that here we let the actor and value networks share the first layer
        - Feel free to change the architecture (e.g. number of hidden layers and the width of each hidden layer) as you like
        - Feel free to add any member variables/functions whenever needed
        TODO:
            1. Initialize the network (including the GAE parameters, shared layer(s), the action layer(s), and the value layer(s))
            2. Random weight initialization of each layer
    """
    def __init__(self):
        super(Policy, self).__init__()

        # Extract the dimensionality of state and action spaces
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.hidden_size = 256
        self.double()  

        ########## YOUR CODE HERE (5~10 lines) ##########
        self.fc_shared = nn.Linear(self.observation_dim, self.hidden_size)
        nn.init.kaiming_uniform_(self.fc_shared.weight, nonlinearity='relu')
        self.action_head = nn.Linear(self.hidden_size, self.action_dim)
        self.value_head = nn.Linear(self.hidden_size, 1)
        nn.init.xavier_uniform_(self.action_head.weight)
        nn.init.xavier_uniform_(self.value_head.weight)
        ########## END OF YOUR CODE ##########

        # action & reward memory
        self.saved_actions = []
        self.rewards = []

    def forward(self, state):
        """
            Forward pass of both policy and value networks
            - The input is the state, and the outputs are the corresponding 
              action probability distirbution and the state value
            TODO:
                1. Implement the forward pass for both the action and the state value
        """

        ########## YOUR CODE HERE (3~5 lines) ##########
        x = F.relu(self.fc_shared(state))
        action_probs = F.softmax(self.action_head(x), dim=-1)
        state_value = self.value_head(x)
        ########## END OF YOUR CODE ##########

        return action_probs, state_value

    def select_action(self, state):
        """
            Select the action given the current state
            - The input is the state, and the output is the action to apply 
            (based on the learned stochastic policy)
            TODO:
                1. Implement the forward pass for both the action and the state value
        """
        ########## YOUR CODE HERE (3~5 lines) ##########
        state = torch.from_numpy(state).double().unsqueeze(0)
        action_probs, state_value = self.forward(state)
        m = Categorical(action_probs)
        action = m.sample()
        ########## END OF YOUR CODE ##########
    
        # save to action buffer
        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))
        return action.item()

    def calculate_loss(self, gamma=0.99):
        """
            Calculate the loss (= policy loss + value loss) to perform backprop later
            TODO:
                1. Calculate rewards-to-go required by REINFORCE with the help of self.rewards
                2. Calculate the policy loss using the policy gradient
                3. Calculate the value loss using either MSE loss or smooth L1 loss
        """

        # Initialize the lists and variables
        R = 0
        
        policy_losses = []
        value_losses = []
        returns = []
        ########## YOUR CODE HERE (8-15 lines) ##########
        for r in self.rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float64)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        for (log_prob, value), R_val in zip(self.saved_actions, returns):
            advantage = R_val - value.squeeze()
            policy_losses.append(-log_prob * advantage)
            # 使用 smooth L1 loss (Huber loss) 平滑 value loss
            value_losses.append(F.smooth_l1_loss(value.squeeze(), torch.tensor([R_val], dtype=torch.float64)))
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        
        ########## END OF YOUR CODE ##########

        return loss

    def clear_memory(self):
        # reset rewards and action buffer
        self.saved_actions.clear()
        self.rewards.clear()

def train(lr=0.0005, gamma=0.99, weight_decay=1e-3):
    model = Policy()
    # Instantiate the policy model and the optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Learning rate scheduler (optional)
    # scheduler = Scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

    # EWMA reward for tracking the learning progress
    ewma_reward = 0

    # run inifinitely many episodes
    for i_episode in count(1):
        # reset environment and episode reward
        state = gym_reset(env)
        done = False
        ep_reward = 0
        t=0
        # Uncomment the following line to use learning rate scheduler
        # scheduler.step()

        # For each episode, only run 9999 steps to avoid entering infinite loop during the learning process
        
        ########## YOUR CODE HERE (10-15 lines) ##########
        while not done:
            action = model.select_action(state)
            state, reward, done, _ = gym_step(env, action)
            model.rewards.append(reward)
            ep_reward += reward
            t+=1
        loss = model.calculate_loss(gamma)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.clear_memory()
        
        ########## END OF YOUR CODE ##########
        
        # update EWMA reward and log the results
        ewma_reward = 0.05 * ep_reward + 0.95 * ewma_reward
        print('Episode {}\tlength: {}\treward: {}\t ewma reward: {}'.format(i_episode, t, ep_reward, ewma_reward))
        
        #Try to use Tensorboard to record the behavior of your implementation 
        ########## YOUR CODE HERE (4-5 lines) ##########
        writer.add_scalar('Loss', loss.item(), i_episode)
        writer.add_scalar('Reward', ep_reward, i_episode)
        writer.add_scalar('Episode_length', t, i_episode)
        writer.add_scalar('EWMA_reward', ewma_reward, i_episode)
        ########## END OF YOUR CODE ##########

        # check if we have "solved" the cart pole problem, use 120 as the threshold in LunarLander-v2
        
        if ewma_reward > 200:
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), CHECKPOINT_DIR / f"LunarLander_baseline_{lr}.pth")
            print(f"🎉 Solved in Episode {i_episode}!")
            break

def test(model_path, n_episodes=10):
    """
        Test the learned model (no change needed)
    """     
    model = Policy()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    for ep in range(1, n_episodes + 1):
        state = gym_reset(env)
        done = False
        reward_total = 0
        while not done:
            action = model.select_action(state)
            state, reward, done, _ = gym_step(env, action)
            reward_total += reward
        print(f"[Test Episode {ep}] Total Reward: {reward_total:.2f}")
    env.close()

if __name__ == '__main__':
    # For reproducibility, fix the random seed
    random_seed = 10
    lr = 0.0005 
    env = gym.make("LunarLander-v2")
    torch.manual_seed(random_seed)

    train(lr=lr)
    test(CHECKPOINT_DIR / f"LunarLander_baseline_{lr}.pth")
