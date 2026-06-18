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

# Define a useful tuple (optional)
SavedAction = namedtuple('SavedAction', 'log_prob')

HW_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = HW_DIR / "checkpoints" / "reinforce"
RUN_DIR = HW_DIR / "runs" / "reinforce"

# Define a tensorboard writer
writer = SummaryWriter(str(RUN_DIR))
        
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
        self.saved_actions = []
        # Extract the dimensionality of state and action spaces
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n if self.discrete else env.action_space.shape[0]
        self.hidden_size = 128
        self.double()
        
        ########## YOUR CODE HERE (5~10 lines) ##########
        self.fc_shared = nn.Linear(self.observation_dim, self.hidden_size)
        nn.init.kaiming_uniform_(self.fc_shared.weight, nonlinearity='relu')
        self.action_head = nn.Linear(self.hidden_size, self.action_dim)
        nn.init.xavier_uniform_(self.action_head.weight)
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
        action_prob = F.softmax(self.action_head(x), dim=-1)
        

        ########## END OF YOUR CODE ##########

        return action_prob

    def select_action(self, state):
        """
            Select the action given the current state
            - The input is the state, and the output is the action to apply 
            (based on the learned stochastic policy)
            TODO:
                1. Implement the forward pass for both the action and the state value
        """
        
        ########## YOUR CODE HERE (3~5 lines) ##########
        state = torch.from_numpy(state).float().unsqueeze(0)
        action_prob= self.forward(state)
        m = Categorical(action_prob)
        action = m.sample()
        ########## END OF YOUR CODE ##########
        
        # save to action buffer
        self.saved_actions.append(SavedAction(m.log_prob(action)))

        return action.item()


    def calculate_loss(self, gamma=0.999):
        """
            Calculate the loss (= policy loss + value loss) to perform backprop later
            TODO:
                1. Calculate rewards-to-go required by REINFORCE with the help of self.rewards
                2. Calculate the policy loss using the policy gradient
                3. Calculate the value loss using either MSE loss or smooth L1 loss
        """
        
        # Initialize the lists and variables
        R = 0
        saved_actions = self.saved_actions
        policy_losses = [] 
        value_losses = [] 
        returns = []

        ########## YOUR CODE HERE (8-15 lines) ##########
        for r in self.rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = returns - returns.mean()
        for saved_action, R in zip(self.saved_actions, returns):
            log_prob = saved_action.log_prob
            policy_losses.append(-log_prob * R)
        loss = torch.stack(policy_losses).sum()

        ########## END OF YOUR CODE ##########
        
        return loss

    def clear_memory(self):
        # reset rewards and action buffer
        del self.rewards[:]
        del self.saved_actions[:]

class GAE:
    def __init__(self, gamma, lambda_, num_steps):
        self.gamma = gamma
        self.lambda_ = lambda_
        self.num_steps = num_steps          # set num_steps = None to adapt full batch

    def __call__(self, rewards, values, done):
        """
        Implement Generalized Advantage Estimation (GAE) for your value prediction
        TODO (1): Pass correct corresponding inputs (rewards, values, and done) into the function arguments
        TODO (2): Calculate the Generalized Advantage Estimation and return the obtained value
        """

        ########## YOUR CODE HERE (8-15 lines) ##########
        advantages = []
        gae = 0
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0 if done else values[i]
            else:
                next_value = values[i+1]
            delta = rewards[i] + self.gamma * next_value - values[i]
            gae = delta + self.gamma * self.lambda_ * gae
            advantages.insert(0, gae)
        return advantages
        ########## END OF YOUR CODE ##########

def train(lr=0.01, gamma=0.99):
    """
        Train the model using SGD (via backpropagation)
        TODO (1): In each episode, 
        1. run the policy till the end of the episode and keep the sampled trajectory
        2. update both the policy and the value network at the end of episode

        TODO (2): In each episode, 
        1. record all the value you aim to visualize on tensorboard (lr, reward, length, ...)
    """
    
    # Instantiate the policy model and the optimizer
    model = Policy()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler (optional)
    # scheduler = Scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.95)
    # EWMA reward for tracking the learning progress
    ewma_reward = 0
    
    # run inifinitely many episodes
    for i_episode in count(1):
        # reset environment and episode reward
        state, _ = env.reset()
        ep_reward = 0
        t = 0

        # Uncomment the following line to use learning rate scheduler
        # scheduler.step()
        
        # For each episode, only run 9999 steps to avoid entering infinite loop during the learning process
        
        ########## YOUR CODE HERE (10-15 lines) ##########
        done = False 
        while not done:
            action = model.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            model.rewards.append(reward)
            ep_reward += reward
            t += 1

        loss = model.calculate_loss(gamma)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.clear_memory()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scheduler.step()

        ########## END OF YOUR CODE ##########
            
        # update EWMA reward and log the results
        ewma_reward = 0.05 * ep_reward + (1 - 0.05) * ewma_reward
        print('Episode {}\tlength: {}\treward: {}\t ewma reward: {}'.format(i_episode, t, ep_reward, ewma_reward))

        #Try to use Tensorboard to record the behavior of your implementation 
        ########## YOUR CODE HERE (4-5 lines) ##########
        writer.add_scalar('Loss', loss.item(), i_episode)
        writer.add_scalar('Reward', ep_reward, i_episode)
        writer.add_scalar('Episode_length', t, i_episode)
        writer.add_scalar('EWMA_reward', ewma_reward, i_episode)
        ########## END OF YOUR CODE ##########

        # check if we have "solved" the cart pole problem, use 120 as the threshold in LunarLander-v2
        if ewma_reward > env.spec.reward_threshold:
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), CHECKPOINT_DIR / 'CartPole_{}.pth'.format(lr))
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(ewma_reward, t))
            break


def test(name, n_episodes=10):
    """
        Test the learned model (no change needed)
    """     
    model = Policy()
    
    model.load_state_dict(torch.load(CHECKPOINT_DIR / name))
    
    render = True
    max_episode_len = 10000
    
    for i_episode in range(1, n_episodes+1):
        state , _= env.reset()
        running_reward = 0
        for t in range(max_episode_len+1):
            action = model.select_action(state)
            state, reward, terminated, truncated, _  = env.step(action)
            done = terminated or truncated
            running_reward += reward
            if render:
                 env.render()
            if done:
                break
        print('Episode {}\tReward: {}'.format(i_episode, running_reward))
    env.close()
    

if __name__ == '__main__':
    # For reproducibility, fix the random seed
    random_seed = 10  
    lr = 0.005  # 更小一點更穩
    env = gym.make('CartPole-v0')
    env.reset(seed=random_seed)
    torch.manual_seed(random_seed)  
    train(lr)
    test(f'CartPole_{lr}.pth')
