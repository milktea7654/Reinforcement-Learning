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
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

HW_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = HW_DIR / "checkpoints" / "reinforce_gae"
RUN_DIR = HW_DIR / "runs" / "reinforce_gae"

# Define a tensorboard writer
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
    def __init__(self, env):
        super(Policy, self).__init__()

        # Extract the dimensionality of state and action spaces
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.hidden_size = 128
        ########## YOUR CODE HERE (5~10 lines) ##########
        self.fc_shared = nn.Linear(self.observation_dim, self.hidden_size)
        nn.init.kaiming_uniform_(self.fc_shared.weight, nonlinearity='relu')
        self.action_head = nn.Linear(self.hidden_size, self.action_dim)
        self.value_head = nn.Linear(self.hidden_size, 1)
        nn.init.xavier_uniform_(self.action_head.weight)
        nn.init.xavier_uniform_(self.value_head.weight)
        
        ########## END OF YOUR CODE ##########
        
        self.saved_actions = []
        self.rewards = []
        self.done_flags = []  

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
        state = torch.from_numpy(state).float().unsqueeze(0)  # 使用 float32
        action_probs, state_value = self.forward(state)
        m = Categorical(action_probs)
        action = m.sample()
        ########## END OF YOUR CODE ##########

        # save to action buffer
        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))
        return action.item()

    def calculate_loss(self, gamma=0.99, lam=0.95):
        """
            Calculate the loss (= policy loss + value loss) to perform backprop later
            TODO:
                1. Calculate rewards-to-go required by REINFORCE with the help of self.rewards
                2. Calculate the policy loss using the policy gradient
                3. Calculate the value loss using either MSE loss or smooth L1 loss
        """
        R = 0
        saved_actions = self.saved_actions
        policy_losses = []
        value_losses = [] 
        returns = []
        
        done_flags = self.done_flags
        values = [sa.value.item() for sa in saved_actions]
        rewards = self.rewards
        gae_estimator = GAE(gamma, lam)
        advantages = gae_estimator(rewards, values, done_flags)

        ########## YOUR CODE HERE (8-15 lines) ##########
        for r in rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns).float()

        advantages = torch.tensor(advantages).float()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        
        for (log_prob, value), R_val, adv in zip(saved_actions, returns, advantages):
            policy_losses.append(-log_prob * adv.detach())
            value_losses.append(F.mse_loss(value.squeeze(), torch.tensor([R_val]).float()))
        
        policy_loss = torch.stack(policy_losses).mean()
        value_loss = torch.stack(value_losses).mean()
        total_loss = policy_loss + value_loss

        ########## END OF YOUR CODE ##########
        
        
        return total_loss

    def clear_memory(self):
        # reset rewards and action buffer
        self.saved_actions.clear()
        self.rewards.clear()
        self.done_flags.clear()

class GAE:
    def __init__(self, gamma, lam):
        self.gamma = gamma
        self.lam = lam

    def __call__(self, rewards, values, done_flags):
        """
        Implement Generalized Advantage Estimation (GAE) for your value prediction
        TODO (1): Pass correct corresponding inputs (rewards, values, and done) into the function arguments
        TODO (2): Calculate the Generalized Advantage Estimation and return the obtained value
        """
        ########## YOUR CODE HERE (8-15 lines) ##########
        advantages = []
        gae = 0
        for i in reversed(range(len(rewards))):
            next_value = 0 if done_flags[i] else (values[i+1] if i+1 < len(values) else 0)
            delta = rewards[i] + self.gamma * next_value - values[i]
            gae = delta + self.gamma * self.lam * gae
            advantages.insert(0, gae)
        return advantages
        ########## END OF YOUR CODE ##########
        
    
def train(env, lr=0.001, gamma=0.99, lam=0.95, max_episodes=50000, batch_size=5):
    """
        Train the model using SGD (via backpropagation)
        TODO (1): In each episode, 
        1. run the policy till the end of the episode and keep the sampled trajectory
        2. update both the policy and the value network at the end of episode

        TODO (2): In each episode, 
        1. record all the value you aim to visualize on tensorboard (lr, reward, length, ...)
    """
    # Instantiate the policy model and the optimizer
    model = Policy(env)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Learning rate scheduler (optional)
    # scheduler = Scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    
    # EWMA reward for tracking the learning progress
    ewma_reward = 0
    episode_count = 0

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
            model.done_flags.append(done)
            ep_reward += reward
            t+=1

        i_episode += 1

        loss = model.calculate_loss(gamma, lam)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        model.clear_memory()
        ########## END OF YOUR CODE ##########
        # update EWMA reward and log the results
        ewma_reward = 0.05 * reward + 0.95 * ewma_reward
        print('Episode {}\tlength: {}\treward: {}\t ewma reward: {}'.format(i_episode, t, ep_reward, ewma_reward))
        #Try to use Tensorboard to record the behavior of your implementation 
        ########## YOUR CODE HERE (4-5 lines) ##########
        writer.add_scalar('Loss', loss.item(), i_episode)
        writer.add_scalar('Reward', ep_reward, i_episode)
        writer.add_scalar('Episode_length', t, i_episode)
        writer.add_scalar('EWMA_reward', ewma_reward, i_episode)
        ########## END OF YOUR CODE ##########
        

        if ewma_reward > 200:
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            model_path = CHECKPOINT_DIR / f"LunarLander_gae_{lr}.pth"
            torch.save(model.state_dict(), model_path)
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(ewma_reward, t))
            break

def test(env, model_path, n_episodes=10):
    """
        Test the learned model (no change needed)
    """     
    model = Policy(env)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    for ep in range(1, n_episodes+1):
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
    lr = 0.001
    env = gym.make("LunarLander-v2")
    torch.manual_seed(random_seed)
    train(env, lr=lr, gamma=0.99, lam=0.95, max_episodes=50000, batch_size=5)
    test(env, CHECKPOINT_DIR / f"LunarLander_gae_{lr}.pth")
