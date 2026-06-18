import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def generalized_advantage_estimation(rewards, values, gamma=0.99, lam=0.95):
    """
    Compute the Generalized Advantage Estimation (GAE).
    
    Args:
        rewards (np.array): Array of rewards collected during an episode.
        values (np.array): Array of value function estimates for each state.
        gamma (float): Discount factor for future rewards.
        lam (float): Lambda parameter controlling bias-variance tradeoff.

    Returns:
        advantages (np.array): Computed advantage values.
        returns (np.array): Computed returns for updating the value function.
    """
    advantages = np.zeros_like(rewards)
    returns = np.zeros_like(rewards)
    next_advantage = 0
    next_value = values[-1]
    
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_value - values[t]
        advantages[t] = delta + gamma * lam * next_advantage
        next_advantage = advantages[t]
        next_value = values[t]
    
    returns = advantages + values  # Compute target returns
    return advantages, returns

class ValueNetwork(nn.Module):
    def __init__(self, input_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Training function for the critic network using Temporal Difference (TD) Learning
def train_critic_td(value_net, optimizer, states, rewards, next_states, dones, gamma=0.99):
    criterion = nn.MSELoss()
    
    values = value_net(states).squeeze()
    next_values = value_net(next_states).squeeze().detach()
    target_values = rewards + gamma * next_values * (1 - dones)
    
    loss = criterion(values, target_values)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Example usage
if __name__ == "__main__":
    input_dim = 4
    value_net = ValueNetwork(input_dim)
    optimizer = optim.Adam(value_net.parameters(), lr=0.01)
    
    # Example batch data
    states = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5]], dtype=torch.float32)
    next_states = torch.tensor([[0.2, 0.3, 0.4, 0.5], [0.3, 0.4, 0.5, 0.6]], dtype=torch.float32)
    rewards = torch.tensor([1.0, 1.5], dtype=torch.float32)
    dones = torch.tensor([0, 1], dtype=torch.float32)  # 1 if episode ended, else 0
    
    # Train the critic with TD learning
    train_critic_td(value_net, optimizer, states, rewards, next_states, dones)
    print("Updated Value Output:", value_net(states).squeeze())
