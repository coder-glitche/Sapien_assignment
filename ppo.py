import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import matplotlib.pyplot as plt

# the neural network model
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        action_logits = self.fc3(x)
        state_value = self.value_head(x)
        return action_logits, state_value

# PPO hyperparameters
gamma = 0.99           # Discount factor
lr = 0.001             # Learning rate
clip_epsilon = 0.2     # Clipping epsilon for PPO
K_epochs = 4           # Number of epochs   to update the policy
T_horizon = 2000       # Number of steps    before updating policy
batch_size = 64

# Initialize environment
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Initialize policy network and optimizer
policy_net = PolicyNetwork(state_dim, action_dim)
optimizer = optim.Adam(policy_net.parameters(), lr=lr)

# compute discounted rewards
def compute_returns(rewards, masks):
    returns = []
    R = 0
    for r, m in zip(rewards[::-1], masks[::-1]):
        R = r + gamma * R * m
        returns.insert(0, R)
    return returns

# PPO update function
def ppo_update(states, actions, log_probs, returns, advantages):
    for _ in range(K_epochs):
        new_logits, state_values = policy_net(states)
        dist = Categorical(logits=new_logits)
        new_log_probs = dist.log_prob(actions)

        # Policy loss
        ratio = torch.exp(new_log_probs - log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = nn.functional.mse_loss(state_values, returns)

        # Total loss
        loss = policy_loss + 0.5 * value_loss

        # Update policy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Training PPO
def train_ppo():
    all_rewards = []
    for episode in range(1000):
        state, _ = env.reset()
        states = []
        actions = []
        rewards = []
        log_probs = []
        masks = []
        done = False
        episode_reward = 0

        while not done:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                logits, value = policy_net(state)
                dist = Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            episode_reward += reward

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            masks.append(1 - done)

            state = next_state

            if len(states) >= T_horizon or done:
                states = torch.cat(states)
                actions = torch.cat(actions)
                log_probs = torch.cat(log_probs)
                returns = compute_returns(rewards, masks)
                returns = torch.FloatTensor(returns).unsqueeze(1)
                advantages = returns - policy_net(states)[1].detach()

                ppo_update(states, actions, log_probs, returns, advantages)

                states = []
                actions = []
                rewards = []
                log_probs = []
                masks = []

        all_rewards.append(episode_reward)

        if episode % 100 == 0:
            avg_reward = np.mean(all_rewards[-100:])
            print(f"Episode {episode}, Average Reward: {avg_reward}")

    return all_rewards

# Train PPO
ppo_rewards = train_ppo()

# Plot PPO rewards
plt.plot(ppo_rewards)
plt.title('PPO Rewards Over Time')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()
