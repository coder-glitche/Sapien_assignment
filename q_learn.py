import warnings
import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Suppress the specific DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym.utils.passive_env_checker")

# Q-learning parameters
alpha = 0.1       # Learning rate
gamma = 0.99      # Discount factor
epsilon = 0.1     # Exploration rate
num_episodes = 500
num_bins = 10     # Number of bins for discretizing each dimension of state space

# Initialize environment
env = gym.make('CartPole-v1')
n_actions = env.action_space.n

# Define the bins for each state dimension
state_bins = [
    np.linspace(-2.4, 2.4, num_bins),  # Position
    np.linspace(-3.0, 3.0, num_bins),  # Velocity
    np.linspace(-0.21, 0.21, num_bins),  # Angle
    np.linspace(-2.0, 2.0, num_bins)   # Angular velocity
]

# Discretize state space
def discretize_state(state):
    state = np.array(state)  
    state_indices = []
    for i in range(len(state)):
        state_indices.append(np.digitize(state[i], bins=state_bins[i]) - 1)
    return tuple(state_indices)

# Initialize Q-table
q_table = defaultdict(lambda: np.zeros(n_actions))

# Q-learning algorithm
def q_learning():
    rewards = []
    for episode in range(num_episodes):
        state, _ = env.reset()  # Extract only the state, ignore the dictionary
        state = discretize_state(state)
        total_reward = 0
        
        done = False
        while not done:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(q_table[state])  # Exploit
            
            next_state, reward, done, truncated, _ = env.step(action)  # Unpack all returned values
            next_state, _ = next_state if isinstance(next_state, tuple) else (next_state, {})  # Handle next state tuple
            next_state = discretize_state(next_state)
            
            best_next_action = np.argmax(q_table[next_state])
            td_target = reward + gamma * q_table[next_state][best_next_action]
            td_error = td_target - q_table[state][action]
            
            q_table[state][action] += alpha * td_error
            
            state = next_state
            total_reward += reward

            if done or truncated:  # End loop if the episode is done or truncated
                break
        
        rewards.append(total_reward)
    
    return rewards

# Run Q-learning
q_learning_rewards = q_learning()

# Plot Q-learning rewards
plt.plot(q_learning_rewards)
plt.title('Q-learning Rewards Over Time')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()
