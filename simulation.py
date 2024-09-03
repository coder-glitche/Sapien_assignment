import pygame
import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import time

# pygame
pygame.init()
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 400
FPS = 60

# environment with render_mode
env = gym.make('CartPole-v1', render_mode='human')
n_actions = env.action_space.n


num_bins = 10
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

# Q-table
q_table = defaultdict(lambda: np.zeros(n_actions))

# Q-learning parameters
alpha = 0.1
gamma = 0.99
epsilon = 0.1
num_episodes = 500

# Q-learning algorithm
def q_learning():
    rewards = []
    durations = []
    start_time = time.time()  # Start timer
    
    for episode in range(num_episodes):
        state, _ = env.reset()  # Extract only the state, ignore the dictionary
        state = discretize_state(state)
        total_reward = 0
        step_count = 0
        
        done = False
        while not done:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(q_table[state])  # Exploit
            
            next_state, reward, done, truncated, _ = env.step(action)  # Unpack all returned values
            next_state, _ = next_state if isinstance(next_state, tuple) else (next_state, {})
            next_state = discretize_state(next_state)
            
            best_next_action = np.argmax(q_table[next_state])
            td_target = reward + gamma * q_table[next_state][best_next_action]
            td_error = td_target - q_table[state][action]
            
            q_table[state][action] += alpha * td_error
            
            state = next_state
            total_reward += reward
            step_count += 1

            if done or truncated:  # End loop if the episode is done or truncated
                break
        
        rewards.append(total_reward)
        durations.append(step_count)
    
    end_time = time.time()  # End timer
    elapsed_time = end_time - start_time
    return rewards, durations, elapsed_time

# Run Q-learning
q_learning_rewards, balance_durations, total_time = q_learning()

# Plot 
plt.plot(q_learning_rewards)
plt.title('Q-learning Rewards Over Time')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()


print(f"Total Rewards in Q-learning: {sum(q_learning_rewards)}")
print(f"Total Time for Q-learning: {total_time:.2f} seconds")

# Simulation 
def simulate_agent():
    state, _ = env.reset()
    state = discretize_state(state)
    done = False
    total_reward = 0

    # Initialize window
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("CartPole Simulation")
    clock = pygame.time.Clock()

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # action from Q-table
        action = np.argmax(q_table[state])
        
        
        next_state, reward, done, truncated, _ = env.step(action)
        next_state, _ = next_state if isinstance(next_state, tuple) else (next_state, {})
        next_state = discretize_state(next_state)
        
        # Render 
        env.render()  
        pygame.display.flip()

        clock.tick(FPS)

        state = next_state
        total_reward += reward

        if done or truncated:
            break

    env.close()
    pygame.quit()
    print(f"Total Reward in Simulation: {total_reward}")

# Run the simulation
simulate_agent()
