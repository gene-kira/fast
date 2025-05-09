import gym
from gym import spaces
import numpy as np

# Install necessary libraries if not already installed
import subprocess
import sys

def install_libraries():
    libs = ['gym', 'numpy']
    for lib in libs:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', lib])

install_libraries()

class MazeEnv(gym.Env):
    def __init__(self, maze_size=(5, 5)):
        self.maze_size = maze_size
        self.start = (0, 0)
        self.goal = (maze_size[0]-1, maze_size[1]-1)
        self.reset()
        
        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space = spaces.Box(low=0, high=1, shape=self.maze_size, dtype=np.int8)
    
    def reset(self):
        self.state = np.zeros(self.maze_size, dtype=np.int8)
        self.state[self.start] = 1
        self.state[self.goal] = 2
        self.current_position = self.start
        return self.state

    def step(self, action):
        actions = {
            0: (0, -1),  # Up
            1: (0, 1),   # Down
            2: (-1, 0),  # Left
            3: (1, 0)    # Right
        }
        
        new_position = tuple(np.array(self.current_position) + np.array(actions[action]))
        
        if not (0 <= new_position[0] < self.maze_size[0]) or not (0 <= new_position[1] < self.maze_size[1]):
            return self.state, -1, False, {}
        
        self.state[self.current_position] = 0
        self.state[new_position] = 1
        self.current_position = new_position
        
        if new_position == self.goal:
            reward = 10
            done = True
        else:
            reward = -0.1  # Encourage exploration with a small negative reward
            done = False
        
        return self.state, reward, done, {}

    def render(self):
        print(self.state)

class QLearningAgent:
    def __init__(self, action_space, state_space, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.99):
        self.action_space = action_space.n
        self.state_space_shape = state_space.shape
        self.q_table = np.zeros((self.state_space_shape + (self.action_space,)))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = 0.1

    def choose_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return np.random.choice(self.action_space)
        else:
            flat_state = state.flatten()
            q_values = self.q_table[tuple(flat_state) + (slice(None),)]
            return np.argmax(q_values)

    def learn(self, state, action, reward, next_state):
        flat_state = state.flatten()
        flat_next_state = next_state.flatten()
        best_next_action = np.argmax(self.q_table[tuple(flat_next_state) + (slice(None),)])
        td_target = reward + self.discount_factor * self.q_table[tuple(flat_next_state) + (best_next_action,)]
        td_error = td_target - self.q_table[tuple(flat_state) + (action,)]
        self.q_table[tuple(flat_state) + (action,)] += self.learning_rate * td_error
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)

def train_agent(env, agent, episodes=1000):
    for episode in range(episodes):
        state = env.reset()
        done = False
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1} completed")

def evaluate_agent(env, agent, episodes=10):
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward

        print(f"Episode {episode + 1} - Total Reward: {total_reward}")

# Set up the environment and agent
env = MazeEnv()
agent = QLearningAgent(env.action_space, env.observation_space)

# Train the agent
train_agent(env, agent)

# Evaluate the trained agent
evaluate_agent(env, agent)
