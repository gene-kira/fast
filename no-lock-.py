import numpy as np
import random
import pygame
import sys
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
from collections import deque

# Constants
WIDTH, HEIGHT = 600, 400
GRID_SIZE = 20
PLAYER_COLOR = (0, 128, 255)
ENEMY_COLOR = (255, 0, 0)
OBSTACLE_COLOR = (0, 0, 0)
BACKGROUND_COLOR = (255, 255, 255)

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# Environment Setup
class Environment:
    def __init__(self):
        self.grid_width = WIDTH // GRID_SIZE
        self.grid_height = HEIGHT // GRID_SIZE
        self.player_pos = np.array([5, 5])
        self.enemy_pos = np.array([10, 10])
        self.obstacles = set()
        self.moving_obstacle_pos = np.array([8, 8])
    
    def reset(self):
        self.player_pos = np.array([5, 5])
        self.enemy_pos = np.array([10, 10])
        self.obstacles = set([(3, 4), (4, 4), (5, 4)])
        self.moving_obstacle_pos = np.array([8, 8])
        return self._get_state()
    
    def step(self, action):
        new_player_pos = self.player_pos + np.array(action)
        if self.is_valid_position(new_player_pos):
            self.player_pos = new_player_pos
        
        # Move moving obstacle
        self.moving_obstacle_pos += np.random.choice([np.array([1, 0]), np.array([-1, 0]), np.array([0, 1]), np.array([0, -1])])
        if not self.is_valid_position(self.moving_obstacle_pos):
            self.moving_obstacle_pos -= np.array(action)
        
        reward = 0
        done = False
        
        if np.array_equal(self.player_pos, self.enemy_pos):
            reward = 100
            done = True
        elif tuple(self.player_pos) in self.obstacles or np.array_equal(self.player_pos, self.moving_obstacle_pos):
            reward = -50
            done = True
        else:
            reward = -1
        
        return self._get_state(), reward, done
    
    def is_valid_position(self, pos):
        x, y = pos
        return 0 <= x < self.grid_width and 0 <= y < self.grid_height and tuple(pos) not in self.obstacles and not np.array_equal(pos, self.moving_obstacle_pos)
    
    def _get_state(self):
        state = np.concatenate((self.player_pos, self.enemy_pos, self.moving_obstacle_pos))
        return state.reshape(1, -1)

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()
    
    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=0.01))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Predictive Model (LSTM)
class PredictiveModel:
    def __init__(self, state_size):
        self.state_size = state_size
        self.model = self._build_model()
    
    def _build_model(self):
        model = Sequential()
        model.add(LSTM(50, input_shape=(None, self.state_size), return_sequences=True))
        model.add(LSTM(50))
        model.add(Dense(2, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=0.01))
        return model
    
    def predict(self, state):
        state = np.expand_dims(state, axis=0)
        return self.model.predict(state)[0]
    
    def train(self, states, targets):
        states = np.array(states)
        targets = np.array(targets)
        self.model.fit(states, targets, epochs=1, verbose=0)

# Initialize Environment and Agents
env = Environment()
state_size = env._get_state().shape[1]
action_size = 4  # up, down, left, right
dqn_agent = DQNAgent(state_size, action_size)
predictive_model = PredictiveModel(state_size)

# Training Loop
num_episodes = 1000
batch_size = 32

def train():
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        
        while not done:
            # Player moves randomly for demonstration
            player_directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
            action = dqn_agent.act(state)
            player_move = player_directions[action]
            
            # Toggle no_lock based on left mouse button press (simulated here)
            if random.random() < 0.1:  # Simulate left mouse button press with probability 0.1
                env.player.toggle_no_lock()
            
            next_state, reward, done = env.step(player_move)
            
            dqn_agent.remember(state, action, reward, next_state, done)
            state = next_state
            
            if len(dqn_agent.memory) > batch_size:
                dqn_agent.replay(batch_size)

def train_predictive_model():
    states = []
    targets = []
    
    for _ in range(100):
        state = env.reset()
        done = False
        
        while not done:
            player_directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
            action = dqn_agent.act(state)
            player_move = player_directions[action]
            
            # Toggle no_lock based on left mouse button press (simulated here)
            if random.random() < 0.1:  # Simulate left mouse button press with probability 0.1
                env.player.toggle_no_lock()
            
            next_state, _, done = env.step(player_move)
            
            predicted_next_state = predictive_model.predict(state)
            target = np.concatenate((env.enemy_pos, env.moving_obstacle_pos))
            targets.append(target)
            
            states.append(np.expand_dims(predictive_model.predict(state), axis=0))
            state = next_state
    
    if len(states) > 1:
        predictive_model.train(states, targets)

# Run Training
train()
train_predictive_model()

# Pygame Visualization
def draw():
    screen.fill(BACKGROUND_COLOR)
    
    # Draw player
    pygame.draw.rect(screen, PLAYER_COLOR, (env.player_pos[0] * GRID_SIZE, env.player_pos[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
    
    # Draw enemy
    pygame.draw.rect(screen, ENEMY_COLOR, (env.enemy_pos[0] * GRID_SIZE, env.enemy_pos[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
    
    # Draw obstacles
    for obstacle in env.obstacles:
        pygame.draw.rect(screen, OBSTACLE_COLOR, (obstacle[0] * GRID_SIZE, obstacle[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
    
    # Draw moving obstacle
    pygame.draw.rect(screen, OBSTACLE_COLOR, (env.moving_obstacle_pos[0] * GRID_SIZE, env.moving_obstacle_pos[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
    
    pygame.display.flip()

# Main Loop
def main():
    state = env.reset()
    done = False
    
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        # Player moves randomly for demonstration
        player_directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        action = dqn_agent.act(state)
        player_move = player_directions[action]
        
        # Toggle no_lock based on left mouse button press (simulated here)
        if random.random() < 0.1:  # Simulate left mouse button press with probability 0.1
            env.player.toggle_no_lock()
        
        next_state, reward, done = env.step(player_move)
        state = next_state
        
        draw()
        clock.tick(10)

if __name__ == "__main__":
    main()
