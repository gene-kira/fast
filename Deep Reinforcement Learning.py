


Integration of Deep Reinforcement Learning
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class DeepReinforcementLearner:
    def __init__(self, state_size, action_size, learning_rate=0.001, discount_factor=0.95):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.memory = []
        self.model = self.build_model()

    def build_model(self):
        """Builds a deep Q-learning model."""
        model = Sequential([
            Dense(24, input_dim=self.state_size, activation='relu'),
            Dense(24, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        """Stores experience for memory replay."""
        self.memory.append((state, action, reward, next_state, done))

    def train(self, batch_size=32):
        """Trains the network using experience replay."""
        if len(self.memory) < batch_size:
            return
        
        batch = np.random.choice(self.memory, batch_size, replace=False)
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target += self.discount_factor * np.max(self.model.predict(next_state)[0])
            
            target_values = self.model.predict(state)
            target_values[0][action] = target
            
            self.model.fit(state, target_values, epochs=1, verbose=0)

    def choose_action(self, state, exploration_rate=0.1):
        """Chooses an action using an exploration-exploitation strategy."""
        if np.random.rand() < exploration_rate:
            return np.random.choice(self.action_size)
        
        return np.argmax(self.model.predict(state)[0])

# Example Execution
learner = DeepReinforcementLearner(state_size=4, action_size=3)
state = np.array([[0.5, 0.3, 0.7, 0.2]])  # Sample input state
action = learner.choose_action(state)
print("Chosen Action:", action)

