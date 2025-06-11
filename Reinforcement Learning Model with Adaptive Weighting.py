import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ---------------------------------------
# Reinforcement Learning Model with Adaptive Weighting
# ---------------------------------------

class AdaptiveRL(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(AdaptiveRL, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)
        self.adaptive_weights = torch.ones(action_dim)  # Dynamic weighting

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action_values = self.fc3(x)
        return action_values * self.adaptive_weights  # Scale actions dynamically

    def update_weights(self, feedback):
        """ Adjust weights based on performance feedback """
        self.adaptive_weights *= (1 + feedback)  # Scale weights adaptively
        self.adaptive_weights = torch.clamp(self.adaptive_weights, 0.1, 2.0)  # Keep values in range

# ---------------------------------------
# RL Training Loop with Adaptive Weighting
# ---------------------------------------

class RLTrainer:
    def __init__(self, state_dim, action_dim, lr=1e-3):
        self.model = AdaptiveRL(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_function = nn.MSELoss()
        self.memory = []  # Store past experience

    def select_action(self, state):
        with torch.no_grad():
            action_values = self.model(state)
        return torch.argmax(action_values).item()

    def store_experience(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def train(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        
        batch = np.random.choice(self.memory, batch_size, replace=False)
        for state, action, reward, next_state in batch:
            self.optimizer.zero_grad()
            predicted_q = self.model(state)[action]
            target_q = reward + 0.99 * torch.max(self.model(next_state))
            loss = self.loss_function(predicted_q, target_q)
            loss.backward()
            self.optimizer.step()

        # Adaptive weight updates based on training feedback
        avg_reward = sum(r for _, _, r, _ in batch) / len(batch)
        self.model.update_weights(torch.tensor(avg_reward))

# ---------------------------------------
# 🚀 RL Training with Adaptive Optimization
# ---------------------------------------

state_dim, action_dim = 64, 10
rl_agent = RLTrainer(state_dim, action_dim)

# Simulated training loop
for _ in range(100):  
    state = torch.randn(1, state_dim)
    action = rl_agent.select_action(state)
    reward = np.random.rand()  # Simulated reward metric
    next_state = torch.randn(1, state_dim)
    rl_agent.store_experience(state, action, reward, next_state)
    rl_agent.train()

print("Adaptive weighting RL optimization complete.")

