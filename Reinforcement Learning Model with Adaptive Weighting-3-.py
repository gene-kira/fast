

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

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
# Multi-Agent RL Trainer with Information Sharing
# ---------------------------------------

class MultiAgentRLTrainer:
    def __init__(self, state_dim, action_dim, num_agents, lr=1e-3):
        self.agents = [RLTrainer(state_dim, action_dim, lr) for _ in range(num_agents)]

    def select_action(self, state):
        actions = [agent.select_action(state) for agent in self.agents]
        return actions[0]  # For simplicity, we use the first agent's action

    def store_experience(self, state, action, reward, next_state):
        for agent in self.agents:
            agent.store_experience(state, action, reward, next_state)

    def train(self, batch_size=32):
        for agent in self.agents:
            agent.train(batch_size)

    def share_information(self):
        avg_weights = torch.mean(torch.stack([agent.model.adaptive_weights for agent in self.agents]), dim=0)
        for agent in self.agents:
            agent.model.adaptive_weights = avg_weights

# ---------------------------------------
# 🚀 RL Training with Adaptive Optimization
# ---------------------------------------

state_dim, action_dim = 64, 10
num_agents = 3
multi_agent_trainer = MultiAgentRLTrainer(state_dim, action_dim, num_agents)

rewards_history = []
weights_history = []

for iteration in range(200):
    state = torch.randn(1, state_dim)
    action = multi_agent_trainer.select_action(state)
    reward = np.random.rand()  # Simulated reward metric
    next_state = torch.randn(1, state_dim)
    multi_agent_trainer.store_experience(state, action, reward, next_state)
    multi_agent_trainer.train()
    multi_agent_trainer.share_information()

    rewards_history.append(reward)
    weights_history.append([agent.model.adaptive_weights.detach().numpy() for agent in multi_agent_trainer.agents])

print("Multi-agent adaptive weighting RL optimization with advanced communication complete.")

# ---------------------------------------
# Visualization Functions
# ---------------------------------------

def plot_reward_trend(rewards):
    plt.plot(rewards, label='Reward Trend')
    plt.xlabel('Iteration')
    plt.ylabel('Reward')
    plt.title('Reward Over Training Iterations')
    plt.legend()
    plt.show()

def plot_weight_evolution(weights_history, agent_idx=0):
    weights = np.array([w[agent_idx] for w in weights_history])
    num_actions = weights.shape[1]
    
    for i in range(num_actions):
        plt.plot(weights[:, i], label=f'Action {i} Weight')
    
    plt.xlabel('Iteration')
    plt.ylabel('Weight Value')
    plt.title(f'Weight Evolution of Agent {agent_idx}')
    plt.legend()
    plt.show()

# Visualize the results
plot_reward_trend(rewards_history)
plot_weight_evolution(weights_history, 0)  # Plot the weight evolution of the first agent for simplicity

print("Training and visualization complete.")
```

### Explanation

1. **AdaptiveRL Class**: This class defines the neural network model with adaptive weighting.
2. **RLTrainer Class**: This class handles the training loop for a single RL agent, including action selection, experience storage, and weight updates.
3. **MultiAgentRLTrainer Class**: This class manages multiple RL agents and includes methods for training, sharing information, and selecting actions.
4. **Training Loop**: The main training loop simulates the environment and trains the multi-agent system over 200 iterations.
5. **Visualization Functions**: These functions plot the reward trend and weight evolution of a specified agent.

### Running the Code

To run the code, you can copy and paste it into a Python environment or Jupyter notebook. The