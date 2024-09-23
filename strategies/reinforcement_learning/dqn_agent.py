import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class DQN(nn.Module):
    # Define the neural network model for DQN with more layers and regularization
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, output_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        if x.size(0) > 1:
            x = torch.relu(self.bn1(self.fc1(x)))
            x = self.dropout(torch.relu(self.bn2(self.fc2(x))))
        else:
            x = torch.relu(self.fc1(x))
            x = self.dropout(torch.relu(self.fc2(x)))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, input_dim, action_space, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, lr=0.001, tau=0.125):
        self.action_space = action_space
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lr = lr
        self.tau = tau 
        
        
        self.memory = deque(maxlen=2000)
        
        
        self.model = DQN(input_dim, action_space)
        self.target_model = DQN(input_dim, action_space)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

        
        self._update_target_model()

    def _update_target_model(self):
        
        self.target_model.load_state_dict(self.model.state_dict())

    def _soft_update_target_model(self):
        # Perform a soft update of target network parameters
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, use_exploration_noise=True):
        # Epsilon-greedy strategy
        if np.random.rand() <= self.epsilon and use_exploration_noise:
            return random.randrange(self.action_space)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        act_values = self.model(state)
        return torch.argmax(act_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

            # Compute the target Q-value
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.target_model(next_state)).item()

            target_f = self.model(state)
            target_f[0][action] = target

            self.optimizer.zero_grad()
            loss = self.criterion(target_f, self.model(state))
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Soft update target model parameters
        self._soft_update_target_model()

    def load(self, name):
        self.model.load_state_dict(torch.load(name))
        self.target_model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)
