import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=500):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_net = QNetwork(state_dim, action_dim)
        self.target_q_net = QNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0
        self.replay_buffer = ReplayBuffer()
        self.update_target()

    def select_action(self, state):
        self.steps_done += 1
        eps = self.epsilon_end + (self.epsilon - self.epsilon_end) * np.exp(-1. * self.steps_done / self.epsilon_decay)
        if np.random.rand() < eps:
            return np.random.uniform(-1, 1, size=(1,))
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_net(state)
            action = q_values.max(1)[1].item()
            # Map discrete action to continuous [-1, 1]
            return np.array([2 * action / (self.action_dim - 1) - 1])

    def update_target(self):
        self.target_q_net.load_state_dict(self.q_net.state_dict())

    def train_step(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state)
        action = torch.LongTensor(((action + 1) * (self.action_dim - 1) / 2).astype(int))
        reward = torch.FloatTensor(reward)
        next_state = torch.FloatTensor(next_state)
        done = torch.FloatTensor(done)
        q_values = self.q_net(state).gather(1, action)
        with torch.no_grad():
            next_q_values = self.target_q_net(next_state).max(1)[0]
        expected_q = reward + self.gamma * next_q_values * (1 - done)
        loss = nn.MSELoss()(q_values.squeeze(), expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, path):
        torch.save({
            'q_net': self.q_net.state_dict(),
            'target_q_net': self.target_q_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.q_net.load_state_dict(checkpoint['q_net'])
        self.target_q_net.load_state_dict(checkpoint['target_q_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer']) 