import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )
    def forward(self, x):
        return self.net(x)

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.net(x)

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, lam=0.95, clip=0.2, epochs=10, batch_size=64):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = lam
        self.clip = clip
        self.epochs = epochs
        self.batch_size = batch_size
        self.memory = []

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().numpy()[0]
        return action

    def store(self, transition):
        self.memory.append(transition)

    def compute_gae(self, rewards, values, dones, next_value):
        gae = 0
        returns = []
        values = values + [next_value]
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.lam * (1 - dones[step]) * gae
            returns.insert(0, gae + values[step])
        return returns

    def train_step(self, next_state):
        states, actions, rewards, dones, values = zip(*self.memory)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        next_value = self.critic(next_state).item()
        returns = self.compute_gae(rewards, list(values), dones, next_value)
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        returns = torch.FloatTensor(returns)
        values = torch.FloatTensor(values)
        advantages = returns - values
        old_log_probs = torch.zeros(len(actions))  # For simplicity, assume deterministic policy
        for _ in range(self.epochs):
            idx = np.arange(len(states))
            np.random.shuffle(idx)
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_idx = idx[start:end]
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]
                # Actor loss
                new_actions = self.actor(batch_states)
                log_probs = -((new_actions - batch_actions) ** 2).mean(dim=1)  # surrogate for log prob
                ratio = torch.exp(log_probs - old_log_probs[batch_idx])
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                # Critic loss
                value_pred = self.critic(batch_states).squeeze()
                critic_loss = (batch_returns - value_pred).pow(2).mean()
                # Update
                self.optimizer_actor.zero_grad()
                actor_loss.backward()
                self.optimizer_actor.step()
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                self.optimizer_critic.step()
        self.memory = []

    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'optimizer_actor': self.optimizer_actor.state_dict(),
            'optimizer_critic': self.optimizer_critic.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.optimizer_actor.load_state_dict(checkpoint['optimizer_actor'])
        self.optimizer_critic.load_state_dict(checkpoint['optimizer_critic']) 