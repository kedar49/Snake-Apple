import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import os
import json
from datetime import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dueling=False):
        super().__init__()
        self.dueling = dueling
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.feature_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        if dueling:
            self.value_stream = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 1)
            )
            self.advantage_stream = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, output_size)
            )
        else:
            self.q_layers = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, output_size)
            )
        
        self.to(device)

    def forward(self, x):
        x = x.to(device)
        features = self.feature_layers(x)
        
        if self.dueling:
            value = self.value_stream(features)
            advantage = self.advantage_stream(features)
            q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
            return q_values
        else:
            return self.q_layers(features)

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        max_priority = max(self.priorities) if self.priorities else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
            self.priorities.append(max_priority)
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
            self.priorities[self.position] = max_priority
            
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            return [], [], [], [], [], []
            
        priorities = np.array(self.priorities[:len(self.buffer)])
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        states, actions, rewards, next_states, dones = zip(*samples)
        return states, actions, rewards, next_states, dones, weights, indices
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def __len__(self):
        return len(self.buffer)

class Agent:
    def __init__(self, input_size=25, hidden_size=512, output_size=3, 
                 use_dueling=True, use_double_dqn=True, use_prioritized_replay=True):
        self.n_games = 0
        self.epsilon = 0.9  # Start with high exploration
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.99  # Higher discount for long-term planning
        self.learning_rate = 0.0001
        
        if use_prioritized_replay:
            self.memory = PrioritizedReplayBuffer(100000)
        else:
            self.memory = deque(maxlen=100000)
        self.use_prioritized_replay = use_prioritized_replay
        
        self.use_double_dqn = use_double_dqn
        self.model = DQN(input_size, hidden_size, output_size, dueling=use_dueling)
        self.target_model = DQN(input_size, hidden_size, output_size, dueling=use_dueling)
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.batch_size = 64
        self.min_memory_size = 1000
        self.target_update_frequency = 10
        self.training_step = 0
        
        self.losses = []
        self.q_values = []
        self.epsilon_history = []
        
        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)

    def get_state(self, state):
        return torch.FloatTensor(state).to(device)

    def remember(self, state, action, reward, next_state, done):
        if self.use_prioritized_replay:
            self.memory.push(state, action, reward, next_state, done)
        else:
            self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) < self.min_memory_size:
            return
            
        if self.use_prioritized_replay:
            states, actions, rewards, next_states, dones, weights, indices = self.memory.sample(self.batch_size)
            if len(states) == 0:
                return
            self.train_step_prioritized(states, actions, rewards, next_states, dones, weights, indices)
        else:
            batch_size = min(self.batch_size, len(self.memory))
            mini_sample = random.sample(self.memory, batch_size)
            states, actions, rewards, next_states, dones = zip(*mini_sample)
            self.train_step(states, actions, rewards, next_states, dones)

    def train_step(self, states, actions, rewards, next_states, dones):
        states = torch.FloatTensor(np.array(states)).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        actions = torch.LongTensor(np.array(actions)).to(device)
        rewards = torch.FloatTensor(np.array(rewards)).to(device)
        dones = torch.BoolTensor(np.array(dones)).to(device)

        current_q_values = self.model(states)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        if self.use_double_dqn:
            next_actions = self.model(next_states).argmax(1)
            next_q_values = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        else:
            next_q_values = self.target_model(next_states).max(1)[0]

        target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        self.optimizer.zero_grad()
        loss = F.mse_loss(current_q_values, target_q_values)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        self.training_step += 1
        if self.training_step % self.target_update_frequency == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        
        self.losses.append(loss.item())
        self.q_values.append(current_q_values.mean().item())

    def train_step_prioritized(self, states, actions, rewards, next_states, dones, weights, indices):
        states = torch.FloatTensor(np.array(states)).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        actions = torch.LongTensor(np.array(actions)).to(device)
        rewards = torch.FloatTensor(np.array(rewards)).to(device)
        dones = torch.BoolTensor(np.array(dones)).to(device)
        weights = torch.FloatTensor(weights).to(device)

        current_q_values = self.model(states)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        if self.use_double_dqn:
            next_actions = self.model(next_states).argmax(1)
            next_q_values = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        else:
            next_q_values = self.target_model(next_states).max(1)[0]

        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        td_errors = (current_q_values - target_q_values).abs().detach().cpu().numpy()

        self.optimizer.zero_grad()
        loss = (weights * F.mse_loss(current_q_values, target_q_values, reduction='none')).mean()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        priorities = td_errors + 1e-6
        self.memory.update_priorities(indices, priorities)
        
        self.training_step += 1
        if self.training_step % self.target_update_frequency == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        
        self.losses.append(loss.item())
        self.q_values.append(current_q_values.mean().item())

    def get_action(self, state, train=True):
        if train:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.epsilon_history.append(self.epsilon)
        
        final_move = [0, 0, 0]
        
        if train and random.random() < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = self.model(state_tensor)
            move = torch.argmax(q_values).item()
            final_move[move] = 1

        return final_move

    def save_model(self, filename=None):
        if filename is None:
            filename = f"agent_model_{self.n_games}.pth"
        
        filepath = os.path.join(self.model_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'n_games': self.n_games,
            'epsilon': self.epsilon,
            'losses': self.losses,
            'q_values': self.q_values,
            'epsilon_history': self.epsilon_history
        }, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.n_games = checkpoint.get('n_games', 0)
            self.epsilon = checkpoint.get('epsilon', 0.01)
            self.losses = checkpoint.get('losses', [])
            self.q_values = checkpoint.get('q_values', [])
            self.epsilon_history = checkpoint.get('epsilon_history', [])
            print(f"Model loaded from {filepath}")
        else:
            print(f"No model found at {filepath}")

    def get_metrics(self):
        return {
            'n_games': self.n_games,
            'epsilon': self.epsilon,
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0,
            'avg_q_value': np.mean(self.q_values[-100:]) if self.q_values else 0,
            'memory_size': len(self.memory)
        }
