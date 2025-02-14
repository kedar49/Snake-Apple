import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

# Set device to CPU
device = torch.device('cpu')

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        self.to(device)

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.linear3(x)

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=100000)
        self.model = DQN(11, 256, 3)  # 11 input states, 256 hidden size, 3 actions
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.batch_size = 1000
        self.min_memory_size = 1000  # Minimum memory size before training

    def get_state(self, state):
        return torch.FloatTensor(state).to(device)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) < self.min_memory_size:
            return  # Don't train if we don't have enough memories
            
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

        # Get predicted Q values with current states
        current_q = self.model(states)
        next_q = self.model(next_states)
        target_q = current_q.clone()

        for idx in range(len(dones)):
            Q_new = rewards[idx]
            if not dones[idx]:
                Q_new = rewards[idx] + self.gamma * torch.max(next_q[idx])
            target_q[idx][torch.argmax(actions[idx]).item()] = Q_new

        # Q_new = reward + gamma * max(next_predicted Q value)
        self.optimizer.zero_grad()
        loss = F.mse_loss(current_q, target_q)
        loss.backward()
        self.optimizer.step()

    def get_action(self, state, train=True):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if train and random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.FloatTensor(state).to(device)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move
