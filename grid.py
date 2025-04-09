# DQN Training on Custom Grid Environment in Colab

# --- Install required libraries ---
!pip install gym tqdm matplotlib
!pip install pygame gym numpy torch matplotlib
# --- Imports ---
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from tqdm import trange
import json
import matplotlib.pyplot as plt
import gym
from gym import spaces

# --- Custom Grid Environment ---
class GridEnv(gym.Env):
    def _init_(self, resource_data, grid_size=5):
        super(GridEnv, self)._init_()
        self.grid_size = grid_size
        self.start_pos = (grid_size - 1, 0)
        self.goal_pos = (0, grid_size - 1)
        self.agent_pos = self.start_pos

        self.action_space = spaces.Discrete(2)  # 0: UP, 1: RIGHT
        self.observation_space = spaces.Discrete(grid_size * grid_size)

        self.resources = {(int(r['x']), int(r['y'])) for r in resource_data}

    def reset(self):
        self.agent_pos = self.start_pos
        return self._get_state(), {}

    def step(self, action):
        x, y = self.agent_pos
        if action == 0 and x > 0:
            x -= 1
        elif action == 1 and y < self.grid_size - 1:
            y += 1
        self.agent_pos = (x, y)

        reward = -0.1
        if self.agent_pos in self.resources:
            reward = 1.0
        done = self.agent_pos == self.goal_pos

        return self._get_state(), reward, done, {}

    def _get_state(self):
        return self.agent_pos[0] * self.grid_size + self.agent_pos[1]

    def render_path(self, path):
        fig, ax = plt.subplots(figsize=(5, 5))
        grid = np.zeros((self.grid_size, self.grid_size))

        for (x, y) in self.resources:
            grid[x, y] = 1  # Resource cell

        for (x, y) in path:
            grid[x, y] = 0.5  # Path cell

        sx, sy = self.start_pos
        gx, gy = self.goal_pos
        ax.imshow(grid, cmap='Pastel1')

        ax.text(sy, sx, 'S', va='center', ha='center', fontsize=12, fontweight='bold')
        ax.text(gy, gx, 'G', va='center', ha='center', fontsize=12, fontweight='bold')
        ax.text(self.agent_pos[1], self.agent_pos[0], 'A', va='center', ha='center', fontsize=12, fontweight='bold')

        ax.set_xticks(np.arange(-0.5, self.grid_size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.grid_size, 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        plt.title("Agent Path Visualization")
        plt.show()

# --- DQN Model ---
class DQN(nn.Module):
    def _init_(self, state_size, action_size):
        super(DQN, self)._init_()
        self.fc1 = nn.Linear(state_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, action_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# --- Training Function ---
def train_dqn(env, episodes=1000, gamma=0.95, epsilon=1.0, epsilon_decay=0.995,
              min_epsilon=0.01, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    memory = deque(maxlen=2000)
    state_size = env.observation_space.n
    action_size = env.action_space.n
    model = DQN(state_size, action_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    rewards = []

    for episode in trange(episodes, desc="Training Progress"):
        state, _ = env.reset()
        total_reward = 0
        done = False
        path = [env.agent_pos]

        while not done:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.eye(state_size)[state].unsqueeze(0).to(device)
                    q_values = model(state_tensor)
                    action = torch.argmax(q_values).item()

            next_state, reward, done, _ = env.step(action)
            path.append(env.agent_pos)
            total_reward += reward
            memory.append((state, action, reward, next_state, done))
            state = next_state

            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                states, actions, rewards_, next_states, dones = zip(*batch)

                states_tensor = torch.stack([torch.eye(state_size)[s] for s in states]).to(device)
                next_states_tensor = torch.stack([torch.eye(state_size)[s] for s in next_states]).to(device)
                actions_tensor = torch.LongTensor(actions).to(device)
                rewards_tensor = torch.FloatTensor(rewards_).to(device)
                dones_tensor = torch.FloatTensor(dones).to(device)

                q_values = model(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze()
                next_q = model(next_states_tensor).detach()
                q_targets = rewards_tensor + gamma * torch.max(next_q, dim=1)[0] * (1 - dones_tensor)

                loss = criterion(q_values, q_targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        rewards.append(total_reward)

    return model, rewards

# --- Load JSON data from file ---
import os
json_file_path = "/extracted_data.json"  # Upload to Colab before running
with open(json_file_path, "r") as f:
    resource_data = json.load(f)

# --- Run Training ---
env = GridEnv(resource_data=resource_data)
model, rewards = train_dqn(env, episodes=1000)

# --- Plot Rewards ---
plt.plot(rewards)
plt.xlabel("Episodes")
plt.ylabel("Cumulative Reward")
plt.title("Training Progress")
plt.grid(True)
plt.show()

# --- Visualize Trained Path ---
def simulate_best_path(env, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state, _ = env.reset()
    done = False
    path = [env.agent_pos]

    while not done:
        with torch.no_grad():
            state_tensor = torch.eye(env.observation_space.n)[state].unsqueeze(0).to(device)
            q_values = model(state_tensor)
            action = torch.argmax(q_values).item()
        next_state, reward, done, _ = env.step(action)
        state = next_state
        path.append(env.agent_pos)

    env.render_path(path)

simulate_best_path(env, model)