import gym
import numpy as np
import pygame
import json
import warnings
import pickle  # For saving and loading the trained model
from gym import spaces

warnings.simplefilter(action='ignore', category=DeprecationWarning)

class GridEnv(gym.Env):
    metadata = {"render_modes": ["human"]}  

    def __init__(self, render_mode=None, json_path="/home/niveditha/Desktop/clg_work/SEM6/extracted_data.json"):
        super().__init__()

        # Load JSON data
        with open(json_path, "r") as file:
            self.resource_data = json.load(file)

        # Scaling the coordinates by 100
        self.size = max(
            max(int(float(data["x_coordinate"]) * 100), int(float(data["y_coordinate"]) * 100))
            for data in self.resource_data.values()
        ) + 1  # Buffer added

        self.window_size = 600  
        self.observation_space = spaces.Discrete(self.size * self.size)  # Flattened grid

        # Action space: 0 (UP), 1 (RIGHT)
        self.action_space = spaces.Discrete(2)

        self.render_mode = render_mode

        # Track resource locations
        self.symbols = {}
        self.reward_map = np.full((self.size, self.size), -0.1)  # Default penalty

        for name, coords in self.resource_data.items():
            scaled_x = int(float(coords["x_coordinate"]) * 100)
            scaled_y = int(float(coords["y_coordinate"]) * 100)
            self.symbols[(scaled_y, scaled_x)] = "📖"
            self.reward_map[scaled_y, scaled_x] = 1  # +1 reward for books

        # Initial positions
        self.agent_pos = np.array([self.size - 1, 0])  # Bottom-left
        self.target_pos = np.array([0, self.size - 1])  # Top-right
        self.visited_cells = []  # Track path for visualization

        # Q-learning
        self.q_table = np.zeros((self.size * self.size, self.action_space.n))  # Q-table

        if self.render_mode == "human":
            self._init_pygame()

    def _init_pygame(self):
        pygame.init()
        self.window = pygame.display.set_mode((self.window_size, self.window_size))
        self.clock = pygame.time.Clock()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = np.array([self.size - 1, 0])
        self.visited_cells = [tuple(self.agent_pos)]
        return self._get_state(), {}

    def step(self, action):
        """
        Actions:
        - 0: Move UP
        - 1: Move RIGHT
        """
        if action == 0 and self.agent_pos[0] > 0:  # UP
            self.agent_pos[0] -= 1
        elif action == 1 and self.agent_pos[1] < self.size - 1:  # RIGHT
            self.agent_pos[1] += 1

        self.visited_cells.append(tuple(self.agent_pos))  # Track path
        done = np.array_equal(self.agent_pos, self.target_pos)
        reward = self.reward_map[self.agent_pos[0], self.agent_pos[1]]

        return self._get_state(), reward, done, {}

    def _get_state(self):
        return self.agent_pos[0] * self.size + self.agent_pos[1]  # Flattened state

    def train_q_learning(self, episodes=10000, alpha=0.1, gamma=0.95, epsilon=0.1):
        """
        Q-learning training loop
        """
        for episode in range(episodes):
            state = self.reset()[0]
            done = False

            while not done:
                if np.random.rand() < epsilon:
                    action = self.action_space.sample()
                else:
                    action = np.argmax(self.q_table[state])

                next_state, reward, done, _ = self.step(action)
                best_next_action = np.argmax(self.q_table[next_state])

                # Update Q-value
                self.q_table[state, action] = self.q_table[state, action] + alpha * (
                    reward + gamma * self.q_table[next_state, best_next_action] - self.q_table[state, action]
                )

                state = next_state

        # Save trained model
        with open("q_table.pkl", "wb") as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self, path="q_table.pkl"):
        with open(path, "rb") as f:
            self.q_table = pickle.load(f)

    def render(self):
        if self.render_mode == "human":
            cell_size = max(self.window_size // self.size, 5)
            font_size = max(cell_size // 2, 10)

            try:
                font = pygame.font.Font("fonts/NotoEmoji-Bold.ttf", font_size)
            except:
                font = pygame.font.Font(None, font_size)

            self.window.fill((255, 255, 255))

            for x in range(self.size):
                for y in range(self.size):
                    rect = pygame.Rect(y * cell_size, x * cell_size, cell_size, cell_size)
                    pygame.draw.rect(self.window, (200, 200, 200), rect, 1)

                    if (x, y) in self.symbols:
                        text = font.render(self.symbols[(x, y)], True, (0, 0, 0))
                        text_rect = text.get_rect(center=(y * cell_size + cell_size // 2, x * cell_size + cell_size // 2))
                        self.window.blit(text, text_rect)

                    if (x, y) in self.visited_cells:
                        pygame.draw.rect(self.window, (0, 255, 0), rect)  # Green path

            pygame.draw.circle(
                self.window,
                (0, 0, 255),
                (self.agent_pos[1] * cell_size + cell_size // 2, self.agent_pos[0] * cell_size + cell_size // 2),
                cell_size // 3
            )

            pygame.draw.rect(
                self.window,
                (255, 0, 0),
                (self.target_pos[1] * cell_size, self.target_pos[0] * cell_size, cell_size, cell_size)
            )

            pygame.display.flip()
            self.clock.tick(10)

    def close(self):
        if hasattr(self, "window"):
            pygame.quit()