# agent.py
from collections import deque
import random

import numpy as np
import torch

from game import BLOCK_SIZE, HEIGHT, WIDTH
from model import Linear_QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
N_ACTIONS = 3
MIN_EPSILON = 10
EPSILON_START = 120
EPSILON_RANDOM_MAX = 200
HIDDEN_SIZE = 512


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)

        self.grid_width = WIDTH // BLOCK_SIZE
        self.grid_height = HEIGHT // BLOCK_SIZE
        self.state_size = 11 + (self.grid_width * self.grid_height)

        self.model = Linear_QNet(self.state_size, HIDDEN_SIZE, N_ACTIONS)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]

        point_l = (head[0] - BLOCK_SIZE, head[1])
        point_r = (head[0] + BLOCK_SIZE, head[1])
        point_u = (head[0], head[1] - BLOCK_SIZE)
        point_d = (head[0], head[1] + BLOCK_SIZE)

        dir_l = game.direction == "LEFT"
        dir_r = game.direction == "RIGHT"
        dir_u = game.direction == "UP"
        dir_d = game.direction == "DOWN"

        binary_state = [
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            dir_l,
            dir_r,
            dir_u,
            dir_d,

            game.food[0] < head[0],
            game.food[0] > head[0],
            game.food[1] < head[1],
            game.food[1] > head[1]
        ]

        grid = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)

        for segment_x, segment_y in game.snake:
            grid_y = segment_y // BLOCK_SIZE
            grid_x = segment_x // BLOCK_SIZE
            if 0 <= grid_y < self.grid_height and 0 <= grid_x < self.grid_width:
                grid[grid_y, grid_x] = 1.0 / 3.0

        head_x, head_y = game.snake[0]
        head_grid_y = head_y // BLOCK_SIZE
        head_grid_x = head_x // BLOCK_SIZE
        if 0 <= head_grid_y < self.grid_height and 0 <= head_grid_x < self.grid_width:
            grid[head_grid_y, head_grid_x] = 2.0 / 3.0

        food_x, food_y = game.food
        food_grid_y = food_y // BLOCK_SIZE
        food_grid_x = food_x // BLOCK_SIZE
        if 0 <= food_grid_y < self.grid_height and 0 <= food_grid_x < self.grid_width:
            grid[food_grid_y, food_grid_x] = 1.0

        state_array = np.array(binary_state, dtype=np.float32)
        return np.concatenate((state_array, grid.flatten()))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = max(MIN_EPSILON, EPSILON_START - self.n_games)
        final_move: list[int] = [0, 0, 0]

        if random.randint(0, EPSILON_RANDOM_MAX) < self.epsilon:
            move: int = random.randint(0, N_ACTIONS - 1)
            final_move[move] = 1
        else:
            state0 = torch.as_tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = int(torch.argmax(prediction).item())
            final_move[move] = 1

        return final_move