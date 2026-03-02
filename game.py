# game.py
import tkinter as tk
import random
import numpy as np

BLOCK_SIZE = 20
WIDTH = 600
HEIGHT = 400
SPEED = 50  # ms
STARVATION_FACTOR = 100
REWARD_DEATH = -10
REWARD_FOOD = 10
REWARD_STEP = -0.01
REWARD_CLOSER = 0.2
REWARD_FARTHER = -0.2

class SnakeGame:

    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Snake AI")
        self.is_running = True
        self.window.protocol("WM_DELETE_WINDOW", self._on_close)

        self.canvas = tk.Canvas(self.window, width=WIDTH, height=HEIGHT, bg="black")
        self.canvas.pack()

        self.reset()

    def _on_close(self):
        self.is_running = False
        if self.window.winfo_exists():
            self.window.destroy()

    def is_open(self):
        return self.is_running and self.window.winfo_exists()

    def reset(self):
        self.direction = "RIGHT"
        self.head = (WIDTH//2, HEIGHT//2)

        self.snake = [
            self.head,
            (self.head[0] - BLOCK_SIZE, self.head[1]),
            (self.head[0] - 2*BLOCK_SIZE, self.head[1])
        ]

        self.score = 0
        self.frame_iteration = 0
        self._place_food()

    def _place_food(self):
        while True:
            x = random.randint(0, (WIDTH - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (HEIGHT - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            self.food = (x, y)
            if self.food not in self.snake:
                return

    def play_step(self, action):
        if not self.is_open():
            return REWARD_DEATH, True, self.score

        self.frame_iteration += 1

        distance_before = abs(self.head[0] - self.food[0]) + abs(self.head[1] - self.food[1])

        self._move(action)
        self.snake.insert(0, self.head)

        reward = REWARD_STEP
        game_over = False

        if self.is_collision() or self.frame_iteration > STARVATION_FACTOR * len(self.snake):
            reward = REWARD_DEATH
            game_over = True
            return reward, game_over, self.score

        if self.head == self.food:
            self.score += 1
            reward = REWARD_FOOD
            self._place_food()
        else:
            self.snake.pop()
            distance_after = abs(self.head[0] - self.food[0]) + abs(self.head[1] - self.food[1])
            if distance_after < distance_before:
                reward += REWARD_CLOSER
            elif distance_after > distance_before:
                reward += REWARD_FARTHER

        try:
            self._update_ui()
            self.window.update()
        except tk.TclError:
            self.is_running = False
            return REWARD_DEATH, True, self.score

        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head

        if pt[0] < 0 or pt[0] >= WIDTH or pt[1] < 0 or pt[1] >= HEIGHT:
            return True

        if pt in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        self.canvas.delete("all")

        for segment in self.snake:
            self.canvas.create_rectangle(
                segment[0], segment[1],
                segment[0]+BLOCK_SIZE,
                segment[1]+BLOCK_SIZE,
                fill="green"
            )

        self.canvas.create_rectangle(
            self.food[0], self.food[1],
            self.food[0]+BLOCK_SIZE,
            self.food[1]+BLOCK_SIZE,
            fill="red"
        )

        self.canvas.create_text(
            50, 10,
            text=f"Score: {self.score}",
            fill="white"
        )

    def _move(self, action):
        clock_wise = ["RIGHT", "DOWN", "LEFT", "UP"]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1,0,0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0,1,0]):
            new_dir = clock_wise[(idx + 1) % 4]
        else:
            new_dir = clock_wise[(idx - 1) % 4]

        self.direction = new_dir

        x, y = self.head
        if self.direction == "RIGHT":
            x += BLOCK_SIZE
        elif self.direction == "LEFT":
            x -= BLOCK_SIZE
        elif self.direction == "UP":
            y -= BLOCK_SIZE
        elif self.direction == "DOWN":
            y += BLOCK_SIZE

        self.head = (x, y)