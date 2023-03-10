import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

SEE_PROGRESS = False # Set this to False to speed up learning
REPLAY_FILE = False # Set this to True to display the best genome from last file

if SEE_PROGRESS:
    pygame.init()
    font = pygame.font.Font('04B_19__.TTF', 30)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED2 = (200, 0, 0)
RED1 = (200, 100, 0)
BLUE2 = (0, 0, 255)
BLUE1 = (0, 100, 255)
GREEN2 = (0, 180, 0)
GREEN1 = (100, 200, 100)
BLACK = (0, 0, 0)

BLOCK_SIZE = 40

class Snake:
    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.reset()

    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.lifetime = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.lifetime += 1
        self.frame_iteration += 1
        # 1. collect user input
        #for event in pygame.event.get():
        #    if event.type == pygame.QUIT:
        #        pygame.quit()
        #        quit()

        # 2. move
        self._move(action)  # update the head
        self.snake.insert(0, self.head)

        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 125:
            reward = self.lifetime+pow(2, self.score)+500*pow(self.score, 2.1)-0.25*pow(self.lifetime, 1.3)*pow(self.score, 1.2)
            game_over = True
            return reward, game_over, self.score

        # 3. place new food or just move
        if self.head == self.food:
            self.frame_iteration = 0
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()

        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False

    def is_wall(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True

        return False

    def is_me(self, pt=None):
        if pt is None:
            pt = self.head
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False

    def _move(self, action):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # right turn r -> d -> l -> u
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # left turn r -> u -> l -> d

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)

class SnakeGameAI:
    def __init__(self, snakes, w, h, display):
        self.w = w
        self.h = h
        self.display = display
        self.clock = pygame.time.Clock()
        self.snakes = snakes

    def update(self, len, max_score):
        self.display.fill(BLACK)
        for x, s in enumerate(self.snakes):
            pygame.draw.rect(self.display, RED1, pygame.Rect(s.food.x, s.food.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, RED2, pygame.Rect(s.food.x + 2, s.food.y + 2, BLOCK_SIZE-4, BLOCK_SIZE-4))
        for x, s in enumerate(self.snakes):
            if s.score < max_score:
                for pt in s.snake:
                    pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                    pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 2, pt.y + 2, BLOCK_SIZE - 4, BLOCK_SIZE - 4))
        for x, s in enumerate(self.snakes):
            if s.score >= max_score:
                for pt in s.snake:
                    pygame.draw.rect(self.display, GREEN1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                    pygame.draw.rect(self.display, GREEN2, pygame.Rect(pt.x + 2, pt.y + 2, BLOCK_SIZE - 4, BLOCK_SIZE - 4))
        left = font.render(str(len)+"/1000 Left", 1, (255, 255, 255))
        score = font.render("Max Score : "+str(max_score), 1, (255, 255, 255))
        self.display.blit(left, (10, 10))
        self.display.blit(score, (10, left.get_height()+10+5))
        pygame.display.flip()
        self.clock.tick(60)