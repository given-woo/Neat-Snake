import random

import neat
import os
import numpy as np
import visualize
import pygame
import math

from game2 import Snake, SnakeGameAI, Direction, Point

def mapFromTo(x,a,b,c,d):
   y=(x-a)/(b-a)*(d-c)+c
   return y

class Agent:
    def __init__(self):
        self.n_games = 0

    def get_state(self, game):
        head = game.snake[0]

        wall_l = 0.
        wall_r = 0.
        wall_u = 0.
        wall_d = 0.

        for i in range(1, 11):
            point_l = Point(head.x - 40 * i, head.y)
            if game.is_wall(point_l):
                wall_l = i / 10
                break
        for i in range(1, 11):
            point_r = Point(head.x + 40 * i, head.y)
            if game.is_wall(point_r):
                wall_r = i / 10
                break
        for i in range(1, 11):
            point_u = Point(head.x, head.y - 40 * i)
            if game.is_wall(point_u):
                wall_u = i / 10
                break
        for i in range(1, 11):
            point_d = Point(head.x, head.y + 40 * i)
            if game.is_wall(point_d):
                wall_d = i / 10
                break

        me_l = 0.
        me_r = 0.
        me_u = 0.
        me_d = 0.

        for i in range(1, 11):
            point_l = Point(head.x - 40 * i, head.y)
            if game.is_me(point_l):
                me_l = i / 10
                break
        for i in range(1, 11):
            point_r = Point(head.x + 40 * i, head.y)
            if game.is_me(point_r):
                me_r = i / 10
                break
        for i in range(1, 11):
            point_u = Point(head.x, head.y - 40 * i)
            if game.is_me(point_u):
                me_u = i / 10
                break
        for i in range(1, 11):
            point_d = Point(head.x, head.y + 40 * i)
            if game.is_me(point_d):
                me_d = i / 10
                break

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Wall Straight
            (dir_l and wall_l) or
            (dir_r and wall_r) or
            (dir_u and wall_u) or
            (dir_d and wall_d)
            ,

            # Wall Right
            (dir_u and wall_r) or
            (dir_d and wall_l) or
            (dir_l and wall_u) or
            (dir_r and wall_d)
            ,

            # Wall Left
            (dir_u and wall_l) or
            (dir_d and wall_r) or
            (dir_l and wall_d) or
            (dir_r and wall_u)
            ,

            # Me Straight
            (dir_l and me_l) or
            (dir_r and me_r) or
            (dir_u and me_u) or
            (dir_d and me_d)
            ,

            # Me Right
            (dir_u and me_r) or
            (dir_d and me_l) or
            (dir_l and me_u) or
            (dir_r and me_d)
            ,

            # Me Left
            (dir_u and me_l) or
            (dir_d and me_r) or
            (dir_l and me_d) or
            (dir_r and me_u)
            ,

            # Food Location
            game.food.x < head.x,
            game.food.x > head.x,
            game.food.y < head.y,
            game.food.y > head.y,

            # Food Distance
            math.sqrt(math.pow(game.food.x - game.head.x, 2) + math.pow(game.food.y - game.head.y, 2))/400,

            # Move Direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
        ]

        return np.array(state), math.sqrt(math.pow(game.food.x-game.head.x, 2)+math.pow(game.food.y-game.head.y, 2))
def main(genomes, config):
    display = pygame.display.set_mode((400, 400))
    pygame.display.set_caption('Snake')

    nets =[]
    ge = []
    snakes = []

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        snakes.append(Snake(400, 400))
        g.fitness = 0
        ge.append(g)

    while len(snakes) > 0:
        S_AI = SnakeGameAI(snakes, 400, 400, display)
        for x, snake in enumerate(snakes):
            agent = Agent()
            state_old, dis_old = agent.get_state(snake)
            output = nets[x].activate(state_old)
            final_move = [0, 0, 0]
            final_move[output.index(max(output))]=1
            reward, done, score = snake.play_step(final_move)
            state_new, dis_new = agent.get_state(snake)
            if dis_new<dis_old:
                reward += 1
            else:
                reward -= 1
            ge[x].fitness += reward

            if done:
                snake.reset()
                snakes.pop(x)
                nets.pop(x)
                ge.pop(x)
        S_AI.update()

def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stat = neat.StatisticsReporter()
    p.add_reporter(stat)

    winner = p.run(main, 5000)
    print('\nBest genome:\n{!s}'.format(winner))

    visualize.draw_net(config, winner, False)
    visualize.plot_stats(stat, ylog=False, view=False)

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)
    # start 1 : 52 am