import random
from multiprocessing import process
import neat
import os
import numpy as np
import visualize
import math
import pygame

from game3 import Snake, Direction, Point, SnakeGameAI

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
        wall_ul = 0.
        wall_ur = 0.
        wall_dl = 0.
        wall_dr = 0.

        for i in range(1, 21):
            point_l = Point(head.x - 20 * i, head.y)
            if game.is_wall(point_l):
                wall_l = i / 20
                break
        for i in range(1, 21):
            point_r = Point(head.x + 20 * i, head.y)
            if game.is_wall(point_r):
                wall_r = i / 20
                break
        for i in range(1, 21):
            point_u = Point(head.x, head.y - 20 * i)
            if game.is_wall(point_u):
                wall_u = i / 20
                break
        for i in range(1, 21):
            point_d = Point(head.x, head.y + 20 * i)
            if game.is_wall(point_d):
                wall_d = i / 20
                break
        for i in range(1, 21):
            point_ul = Point(head.x - 20 * i, head.y - 20 * i)
            if game.is_wall(point_ul):
                wall_ul = i / 20
                break
        for i in range(1, 21):
            point_ur = Point(head.x + 20 * i, head.y - 20 * i)
            if game.is_wall(point_ur):
                wall_ur = i / 20
                break
        for i in range(1, 21):
            point_dl = Point(head.x - 20 * i, head.y + 20 * i)
            if game.is_wall(point_dl):
                wall_dl = i / 20
                break
        for i in range(1, 21):
            point_dr = Point(head.x + 20 * i, head.y + 20 * i)
            if game.is_wall(point_dr):
                wall_dr = i / 20
                break

        me_l = 0.
        me_r = 0.
        me_u = 0.
        me_d = 0.
        me_ul = 0.
        me_ur = 0.
        me_dl = 0.
        me_dr = 0.

        for i in range(1, 21):
            point_l = Point(head.x - 20 * i, head.y)
            if game.is_me(point_l):
                me_l = i / 20
                break
        for i in range(1, 21):
            point_r = Point(head.x + 20 * i, head.y)
            if game.is_me(point_r):
                me_r = i / 20
                break
        for i in range(1, 21):
            point_u = Point(head.x, head.y - 20 * i)
            if game.is_me(point_u):
                me_u = i / 20
                break
        for i in range(1, 21):
            point_d = Point(head.x, head.y + 20 * i)
            if game.is_me(point_d):
                me_d = i / 20
                break
        for i in range(1, 21):
            point_ul = Point(head.x - 20 * i, head.y - 20 * i)
            if game.is_me(point_ul):
                me_ul = i / 20
                break
        for i in range(1, 21):
            point_ur = Point(head.x + 20 * i, head.y - 20 * i)
            if game.is_me(point_ur):
                me_ur = i / 20
                break
        for i in range(1, 21):
            point_dl = Point(head.x - 20 * i, head.y + 20 * i)
            if game.is_me(point_dl):
                me_dl = i / 20
                break
        for i in range(1, 21):
            point_dr = Point(head.x + 20 * i, head.y + 20 * i)
            if game.is_me(point_dr):
                me_dr = i / 20
                break

        food_l = 0.
        food_r = 0.
        food_u = 0.
        food_d = 0.
        food_ul = 0.
        food_ur = 0.
        food_dl = 0.
        food_dr = 0.

        for i in range(1, 21):
            point_l = Point(head.x - 20 * i, head.y)
            if point_l==game.food:
                food_l = i / 20
                break
        for i in range(1, 21):
            point_r = Point(head.x + 20 * i, head.y)
            if point_r==game.food:
                food_r = i / 20
                break
        for i in range(1, 21):
            point_u = Point(head.x, head.y - 20 * i)
            if point_u==game.food:
                food_u = i / 20
                break
        for i in range(1, 21):
            point_d = Point(head.x, head.y + 20 * i)
            if point_d==game.food:
                food_d = i / 20
                break
        for i in range(1, 21):
            point_ul = Point(head.x - 20 * i, head.y - 20 * i)
            if point_ul==game.food:
                food_ul = i / 20
                break
        for i in range(1, 21):
            point_ur = Point(head.x + 20 * i, head.y - 20 * i)
            if point_ur==game.food:
                food_ur = i / 20
                break
        for i in range(1, 21):
            point_dl = Point(head.x - 20 * i, head.y + 20 * i)
            if point_dl==game.food:
                food_dl = i / 20
                break
        for i in range(1, 21):
            point_dr = Point(head.x + 20 * i, head.y + 20 * i)
            if point_dr==game.food:
                food_dr = i / 20
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

            # Wall Back
            (dir_u and wall_d) or
            (dir_d and wall_u) or
            (dir_l and wall_r) or
            (dir_r and wall_l)
            ,

            # Wall Straight Right
            (dir_l and wall_ul) or
            (dir_r and wall_dr) or
            (dir_u and wall_ur) or
            (dir_d and wall_dl)
            ,

            # Wall Straight Left
            (dir_l and wall_dl) or
            (dir_r and wall_ur) or
            (dir_u and wall_ul) or
            (dir_d and wall_dr)
            ,

            # Wall Back Right
            (dir_l and wall_dr) or
            (dir_r and wall_ul) or
            (dir_u and wall_dl) or
            (dir_d and wall_ur)
            ,

            # Wall Back Left
            (dir_l and wall_ur) or
            (dir_r and wall_dl) or
            (dir_u and wall_dr) or
            (dir_d and wall_ul)
            ,

            # me Straight
            (dir_l and me_l) or
            (dir_r and me_r) or
            (dir_u and me_u) or
            (dir_d and me_d)
            ,

            # me Right
            (dir_u and me_r) or
            (dir_d and me_l) or
            (dir_l and me_u) or
            (dir_r and me_d)
            ,

            # me Left
            (dir_u and me_l) or
            (dir_d and me_r) or
            (dir_l and me_d) or
            (dir_r and me_u)
            ,

            # me Back
            (dir_u and me_d) or
            (dir_d and me_u) or
            (dir_l and me_r) or
            (dir_r and me_l)
            ,

            # me Straight Right
            (dir_l and me_ul) or
            (dir_r and me_dr) or
            (dir_u and me_ur) or
            (dir_d and me_dl)
            ,

            # me Straight Left
            (dir_l and me_dl) or
            (dir_r and me_ur) or
            (dir_u and me_ul) or
            (dir_d and me_dr)
            ,

            # me Back Right
            (dir_l and me_dr) or
            (dir_r and me_ul) or
            (dir_u and me_dl) or
            (dir_d and me_ur)
            ,

            # me Back Left
            (dir_l and me_ur) or
            (dir_r and me_dl) or
            (dir_u and me_dr) or
            (dir_d and me_ul)
            ,
            
            # food Straight
            (dir_l and food_l) or
            (dir_r and food_r) or
            (dir_u and food_u) or
            (dir_d and food_d)
            ,

            # food Right
            (dir_u and food_r) or
            (dir_d and food_l) or
            (dir_l and food_u) or
            (dir_r and food_d)
            ,

            # food Left
            (dir_u and food_l) or
            (dir_d and food_r) or
            (dir_l and food_d) or
            (dir_r and food_u)
            ,

            # food Back
            (dir_u and food_d) or
            (dir_d and food_u) or
            (dir_l and food_r) or
            (dir_r and food_l)
            ,

            # food Straight Right
            (dir_l and food_ul) or
            (dir_r and food_dr) or
            (dir_u and food_ur) or
            (dir_d and food_dl)
            ,

            # food Straight Left
            (dir_l and food_dl) or
            (dir_r and food_ur) or
            (dir_u and food_ul) or
            (dir_d and food_dr)
            ,

            # food Back Right
            (dir_l and food_dr) or
            (dir_r and food_ul) or
            (dir_u and food_dl) or
            (dir_d and food_ur)
            ,

            # food Back Left
            (dir_l and food_ur) or
            (dir_r and food_dl) or
            (dir_u and food_dr) or
            (dir_d and food_ul)
            ,
        ]

        return np.array(state), math.sqrt(math.pow(game.food.x-game.head.x, 2)+math.pow(game.food.y-game.head.y, 2))


agent = Agent()

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
            state_old, dis_old = agent.get_state(snake)
            output = nets[x].activate(state_old)
            final_move = [0, 0, 0]
            final_move[np.argmax(output)]=1
            done = False
            reward, done, score = snake.play_step(final_move)
            state_new, dis_new = agent.get_state(snake)
            if dis_new<dis_old:
                reward += .05
            else:
                reward -= .05
            ge[x].fitness += reward
            if done:
                snake.reset()
                snakes.pop(x)
                nets.pop(x)
                ge.pop(x)
        S_AI.update(len(snakes))

def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stat = neat.StatisticsReporter()
    p.add_reporter(stat)

    winner = p.run(main, 500)
    print('\nBest genome:\n{!s}'.format(winner))

    visualize.draw_net(config, winner, False)
    visualize.plot_stats(stat, ylog=False, view=False)

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)