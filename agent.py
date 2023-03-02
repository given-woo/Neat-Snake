import neat
import os
import numpy as np
import visualize
import math
import pygame
import pickle
import multiprocessing as mp

from game import Snake, Direction, Point, SnakeGameAI

SEE_PROGRESS = True # Set this to False to speed up learning
REPLAY_FILE = True # Set this to True to display the best genome from last file

BLOCK_WIDTH = 40 # Size of the block
BLOCKS = 10 # Number of blocks ex) 20 -> (20, 20)
GENS = 10000 # Generation to run

def replay_genome(config_path, genome_path="winner.pkl"):
    # Load requried NEAT config
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                neat.DefaultStagnation, config_path)
    # Unpickle saved winner
    with open(genome_path, "rb") as f:
        genome = pickle.load(f)
    print(genome)
    # Convert loaded genome into required data structure
    genomes = [(1, genome)]

    # Call game with only the loaded genome
    while True:
        main_with_progress(genomes, config)
    # return genomes

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

        me_l = 0.
        me_r = 0.
        me_u = 0.
        me_d = 0.
        me_ul = 0.
        me_ur = 0.
        me_dl = 0.
        me_dr = 0.

        food_l = 0.
        food_r = 0.
        food_u = 0.
        food_d = 0.
        food_ul = 0.
        food_ur = 0.
        food_dl = 0.
        food_dr = 0.

        for i in range(1, BLOCKS+1):
            point_l = Point(head.x - BLOCK_WIDTH * i, head.y)
            if game.is_wall(point_l):
                wall_l = max(wall_l, 1 / i)
            if game.is_me(point_l):
                me_l = max(me_l, 1 / i)
            if point_l == game.food:
                food_l = max(food_l, 1 / i)

            point_r = Point(head.x + BLOCK_WIDTH * i, head.y)
            if game.is_wall(point_r):
                wall_r = max(wall_r, 1 / i)
            if game.is_me(point_r):
                me_r = max(me_r, 1 / i)
            if point_r == game.food:
                food_r = max(food_r, 1 / i)

            point_u = Point(head.x, head.y - BLOCK_WIDTH * i)
            if game.is_wall(point_u):
                wall_u = max(wall_u, 1 / i)
            if game.is_me(point_u):
                me_u = max(me_u, 1 / i)
            if point_u == game.food:
                food_u = max(food_u, 1 / i)

            point_d = Point(head.x, head.y + BLOCK_WIDTH * i)
            if game.is_wall(point_d):
                wall_d = max(wall_d, 1 / i)
            if game.is_me(point_d):
                me_d = max(me_d, 1 / i)
            if point_d == game.food:
                food_d = max(food_d, 1 / i)

            point_ul = Point(head.x - BLOCK_WIDTH * i, head.y - BLOCK_WIDTH * i)
            if game.is_wall(point_ul):
                wall_ul = max(wall_ul, 1 / i)
            if game.is_me(point_ul):
                me_ul = max(me_ul, 1 / i)
            if point_ul == game.food:
                food_ul = max(food_ul, 1 / i)

            point_ur = Point(head.x + BLOCK_WIDTH * i, head.y - BLOCK_WIDTH * i)
            if game.is_wall(point_ur):
                wall_ur = max(wall_ur, 1 / i)
            if game.is_me(point_ur):
                me_ur = max(me_ur, 1 / i)
            if point_ur == game.food:
                food_ur = max(food_ur, 1 / i)

            point_dl = Point(head.x - BLOCK_WIDTH * i, head.y + BLOCK_WIDTH * i)
            if game.is_wall(point_dl):
                wall_dl = max(wall_dl, 1 / i)
            if game.is_me(point_dl):
                me_dl = max(me_dl, 1 / i)
            if point_dl == game.food:
                food_dl = max(food_dl, 1 / i)

            point_dr = Point(head.x + BLOCK_WIDTH * i, head.y + BLOCK_WIDTH * i)
            if game.is_wall(point_dr):
                wall_dr = max(wall_dr, 1 / i)
            if game.is_me(point_dr):
                me_dr = max(me_dr, 1 / i)
            if point_dr == game.food:
                food_dr = max(food_dr, 1 / i)


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

            # Wall Back
            (dir_l and wall_r) or
            (dir_r and wall_l) or
            (dir_u and wall_d) or
            (dir_d and wall_u)
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

            # Self Straight
            (dir_l and me_l) or
            (dir_r and me_r) or
            (dir_u and me_u) or
            (dir_d and me_d)
            ,

            # Self Back
            (dir_l and me_r) or
            (dir_r and me_l) or
            (dir_u and me_d) or
            (dir_d and me_u)
            ,

            # Self Right
            (dir_u and me_r) or
            (dir_d and me_l) or
            (dir_l and me_u) or
            (dir_r and me_d)
            ,

            # Self Left
            (dir_u and me_l) or
            (dir_d and me_r) or
            (dir_l and me_d) or
            (dir_r and me_u)
            ,

            # Self Straight Right
            (dir_l and me_ul) or
            (dir_r and me_dr) or
            (dir_u and me_ur) or
            (dir_d and me_dl)
            ,

            # Self Straight Left
            (dir_l and me_dl) or
            (dir_r and me_ur) or
            (dir_u and me_ul) or
            (dir_d and me_dr)
            ,

            # Self Back Right
            (dir_l and me_dr) or
            (dir_r and me_ul) or
            (dir_u and me_dl) or
            (dir_d and me_ur)
            ,

            # Self Back Left
            (dir_l and me_ur) or
            (dir_r and me_dl) or
            (dir_u and me_dr) or
            (dir_d and me_ul)
            ,

            # Food Straight
            (dir_l and food_l) or
            (dir_r and food_r) or
            (dir_u and food_u) or
            (dir_d and food_d)
            ,

            # Food Back
            (dir_l and food_r) or
            (dir_r and food_l) or
            (dir_u and food_d) or
            (dir_d and food_u)
            ,

            # Food Right
            (dir_u and food_r) or
            (dir_d and food_l) or
            (dir_l and food_u) or
            (dir_r and food_d)
            ,

            # Food Left
            (dir_u and food_l) or
            (dir_d and food_r) or
            (dir_l and food_d) or
            (dir_r and food_u)
            ,

            # Food Straight Right
            (dir_l and food_ul) or
            (dir_r and food_dr) or
            (dir_u and food_ur) or
            (dir_d and food_dl)
            ,

            # Food Straight Left
            (dir_l and food_dl) or
            (dir_r and food_ur) or
            (dir_u and food_ul) or
            (dir_d and food_dr)
            ,

            # Food Back Right
            (dir_l and food_dr) or
            (dir_r and food_ul) or
            (dir_u and food_dl) or
            (dir_d and food_ur)
            ,

            # Food Back Left
            (dir_l and food_ur) or
            (dir_r and food_dl) or
            (dir_u and food_dr) or
            (dir_d and food_ul)
            ,
        ]

        return np.array(state), math.sqrt(math.pow(game.food.x-game.head.x, 2)+math.pow(game.food.y-game.head.y, 2))


agent = Agent()

def main(genomes, config):
    nets =[]
    ge = []
    snakes = []

    max_score = 0

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        snakes.append(Snake(BLOCK_WIDTH*BLOCKS, BLOCK_WIDTH*BLOCKS))
        g.fitness = 0
        ge.append(g)

    while len(snakes) > 0:
        for x, snake in enumerate(snakes):
            state_old, dis_old = agent.get_state(snake)
            output = nets[x].activate(state_old)
            final_move = [0, 0, 0]
            final_move[np.argmax(output)]=1
            reward, done, score = snake.play_step(final_move)
            if score > max_score:
                max_score = score
            ge[x].fitness = reward
            if done:
                snake.reset()
                snakes.pop(x)
                nets.pop(x)
                ge.pop(x)

def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stat = neat.StatisticsReporter()
    p.add_reporter(stat)

    winner = p.run(main, GENS)
    print('\nBest genome:\n{!s}'.format(winner))

    visualize.draw_net(config, winner, False)
    visualize.plot_stats(stat, ylog=False, view=False)

    with open("winner.pkl", "wb") as f:
        pickle.dump(winner, f)
        f.close()

def main_with_progress(genomes, config):
    display = pygame.display.set_mode((BLOCK_WIDTH*BLOCKS, BLOCK_WIDTH*BLOCKS))
    pygame.display.set_caption('Snake')

    nets =[]
    ge = []
    snakes = []

    max_score = 0

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        snakes.append(Snake(BLOCK_WIDTH*BLOCKS, BLOCK_WIDTH*BLOCKS))
        g.fitness = 0
        ge.append(g)

    while len(snakes) > 0:
        S_AI = SnakeGameAI(snakes, BLOCK_WIDTH*BLOCKS, BLOCK_WIDTH*BLOCKS, display)
        for x, snake in enumerate(snakes):
            state_old, dis_old = agent.get_state(snake)
            output = nets[x].activate(state_old)
            final_move = [0, 0, 0]
            final_move[np.argmax(output)]=1
            reward, done, score = snake.play_step(final_move)
            if score > max_score:
                max_score = score
            ge[x].fitness = reward
            if done:
                snake.reset()
                snakes.pop(x)
                nets.pop(x)
                ge.pop(x)
        S_AI.update(len(snakes), max_score)

def run_with_progress(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stat = neat.StatisticsReporter()
    p.add_reporter(stat)

    winner = p.run(main_with_progress, GENS)
    print('\nBest genome:\n{!s}'.format(winner))

    visualize.draw_net(config, winner, False)
    visualize.plot_stats(stat, ylog=False, view=False)

    with open("winner.pkl", "wb") as f:
        pickle.dump(winner, f)
        f.close()

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    if REPLAY_FILE:
        replay_genome(config_path)
    elif SEE_PROGRESS:
        run_with_progress(config_path)
    else:
        run(config_path)