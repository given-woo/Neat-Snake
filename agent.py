import neat
import os
import numpy as np
import visualize
import pygame

from game import SnakeGameAI, Direction, Point

class Agent:
    def __init__(self):
        self.n_games = 0

    def get_state(self, game):
        head = game.snake[0]

        danger_l = 0
        danger_r = 0
        danger_u = 0
        danger_d = 0
        danger_ul = 0
        danger_ur = 0
        danger_dl = 0
        danger_dr = 0

        for i in range(0, 5):
            point_l = Point(head.x - 20 * (i + 1), head.y)
            if game.is_collision(point_l):
                danger_l = 1 / (i+1)
        for i in range(0, 5):
            point_r = Point(head.x + 20 * (i + 1), head.y)
            if game.is_collision(point_r):
                danger_r = 1 / (i+1)
        for i in range(0, 5):
            point_u = Point(head.x, head.y - 20 * (i + 1))
            if game.is_collision(point_u):
                danger_u = 1 / (i+1)
        for i in range(0, 5):
            point_d = Point(head.x, head.y + 20 * (i + 1))
            if game.is_collision(point_d):
                danger_d = 1 / (i+1)
        for i in range(0, 5):
            point_ul = Point(head.x - 20 * (i + 1), head.y - 20 * (i + 1))
            if game.is_collision(point_ul):
                danger_ul = 1 / (i+1)
        for i in range(0, 5):
            point_ur = Point(head.x + 20 * (i + 1), head.y - 20 * (i + 1))
            if game.is_collision(point_ur):
                danger_ur = 1 / (i+1)
        for i in range(0, 5):
            point_dl = Point(head.x - 20 * (i + 1), head.y + 20 * (i + 1))
            if game.is_collision(point_dl):
                danger_dl = 1 / (i+1)
        for i in range(0, 5):
            point_dr = Point(head.x + 20 * (i + 1), head.y + 20 * (i + 1))
            if game.is_collision(point_dr):
                danger_dr = 1 / (i+1)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger
            danger_l,
            danger_r,
            danger_u,
            danger_d,
            danger_ul,
            danger_ur,
            danger_dl,
            danger_dr,

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
        ]

        return np.array(state, dtype=int)

def main(genomes, config):
    display = pygame.display.set_mode((400, 400))

    agent = Agent()
    nets =[]
    ge = []
    snakes = []

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        snakes.append(SnakeGameAI(display, 400, 400))
        pygame.display.set_caption('Snake')
        g.fitness = 0
        ge.append(g)

    while len(snakes) > 0:
        for x, snake in enumerate(snakes):
            state_old = agent.get_state(snake)
            output = nets[x].activate(state_old.tolist())
            final_move = [0, 0, 0]
            for i in range(0, 3):
                if output[i] > output[(i + 1) % 3] and output[i] > output[(i + 2) % 3]:
                    final_move[i] = 1
            reward, done, score = snake.play_step(final_move)
            ge[x].fitness += reward
            while not done:
                state_old = agent.get_state(snake)
                output = nets[x].activate(state_old.tolist())
                final_move = [0, 0, 0]
                for i in range(0, 3):
                    if output[i]>output[(i+1)%3] and output[i]>output[(i+2)%3]:
                        final_move[i]=1
                reward, done, score = snake.play_step(final_move)
                ge[x].fitness += reward

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

    winner = p.run(main, 200)
    print('\nBest genome:\n{!s}'.format(winner))

    node_names = {-1: 'danger-straight', -2: 'danger-right', -3: 'danger-left', -4: 'danger-down', -5: 'danger-straight&left', -6: 'danger-straight&right', -7: 'danger-down&left', -8: 'danger-down&right', -9: 'move(l)', -10: 'move(r)', -11: 'move(u)', -12: 'move(d)', -13: 'food(l)', -14: 'food(r)', -15: 'food(u)', -16: 'food(d)', 0: 'straight', 1: 'right', 2: 'left'}
    visualize.draw_net(config, winner, False, node_names=node_names)
    visualize.plot_stats(stat, ylog=False, view=False)

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)
    # start 9 : 25 pm
    # end