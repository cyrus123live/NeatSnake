# Taken from husano896's PR thread (slightly modified)
import pygame
# import PlayerSegment as pg
from pygame.locals import *
from datetime import datetime
from datetime import time
import neat
import random
import os
import sys

red = (205, 0, 0)
blue = (30,30,205)
black = (0, 0, 0)

WIDTH = 900
HEIGHT = 600
SNAKESIZE = 20

FRAMESUNTILSTARVATION = 400

generation = 0

TOTALGENERATIONS = 750

FRAMETIME = 50000 # in microseconds
#FRAMETIME = 500000
VISIBLEGENERATION = 2
SNAKESDRAWN = 50
SIZECAP = 30

draw = False
drawAll = False

k = 0 # key pressed

class Snake():

    def __init__(self, color):
        x = (WIDTH + SNAKESIZE) / 2
        y = 0
        self.fruit = [random.randrange(0, WIDTH - SNAKESIZE, SNAKESIZE), random.randrange(0, HEIGHT - SNAKESIZE, SNAKESIZE)]
        self.timeLastAte = datetime.now()
        self.segments = [[x, y], [x, y + SNAKESIZE], [x, y + SNAKESIZE]]
        self.direction = "down"
        self.prevDirection = "down"
        self.decision = "straight"
        self.color = color
        self.size = 3
        self.framesSinceAte = 0

def drawSnake(snake):
    if snake.size > SIZECAP:
        snake.size = SIZECAP
    for i in range(snake.size):
        seg = snake.segments[len(snake.segments) - i - 1]
        pygame.draw.rect(screen,snake.color,(seg[0],seg[1], SNAKESIZE, SNAKESIZE))

    pygame.draw.rect(screen,snake.color,(snake.fruit[0],snake.fruit[1], SNAKESIZE, SNAKESIZE))

def distance(x1, y1, x2, y2):
    return root(((x1-x2) ** 2)-((y1-y2) ** 2))

def distanceAhead(snake):

    direction = snake.direction
    x = snake.segments[len(snake.segments) - 1][0]
    y = snake.segments[len(snake.segments) - 1][1]

    distance = 1.0

    if direction == "right":
        distance = (WIDTH - x - SNAKESIZE) / WIDTH
    elif direction == "left":
        distance = x / WIDTH
    elif direction == "up":
        distance = y / HEIGHT
    elif direction == "down":
        distance = (HEIGHT - y - SNAKESIZE) / HEIGHT

    return distance

def getSnakeInfo(snake):

    x = snake.segments[len(snake.segments) - 1][0]
    y = snake.segments[len(snake.segments) - 1][1]

    dist_straight_wall = ""
    dist_straight_fruit = ""
    dist_right_wall = ""
    dist_right_fruit = ""
    dist_left_wall = ""
    dist_left_fruit = ""

    dist_fruit_y = abs(snake.fruit[1] - y) - SNAKESIZE
    dist_fruit_x = abs(snake.fruit[0] - x) - SNAKESIZE

    if snake.direction == "right":

        dist_straight_wall = WIDTH - x - SNAKESIZE
        dist_right_wall = HEIGHT - y - SNAKESIZE
        dist_left_wall = y

        dist_left_fruit = y - snake.fruit[1] - SNAKESIZE
        dist_right_fruit = snake.fruit[1] - y - SNAKESIZE
        dist_straight_fruit = snake.fruit[0] - x - SNAKESIZE

    elif snake.direction == "left":

        dist_straight_wall = x
        dist_right_wall = y
        dist_left_wall = HEIGHT - y - SNAKESIZE

        dist_left_fruit = snake.fruit[1] - y - SNAKESIZE
        dist_right_fruit = y - snake.fruit[1] - SNAKESIZE
        dist_straight_fruit = x - snake.fruit[0] - SNAKESIZE

    elif snake.direction == "up":

        dist_straight_wall = y
        dist_right_wall = WIDTH - x - SNAKESIZE
        dist_left_wall = x

        dist_left_fruit = x - snake.fruit[0] - SNAKESIZE
        dist_right_fruit = x - snake.fruit[0] - x - SNAKESIZE
        dist_straight_fruit = y - snake.fruit[1] - SNAKESIZE

    elif snake.direction == "down":

        dist_straight_wall = HEIGHT - y - SNAKESIZE
        dist_right_wall = x
        dist_left_wall = WIDTH - x - SNAKESIZE

        dist_left_fruit = snake.fruit[0] - x - SNAKESIZE
        dist_right_fruit =  x - snake.fruit[0] - SNAKESIZE
        dist_straight_fruit = snake.fruit[1] - y - SNAKESIZE

    #info = [dist_straight_wall/SNAKESIZE, abs(snake.fruit[0] - x)/SNAKESIZE, abs(snake.fruit[1] - y)/SNAKESIZE]
    #info = [dist_straight_wall, dist_straight_fruit, dist_right_fruit, dist_left_fruit]
    info = [dist_straight_wall, dist_straight_fruit, dist_right_wall, dist_right_fruit, dist_left_wall, dist_left_fruit]
    #info = [(x/WIDTH),(y/HEIGHT),abs(snake.fruit[0] - x)/WIDTH,abs(snake.fruit[1] - y)/HEIGHT, distanceAhead(snake)]

    # info = [-1 if x < 0 else x for x in info]

    return info

def interpretSnake(snake, output):

    index = output.index(max(output))
    if index == 0:
        snake.decision = "straight"
    elif index == 1:
        snake.decision = "right"
    elif index == 2:
        snake.decision = "left"

    if snake.direction == "down":
        if snake.decision == "right":
            snake.direction = "left"
        elif snake.decision == "left":
            snake.direction = "right"

    elif snake.direction == "up":
        if snake.decision == "right":
            snake.direction = "right"
        elif snake.decision == "left":
            snake.direction = "left"

    elif snake.direction == "left":
        if snake.decision == "right":
            snake.direction = "up"
        elif snake.decision == "left":
            snake.direction = "down"

    elif snake.direction == "right":
        if snake.decision == "right":
            snake.direction = "down"
        elif snake.decision == "left":
            snake.direction = "up"

def run(config_path):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    # p.add_reporter(stats)
    # p.add_reporter(neat.Checkpointer(5))

    # Run 300 generations.
    winner = p.run(eval_genomes, TOTALGENERATIONS)

    while(True):
        simulate(winner, config)
    print(winner)

def simulate(winner, config):

    snake = Snake((random.randrange(0, 250), random.randrange(0, 250), random.randrange(0, 250), 123))
    ge = winner
    net = neat.nn.FeedForwardNetwork.create(ge, config)

    t = datetime(2002, 1, 1) # time frame was last drawn

    # x, y coordinate of head of current snake
    x = 0
    y = 0

    while True:

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                return
            if event.type == KEYDOWN:
                return

        duration = datetime.now() - t

        if duration.microseconds > FRAMETIME:

            screen.fill(black)

            x = snake.segments[len(snake.segments) - 1][0]
            y = snake.segments[len(snake.segments) - 1][1]

            # check for collision with wall or going too long without eating
            if (x > WIDTH - SNAKESIZE or x < 0 or y > HEIGHT - SNAKESIZE or y < 0 or snake.framesSinceAte > FRAMESUNTILSTARVATION) and snake != None:
                break

            info = getSnakeInfo(snake)
            output = net.activate(info)

            interpretSnake(snake, output)

            # move snake
            if snake.direction == "left":
                x -= SNAKESIZE
            elif snake.direction == "right":
                x += SNAKESIZE
            elif snake.direction == "up":
                y -= SNAKESIZE
            elif snake.direction == "down":
                y += SNAKESIZE

            snake.segments.append([x, y])

            drawSnake(snake)

            # check for collision with fruit
            if x == snake.fruit[0] and y == snake.fruit[1]:
                    snake.size += 3

                    # spawn new random fruit for snake
                    snake.fruit = [random.randrange(0, WIDTH - SNAKESIZE, SNAKESIZE), random.randrange(0, HEIGHT - SNAKESIZE, SNAKESIZE)]
                    snake.framesSinceAte = 0
                    #print(f"Snake {i} just ate a fruit")
            else:
                snake.framesSinceAte += 1

            pygame.display.flip()

            t = datetime.now()

def eval_genomes(genomes, config):

    global generation, draw, drawAll

    snakes = []
    ge = []
    nets = []
    simulated = False

    for genome_id, genome in genomes:

        snakes.append(Snake((random.randrange(0, 250), random.randrange(0, 250), random.randrange(0, 250))))
        genome.fitness = 0
        ge.append(genome)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)

    t = datetime(2002, 1, 1) # time frame was last drawn

    # x, y coordinate of head of current snake
    x = 0
    y = 0

    if len(sys.argv) > 1 and (sys.argv[1] == "True" or sys.argv[1] == "1"):
        draw = True

    if len(sys.argv) > 2 and (sys.argv[1] == "True" or sys.argv[1] == "1"):
        drawAll = True
        draw = False

    while True:

        if drawAll and generation % VISIBLEGENERATION == 0:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    return
                if event.type == KEYDOWN:
                    generation += 1
                    return

            screen.fill(black)

        for i, snake in enumerate(snakes):

            x = snake.segments[len(snake.segments) - 1][0]
            y = snake.segments[len(snake.segments) - 1][1]

            # check for collision with wall or going too long without eating
            if (x > WIDTH - SNAKESIZE or x < 0 or y > HEIGHT - SNAKESIZE or y < 0 or snake.framesSinceAte > FRAMESUNTILSTARVATION) and snake != None:
                del(snakes[i])
                del(ge[i])
                del(nets[i])
                #print(f"snake {i} has died")
                continue

            info = getSnakeInfo(snake)
            output = nets[i].activate(info)

            interpretSnake(snake, output)

            # move snake
            if snake.direction == "left":
                x -= SNAKESIZE
            elif snake.direction == "right":
                x += SNAKESIZE
            elif snake.direction == "up":
                y -= SNAKESIZE
            elif snake.direction == "down":
                y += SNAKESIZE

            # reward snakes for exploring new area
            if not([x, y] in snake.segments) and ge[i].fitness%100 != 99:
                ge[i].fitness += 1

            snake.segments.append([x, y])

            #Give them points for trying
            # if (info[1] == 0 and snake.decision == "straight") and ge[i].fitness % 100 != 99:
            #     ge[i].fitness += 10

            # check for collision with fruit
            if x == snake.fruit[0] and y == snake.fruit[1]:
                    snake.size += 3
                    ge[i].fitness += 100 * ((FRAMESUNTILSTARVATION - 10 - snake.framesSinceAte) / (FRAMESUNTILSTARVATION - 10))

                    # spawn new random fruit for snake
                    snake.fruit = [random.randrange(0, WIDTH - SNAKESIZE, SNAKESIZE), random.randrange(0, HEIGHT - SNAKESIZE, SNAKESIZE)]
                    snake.framesSinceAte = 0
                    #print(f"Snake {i} just ate a fruit")
            else:
                snake.framesSinceAte += 1


            killed = False
            #  check for collision with itself
            # for j in range(snake.size):
            #      seg = snake.segments[len(snake.segments) - 1 - j]
            #
            #      if x == seg[0] and y == seg[1] and j != 0:
            #          killed = True
            #          #print(f"snake {i} has died of collision")

            if drawAll and i < SNAKESDRAWN and generation % VISIBLEGENERATION == 0:
                drawSnake(snake)

            if killed or ge[i].fitness >= 30000:
                del(snakes[i])
                del(ge[i])
                del(nets[i])
                #print(f"snake {i} has died")
                continue

        t = datetime.now()

        if len(snakes) == 1 and not(simulated) and generation % VISIBLEGENERATION == 0 and generation != 0 and bool(draw):
            simulate(ge[0], config)
            simulated = True

        if drawAll and generation % VISIBLEGENERATION == 0:
            pygame.display.flip()

        if len(snakes) == 0:
            generation += 1
            break

if __name__ == "__main__":

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT)) # x = 630, y = 480
    pygame.display.set_caption("Snake")
    clock = pygame.time.Clock()

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')

    run(config_path)
