import numpy as np
import random 
from collections import deque
import math
import matplotlib.pyplot as plt

class Snake():
    def __init__(self, h=10, w=10, live_draw=False):
        self.h = h
        self.w = w
        self.snake_locs = deque()
        self.apple = tuple()
        self.board = np.zeros((self.h, self.w))
        self.score = 0
        self.grow = False
        self.actions = ['up', 'down', 'left', 'right']
        self.reset()
        self.live_draw = live_draw
        plt.ion()
        self.fig1, ax1 = plt.subplots()
        self.axim1 = ax1.imshow(self.board*100)
        self.draw()

        self.invalid = -3
        self.loss = -10
        self.score = 10
        self.normal = -1

    def reset(self):
        self.score = 0 
        self.grow = False
        self.board.fill(0)
        del self.snake_locs
        snake_head = (random.randint(0, self.h-1), random.randint(3, self.w-1))
        self.snake_locs = deque([(snake_head[0], snake_head[1]-i) for i in range(1, 4)])
        
        self.apple = self.spawn_apple()

        for x, y in self.snake_locs:
            self.board[x][y] = 1 
        
        x, y = self.apple
        self.board[x][y] = 2

    def spawn_apple(self):
        apple = (random.randint(0, self.h-1), random.randint(0, self.w-1))
        if apple in self.snake_locs:
            return self.spawn_apple()
        else:
            return apple

    def get_state(self):
        return (tuple(self.snake_locs), self.apple)

    def step(self, action):
        if action not in self.actions or not self.is_applicable(action):
            return self.invalid

        if action == "up":
            self.up()
        elif action == "down":
            self.down()
        elif action == "left":
            self.left()
        elif action == "right":
            self.right()

        self.board.fill(0)
        for x, y in self.snake_locs:
            if not 0 <= x <= self.h-1:
                self.reset() 
                self.draw()
                return self.loss
            if not 0 <= y <= self.w-1:
                self.reset()
                self.draw()
                return self.loss

            self.board[x][y] = 1 

            if self.snake_locs.count((x, y)) > 1:
                self.reset() 
                self.draw()
                return self.loss

        x, y = self.apple
        self.board[x][y] = 2

        prv = self.score

        self.try_eat()

        self.draw()
        
        if prv-self.score != 0:
            return self.score
        else:
            return self.normal

    def is_applicable(self, action, state=None):
        if not state:
            snk_locs, apl_loc = self.get_state()
        else:
            snk_locs, apl_loc = state[0], state[1]

        if action == 'up':
            head = snk_locs[0]
            if (head[0]-1, head[1]) == snk_locs[1]:
                return False
        if action == 'down':
            head = snk_locs[0]
            if (head[0]+1, head[1]) == snk_locs[1]:
                return False
        if action == 'left':
            head = snk_locs[0]
            if (head[0], head[1]-1) == snk_locs[1]:
                return False
        if action == 'right':
            head = snk_locs[0]
            if (head[0], head[1]+1) == snk_locs[1]:
                return False
        return True

    def get_applicable_actions(self):
        applicable = []
        for act in self.actions:
            if self.is_applicable(act):
                applicable.append(act)
        return applicable
    
    def get_applicable_actions_from(self, state):
        applicable = []
        for act in self.actions:
            if self.is_applicable(act,state):
                applicable.append(act)
        return applicable

    def up(self):
        if not self.grow:
            self.snake_locs.pop()
        
        old_head = self.snake_locs[0]
        new_head = (old_head[0]-1, old_head[1])

        self.snake_locs.appendleft(new_head)

    def down(self):
        if not self.grow:
            self.snake_locs.pop()
        
        old_head = self.snake_locs[0]
        new_head = (old_head[0]+1, old_head[1])

        self.snake_locs.appendleft(new_head)

    def left(self):
        if not self.grow:
            self.snake_locs.pop()
        
        old_head = self.snake_locs[0]
        new_head = (old_head[0], old_head[1]-1)

        self.snake_locs.appendleft(new_head)

    def right(self):
        if not self.grow:
            self.snake_locs.pop()
        
        old_head = self.snake_locs[0]
        new_head = (old_head[0], old_head[1]+1)

        self.snake_locs.appendleft(new_head)

    def try_eat(self):
        if self.apple in self.snake_locs:
            self.score += 1
            self.grow = True
            self.apple = self.spawn_apple()
            return True
        else:
            self.grow = False
            return False
    
    def manhatten_to_apple(self):
        x1, y1 = self.snake_locs[0]
        x2, y2 = self.apple
        return abs(x1-x2) + abs(y1-y2)

    def euclid_to_apple(self):
        x1, y1 = self.snake_locs[0]
        x2, y2 = self.apple
        return math.sqrt(((x1 - x2)**2+(y1 - y2)**2))

    def draw(self):
        if self.live_draw:
            self.axim1.set_data(self.board*100)
            self.fig1.canvas.flush_events()
    
    def set_state(self, snk_locs, apl_loc):
        self.board.fill(0)
        for x, y in snk_locs:
            self.board[x][y] = 1 
        
        x, y = apl_loc
        self.board[x][y] = 2
