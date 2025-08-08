import pygame
import math
import numpy as np
import matplotlib.pyplot as plt
import statistics
import torch
import torch.nn as nn
import torch.optim as optim
import random
import itertools
from collections import deque
import imageio.v2 as imageio
import imageio as iio
import os

class Board:
    def __init__(self):
        self.width, self.height = 400, 400
        self.map_size = 8
        self.tile_size = self.width / self.map_size
        self.view_range = self.width / 1.5
        self.fov = 60  # 90
        self.no_ofrays = 8
        self.max_speed = 1.5 * self.map_size / math.pi
        self.max_rotate = 15  # math.pi
        self.fps = 12

        self.player_angle = 0
        self.player_x = self.width / 2
        self.player_y = self.height / 2
        self.player_radius = self.tile_size * 0.2
        self.distance = 0
        self.hunger = 24

        self.object_radius = self.tile_size * 0.4
        self.obj_x, self.obj_y = 350, 350

        self.board = []
        self.count = 0

    def maze(self, rnd=False, bll=True):
        if rnd == False:
            self.board = [['#', '#', '#', '#', '#', '#', '#', '#'],
                          ['#', ' ', '#', '#', ' ', ' ', '#', '#'],
                          ['#', ' ', ' ', ' ', ' ', ' ', ' ', '#'],
                          ['#', ' ', ' ', ' ', ' ', ' ', ' ', '#'],
                          ['#', ' ', '#', ' ', ' ', ' ', '#', '#'],
                          ['#', ' ', '#', '#', ' ', ' ', ' ', '#'],
                          ['#', ' ', ' ', ' ', ' ', ' ', '#', '#'],
                          ['#', '#', '#', '#', '#', '#', '#', '#']]

        else:
            for i in range(self.map_size):
                self.board.append([])
                for j in range(self.map_size):
                    self.board[i].append(' ')
            for row in range(self.map_size):
                for col in range(self.map_size):
                    if row == 0 or row == self.map_size - 1 or col == 0 or col == self.map_size - 1:
                        self.board[row][col] = '#'

            for i in range(int((self.map_size-2) ** 2 / 3)):
                while True:
                    c = [random.randint(1, self.map_size - 2), random.randint(1, self.map_size - 2)]
                    if self.board[c[0]][c[1]] != '#':
                        if 2 < c[0] < self.map_size - 4 and 2 < c[1] < self.map_size - 4:
                            self.board[c[0]][c[1]] = '#'
                            areas = self.no_ofAreas(self.board)
                            if areas == 1:
                                break
                            else:
                                self.board[c[0]][c[1]] = ' '
                        else:
                            r =random.randint(1,3)
                            if r != 1:
                                self.board[c[0]][c[1]] = '#'
                                areas = self.no_ofAreas(self.board)
                                if areas == 1:
                                    break
                                else:
                                    self.board[c[0]][c[1]] = ' '

            start = [int(self.player_x // self.tile_size), int(self.player_y // self.tile_size)]
            for val in [[i-1, j-1] for i in range(start[0], start[0] + 2) for j in range(start[1], start[1] + 2)]:
                #print(val)
                self.board[val[0]][val[1]] = ' '

        if bll == True:
            self.ball()

    def no_ofAreas(self, grid):
        if len(grid) == 0:
            return 0
        self.count = sum(grid[i][j] == ' ' for i in range(self.map_size) for j in range(self.map_size))
        parent = [i for i in range(self.map_size ** 2)]

        def find(x):
            if parent[x] != x:
                return find(parent[x])
            return parent[x]

        def union(x, y):
            xroot, yroot = find(x), find(y)
            if xroot == yroot:
                return
            parent[xroot] = yroot
            self.count -= 1

        for i in range(self.map_size):
            for j in range(self.map_size):
                if grid[i][j] == '#':
                    continue
                index = i * self.map_size + j
                if j < self.map_size - 1 and grid[i][j + 1] == ' ':
                    union(index, index + 1)
                if i < self.map_size - 1 and grid[i + 1][j] == ' ':
                    union(index, index + self.map_size)
        return self.count

    def ball(self):
        while True:
            c = [random.randint(1, self.map_size - 2), random.randint(1, self.map_size - 2)]
            if c[0] != self.player_x // self.tile_size and c[1] != self.player_y // self.tile_size and self.board[c[0]][c[1]] != '#':
                self.obj_x = (c[0] + 0.5) * self.tile_size
                self.obj_y = (c[1] + 0.5) * self.tile_size
                break
class Player:
    def __init__(self, x, y, angle):
        self.alive = True
        self.x = x
        self.y = y
        self.angle = angle
        self.forward = 1
        self.backward = 1
        self.fwd = Ray(start_x=self.x, start_y=self.y, start_angle=self.angle, offset=0)
        self.bkwd = Ray(start_x=self.x, start_y=self.y, start_angle=self.angle, offset=180)
        self.move_x = 1
        self.move_y = 1
        self.score = 0.5
        self.hunger = 0
        self.found = 0

    def update(self, x, y, angle):
        self.x = x
        self.y = y
        self.angle = angle

    def check_move(self):
        self.fwd.update_params(player_x=self.x, player_y=self.y, player_angle=self.angle)
        self.bkwd.update_params(player_x=self.x, player_y=self.y, player_angle=self.angle)
        self.fwd.check_move()
        self.bkwd.check_move()
        self.fwd.check_intercept()
        self.bkwd.check_intercept()

        if self.fwd.object != (0, 0, 255):
            if self.fwd.distance <= board.player_radius:
                self.alive = False
                self.score -= 0.5
        elif self.fwd.object == (0, 0, 255):
            if self.fwd.distance <= board.player_radius:
                self.score += 0.5
                self.hunger = 0
                self.found += 1
                # print(self.score)
            else:
                pass
        self.hunger += 1
        if self.hunger % board.fps == 0:
            pass  # print(f"Hunger: {self.hunger/board.fps}")
        if self.hunger > board.hunger * board.fps:
            self.alive = False
            # self.score = 0

        if self.fwd.distance <= board.player_radius and self.fwd.object == (0, 0, 255):
            board.ball()
        elif self.bkwd.distance <= board.player_radius and self.bkwd.object == (0, 0, 255):
            board.ball()
class Ray:
    def __init__(self, start_x, start_y, start_angle, offset):
        self.x = start_x
        self.y = start_y
        self.offset = offset
        self.angle = start_angle + self.offset
        self.side = 'x'
        self.distance = 0
        self.object = (0, 0, 0)

    def update_params(self, player_x, player_y, player_angle):
        self.x = player_x
        self.y = player_y
        self.angle = player_angle + self.offset

    def check_move(self):
        rx = math.cos(math.radians(self.angle))
        ry = math.sin(math.radians(self.angle))
        map_x = self.x // board.tile_size
        map_y = self.y // board.tile_size

        t_max_x = self.x / board.tile_size - map_x
        if rx > 0:
            t_max_x = 1 - t_max_x
        t_max_y = self.y / board.tile_size - map_y
        if ry > 0:
            t_max_y = 1 - t_max_y

        while True:
            if ry == 0 or t_max_x < t_max_y * abs(rx / ry):
                self.side = 'x'
                map_x += 1 if rx > 0 else -1
                t_max_x += 1
                if map_x < 0 or map_x >= board.map_size:
                    break
            else:
                self.side = 'y'
                map_y += 1 if ry > 0 else -1
                t_max_y += 1
                if map_x < 0 or map_y >= board.map_size:
                    break
            if board.board[int(map_x)][int(map_y)] == "#":
                self.object = (0, 0, 255)
                break

        if self.side == 'x':
            x = (map_x + (1 if rx < 0 else 0)) * board.tile_size
            y = board.player_y + (x - board.player_x) * ry / rx
            direction = 'r' if x >= board.player_x else 'l'
        else:
            y = (map_y + (1 if ry < 0 else 0)) * board.tile_size
            x = board.player_x + (y - board.player_y) * rx / ry
            direction = 'd' if y >= board.player_y else 'u'
        self.distance = math.hypot(x - self.x, y - self.y)
        self.object = (125, 125, 0)

    def check_intercept(self):
        dx = math.cos(math.radians(self.angle))
        dy = math.sin(math.radians(self.angle))
        fx = self.x - board.obj_x
        fy = self.y - board.obj_y
        a = dx ** 2 + dy ** 2
        b = 2 * (fx * dx + fy * dy)
        c = fx ** 2 + fy ** 2 - board.object_radius ** 2
        discriminant = b ** 2 - 4 * a * c
        if discriminant >= 0:
            discriminant = math.sqrt(discriminant)
            t1 = (-b - discriminant) / (2 * a)
            t2 = (-b + discriminant) / (2 * a)
            t = min(t for t in [t1, t2] if t > 0) if any(t > 0 for t in [t1, t2]) else None
            if t and t < board.view_range and t < self.distance:
                self.end_x = self.x + dx * t
                self.distance = t
                self.end_y = self.y + dy * t
                self.colour = (0, 125, 0)
                self.object = (0, 0, 255)
                return
        self.end_x = self.x + dx * board.view_range
        self.end_y = self.y + dy * board.view_range
        self.colour = (125, 0, 0)
        if board.view_range < self.distance:
            self.distance = board.view_range
            self.object = (0, 0, 0)
        else:
            self.end_x = self.x + dx * self.distance
            self.end_y = self.y + dy * self.distance
            self.object = (125, 125, 0)

class Run:
    def __init__(self):
        self.ticks = 0
    def render_init(self):
        pygame.init()
        self.board_surf = pygame.display.set_mode((board.width, board.height))
        '''self.rend_surf = pygame.display.set_mode((board.width * 2, board.width))
        self.board_surf.blit(self.rend_surf, (0, 0))'''
    def step(self):
        #print(player.score)
        player.update(x=board.player_x, y=board.player_y, angle=board.player_angle)
        player.check_move()
        for ray in rays:
            ray.check_move()
            ray.check_intercept()
            ray.update_params(player_x=board.player_x, player_y=board.player_y, player_angle=board.player_angle)
            #print(f'check: {ray.distance}')
            if ray.object == (0, 0, 255):# and ray.offset == 0:
                player.score += abs(0.05 * math.cos(ray.offset)) * (ray.distance/board.view_range) #(abs(math.cos(ray.offset)) ** 4 * math.e ** (-ray.distance / 40)) / (board.fps * board.no_ofrays)
        #player.update(x=board.player_x, y=board.player_y, angle=board.player_angle)
        #player.check_move()
        #player.score += 0.001
    def render_step(self):
        self.board_surf.fill((0, 0, 0))
        edges = []
        for ray in rays:
            if ray.angle == player.angle:
                ray.colour = (ray.colour[0] * 2, ray.colour[1] * 2, ray.colour[2] * 2)
            pygame.draw.line(self.board_surf, ray.colour, start_pos=(board.player_x, board.player_y),
                             end_pos=(float(ray.end_x), float(ray.end_y)))
            edges.append((float(ray.end_x), float(ray.end_y)))

        for col in range(board.map_size):
            for row in range(board.map_size):
                if board.board[col][row] == '#':
                    rect = (col * board.tile_size, row * board.tile_size, board.tile_size, board.tile_size)
                    pygame.draw.rect(self.board_surf, (125, 125, 125), rect)

        pygame.draw.circle(self.board_surf, (0, 125, 0), (board.player_x, board.player_y), radius=board.player_radius)
        pygame.draw.circle(self.board_surf, (0, 0, 125), (board.obj_x, board.obj_y), radius=board.object_radius)
        pygame.draw.lines(self.board_surf, (255, 255, 255), closed=False, points=edges)

        '''try:
            for i, ray in enumerate(rays):
                h = round(10000 / ray.distance)
                rend_width = board.width // len(rays)
                colour = pygame.Color((0, 0, 0)).lerp(ray.object, min(h / 256, 1))
                rect = pygame.Rect(board.width + i * rend_width, board.width // 2 - h // 2, rend_width, h)
                pygame.draw.rect(self.rend_surf, colour, rect)
        except:
            pass'''

        #pygame.display.update()
        #self.clock.tick(board.fps)


os.makedirs("frames", exist_ok=True)
frame_count = 0


board = Board()
board.maze(rnd=True,bll=False)
player = Player(x=board.player_x, y=board.player_y, angle=board.player_angle)
rays = []
offsets = []
for i in np.arange(-board.fov / 2, board.fov / 2 + 1, board.fov / board.no_ofrays):
    offsets.append(i)
    r = Ray(start_x=board.player_x, start_y=board.player_y, start_angle=board.player_angle, offset=i)
    rays.append(r)
if 0 not in offsets:
    r = Ray(start_x=board.player_x, start_y=board.player_y, start_angle=board.player_angle, offset=0)
    rays.append(r)


if __name__ == '__main__':
    # Training Loop
    loop = 0

    pygame.init()
    clock = pygame.time.Clock()

    run = Run()
    run.render_init()

    # -- Game Loop --
    while player.alive:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break

        '''
        keys = pygame.key.get_pressed()
        board.player_angle += (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]) * board.max_rotate
        speed = keys[pygame.K_UP] * board.max_speed
        # speed = ((keys[pygame.K_UP]*player.forward) - keys[pygame.K_DOWN]*player.backward) * board.max_speed
        # speed = ((keys[pygame.K_UP]) - keys[pygame.K_DOWN]) * board.max_speed
        '''
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] != 0:
            speed = board.max_speed# / math.pi
            board.player_angle -= board.max_rotate
        elif keys[pygame.K_RIGHT] != 0:
            speed = board.max_speed# / math.pi
            board.player_angle += board.max_rotate
        else:
            speed = board.max_speed

        board.player_x += math.cos(math.radians(board.player_angle)) * speed  # * player.move_x
        board.player_y += math.sin(math.radians(board.player_angle)) * speed  # * player.move_y

        run.step()
        run.render_step()

        pygame.display.update()
        clock.tick(board.fps)

        if player.alive == False:
            print(f'dead\t{player.score}')
            break

    pygame.quit()
