import pygame
import math
import numpy as np
import random
import csv

class Board:
    def __init__(self):
        self.width, self.height = 300, 300
        self.map_size = 10
        self.tile_size = self.width / self.map_size
        self.view_range = self.width / 2.5
        self.fov = 60 # 90
        self.no_ofrays = 4
        self.max_speed = 5
        self.max_rotate = 10# math.pi
        self.fps = 12

        self.player_angle = 0
        self.player_x = self.width / 2
        self.player_y = self.height / 2
        self.player_radius = self.tile_size * 0.2
        self.distance = 0
        self.hunger = 18

        self.object_radius = self.tile_size * 0.4
        self.obj_x, self.obj_y = 350, 350

        self.board = []
        self.count = 0

        self.maze()

    def maze(self):
        self.board = [['#', '#', '#', '#', '#', '#', '#', '#', '#', '#'],
                      ['#', ' ', '#', ' ', ' ', ' ', ' ', ' ', ' ', '#'],
                      ['#', ' ', '#', ' ', ' ', ' ', ' ', ' ', ' ', '#'],
                      ['#', ' ', '#', '#', ' ', ' ', '#', ' ', ' ', '#'],
                      ['#', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '#'],
                      ['#', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '#'],
                      ['#', ' ', '#', ' ', ' ', ' ', '#', '#', ' ', '#'],
                      ['#', ' ', '#', '#', ' ', ' ', ' ', ' ', ' ', '#'],
                      ['#', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '#', '#'],
                      ['#', '#', '#', '#', '#', '#', '#', '#', '#', '#']]

        '''
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
        '''

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
            c = [random.randint(1, self.map_size-2), random.randint(1, self.map_size-2)]
            if c[0] != self.player_x//self.tile_size and c[1] != self.player_y//self.tile_size and self.board[c[0]][c[1]] != '#':
                self.obj_x = (c[0]+0.5)*self.tile_size
                self.obj_y = (c[1]+0.5)*self.tile_size
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

        if self.fwd.object != (0,0,255):
            if self.fwd.distance <= board.player_radius:
                self.alive = False
                self.score -= 0.5
        elif self.fwd.object == (0,0,255):
            if self.fwd.distance <= board.player_radius:
                self.score += 0.5#1
                self.hunger = 0
                #print(self.score)
            else:
                pass
        self.hunger += 1
        if self.hunger%board.fps == 0:
            pass # print(f"Hunger: {self.hunger/board.fps}")
        if self.hunger > board.hunger * board.fps:
            self.alive = False
            #self.score = 0

        if self.fwd.distance <= board.player_radius and self.fwd.object == (0, 0, 255):
            board.ball()
        elif self.bkwd.distance <= board.player_radius and self.bkwd.object == (0, 0, 255):
            board.ball()

class Ray:
    def __init__(self,start_x, start_y, start_angle, offset):
        self.x = start_x
        self.y = start_y
        self.offset = offset
        self.angle = start_angle + self.offset
        self.side = 'x'
    def update_params(self,player_x, player_y, player_angle):
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

# -- NN States --

from collections import deque
from main import model # Your trained Q-network

# RL Hyperparameters
gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.999992 # 0.9999
epsilon_min = 0.1
batch_size = 64
memory = deque(maxlen=200000)

def get_state():
    state = [round(board.player_x,1), round(board.player_y,1), round(board.obj_x,1), round(board.obj_y,1), player.hunger]
    for ray in rays:
        state.append(round(ray.distance,1))
        if ray.object == (0, 0, 255):
            state.append(True)
        else:
            state.append(False)
    return np.array(state)
def choose_action(state):
    #print(epsilon)
    if random.random() < epsilon:
        return random.choice([0,1,1,2,2])
    q_values = model.forward(state.reshape(1, -1), training=False)
    return np.argmax(q_values)
def remember(s, a, r, s_, done):
    memory.append((s, a, r, s_, done))
def replay():
    if len(memory) < batch_size:
        return
    batch = random.sample(memory, batch_size)
    for s, a, r, s_, done in batch:
        target = model.forward(s.reshape(1, -1), training=False)
        if done:
            target[0][a] = r
        else:
            t = model.forward(s_.reshape(1, -1), training=False)
            target[0][a] = r + gamma * np.max(t)
        model.train(s.reshape(1, -1), target, epochs=1, batch_size=1, print_every=1)

# Training Loop
loop = 0
rend_l = 50
with open("Data\\Values.txt", "a", newline="") as f:
    f.write(f'Gamma: {gamma}')
    f.write(f'Epsilon_decay: {epsilon_decay}')
    f.write(f'Epsilon_min: {epsilon_min}')
    f.write(f'Batch_size: {batch_size}\n')

    f.write(f'loop\tplayer.score\tticks\tepsilon')
while True:
    # -- Initilisation --
    board = Board()

    player = Player(x=board.player_x, y=board.player_y, angle=board.player_angle)
    player.check_move()

    rays = []
    for i in np.arange(-board.fov/2, board.fov/2+1, board.fov/board.no_ofrays):
        r = Ray(start_x=board.player_x, start_y=board.player_y, start_angle=board.player_angle, offset=i)
        rays.append(r)

    # -- Render on loops --
    loop += 1
    print(f'Loop: {loop}')
    #print(loop % rend_l)

    if loop % rend_l == 0:
        pygame.init()
        board_surf = pygame.display.set_mode((board.width, board.height))
        rend_surf = pygame.display.set_mode((board.width * 2, board.width))
        board_surf.blit(rend_surf, (0, 0))
        clock = pygame.time.Clock()

        with open(str(f'Data\\Loop_{loop}_Training_Data.txt'), "w", newline="") as f:
            writer = csv.writer(f)
            for row in board.board:
                writer.writerow(row)
            header = ['p_x', '\tp_y', '\to_x', '\to_y'] + [f'\tray_{i}_values' for i in range(len(rays))] + ['\taction']
            writer.writerow(header)

    ticks = 0


    # -- Game Loop --
    while player.alive:
        if loop % rend_l == 0:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    break
        '''
            keys = pygame.key.get_pressed()
        board.player_angle += (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]) * board.max_rotate
        speed = keys[pygame.K_UP] * board.max_speed
        #speed = ((keys[pygame.K_UP]*player.forward) - keys[pygame.K_DOWN]*player.backward) * board.max_speed
        #speed = ((keys[pygame.K_UP]) - keys[pygame.K_DOWN]) * board.max_speed
    
        board.player_x += math.cos(math.radians(board.player_angle)) * speed # * player.move_x
        board.player_y += math.sin(math.radians(board.player_angle)) * speed # * player.move_y
    
        player.update(x=board.player_x, y=board.player_y, angle=board.player_angle)
    
        if player.alive == False:
            print('dead')
            break
            '''

        # -- Maze State/Render --
        if loop % rend_l == 0:
            board_surf.fill((0,0,0))

        for ray in rays:
            ray.check_move()
            ray.update_params(player_x=board.player_x, player_y=board.player_y, player_angle=board.player_angle)
            ray.check_intercept()
            if ray.object == (0,0,255):
                player.score += abs(math.cos(ray.angle)) * math.e ** (-ray.distance / 40) / board.no_ofrays # (20 * board.no_ofrays)
            if ray.angle == player.angle:
                distance = ray.distance
                ray.colour = (ray.colour[0]*2, ray.colour[1]*2, ray.colour[2]*2)
            if loop % rend_l == 0:
                pygame.draw.line(board_surf, ray.colour, start_pos=(board.player_x, board.player_y), end_pos=(float(ray.end_x), float(ray.end_y)))

        player.check_move()
    #print("score:",player.score)

        if loop % rend_l == 0:
            for col in range(board.map_size):
                for row in range(board.map_size):
                    if board.board[col][row] == '#':
                        rect = (col*board.tile_size, row*board.tile_size, board.tile_size, board.tile_size)
                        pygame.draw.rect(board_surf,(125, 125, 125), rect)

        if loop % rend_l == 0:
            pygame.draw.circle(board_surf,  (0, 125, 0), (board.player_x, board.player_y), radius=board.player_radius)
            pygame.draw.circle(board_surf,  (0, 0, 125), (board.obj_x, board.obj_y), radius=board.object_radius)

        #lines = []
        #for ray in rays:
         #   lines.append([ray.end_x, ray.end_y])
        #pygame.draw.lines(board_surf,(255,255,255), closed=False ,points=lines)

        if loop % rend_l == 0:
            try:
                for i, ray in enumerate(rays):
                    h = round(10000 / ray.distance)
                    rend_width = board.width // len(rays)
                    colour = pygame.Color((0, 0, 0)).lerp(ray.object, min(h / 256, 1))
                    rect = pygame.Rect(board.width + i * rend_width, board.width // 2 - h // 2, rend_width, h)
                    pygame.draw.rect(rend_surf, colour, rect)
            except: pass


        '''inputs = [board.player_x, board.player_y, board.obj_x, board.obj_y]
        for ray in rays:
            inputs.append(ray.distance)
        input_data = np.array([inputs])
    
        output = model.forward(input_data, training=False)
        action = np.argmax(output)
    
        print(action)
    
        if action == 0:
            speed = board.max_speed
        elif action == 1:
            speed = 0
            board.player_angle += board.max_rotate
        elif action == 2:
            speed = 0
            board.player_angle -= board.max_rotate'''

        if loop % rend_l == 0:
            pygame.display.flip()
        #clock.tick(board.fps)
        ticks += 1
        #if ticks%(board.fps*20) == 0:
         #   print('\ntick:', ticks)
          #  print('secs:', ticks/board.fps)

        ## -- NN Actions --
        done = True
        if player.alive == False:
            done = True

        state = get_state()
        action = choose_action(state)
        #print(action)

        # Apply action
        speed = 0
        if action == 0:  # forward
            speed = board.max_speed
        elif action == 1:
            board.player_angle -= board.max_rotate
        elif action == 2:  # right
            board.player_angle += board.max_rotate
        board.player_x += math.cos(math.radians(board.player_angle)) * speed
        board.player_y += math.sin(math.radians(board.player_angle)) * speed

        next_state = get_state()
        # s, a, r, s_, done
        remember(state, action, player.score, next_state, not player.alive)
        replay()

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        player.update(x=board.player_x, y=board.player_y, angle=board.player_angle)

        if loop % rend_l == 0:
            with open(str(f'Data\\Loop_{loop}_Training_Data.txt'), "a", newline="") as f:
                #print(state,'\n')
                for s in state:
                    f.write(str(s))
                    f.write('\t')
                f.write('\n')

        #player.score += 0.002

    if loop % rend_l == 0:
        pygame.quit()
        model.save(str(f"Data\\Loop_{loop}_Model.pkl"))

    print(f'Score: {player.score}')
    print(f'Epsilon: {epsilon}')
    with open("Data\\Values.txt", "a", newline="") as f:
        f.write(f'\n{loop}\t{player.score}\t{ticks}\t{epsilon}')


    '''input_size = 30
    X = np.random.rand(1000, input_size)  # input_size = number of features
    y = np.random.randint(0, 3, size=(1000,))
    X_test = np.random.rand(100, input_size)
    y_test = np.random.randint(0, 3, size=(100,))
    for i in range(training):
        print(f'Training session {i + 1}...')
        model.train(X, y, validation_data=(X_test, y_test), epochs=100, print_every=10)
        #print(f'Session {i + 1} complete')
    print(f'Loop {loop} training complete')'''