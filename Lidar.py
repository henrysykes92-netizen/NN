import itertools

from Environment import *

class Lidar:
    def __init__(self):
        self.memory = 200
        self.area = list(itertools.repeat([0,0], self.memory))
        self.objects = list(itertools.repeat([0,0], int(self.memory / 4)))
    def render_init(self):
        pygame.init()
        self.clock = pygame.time.Clock()
        self.area_surf = pygame.display.set_mode((board.width, board.height))

    def bounce(self, x, y, obj=False):
        self.point_x = x
        self.point_y = y
        if obj == True:
            self.objects.append([self.point_x,self.point_y])
            self.objects.pop(0)
        else:
            self.area.append([self.point_x,self.point_y])
            self.area.pop(0)
    def render(self):
        self.area_surf.fill((0, 0, 0))
        #print(self.area)
        for point in self.area:
            if point != 0: #None:
                pos = (point[0], point[1])
                pygame.draw.circle(self.area_surf, (125, 125, 125), pos, radius=1)
        for point in self.objects:
            if point != 0: #None:
                pos = (point[0], point[1])
                pygame.draw.circle(self.area_surf, (0, 125, 125), pos, radius=1)
        pygame.draw.circle(self.area_surf, (0, 125, 0), (board.player_x, board.player_y), radius=board.player_radius)
        #pygame.draw.circle(self.board_surf, (0, 0, 125), (board.obj_x, board.obj_y), radius=board.object_radius)

if __name__ == '__main__':
    lidar = Lidar()
    # Training Loop
    pygame.init()
    clock = pygame.time.Clock()

    run = Run()
    lidar.render_init()

    # -- Game Loop --
    found = 0
    while player.alive:

        #board.player_x += math.cos(math.radians(board.player_angle)) * board.max_speed  # speed
        #board.player_y += math.sin(math.radians(board.player_angle)) * board.max_speed  # speed

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break
        if player.found != found:
            found = player.found
            lidar.objects = [None] * int(lidar.memory / 40)

        keys = pygame.key.get_pressed()
        board.player_angle += (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]) * board.max_rotate
        speed = keys[pygame.K_UP] * board.max_speed
        # speed = ((keys[pygame.K_UP]*player.forward) - keys[pygame.K_DOWN]*player.backward) * board.max_speed
        # speed = ((keys[pygame.K_UP]) - keys[pygame.K_DOWN]) * board.max_speed

        board.player_x += math.cos(math.radians(board.player_angle)) * speed  # * player.move_x
        board.player_y += math.sin(math.radians(board.player_angle)) * speed  # * player.move_y

        run.step()

        for ray in rays:
            #print(ray.object)
            if ray.object == (125, 125, 0):
                #print((ray.end_x, ray.end_y))
                lidar.bounce(ray.end_x, ray.end_y, False)
            if ray.object == (0, 0, 255):
                lidar.bounce(ray.end_x, ray.end_y, True)

        lidar.render()
        pygame.display.flip()
        clock.tick(board.fps)




pygame.quit()
