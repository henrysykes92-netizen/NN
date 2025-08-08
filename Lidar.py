import math

from Environment import *

class Lidar:
    def __init__(self, mem=True):
        self.mem = mem
        if self.mem == True:
            self.memory = 200
            self.area = [[0,0]] * self.memory
            self.objects = [[0,0]] * int(self.memory / 4)
        else:
            self.area = []
            self.objects = []

    def render_init(self):
        pygame.init()
        self.clock = pygame.time.Clock()
        self.area_surf = pygame.display.set_mode((board.width, board.height))

    def step(self):
        for ray in rays:
            if ray.object == (125, 125, 0):
                self.bounce(ray.end_x, ray.end_y, False)
            if ray.object == (0, 0, 255):
                self.bounce(ray.end_x, ray.end_y, True)

    def bounce(self, x, y, obj=False):
        self.point_x = x
        self.point_y = y
        p = [self.point_x,self.point_y]
        if obj == True:
            self.objects.append(p)
            if self.mem == True:
                self.objects.pop(0)
        else:
            self.area.append(p)
            if self.mem == True:
                self.area.pop(0)
    def render_step(self):
        self.area_surf.fill((0, 0, 0))
        for point in self.area:
            if point != 0:
                pos = (point[0], point[1])
                pygame.draw.circle(self.area_surf, (125, 125, 125), pos, radius=1)
        for point in self.objects:
            if point != 0:
                pos = (point[0], point[1])
                pygame.draw.circle(self.area_surf, (0, 125, 125), pos, radius=1)
        pygame.draw.circle(self.area_surf, (0, 125, 0), (board.player_x, board.player_y), radius=board.player_radius)
        end_c = (board.player_x+(board.player_radius*math.cos(math.radians(player.angle))),
                 board.player_y+(board.player_radius*math.sin(math.radians(player.angle))))
        pygame.draw.line(self.area_surf, (255,0,0), (board.player_x,board.player_y), end_c)

if __name__ == '__main__':
    pygame.init()
    clock = pygame.time.Clock()

    run = Run()
    lidar = Lidar()
    lidar.render_init()

    # -- Game Loop --
    prev_f = player.found
    while player.alive:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break

        keys = pygame.key.get_pressed()
        board.player_angle += (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]) * board.max_rotate
        speed = keys[pygame.K_UP] * board.max_speed
        board.player_x += math.cos(math.radians(board.player_angle)) * speed
        board.player_y += math.sin(math.radians(board.player_angle)) * speed

        run.step()
        lidar.step()

        if prev_f != player.found:
            lidar.objects = [0] * int(lidar.memory / 4)
            prev_f = player.found

        lidar.render_step()
        pygame.display.flip()
        clock.tick(board.fps)

pygame.quit()
