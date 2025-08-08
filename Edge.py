import random

import pygame

from Lidar import *

class Edge():
    def __init__(self):
        self.horiz = []
        self.vert = []
        for i in range(board.map_size):
            self.horiz.append([])
            self.vert.append([])

    def line(self, p1, p2):
        print(p1,p2)


if __name__ == '__main__':
    pygame.init()
    clock = pygame.time.Clock()

    run = Run()
    lidar = Lidar(mem=False)
    lidar.render_init()

    board.maze(rnd=True,bll=False)

    edge = Edge()

    # -- Game Loop --
    while player.alive:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break

        board.player_angle += 3 # random.randint(0,360)
        board.obj_x = board.width
        board.obj_y = board.height

        run.step()
        lidar.step()

        for i in range(len(edge.vert)):
            for p in lidar.area:
                if round(p[0], 4) % board.tile_size == 0:
                    #print(int(p[0]//board.tile_size))
                    edge.vert[int(p[0]//board.tile_size)].append(p)

        #for p in lidar.area:
            #print(round(p[0], 4), round(p[1], 4))
        for v in edge.vert:
            print(v)
        print('')

        lidar.render_step()
        pygame.display.flip()
        clock.tick(board.fps)

        lidar.area = []

pygame.quit()
