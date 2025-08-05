import imageio.v2 as imageio
import pygame
import os

# Ensure the output directory exists
os.makedirs("Loss", exist_ok=True)

pygame.init()
board = pygame.display.set_mode((600, 400))

# Frame 1
board.fill((0, 0, 0))
pygame.draw.line(board, color=(255, 255, 255), start_pos=(250, 100), end_pos=(250, 300), width=5)
frame_path = "Loss/frame_l1.png"
gif_path = "Loss/frame_l1.gif"
pygame.image.save(pygame.display.get_surface(), frame_path)
image = imageio.imread(frame_path)
imageio.mimsave(gif_path, [image], duration=1)

# Frame 2
board.fill((0, 0, 0))
pygame.draw.line(board, color=(255, 255, 255), start_pos=(250, 100), end_pos=(250, 300), width=5)
pygame.draw.line(board, color=(255, 255, 255), start_pos=(350, 100), end_pos=(350, 300), width=5)
frame_path = "Loss/frame_l2.png"
gif_path = "Loss/frame_l2.gif"
pygame.image.save(pygame.display.get_surface(), frame_path)
image = imageio.imread(frame_path)
imageio.mimsave(gif_path, [image], duration=1)

# Frame 3
frame_path = "Loss/frame_l3.png"
gif_path = "Loss/frame_l3.gif"
pygame.image.save(pygame.display.get_surface(), frame_path)
image = imageio.imread(frame_path)
imageio.mimsave(gif_path, [image], duration=1)

# Frame 4
board.fill((0, 0, 0))
pygame.draw.line(board, color=(255, 255, 255), start_pos=(250, 100), end_pos=(250, 300), width=5)
pygame.draw.line(board, color=(255, 255, 255), start_pos=(350, 300), end_pos=(550, 300), width=5)
frame_path = "Loss/frame_l4.png"
gif_path = "Loss/frame_l4.gif"
pygame.image.save(pygame.display.get_surface(), frame_path)
image = imageio.imread(frame_path)
imageio.mimsave(gif_path, [image], duration=1)

pygame.quit()
