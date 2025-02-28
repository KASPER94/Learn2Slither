import pygame
import os

os.environ["SDL_VIDEODRIVER"] = "x11"  
os.environ["SDL_RENDER_DRIVER"] = "software" 
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1" 
os.environ["MESA_LOADER_DRIVER_OVERRIDE"] = "swrast"  

pygame.init()
screen = pygame.display.set_mode((400, 300))
pygame.display.set_caption("Test Pygame")

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((0, 255, 0))  # Fond vert
    pygame.display.flip()

pygame.quit()
