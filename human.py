import pygame
import random

# Initialize Pygame
pygame.init()

# Game window dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Space Dodger")

# Game clock and FPS setting
clock = pygame.time.Clock()
FPS = 30

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Spaceship properties
spaceship_width, spaceship_height = 40, 60
spaceship_x = WIDTH // 2 - spaceship_width // 2
spaceship_y = HEIGHT - spaceship_height - 10
spaceship_speed = 10
spaceship = pygame.Rect(spaceship_x, spaceship_y, spaceship_width, spaceship_height)

# Asteroid properties
asteroid_min_size, asteroid_max_size = 10, 80
asteroid_min_speed, asteroid_max_speed = 2, 20
asteroid_spawn_rate = 20  # Frames
asteroids = []

# Game loop
running = True
frame_count = 0

while running:
    screen.fill(BLACK)

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # AI agent's action (to be integrated)
    # For now, using keyboard for control
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        spaceship.x -= spaceship_speed
    if keys[pygame.K_RIGHT]:
        spaceship.x += spaceship_speed

    # Keep spaceship within screen bounds
    spaceship.x = max(0, min(spaceship.x, WIDTH - spaceship_width))

    # Asteroid spawning
    if frame_count % asteroid_spawn_rate == 0:
        asteroid_size = random.randint(asteroid_min_size, asteroid_max_size)
        asteroid_x = random.randint(0, WIDTH - asteroid_size)
        asteroid_speed = random.randint(asteroid_min_speed, asteroid_max_speed)
        asteroids.append([pygame.Rect(asteroid_x, 0, asteroid_size, asteroid_size), asteroid_speed])

    # Update asteroid positions
    for asteroid in asteroids:
        asteroid[0].y += asteroid[1]
        if asteroid[0].y > HEIGHT:
            asteroids.remove(asteroid)

    # Collision detection
    for asteroid in asteroids:
        if spaceship.colliderect(asteroid[0]):
            print("Collision detected!")
            running = False

    # Draw spaceship
    pygame.draw.rect(screen, WHITE, spaceship)

    # Draw asteroids
    for asteroid in asteroids:
        pygame.draw.rect(screen, WHITE, asteroid[0])

    pygame.display.flip()
    frame_count += 1
    clock.tick(FPS)

pygame.quit()
