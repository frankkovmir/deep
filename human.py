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

# Load the entire sprite sheet
sprite_sheet = pygame.image.load('assets/ship.png')
sprite_width, sprite_height = sprite_sheet.get_size()
sprite_width //= 5  # Assuming 5 sprites in a row
sprite_height //= 2  # Assuming 2 rows
scale_factor = 2

# Extract the middle 3 sprites from the top row
spaceships = []

for i in range(5):
    rect = pygame.Rect(i * sprite_width, 0, sprite_width, sprite_height)
    spaceship_sprite = sprite_sheet.subsurface(rect)
    scaled_sprite = pygame.transform.scale(spaceship_sprite, (sprite_width * scale_factor, sprite_height * scale_factor))
    spaceships.append(scaled_sprite)

# Animation variables
current_spaceship_index = 0
animation_frame_count = 0
animation_speed = 5  # Lower is faster

# Initial spaceship setup
current_spaceship = spaceships[current_spaceship_index]
spaceship_width, spaceship_height = current_spaceship.get_size()
spaceship_x = WIDTH // 2 - spaceship_width // 2
spaceship_y = HEIGHT - spaceship_height - 10
spaceship_speed = 10
spaceship_rect = pygame.Rect(spaceship_x, spaceship_y, spaceship_width, spaceship_height)

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
        spaceship_rect.x -= spaceship_speed
    if keys[pygame.K_RIGHT]:
        spaceship_rect.x += spaceship_speed

    # Keep spaceship within screen bounds
    spaceship_rect.x = max(0, min(spaceship_rect.x, WIDTH - spaceship_width))

    # Animation

    animation_frame_count += 1
    if animation_frame_count >= animation_speed:
        animation_frame_count = 0
        current_spaceship_index = (current_spaceship_index + 1) % len(spaceships)
        current_spaceship = spaceships[current_spaceship_index]


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
        if spaceship_rect.colliderect(asteroid[0]):
            print("Collision detected!")
            running = False

    # Draw spaceship image
    screen.blit(current_spaceship, spaceship_rect.topleft)

    # Draw asteroids
    for asteroid in asteroids:
        pygame.draw.rect(screen, WHITE, asteroid[0])

    pygame.display.flip()
    frame_count += 1
    clock.tick(FPS)

pygame.quit()
