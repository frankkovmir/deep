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
    scaled_sprite = pygame.transform.scale(spaceship_sprite,
                                           (sprite_width * scale_factor, sprite_height * scale_factor))
    spaceships.append(scaled_sprite)

# Load asteroid sprite sheet
asteroid_sheet = pygame.image.load('assets/asteroids.png')
asteroid_sprite_width, asteroid_sprite_height = asteroid_sheet.get_size()
asteroid_sprite_width //= 8  # Assuming 4 sprites in a row
asteroid_sprite_height //= 8  # Assuming 4 rows

# Extract individual asteroid sprites
asteroid_sprites = []
for row in range(8):
    for col in range(8):
        rect = pygame.Rect(col * asteroid_sprite_width, row * asteroid_sprite_height, asteroid_sprite_width,
                           asteroid_sprite_height)
        asteroid_sprite = asteroid_sheet.subsurface(rect)
        asteroid_sprites.append(asteroid_sprite)

# Asteroid properties
asteroid_min_size, asteroid_max_size = 10, 200
asteroid_min_speed, asteroid_max_speed = 2, 20
asteroid_spawn_rate = 20  # Frames
asteroids = []

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

# Generate mask for spaceship
spaceship_mask = pygame.mask.from_surface(current_spaceship)

# Generate masks for asteroids
asteroid_masks = [pygame.mask.from_surface(sprite) for sprite in asteroid_sprites]

# Game loop
running = True
frame_count = 0

while running:
    screen.fill(BLACK)

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Spaceship control
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
        spaceship_mask = pygame.mask.from_surface(current_spaceship)

    # Asteroid spawning
    if frame_count % asteroid_spawn_rate == 0:
        asteroid_size = random.randint(asteroid_min_size, asteroid_max_size)
        asteroid_x = random.randint(0, WIDTH - asteroid_size)
        asteroid_speed = random.randint(asteroid_min_speed, asteroid_max_speed)
        asteroid_sprite = random.choice(asteroid_sprites)
        scaled_sprite = pygame.transform.scale(asteroid_sprite, (asteroid_size, asteroid_size))
        asteroid_mask = pygame.mask.from_surface(scaled_sprite)
        asteroids.append(
            [scaled_sprite, pygame.Rect(asteroid_x, 0, asteroid_size, asteroid_size), asteroid_speed, asteroid_mask])

    # Update asteroid positions and collision detection
    for asteroid in asteroids:
        asteroid[1].y += asteroid[2]  # Update y-position
        if asteroid[1].y > HEIGHT:
            asteroids.remove(asteroid)

        # Collision detection
        offset_x = asteroid[1].x - spaceship_rect.x
        offset_y = asteroid[1].y - spaceship_rect.y
        if spaceship_mask.overlap(asteroid[3], (offset_x, offset_y)):
            print("Collision detected!")
            running = False

    # Draw spaceship image
    screen.blit(current_spaceship, spaceship_rect.topleft)

    # Draw asteroids
    for asteroid in asteroids:
        screen.blit(asteroid[0], asteroid[1])

    pygame.display.flip()
    frame_count += 1
    clock.tick(FPS)

pygame.quit()
