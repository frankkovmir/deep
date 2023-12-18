import pygame
import random
import cv2
import time

class SpaceDodgerGame:
    def __init__(self):
        pygame.init()

        # Load asteroid sprite sheet
        asteroid_sheet = pygame.image.load('assets/asteroids.png')
        asteroid_sprite_width, asteroid_sprite_height = asteroid_sheet.get_size()
        asteroid_sprite_width //= 8  # Assuming 8 sprites in a row
        asteroid_sprite_height //= 8  # Assuming 8 rows

        # Extract individual asteroid sprites
        self.asteroid_sprites = []
        for row in range(8):
            for col in range(8):
                rect = pygame.Rect(col * asteroid_sprite_width, row * asteroid_sprite_height, asteroid_sprite_width, asteroid_sprite_height)
                asteroid_sprite = asteroid_sheet.subsurface(rect)
                self.asteroid_sprites.append(asteroid_sprite)


        # Load spaceship sprites
        sprite_sheet = pygame.image.load('assets/ship.png')
        sprite_width, sprite_height = sprite_sheet.get_size()
        sprite_width //= 5  # Assuming 5 sprites in a row
        sprite_height //= 2  # Assuming 2 rows
        scale_factor = 2  # Scale factor for spaceship
        self.spaceships = []
        for i in range(5):
            rect = pygame.Rect(i * sprite_width, 0, sprite_width, sprite_height)
            spaceship_sprite = sprite_sheet.subsurface(rect)
            scaled_sprite = pygame.transform.scale(spaceship_sprite,
                                                   (sprite_width * scale_factor, sprite_height * scale_factor))
            self.spaceships.append(scaled_sprite)

        # Spaceship animation variables
        self.current_spaceship_index = 0
        self.animation_frame_count = 0
        self.animation_speed = 5  # Lower is faster

        # Game window dimensions
        self.width, self.height = 800, 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Space Dodger")

        # Spaceship properties
        self.spaceship_width, self.spaceship_height = self.spaceships[0].get_size()
        self.spaceship_speed = 10
        self.spaceship = None

        # Points counter
        self.points = 0
        self.last_points_time = time.time()

        # Asteroid properties
        self.asteroid_min_size, self.asteroid_max_size = 10, 80
        self.asteroid_min_speed, self.asteroid_max_speed = 2, 20
        self.asteroid_spawn_rate = 20
        self.asteroids = []

        self.reset()

    def reset(self):
        self.spaceship = pygame.Rect(self.width // 2 - self.spaceship_width // 2,
                                     self.height - self.spaceship_height - 10,
                                     self.spaceship_width, self.spaceship_height)
        self.asteroids = []
        self.frame_count = 0
        return self.get_state()

    def update_spaceship_animation(self):
        self.animation_frame_count += 1
        if self.animation_frame_count >= self.animation_speed:
            self.animation_frame_count = 0
            self.current_spaceship_index = (self.current_spaceship_index + 1) % len(self.spaceships)

    def step(self, action):
        # action: 0 - move left, 1 - move right
        if action == 0 and self.spaceship.x > 0:
            self.spaceship.x -= self.spaceship_speed
        elif action == 1 and self.spaceship.x < self.width - self.spaceship_width:
            self.spaceship.x += self.spaceship_speed

        # Spawn asteroids
        if self.frame_count % self.asteroid_spawn_rate == 0:
            asteroid_size = random.randint(self.asteroid_min_size, self.asteroid_max_size)
            asteroid_x = random.randint(0, self.width - asteroid_size)
            asteroid_speed = random.randint(self.asteroid_min_speed, self.asteroid_max_speed)

            # Select a random sprite and scale it to the asteroid size
            asteroid_sprite = random.choice(self.asteroid_sprites)
            scaled_sprite = pygame.transform.scale(asteroid_sprite, (asteroid_size, asteroid_size))
            self.asteroids.append(
                [scaled_sprite, pygame.Rect(asteroid_x, 0, asteroid_size, asteroid_size), asteroid_speed])

        # Update asteroid positions and check for collisions
        done = False
        for asteroid in self.asteroids:
            asteroid[1].y += asteroid[2]
            if asteroid[1].y > self.height:
                self.asteroids.remove(asteroid)
            if self.spaceship.colliderect(asteroid[1]):
                done = True

        reward = 1 if not done else -100  # Reward of 1 for surviving a frame, -100 for collision
        self.frame_count += 1

        # Update points counter
        current_time = time.time()
        if current_time - self.last_points_time >= 1:  # Increase points every second
            self.points += 1
            self.last_points_time = current_time

        return self.get_state(), reward, done

    def get_state(self):
        # Capture the current screen state
        pygame.display.flip()
        data = pygame.surfarray.array3d(self.screen)
        data = cv2.resize(cv2.cvtColor(data, cv2.COLOR_RGB2GRAY), (84, 84))
        return data

    def render(self):
        self.screen.fill((0, 0, 0))  # Black background

        # Draw current spaceship sprite
        current_spaceship = self.spaceships[self.current_spaceship_index]
        self.screen.blit(current_spaceship, self.spaceship.topleft)

        for asteroid in self.asteroids:
            self.screen.blit(asteroid[0], asteroid[1].topleft)  # Use the sprite for drawing

        font = pygame.font.Font(None, 36)
        text = font.render(f"Points: {self.points}", True, (255, 255, 255))
        self.screen.blit(text, (10, 10))
