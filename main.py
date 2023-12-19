import pygame
import random
import cv2
import time
from pygame import mixer

class SpaceDodgerGame:
    def __init__(self):
        pygame.init()
        mixer.init()
        mixer.music.load('assets/neon-gaming-128925.mp3')
        mixer.music.play(-1)
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

        self.spaceship_masks = [pygame.mask.from_surface(sprite) for sprite in self.spaceships]

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
        self.spaceship_mask = self.spaceship_masks[0]
        return self.get_state()

    def update_spaceship_animation(self):
        self.animation_frame_count += 1
        if self.animation_frame_count >= self.animation_speed:
            self.animation_frame_count = 0
            self.current_spaceship_index = (self.current_spaceship_index + 1) % len(self.spaceships)
            self.spaceship_mask = self.spaceship_masks[self.current_spaceship_index]

    def step(self, action):
        if action == 0 and self.spaceship.x > 0:
            self.spaceship.x -= min(self.spaceship_speed, self.spaceship.x)
        elif action == 1 and self.spaceship.x < self.width - self.spaceship_width:
            self.spaceship.x += min(self.spaceship_speed, self.width - self.spaceship_width - self.spaceship.x)

        if self.frame_count % self.asteroid_spawn_rate == 0:
            asteroid_size = random.randint(self.asteroid_min_size, self.asteroid_max_size)
            asteroid_x = random.randint(0, self.width - asteroid_size)
            edge_bias = random.random() < 0.2  # 20% chance to spawn near edges
            if edge_bias:
                # Choose left or right edge
                asteroid_x = 0 if random.random() < 0.5 else self.width - asteroid_size
            else:
                asteroid_x = random.randint(0, self.width - asteroid_size)
            asteroid_speed = random.randint(self.asteroid_min_speed, self.asteroid_max_speed)
            asteroid_sprite = random.choice(self.asteroid_sprites)
            scaled_sprite = pygame.transform.scale(asteroid_sprite, (asteroid_size, asteroid_size))
            self.asteroids.append(
                [scaled_sprite, pygame.Rect(asteroid_x, 0, asteroid_size, asteroid_size), asteroid_speed])

        done = False
        for asteroid in self.asteroids:
            asteroid[1].y += asteroid[2]
            if asteroid[1].y > self.height:
                self.asteroids.remove(asteroid)

            # Collision detection using masks
            offset_x = asteroid[1].x - self.spaceship.x
            offset_y = asteroid[1].y - self.spaceship.y
            asteroid_mask = pygame.mask.from_surface(asteroid[0])
            if self.spaceship_mask.overlap(asteroid_mask, (offset_x, offset_y)):
                done = True
                break

        corner_penalty = 0
        corner_threshold = 50
        movement_penalty = -0.1  # Small penalty for moving

        if self.spaceship.x < corner_threshold or self.spaceship.x > self.width - self.spaceship_width - corner_threshold:
            corner_penalty = -0.5  # Penalty for being in the corner

        reward = 1 + movement_penalty + corner_penalty if not done else -100

        self.frame_count += 1

        current_time = time.time()
        if current_time - self.last_points_time >= 1:
            self.points += 1
            self.last_points_time = current_time

        return self.get_state(), reward, done

    def get_state(self):
        pygame.display.flip()
        data = pygame.surfarray.array3d(self.screen)
        data = cv2.resize(cv2.cvtColor(data, cv2.COLOR_RGB2GRAY), (84, 84))
        return data

    def render(self):
        self.screen.fill((0, 0, 0))

        current_spaceship = self.spaceships[self.current_spaceship_index]
        self.screen.blit(current_spaceship, self.spaceship.topleft)

        for asteroid in self.asteroids:
            self.screen.blit(asteroid[0], asteroid[1].topleft)

        font = pygame.font.Font(None, 36)
        text = font.render(f"Points: {self.points}", True, (255, 255, 255))
        self.screen.blit(text, (10, 10))
