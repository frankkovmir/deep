"""
author: Frank Kovmir
frankkovmir@gmail.com
"""

import pygame
import random

class SpaceDodgerGame:
    def __init__(self):
        pygame.init()

        self.clock = pygame.time.Clock()
        self.dodge_reward = 0.1
        self.asteroids_per_level = 1
        self.current_level = 1
        self.spawned_asteroids_count = 0

        # hole parameter
        self.hole_position = 0
        self.hole_size = 0

        # Load asteroid sprite sheet
        asteroid_sheet = pygame.image.load('assets/asteroids.png')
        asteroid_sprite_width, asteroid_sprite_height = asteroid_sheet.get_size()
        asteroid_sprite_width //= 8
        asteroid_sprite_height //= 8

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
        sprite_width //= 5
        sprite_height //= 2
        scale_factor = 2
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
        self.width, self.height = 288, 512
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Space Dodger")

        # Spaceship properties
        self.spaceship_width, self.spaceship_height = self.spaceships[0].get_size()
        self.spaceship_speed = 3
        self.spaceship = None

        # Asteroid properties
        self.asteroid_size = 40
        self.asteroid_speed = 1
        self.asteroid_spawn_rate = 100 # not spawning simultatnously
        self.asteroids = []
        self.reset()

    def reset(self):
        self.spaceship = pygame.Rect(self.width // 2 - self.spaceship_width // 2,
                                     self.height - self.spaceship_height - 10,
                                     self.spaceship_width, self.spaceship_height)
        self.asteroids = []
        self.frame_count = 0
        self.spaceship_mask = self.spaceship_masks[0]
        self.current_level = 1
        self.spawned_asteroids_count = 0
        self.asteroid_speed = 1
        return self.get_state()

    def update_spaceship_animation(self):
        self.animation_frame_count += 1
        if self.animation_frame_count >= self.animation_speed:
            self.animation_frame_count = 0
            self.current_spaceship_index = (self.current_spaceship_index + 1) % len(self.spaceships)
            self.spaceship_mask = self.spaceship_masks[self.current_spaceship_index]

    def spawn_asteroid_chain(self):
        hole_position = random.randint(self.asteroid_size * 2, self.width - self.asteroid_size * 3)
        hole_size = random.randint(self.asteroid_size * 2, self.asteroid_size * 3)

        for x in range(0, self.width, self.asteroid_size):
            if not (hole_position < x < hole_position + hole_size):
                asteroid_sprite = random.choice(self.asteroid_sprites)
                scaled_sprite = pygame.transform.scale(asteroid_sprite, (self.asteroid_size, self.asteroid_size))
                self.asteroids.append(
                    [scaled_sprite, pygame.Rect(x, 0, self.asteroid_size, self.asteroid_size),
                     self.asteroid_speed])

        # Store the hole position and size
        self.hole_position = hole_position
        self.hole_size = hole_size

    def step(self, action):
        collision_penalty = -1
        successful_hole_navigation_reward = 0.5  # Reward for successfully navigating through the hole
        reward = 0
        done = False

        # Movement actions
        if action == 0 and self.spaceship.x > 0:
            self.spaceship.x -= min(self.spaceship_speed, self.spaceship.x)
        elif action == 1 and self.spaceship.x < self.width - self.spaceship_width:
            self.spaceship.x += min(self.spaceship_speed, self.width - self.spaceship_width - self.spaceship.x)

        # Spawn a chain of asteroids with a hole
        if self.spawned_asteroids_count < self.asteroids_per_level:
            if self.frame_count % self.asteroid_spawn_rate == 0:
                self.spawn_asteroid_chain()
                self.spawned_asteroids_count += 1

        # Calculate the hole's position and size
        if self.asteroids:
            all_x_positions = [asteroid[1].x for asteroid in self.asteroids]
            hole_start = max(0, min(all_x_positions) - self.asteroid_size)
            hole_end = min(self.width, max(all_x_positions) + self.asteroid_size)
        else:
            hole_start = 0
            hole_end = self.width

        # Check collision and update dodged asteroids count
        for asteroid in self.asteroids:
            asteroid_sprite, asteroid_rect, _ = asteroid
            prev_y = asteroid_rect.y
            asteroid_rect.y += self.asteroid_speed
            new_y = asteroid_rect.y

            asteroid_mask = pygame.mask.from_surface(asteroid_sprite)
            offset_x = asteroid_rect.x - self.spaceship.x
            offset_y = new_y - self.spaceship.y

            if self.spaceship_mask.overlap(asteroid_mask, (offset_x, offset_y)):
                done = True
                reward = collision_penalty
                self.reset()
                break

            if prev_y < self.height and new_y >= self.height:
                if hole_start <= self.spaceship.x <= hole_end:
                    reward += successful_hole_navigation_reward

        # Remove asteroids that have gone off-screen
        self.asteroids = [asteroid for asteroid in self.asteroids if asteroid[1].y <= self.height]

        # Check if all asteroids have been dodged or gone off-screen
        if len(self.asteroids) == 0 and self.spawned_asteroids_count >= self.asteroids_per_level:
            self.prepare_next_level()
            done = False

        self.frame_count += 1

        # print(f'state {self.get_state()}')
        # print(f'reward {reward}')
        # print(f'done {done}')

        return self.get_state(), reward, done
    def prepare_next_level(self):
        self.current_level += 1
        #self.asteroids_per_level += 1
        self.asteroid_speed += 0.5
        self.spawned_asteroids_count = 0
        self.asteroids = []
        print(f"Moving to Level: {self.current_level}, Asteroids This Level: {self.asteroids_per_level}")

    def get_state(self):

        pygame.display.flip()

        state = [
            self.spaceship.x / self.width,  # Spaceship X position
            self.spaceship.y / self.height,  # Spaceship Y position
            self.hole_position / self.width,  # Hole X position
            self.hole_size / self.width,  # Hole size
            self.asteroid_speed
        ]
        #print(state)
        return state

    def render(self,action):
        self.screen.fill((0, 0, 0))

        current_spaceship = self.spaceships[self.current_spaceship_index]
        self.screen.blit(current_spaceship, self.spaceship.topleft)

        for asteroid in self.asteroids:
            self.screen.blit(asteroid[0], asteroid[1].topleft)

        font = pygame.font.Font(None, 36)
        level_text = font.render(f"Level: {self.current_level}", True, (255, 255, 255))
        action_text = font.render(f"Action: {action}", True, (255, 255, 255))
        self.screen.blit(action_text, (10, 50))
        self.clock.tick(60)
        self.screen.blit(level_text, (10, 10))
