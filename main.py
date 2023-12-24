import pygame
import random
from pygame import mixer

class SpaceDodgerGame:
    def __init__(self):
        pygame.init()
        mixer.init()

        self.clock = pygame.time.Clock()
        self.dodge_reward = 0.1
        self.asteroids_per_level = 1
        self.current_level = 1
        self.spawned_asteroids_count = 0

        mixer.music.load('assets/neon-gaming-128925.mp3')
        mixer.music.play(-1)

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
        self.asteroid_spawn_rate = 60 # not spawning simultatnously
        self.asteroids = []
        self.reset()

    def reset(self):
        self.spaceship = pygame.Rect(self.width // 2 - self.spaceship_width // 2,
                                     self.height - self.spaceship_height - 10,
                                     self.spaceship_width, self.spaceship_height)
        self.asteroids = []
        self.frame_count = 0
        self.spaceship_mask = self.spaceship_masks[0]
        self.current_level = 1  # Reset to level 1
        self.spawned_asteroids_count = 0
        self.asteroid_speed = 1  # Reset the asteroid speed to the initial value
        return self.get_state()

    def update_spaceship_animation(self):
        self.animation_frame_count += 1
        if self.animation_frame_count >= self.animation_speed:
            self.animation_frame_count = 0
            self.current_spaceship_index = (self.current_spaceship_index + 1) % len(self.spaceships)
            self.spaceship_mask = self.spaceship_masks[self.current_spaceship_index]

    def step(self, action):

        level_complete_reward = 1
        collision_penalty = -1
        reward = 0
        done = False

        # Movement actions
        if action == 1 and self.spaceship.x > 0:
            self.spaceship.x -= min(self.spaceship_speed, self.spaceship.x)
        elif action == 2 and self.spaceship.x < self.width - self.spaceship_width:
            self.spaceship.x += min(self.spaceship_speed, self.width - self.spaceship_width - self.spaceship.x)
        elif action == 0 and self.spaceship.x < self.width - self.spaceship_width:
            pass

        # Spawn a fixed number of asteroids per level
        if self.spawned_asteroids_count < self.asteroids_per_level:
            if self.frame_count % self.asteroid_spawn_rate == 0:
                asteroid_x = random.randint(0, self.width - self.asteroid_size)
                asteroid_sprite = random.choice(self.asteroid_sprites)
                scaled_sprite = pygame.transform.scale(asteroid_sprite, (self.asteroid_size, self.asteroid_size))
                self.asteroids.append(
                    [scaled_sprite, pygame.Rect(asteroid_x, 0, self.asteroid_size, self.asteroid_size),
                     self.asteroid_speed])
                self.spawned_asteroids_count += 1

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
                # Reward for dodging when the asteroid just crosses the bottom of the screen
                reward += self.dodge_reward

        # Remove asteroids that have gone off-screen
        self.asteroids = [asteroid for asteroid in self.asteroids if asteroid[1].y <= self.height]
        #print("Asteroids remaining after removal:", len(self.asteroids))

        # Check if all asteroids have been dodged or gone off-screen
        if len(self.asteroids) == 0 and self.spawned_asteroids_count >= self.asteroids_per_level:
            reward += level_complete_reward
            self.prepare_next_level()
            done = False

        self.frame_count += 1
        #print("Total Reward This Step:", reward)

        return self.get_state(), reward, done

    def prepare_next_level(self):
        self.current_level += 1
        self.asteroids_per_level += 1
        self.spawned_asteroids_count = 0
        self.asteroid_speed += 0.5
        self.asteroids = []
        print(f"Moving to Level: {self.current_level}, Asteroids This Level: {self.asteroids_per_level}")

    def get_state(self):
        """folde inputs werden übergeben
        1 Spaceship X position (normalized)
        2 Spaceship Y position (normalized)
        3 Nearest asteroid X position (normalized)
        4 Nearest asteroid Y position (normalized)
        5 Asteroid speed (normalized)
        6 Distance to left edge (normalized)
        7 Distance to right edge (normalized)
        8 Horizontal distance to nearest asteroid (normalized)
        9 Vertical distance to nearest asteroid (normalized)
        10 Current level (normalized)"""

        pygame.display.flip()
        state = [
            self.spaceship.x / self.width,
            self.spaceship.y / self.height
        ]

        # Distances to the screen edges
        distance_to_left_edge = self.spaceship.x / self.width
        distance_to_right_edge = (self.width - (self.spaceship.x + self.spaceship_width)) / self.width
        state.extend([distance_to_left_edge, distance_to_right_edge])

        # Information about the nearest asteroid
        if self.asteroids:
            nearest_asteroid = min(self.asteroids, key=lambda a: a[1].y)
            asteroid_x = nearest_asteroid[1].x / self.width
            asteroid_y = nearest_asteroid[1].y / self.height
            state.extend([asteroid_x, asteroid_y, self.asteroid_speed / 10])
            # Horizontal and vertical distances to the nearest asteroid
            horizontal_distance = (nearest_asteroid[1].x - self.spaceship.x) / self.width
            vertical_distance = (nearest_asteroid[1].y - self.spaceship.y) / self.height
            state.extend([horizontal_distance, vertical_distance])
        else:
            state.extend([0, 0, 0, 0, 0])

        # Level information
        state.append(self.current_level / 10)

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
