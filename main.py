import pygame
import random
import cv2
import time

class SpaceDodgerGame:
    def __init__(self):
        pygame.init()

        # Game window dimensions
        self.width, self.height = 800, 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Space Dodger")

        # Spaceship properties
        self.spaceship_width, self.spaceship_height = 40, 60
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
            self.asteroids.append([pygame.Rect(asteroid_x, 0, asteroid_size, asteroid_size), asteroid_speed])

        # Update asteroid positions and check for collisions
        done = False
        for asteroid in self.asteroids:
            asteroid[0].y += asteroid[1]
            if asteroid[0].y > self.height:
                self.asteroids.remove(asteroid)
            if self.spaceship.colliderect(asteroid[0]):
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
        pygame.draw.rect(self.screen, (255, 255, 255), self.spaceship)  # Draw spaceship
        for asteroid in self.asteroids:
            pygame.draw.rect(self.screen, (255, 255, 255), asteroid[0])  # Draw asteroids

        font = pygame.font.Font(None, 36)
        text = font.render(f"Points: {self.points}", True, (255, 255, 255))
        self.screen.blit(text, (10, 10))
