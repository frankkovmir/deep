import pygame
from random import randint

class Obstacle(pygame.sprite.Sprite):
    base_speed = -5  # Initial speed of obstacles

    def __init__(self, type):
        super().__init__()

        if type == 'fly':
            fly_1 = pygame.image.load('graphics/fly/fly1.png').convert_alpha()
            fly_2 = pygame.image.load('graphics/fly/fly2.png').convert_alpha()
            self.frames = [fly_1, fly_2]
            y_pos = 245
        else:
            snail_1 = pygame.image.load('graphics/snail/snail1.png').convert_alpha()
            snail_2 = pygame.image.load('graphics/snail/snail2.png').convert_alpha()
            self.frames = [snail_1, snail_2]
            y_pos = 300

        self.animation_index = 0
        self.image = self.frames[self.animation_index]
        self.rect = self.image.get_rect(midbottom=(randint(1300, 1500), y_pos))
        self.speed = Obstacle.base_speed  # Initialize speed with base speed

    def animation_state(self):
        self.animation_index += 0.1 
        if self.animation_index >= len(self.frames):
            self.animation_index = 0
        self.image = self.frames[int(self.animation_index)]

    def update(self, score):
        self.animation_state()

        # Calculate speed based on score
        speed_increase = score // 15
        self.speed = Obstacle.base_speed - speed_increase

        # Update obstacle's position using the speed
        self.rect.x += self.speed

        self.destroy()

    def destroy(self):
        if self.rect.x <= -100: 
            self.kill()
