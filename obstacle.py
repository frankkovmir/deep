import pygame
from random import randint

class Obstacle(pygame.sprite.Sprite):
    base_speed = -5

    def __init__(self, type):
        super().__init__()

        if type == 'fly':
            fly_1 = pygame.image.load('graphics/fly/fly1.png').convert_alpha()
            fly_2 = pygame.image.load('graphics/fly/fly2.png').convert_alpha()
            self.frames = [fly_1, fly_2]
            y_pos = 245
        else:
            enemy_frames = [pygame.image.load(f'graphics/enemy/enemy{i}.png').convert_alpha() for i in range(1, 7)]
            self.frames = enemy_frames
            y_pos = 300

        self.animation_index = 0
        self.image = self.frames[self.animation_index]
        self.rect = self.image.get_rect(midbottom=(randint(1300, 1500), y_pos))
        self.speed = Obstacle.base_speed 

    def animation_state(self):
        self.animation_index += 0.1 
        if self.animation_index >= len(self.frames):
            self.animation_index = 0
        self.image = self.frames[int(self.animation_index)]

    def update(self, score):
        self.animation_state()
        speed_increase = score // 15
        self.speed = Obstacle.base_speed - speed_increase
        self.rect.x += self.speed
        self.destroy()

    def destroy(self):
        if self.rect.x <= -100: 
            self.kill()
