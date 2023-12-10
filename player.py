import pygame

class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.load_images()
        self.player_index = 0
        self.image = self.player_stand[0]
        self.rect = self.image.get_rect(midbottom=(80, 300))
        self.gravity = 0

        self.jump_sound = pygame.mixer.Sound('audio/jump.mp3')
        self.jump_sound.set_volume(0.5)

    def load_images(self):
        self.player_walk = [pygame.image.load(f'graphics/player/player_walk_{i}.png').convert_alpha() for i in range(1, 7)]
        self.player_jump = [pygame.image.load(f'graphics/player/jump_{i}.png').convert_alpha() for i in range(1, 4)]
        self.player_stand = [pygame.image.load(f'graphics/player/player_stand_{i}.png').convert_alpha() for i in range(1, 6)]

    def player_input(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE] and self.rect.bottom >= 300:
            self.gravity = -20
            self.jump_sound.play()

    def apply_gravity(self):
        self.gravity += 1
        self.rect.y += self.gravity
        if self.rect.bottom >= 300:
            self.rect.bottom = 300

    def animation_state(self):
        if self.rect.bottom < 300:  # In air
            self.image = self.player_jump[int(self.player_index) % len(self.player_jump)]
        else:  # On ground
            if self.gravity < 0:  # Rising in jump
                self.image = self.player_jump[0]
            else:
                self.player_index += 0.1
                if self.player_index >= len(self.player_walk):
                    self.player_index = 0
                self.image = self.player_walk[int(self.player_index)]

    def update(self):
        self.player_input()
        self.apply_gravity()
        self.animation_state()
