import pygame
import settings
from random import choice
import math
import pygame
from random import randint

def display_score(screen, start_time, test_font):
    current_time = int(pygame.time.get_ticks() / 1000) - start_time
    score_surf = test_font.render(f'Punkte: {current_time}', False, (64, 64, 64))
    score_rect = score_surf.get_rect(center=(400, 50))
    screen.blit(score_surf, score_rect)
    return current_time

def collision_sprite(player, obstacle_group):
    if pygame.sprite.spritecollide(player.sprite, obstacle_group, False):
        obstacle_group.empty()
        return False
    else: 
        return True

class Obstacle(pygame.sprite.Sprite):
    base_speed = -5

    def __init__(self, type):
        super().__init__()

        if type == 'fly':
            fly_1 = pygame.image.load('graphics/fly/fly1.png').convert_alpha()
            fly_2 = pygame.image.load('graphics/fly/fly2.png').convert_alpha()
            self.frames = [fly_1, fly_2]
            y_pos = 515
        else:
            enemy_frames = [pygame.image.load(f'graphics/enemy/enemy{i}.png').convert_alpha() for i in range(1, 7)]
            self.frames = enemy_frames
            y_pos = 555

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

class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.load_images()
        self.player_index = 0
        self.image = self.player_stand[0]
        self.rect = self.image.get_rect(midbottom=(80, 555))
        self.gravity = 0
        self.ducking = False
        self.original_rect_height = self.rect.height
        self.duck_rect_height = self.original_rect_height * 0.85
        self.jump_sound = pygame.mixer.Sound('audio/jump.mp3')
        self.jump_sound.set_volume(0.5)

    def load_images(self):
        self.player_walk = [pygame.image.load(f'graphics/player/player_walk_{i}.png').convert_alpha() for i in range(1, 7)]
        self.player_jump = [pygame.image.load(f'graphics/player/jump_{i}.png').convert_alpha() for i in range(1, 4)]
        self.player_stand = [pygame.image.load(f'graphics/player/player_stand_{i}.png').convert_alpha() for i in range(1, 6)]
        self.player_duck_images = [pygame.image.load(f'graphics/player/player_duck_{i}.png').convert_alpha() for i in range(1, 4)]

    def player_input(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE] and self.rect.bottom >= 555:
            self.gravity = -20
            self.jump_sound.play()
        if keys[pygame.K_s] or keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]:
            self.ducking = True
        else:
            self.ducking = False

    def apply_gravity(self):
        self.gravity += 1
        self.rect.y += self.gravity
        if self.rect.bottom >= 555:
            self.rect.bottom = 555

    def animation_state(self):
        if self.rect.bottom < 555:
            self.image = self.player_jump[int(self.player_index) % len(self.player_jump)]
        else:  # On ground
            if self.gravity < 0:
                self.image = self.player_jump[0]
            else:
                self.player_index += 0.1
                if self.player_index >= len(self.player_walk):
                    self.player_index = 0
                self.image = self.player_walk[int(self.player_index)]

    def update(self):
        self.player_input()
        self.apply_gravity()
        if self.ducking:
            self.image = self.player_duck_images[0]
            self.rect.height = self.duck_rect_height
            self.rect.y += self.original_rect_height - self.duck_rect_height
        else:
            self.animation_state()
            self.rect.height = self.original_rect_height

class PirateBayGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(settings.SCREEN_SIZE)
        pygame.display.set_caption(settings.CAPTION)
        self.clock = pygame.time.Clock()
        self.test_font = pygame.font.Font('font/Pixeltype.ttf', 50)
        self.bg_music = pygame.mixer.Sound('audio/music.wav')
        self.bg_music.play(loops=-1)
        self._reset_game()

    def _load_resources(self):
        self.ground_surface_original = pygame.image.load('graphics/bg.png').convert()
        self.ground_surface = pygame.transform.scale(self.ground_surface_original, (settings.SCREEN_SIZE[0], settings.SCREEN_SIZE[1]))
        self.ground_surface_width = self.ground_surface.get_width()
        self.tiles = math.ceil(settings.SCREEN_SIZE[0] / self.ground_surface_width) + 1
        self.scroll = 0

    def _set_up_timers(self):
        self.obstacle_timer = pygame.USEREVENT + 1
        pygame.time.set_timer(self.obstacle_timer, 1500)

    def _reset_game(self):
        self.game_active = True
        self.start_time = int(pygame.time.get_ticks() / 1000)  # Reset start time
        self.score = 0
        self.player = pygame.sprite.GroupSingle(Player())
        self.obstacle_group = pygame.sprite.Group()
        self._load_resources()
        self._set_up_timers()

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False

            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.player.sprite.player_input()

            if event.type == self.obstacle_timer:
                self.obstacle_group.add(Obstacle(choice(['fly', 'snail', 'snail', 'snail'])))

        return True

    def _update_game_logic(self):
        self.scroll -= 4
        if self.scroll <= -self.ground_surface_width:
            self.scroll = 0

        self.player.update()
        self.obstacle_group.update(self.score)
        self.game_active = collision_sprite(self.player, self.obstacle_group)
        if not self.game_active:
            self._reset_game()

    def _render(self):
        for i in range(self.tiles):
            self.screen.blit(self.ground_surface, (i * self.ground_surface_width + self.scroll, settings.SCREEN_SIZE[1] - self.ground_surface.get_height()))

        self.score = display_score(self.screen, self.start_time, self.test_font)
        self.player.draw(self.screen)
        self.obstacle_group.draw(self.screen)
        pygame.display.update()

    def play_step(self):
        if not self._handle_events():
            return False
        self._update_game_logic()
        self._render()
        self.clock.tick(60)
        return True

if __name__ == '__main__':
    game = PirateBayGame()
    while game.play_step():
        pass
    pygame.quit()
