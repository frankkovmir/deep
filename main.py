import pygame
from player import Player
from obstacle import Obstacle
from utilities import display_score, collision_sprite
import settings
from random import choice

def main():
    pygame.init()
    screen = pygame.display.set_mode(settings.SCREEN_SIZE)
    pygame.display.set_caption(settings.CAPTION)
    clock = pygame.time.Clock()
    test_font = pygame.font.Font('font/Pixeltype.ttf', 50)
    game_active = False
    start_time = 0
    score = 0
    bg_music = pygame.mixer.Sound('audio/music.wav')
    bg_music.play(loops=-1)

    # Player and Obstacle Groups
    player = pygame.sprite.GroupSingle()
    player.add(Player())
    obstacle_group = pygame.sprite.Group()

    # Loading the sky and ground surfaces
    sky_surface = pygame.image.load('graphics/Sky.png').convert()
    ground_surface = pygame.image.load('graphics/ground.png').convert()

    # Intro screen elements
    player_stand = pygame.image.load('graphics/player/player_stand.png').convert_alpha()
    player_stand = pygame.transform.rotozoom(player_stand, 0, 2)
    player_stand_rect = player_stand.get_rect(center=(400, 200))
    game_name = test_font.render('Pixel Runner', False, (111, 196, 169))
    game_name_rect = game_name.get_rect(center=(400, 80))
    game_message = test_font.render('Press space to run', False, (111, 196, 169))
    game_message_rect = game_message.get_rect(center=(400, 330))

    # Timer for obstacle generation
    obstacle_timer = pygame.USEREVENT + 1
    pygame.time.set_timer(obstacle_timer, 1500)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

            if game_active:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        player.sprite.player_input()

            else:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    game_active = True
                    start_time = int(pygame.time.get_ticks() / 1000)

            if game_active:
                if event.type == obstacle_timer:
                    obstacle_group.add(Obstacle(choice(['fly', 'snail', 'snail', 'snail'])))

        if game_active:
            screen.blit(sky_surface, (0, 0))
            screen.blit(ground_surface, (0, 300))
            score = display_score(screen, start_time, test_font)
            
            player.draw(screen)
            player.update()

            obstacle_group.draw(screen)
            obstacle_group.update()

            game_active = collision_sprite(player, obstacle_group)
            
        else:
            screen.fill((94, 129, 162))
            screen.blit(player_stand, player_stand_rect)
            screen.blit(game_name, game_name_rect)

            if score == 0:
                screen.blit(game_message, game_message_rect)
            else:
                score_message = test_font.render(f'Your score: {score}', False, (111, 196, 169))
                score_message_rect = score_message.get_rect(center=(400, 330))
                screen.blit(score_message, score_message_rect)

        pygame.display.update()
        clock.tick(60)

if __name__ == '__main__':
    main()