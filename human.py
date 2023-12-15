import pygame
from player import Player
from obstacle import Obstacle
from utilities import display_score, collision_sprite
import settings
from random import choice
import math

def main():
    pygame.init()
    # Initialisiere das Pygame-Fenster
    screen = pygame.display.set_mode(settings.SCREEN_SIZE)
    pygame.display.set_caption(settings.CAPTION)
    clock = pygame.time.Clock()
    test_font = pygame.font.Font('font/Pixeltype.ttf', 50)
    game_active = False
    start_time = 0
    score = 0
    bg_music = pygame.mixer.Sound('audio/music.wav')
    bg_music.play(loops=-1)

    # Spieler- und Hindernisgruppen
    player = pygame.sprite.GroupSingle()
    player.add(Player())
    obstacle_group = pygame.sprite.Group()

    # Lade Bodenoberflächen
    ground_surface_original = pygame.image.load('graphics/bg.png').convert()

    # Skaliere die Bilder auf die neue Auflösung + Scrolling
    ground_surface = pygame.transform.scale(ground_surface_original, (settings.SCREEN_SIZE[0], settings.SCREEN_SIZE[1])) 
    ground_surface_width = ground_surface.get_width()
    tiles = math.ceil(settings.SCREEN_SIZE[0] / ground_surface_width) + 1
    scroll = 0

    # Lade alle stehenden Bilder für die Animation des Spielers
    player_stand_images = [pygame.image.load(f'graphics/player/player_stand_{i}.png').convert_alpha() for i in range(1, 6)]
    player_stand_index = 0

    # Elemente für den Einführungsbildschirm
    game_name = test_font.render('PirateBay', False, (111, 196, 169))
    game_name_rect = game_name.get_rect(center=(640, 160))
    game_message = test_font.render('Leertaste zum Starten und Springen, "S" oder "STRG" zum Ducken', False, (111, 196, 169))
    game_message_rect = game_message.get_rect(center=(640, 660))

    # Timer für die Generierung von Hindernissen
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

        scroll -= 4
        if scroll <= -ground_surface_width:
            scroll = 0

        if game_active:
            for i in range(2):  # Two images are enough to cover the screen without tearing
                screen.blit(ground_surface, (i * ground_surface_width + scroll, settings.SCREEN_SIZE[1] - ground_surface.get_height()))
            score = display_score(screen, start_time, test_font)
            
            player.draw(screen)
            player.update()

            obstacle_group.draw(screen)
            obstacle_group.update(score)

            game_active = collision_sprite(player, obstacle_group)
            
        else:
            screen.fill((94, 129, 162))
            player_stand_index += 0.1
            if player_stand_index >= len(player_stand_images):
                player_stand_index = 0
            player_stand = pygame.transform.rotozoom(player_stand_images[int(player_stand_index)], 0, 2)
            player_stand_rect = player_stand.get_rect(center=(640, 360))
            screen.blit(player_stand, player_stand_rect)
            screen.blit(game_name, game_name_rect)

            if score == 0:
                screen.blit(game_message, game_message_rect)
            else:
                score_message = test_font.render(f'Dein Punktestand: {score}', False, (111, 196, 169))
                score_message_rect = score_message.get_rect(center=(640, 660))
                screen.blit(score_message, score_message_rect)

        pygame.display.update()
        clock.tick(60)

if __name__ == '__main__':
    main()