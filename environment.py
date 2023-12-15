import pygame

def collision_sprite(player_group, obstacle_group):
        """
        Check for collisions between the player and obstacles.
        :param player_group: GroupSingle containing the player sprite.
        :param obstacle_group: Group containing obstacle sprites.
        :return: True if a collision is detected, False otherwise.
        """
        if pygame.sprite.spritecollideany(player_group.sprite, obstacle_group):
            return True
        return False

class RLEnvironment:
    def __init__(self, player, obstacles, screen_width):
        self.player = player
        self.obstacles = obstacles
        self.screen_width = screen_width

    def get_state(self):
        player_x = self.player.sprite.rect.x
        player_y = self.player.sprite.rect.y

        # Filter obstacles that are ahead of the player
        ahead_obstacles = [o for o in self.obstacles if o.rect.x > player_x]

        # Debugging: Print the number of obstacles ahead of the player

        if ahead_obstacles:
            nearest_obstacle = min(ahead_obstacles, key=lambda o: o.rect.x)
            distance = nearest_obstacle.rect.x - player_x
            obstacle_type = 1 if nearest_obstacle.type == 'fly' else 0
        else:
            distance = self.screen_width  # Max distance if no obstacle ahead
            obstacle_type = 0

        normalized_distance = distance / self.screen_width
        normalized_player_y = player_y / self.screen_width

        return [normalized_distance, obstacle_type, normalized_player_y]

    def step(self, action):
        # Perform the specified action (jump or duck)
        player_sprite = self.player.sprite  # Get the actual Player instance
        if action == 0:  # Jump
            player_sprite.jump()
        if action == 1:  # Duck
            player_sprite.duck()

        # Check for collisions and calculate reward and done flag
        done = self.is_game_over()
        reward = -10 if done else 1  # Negative reward for collision, positive for survival

        # Update player state based on action
        if action == 1:  # Duck
            self.player.sprite.ducking = True
            self.player.sprite.update()
        else:
            self.player.sprite.ducking = False
            self.player.sprite.update()

        # Check if ducking is allowed
        if not self.is_obstacle_close():
            if action == 1:  # If trying to duck without an obstacle nearby
                reward -= 0.2  # Penalize unnecessary ducking
                action = -1  # Reset action to no action

        # Return updated state, reward, and done flag
        return self.get_state(), reward, done

    def is_obstacle_close(self):
        # Define logic to determine if an obstacle is close enough to require action
        nearest_obstacle_sprites = [o.sprite for o in self.obstacles if hasattr(o, 'sprite')]
        if nearest_obstacle_sprites:
            nearest_obstacle = min(nearest_obstacle_sprites, key=lambda o: o.rect.x)
            distance = nearest_obstacle.rect.x - self.player.sprite.rect.x
            return distance < 384
        return False

    
    def is_game_over(self):
        # Implement the actual game over condition
        return collision_sprite(self.player, self.obstacles)

    def reset(self):
        # Reset the player's position
        self.player.sprite.reset_position()

        # Clear obstacles
        self.obstacles.empty()

        # Reset any other game state as needed
        return self.get_state()