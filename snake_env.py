import pygame
import numpy as np
from enum import Enum
import random
import os
from game_assets import GameAssets

# Center the pygame window
os.environ['SDL_VIDEO_CENTERED'] = '1'

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 640
GRID_SIZE = 20
HEADER_HEIGHT = 40
FPS = 10

BACKGROUND_COLOR = (255, 255, 255)
GRID_COLOR = (200, 200, 200)
HEADER_COLOR = (240, 240, 240)
APPLE_COLOR = (255, 69, 58)
SNAKE1_COLOR = (0, 122, 255)
SNAKE2_COLOR = (255, 45, 85)
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

class SnakeGameEnv:
    def __init__(self, width=WINDOW_WIDTH, height=WINDOW_HEIGHT, grid_size=GRID_SIZE):
        self.width = width
        self.height = height - HEADER_HEIGHT
        self.grid_size = grid_size
        
        pygame.init()
        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        pygame.display.set_caption('Snake\'s & The Golden Apple - Enhanced RL Training')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 28, bold=True)
        
        self.assets = GameAssets()
        self.apple_img = self.assets.get_texture('golden_apple')
        self.snake_head_img = self.assets.get_texture('snake_head')
        self.snake_body_img = self.assets.get_texture('snake_body')
        
        if self.snake_head_img:
            self.snake_head_img = pygame.transform.scale(self.snake_head_img, (self.grid_size, self.grid_size))
        if self.snake_body_img:
            self.snake_body_img = pygame.transform.scale(self.snake_body_img, (self.grid_size, self.grid_size))
        if self.apple_img:
            self.apple_img = pygame.transform.scale(self.apple_img, (self.grid_size, self.grid_size))
        
        self.particles = []
        self.particle_lifetime = 20
        
        self.paused = False
        self.show_controls = True
        self.training_speed = 1.0
        
        self.reset()

    def reset(self):
        self.snake1 = [(self.width // 4 // self.grid_size * self.grid_size, 
                        self.height // 2 // self.grid_size * self.grid_size)]
        self.snake2 = [(3 * self.width // 4 // self.grid_size * self.grid_size, 
                        self.height // 2 // self.grid_size * self.grid_size)]
        
        self.snake1_direction = Direction.RIGHT
        self.snake2_direction = Direction.LEFT
        
        self.food = self._place_food()
        self.score1 = 0
        self.score2 = 0
        self.frame_iteration = 0
        self.trail1 = []
        self.trail2 = []
        self.particles = []
        pygame.event.pump()
        return self._get_state()

    def _create_particles(self, pos, color):
        for _ in range(10):
            angle = random.uniform(0, 2 * np.pi)
            speed = random.uniform(2, 5)
            self.particles.append({
                'pos': list(pos),
                'vel': [speed * np.cos(angle), speed * np.sin(angle)],
                'lifetime': self.particle_lifetime,
                'color': color
            })

    def _update_particles(self):
        for particle in self.particles[:]:
            particle['pos'][0] += particle['vel'][0]
            particle['pos'][1] += particle['vel'][1]
            particle['lifetime'] -= 1
            if particle['lifetime'] <= 0:
                self.particles.remove(particle)

    def _draw_particles(self):
        for particle in self.particles:
            alpha = int(255 * (particle['lifetime'] / self.particle_lifetime))
            color = (*particle['color'], alpha)
            surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.circle(surf, color, (2, 2), 2)
            self.screen.blit(surf, (particle['pos'][0], particle['pos'][1] + HEADER_HEIGHT))

    def render(self, game_number):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.close()
                    return False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_h:
                    self.show_controls = not self.show_controls
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.training_speed = min(5.0, self.training_speed + 0.5)
                elif event.key == pygame.K_MINUS:
                    self.training_speed = max(0.1, self.training_speed - 0.5)
                elif event.key == pygame.K_r:
                    self.training_speed = 1.0
            elif event.type == pygame.VIDEORESIZE:
                width, height = event.size
                width = max(400, width)
                height = max(320 + HEADER_HEIGHT, height)
                width = (width // self.grid_size) * self.grid_size
                height = ((height - HEADER_HEIGHT) // self.grid_size) * self.grid_size + HEADER_HEIGHT
                self.width = width
                self.height = height - HEADER_HEIGHT
                self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
                self._adjust_positions_after_resize()
        # Fill background
        self.screen.fill(BACKGROUND_COLOR)
        
        # Draw grid pattern with proper alignment
        for x in range(0, self.width + self.grid_size, self.grid_size):
            for y in range(HEADER_HEIGHT, self.height + HEADER_HEIGHT + self.grid_size, self.grid_size):
                pygame.draw.rect(self.screen, GRID_COLOR, 
                               (x, y, self.grid_size, self.grid_size), 1)
        
        # Draw header with gradient
        header_surface = pygame.Surface((self.width, HEADER_HEIGHT))
        for i in range(HEADER_HEIGHT):
            pygame.draw.line(header_surface, 
                           (max(30, 33 - i//2), max(33, 36 - i//2), max(40, 43 - i//2)),
                           (0, i), (self.width, i))
        self.screen.blit(header_surface, (0, 0))
        
        self._draw_enhanced_ui(game_number)
        
        # Update and draw particles
        self._update_particles()
        self._draw_particles()
        
        # Draw enhanced apple with glow effect
        if self.apple_img:
            # Create glow effect
            glow_size = self.grid_size + 4
            glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (255, 215, 0, 100), 
                             (glow_size//2, glow_size//2), glow_size//2)
            self.screen.blit(glow_surf, 
                            (self.food[0] - 2, self.food[1] + HEADER_HEIGHT - 2))
            # Draw apple sprite
            self.screen.blit(self.apple_img, 
                            (self.food[0] + 2, self.food[1] + HEADER_HEIGHT + 2))
        else:
            # Fallback to enhanced circle
            glow_size = self.grid_size + 4
            glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (255, 215, 0, 100), 
                             (glow_size//2, glow_size//2), glow_size//2)
            self.screen.blit(glow_surf, 
                            (self.food[0] - 2, self.food[1] + HEADER_HEIGHT - 2))
            food_rect = pygame.Rect(self.food[0], self.food[1] + HEADER_HEIGHT, 
                                  self.grid_size - 2, self.grid_size - 2)
            pygame.draw.ellipse(self.screen, APPLE_COLOR, food_rect)
        
        # Draw enhanced trails with glowing effect for snake1
        for i, pos in enumerate(self.trail1):
            alpha = max(0, 255 - int(255 * (i / len(self.trail1))))
            size_reduction = int(i * 0.5)
            trail_surf = pygame.Surface((self.grid_size - size_reduction, 
                                       self.grid_size - size_reduction), pygame.SRCALPHA)
            trail_surf.fill((*SNAKE1_COLOR[:3], alpha))
            rect = pygame.Rect(pos[0] + size_reduction//2, 
                             pos[1] + HEADER_HEIGHT + size_reduction//2, 
                             self.grid_size - size_reduction, 
                             self.grid_size - size_reduction)
            self.screen.blit(trail_surf, rect.topleft)
        
        # Draw enhanced trails with glowing effect for snake2
        for i, pos in enumerate(self.trail2):
            alpha = max(0, 255 - int(255 * (i / len(self.trail2))))
            size_reduction = int(i * 0.5)
            trail_surf = pygame.Surface((self.grid_size - size_reduction, 
                                       self.grid_size - size_reduction), pygame.SRCALPHA)
            trail_surf.fill((*SNAKE2_COLOR[:3], alpha))
            rect = pygame.Rect(pos[0] + size_reduction//2, 
                             pos[1] + HEADER_HEIGHT + size_reduction//2, 
                             self.grid_size - size_reduction, 
                             self.grid_size - size_reduction)
            self.screen.blit(trail_surf, rect.topleft)
        
        # Draw enhanced snakes with enhanced effects
        self._draw_snake(self.snake1, SNAKE1_COLOR, is_snake1=True)
        self._draw_snake(self.snake2, SNAKE2_COLOR, is_snake1=False)
        
        pygame.display.flip()
        self.clock.tick(FPS)

    def _draw_snake(self, snake, base_color, is_snake1):
        n = len(snake)
        for idx, pt in enumerate(snake):
            rect = pygame.Rect(pt[0], pt[1] + HEADER_HEIGHT, 
                             self.grid_size - 2, self.grid_size - 2)
            if idx == 0:  # Head
                # Draw glow effect for head
                glow_size = self.grid_size + 4
                glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
                glow_color = (*base_color[:3], 100)
                pygame.draw.circle(glow_surf, glow_color, 
                                 (glow_size//2, glow_size//2), glow_size//2)
                self.screen.blit(glow_surf, 
                                (pt[0] - 2, pt[1] + HEADER_HEIGHT - 2))
                
                if self.snake_head_img:
                    head_img = self.snake_head_img.copy()
                    if not is_snake1:
                        # Color the head for snake2
                        head_img.fill(SNAKE2_COLOR, special_flags=pygame.BLEND_MULT)
                    self.screen.blit(head_img, rect.topleft)
                else:
                    pygame.draw.rect(self.screen, base_color, rect, border_radius=5)
            else:  # Body
                # Calculate gradient color
                factor = 1 - (idx / max(n - 1, 1)) * 0.7
                color = tuple(int(c * factor) for c in base_color[:3])
                pygame.draw.rect(self.screen, color, rect, border_radius=5)

    def _move_snake(self, snake_num, action):
        # Original turning logic using [straight, right, left]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        snake = self.snake1 if snake_num == 1 else self.snake2
        direction = self.snake1_direction if snake_num == 1 else self.snake2_direction
        idx = clock_wise.index(direction)
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = clock_wise[(idx + 1) % 4]
        else:  # [0, 0, 1]
            new_dir = clock_wise[(idx - 1) % 4]
        if snake_num == 1:
            self.snake1_direction = new_dir
        else:
            self.snake2_direction = new_dir
        
        x, y = snake[0]
        if new_dir == Direction.RIGHT:
            x += self.grid_size
        elif new_dir == Direction.LEFT:
            x -= self.grid_size
        elif new_dir == Direction.DOWN:
            y += self.grid_size
        elif new_dir == Direction.UP:
            y -= self.grid_size
        new_head = (x, y)
        
        # Check for collisions
        if x < 0 or x >= self.width or y < 0 or y >= self.height or new_head in snake:
            return -10, True
        
        snake.insert(0, new_head)
        # Update onion-skin trail
        if snake_num == 1:
            self.trail1.insert(0, new_head)
            self.trail1 = self.trail1[:10]
        else:
            self.trail2.insert(0, new_head)
            self.trail2 = self.trail2[:10]
        
        reward = 0
        
        survival_reward = 0.1 * (1.0 - self.frame_iteration / 2000.0)
        reward += survival_reward
        
        old_dist = abs(snake[1][0] - self.food[0]) + abs(snake[1][1] - self.food[1])
        new_dist = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])
        distance_improvement = old_dist - new_dist
        
        if distance_improvement > 0:
            reward += 1.0 * (distance_improvement / self.grid_size)
        else:
            reward -= 0.3 * (abs(distance_improvement) / self.grid_size)
        
        if new_head == self.food:
            base_food_reward = 20.0
            length_bonus = len(snake) * 0.8
            speed_bonus = max(0, 5.0 - self.frame_iteration / 100.0)
            
            other_snake = self.snake2 if snake_num == 1 else self.snake1
            my_score = self.score1 if snake_num == 1 else self.score2
            other_score = self.score2 if snake_num == 1 else self.score1
            competitive_bonus = 0
            if my_score > other_score:
                competitive_bonus = 5.0
            elif my_score == other_score:
                competitive_bonus = 2.0
            
            total_food_reward = base_food_reward + length_bonus + speed_bonus + competitive_bonus
            reward += total_food_reward
            
            if snake_num == 1:
                self.score1 += 1
                self._create_particles(new_head, SNAKE1_COLOR)
            else:
                self.score2 += 1
                self._create_particles(new_head, SNAKE2_COLOR)
            self.food = self._place_food()
        else:
            snake.pop()
        
        other_snake = self.snake2 if snake_num == 1 else self.snake1
        min_distance = float('inf')
        for part in other_snake:
            distance = abs(new_head[0] - part[0]) + abs(new_head[1] - part[1])
            min_distance = min(min_distance, distance)
        
        if min_distance <= self.grid_size:
            reward -= 1.0
        elif min_distance <= self.grid_size * 2:
            reward -= 0.2
        else:
            reward += 0.1
        
        my_food_distance = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])
        opponent_food_distance = abs(other_snake[0][0] - self.food[0]) + abs(other_snake[0][1] - self.food[1])
        
        if my_food_distance < opponent_food_distance:
            reward += 0.5
        elif my_food_distance > opponent_food_distance * 1.5:
            reward -= 0.3
        
        if len(snake) > 1:
            efficiency = len(snake) / max(self.frame_iteration, 1)
            reward += efficiency * 0.1
        
        if self.frame_iteration > 100:
            reward += 0.05
        
        if snake_num == 1:
            self.snake1 = snake
        else:
            self.snake2 = snake
        
        return reward, False

    def step(self, action1, action2):
        self.frame_iteration += 1
        
        # Move both snakes and get rewards/done status
        reward1, done1 = self._move_snake(1, action1)
        reward2, done2 = self._move_snake(2, action2)
        
        # Game is done if either snake collides
        done = done1 or done2
        
        # Get new states for both snakes
        state1, state2 = self._get_state()
        
        return (state1, state2), reward1, reward2, done

    def close(self):
        pygame.quit()

    def _get_state(self):
        # Get state for snake1
        state1 = self._get_snake_state(self.snake1, self.snake1_direction, self.snake2)
        # Get state for snake2
        state2 = self._get_snake_state(self.snake2, self.snake2_direction, self.snake1)
        return state1, state2

    def _get_snake_state(self, snake, direction, other_snake):
        head = snake[0]
        point_l = (head[0] - self.grid_size, head[1])
        point_r = (head[0] + self.grid_size, head[1])
        point_u = (head[0], head[1] - self.grid_size)
        point_d = (head[0], head[1] + self.grid_size)

        dir_l = direction == Direction.LEFT
        dir_r = direction == Direction.RIGHT
        dir_u = direction == Direction.UP
        dir_d = direction == Direction.DOWN

        # Check for danger in each direction
        danger_straight = (dir_r and self._is_collision(point_r, snake, other_snake)) or \
                          (dir_l and self._is_collision(point_l, snake, other_snake)) or \
                          (dir_u and self._is_collision(point_u, snake, other_snake)) or \
                          (dir_d and self._is_collision(point_d, snake, other_snake))

        danger_right = (dir_u and self._is_collision(point_r, snake, other_snake)) or \
                       (dir_d and self._is_collision(point_l, snake, other_snake)) or \
                       (dir_l and self._is_collision(point_u, snake, other_snake)) or \
                       (dir_r and self._is_collision(point_d, snake, other_snake))

        danger_left = (dir_u and self._is_collision(point_l, snake, other_snake)) or \
                      (dir_d and self._is_collision(point_r, snake, other_snake)) or \
                      (dir_l and self._is_collision(point_d, snake, other_snake)) or \
                      (dir_r and self._is_collision(point_u, snake, other_snake))

        state = [
            danger_straight,
            danger_right,
            danger_left,
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            self.food[0] < head[0],
            self.food[0] > head[0],
            self.food[1] < head[1],
            self.food[1] > head[1],
            abs(self.food[0] - head[0]) / self.width,
            abs(self.food[1] - head[1]) / self.height,
            head[0] / self.width,
            (self.width - head[0]) / self.width,
            head[1] / self.height,
            (self.height - head[1]) / self.height,
            len(snake) / 50.0,
            self._get_opponent_relative_position(head, other_snake),
            self._get_opponent_distance(head, other_snake),
            self._is_opponent_closer_to_food(head, other_snake),
            *self._get_recent_actions(snake),
            self.frame_iteration / 1000.0,
        ]
        return np.array(state, dtype=float)

    def _get_opponent_relative_position(self, head, other_snake):
        if not other_snake:
            return 0
        opponent_head = other_snake[0]
        dx = opponent_head[0] - head[0]
        dy = opponent_head[1] - head[1]
        
        if abs(dx) > abs(dy):
            return 1 if dx > 0 else 2
        else:
            return 3 if dy > 0 else 4

    def _get_opponent_distance(self, head, other_snake):
        if not other_snake:
            return 1.0
        opponent_head = other_snake[0]
        distance = abs(head[0] - opponent_head[0]) + abs(head[1] - opponent_head[1])
        return min(distance / (self.width + self.height), 1.0)

    def _is_opponent_closer_to_food(self, head, other_snake):
        if not other_snake:
            return 0
        my_distance = abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])
        opponent_distance = abs(other_snake[0][0] - self.food[0]) + abs(other_snake[0][1] - self.food[1])
        return 1 if opponent_distance < my_distance else 0

    def _get_recent_actions(self, snake):
        return [0, 0, 0]

    def _draw_enhanced_ui(self, game_number):
        score_font = pygame.font.SysFont("Arial", 20, bold=True)
        small_font = pygame.font.SysFont("Arial", 14)
        
        # Score panels with modern design
        panel_height = 35
        score1_rect = pygame.Rect(15, 8, 200, panel_height)
        score2_rect = pygame.Rect(230, 8, 200, panel_height)
        info_rect = pygame.Rect(self.width - 200, 8, 180, panel_height)
        
        # Gradient backgrounds
        self._draw_gradient_rect(score1_rect, SNAKE1_COLOR, (0, 0, 0, 50))
        self._draw_gradient_rect(score2_rect, SNAKE2_COLOR, (0, 0, 0, 50))
        self._draw_gradient_rect(info_rect, (40, 40, 40), (0, 0, 0, 50))
        
        # Score text with shadows
        text1 = score_font.render(f'Bluessy: {self.score1}', True, (255, 255, 255))
        text2 = score_font.render(f'Redish: {self.score2}', True, (255, 255, 255))
        text_game = small_font.render(f'Game: {game_number}', True, (200, 200, 200))
        
        self.screen.blit(text1, (25, 15))
        self.screen.blit(text2, (240, 15))
        self.screen.blit(text_game, (self.width - 190, 20))
        
        # Performance indicators
        if hasattr(self, 'frame_iteration'):
            steps_text = small_font.render(f'Steps: {self.frame_iteration}', True, (180, 180, 180))
            self.screen.blit(steps_text, (self.width - 350, 20))
        
        speed_text = small_font.render(f'Speed: {self.training_speed:.1f}x', True, (180, 180, 180))
        self.screen.blit(speed_text, (self.width - 350, 35))
        
        # Pause overlay
        if self.paused:
            self._draw_pause_overlay()
        
        # Control overlay
        if self.show_controls:
            self._draw_control_overlay()

    def _draw_gradient_rect(self, rect, color1, color2):
        for i in range(rect.height):
            ratio = i / rect.height
            r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
            g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
            b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
            a = int(color1[3] * (1 - ratio) + color2[3] * ratio) if len(color1) > 3 else 255
            pygame.draw.line(self.screen, (r, g, b, a), (rect.x, rect.y + i), (rect.x + rect.width, rect.y + i))

    def _draw_pause_overlay(self):
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, HEADER_HEIGHT))
        
        pause_font = pygame.font.SysFont("Arial", 64, bold=True)
        pause_text = pause_font.render("PAUSED", True, (255, 100, 100))
        pause_rect = pause_text.get_rect(center=(self.width // 2, self.height // 2 + HEADER_HEIGHT))
        
        # Glow effect
        for i in range(5):
            glow_color = (255, 100, 100, 50 - i * 10)
            glow_text = pause_font.render("PAUSED", True, glow_color)
            glow_rect = glow_text.get_rect(center=(pause_rect.centerx + i, pause_rect.centery + i))
            self.screen.blit(glow_text, glow_rect)
        
        self.screen.blit(pause_text, pause_rect)

    def _draw_control_overlay(self):
        overlay_font = pygame.font.SysFont("Arial", 12)
        controls = [
            "CONTROLS",
            "SPACE - Pause/Resume",
            "H - Toggle Help",
            "+/- - Speed Control",
            "R - Reset Speed",
            "ESC - Exit"
        ]
        
        overlay_surface = pygame.Surface((180, 100), pygame.SRCALPHA)
        overlay_surface.fill((0, 0, 0, 180))
        
        for i, control in enumerate(controls):
            color = (255, 255, 0) if i == 0 else (255, 255, 255)
            text = overlay_font.render(control, True, color)
            overlay_surface.blit(text, (10, 5 + i * 15))
        
        self.screen.blit(overlay_surface, (self.width - 190, HEADER_HEIGHT + 10))

    def _is_collision(self, point, snake, other_snake):
        if point[0] < 0 or point[0] >= self.width or \
           point[1] < 0 or point[1] >= self.height:
            return True
        if point in snake[1:] or point in other_snake:
            return True
        return False

    def _place_food(self):
        while True:
            x = random.randint(0, (self.width - self.grid_size) // self.grid_size) * self.grid_size
            y = random.randint(0, (self.height - self.grid_size) // self.grid_size) * self.grid_size
            food_pos = (x, y)
            if food_pos not in self.snake1 and food_pos not in self.snake2:
                return food_pos

    def _adjust_positions_after_resize(self):
        for i, pos in enumerate(self.snake1):
            x, y = pos
            x = min(max(0, x), self.width - self.grid_size)
            y = min(max(0, y), self.height - self.grid_size)
            self.snake1[i] = (x, y)
        
        for i, pos in enumerate(self.snake2):
            x, y = pos
            x = min(max(0, x), self.width - self.grid_size)
            y = min(max(0, y), self.height - self.grid_size)
            self.snake2[i] = (x, y)
        
        x, y = self.food
        x = min(max(0, x), self.width - self.grid_size)
        y = min(max(0, y), self.height - self.grid_size)
        self.food = (x, y)

if __name__ == "__main__":
    env = SnakeGameEnv()
    game_number = 1
    done = False
    action1 = np.array([1, 0, 0])
    action2 = np.array([1, 0, 0])
    while not done:
        state, reward1, reward2, done = env.step(action1, action2)
        env.render(game_number)
    env.close()
