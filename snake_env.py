import pygame
import numpy as np
from enum import Enum
import random
import os
<<<<<<< HEAD
from game_assets import GameAssets
=======
>>>>>>> e884ccb65626f5b0ab629333013f9ec33c7268db

# Center the pygame window
os.environ['SDL_VIDEO_CENTERED'] = '1'

# Configuration Constants
<<<<<<< HEAD
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 640   # 40 pixels reserved for header
GRID_SIZE = 20
HEADER_HEIGHT = 40
FPS = 10  # Decreased for slower movement

# Colors
BACKGROUND_COLOR = (255, 255, 255)  # White background
GRID_COLOR = (200, 200, 200)  # Light gray for grid
HEADER_COLOR = (240, 240, 240)  # Slightly darker than background
APPLE_COLOR = (255, 69, 58)  # Vibrant red for apple
SNAKE1_COLOR = (0, 122, 255)  # Bright blue
SNAKE2_COLOR = (255, 45, 85)  # Bright red
=======
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 520   # 40 pixels reserved for header
GRID_SIZE = 20
HEADER_HEIGHT = 40
FPS = 10
>>>>>>> e884ccb65626f5b0ab629333013f9ec33c7268db

# --- Direction Enumeration ---
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# --- Main Game Environment ---
class SnakeGameEnv:
    def __init__(self, width=WINDOW_WIDTH, height=WINDOW_HEIGHT, grid_size=GRID_SIZE):
        self.width = width
        self.height = height - HEADER_HEIGHT  # Playable area
        self.grid_size = grid_size
        
        pygame.init()
<<<<<<< HEAD
        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        pygame.display.set_caption('Snake\'s & The Golden Apple')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 28, bold=True)
        
        # Initialize game assets
        self.assets = GameAssets()
        self.apple_img = self.assets.get_texture('golden_apple')
        self.snake_head_img = self.assets.get_texture('snake_head')
        self.snake_body_img = self.assets.get_texture('snake_body')
        
        # Initialize particle system for visual effects
        self.particles = []
        self.particle_lifetime = 20
=======
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption('Competing Snakes RL')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Calibri", 28)
        
        # Attempt to load snake images (if not found, use drawing fallback)
        try:
            self.snake_head_img = pygame.image.load("snake_head.png").convert_alpha()
            self.snake_head_img = pygame.transform.scale(self.snake_head_img, (grid_size, grid_size))
        except Exception:
            self.snake_head_img = None
        try:
            self.snake_body_img = pygame.image.load("snake_body.png").convert_alpha()
            self.snake_body_img = pygame.transform.scale(self.snake_body_img, (grid_size, grid_size))
        except Exception:
            self.snake_body_img = None
>>>>>>> e884ccb65626f5b0ab629333013f9ec33c7268db
        
        self.reset()

    def reset(self):
<<<<<<< HEAD
        # Initialize snakes at opposite sides
=======
        # Original snake initialization
>>>>>>> e884ccb65626f5b0ab629333013f9ec33c7268db
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
<<<<<<< HEAD
        self.trail1 = []  # Trail for visual effect
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
        # Process window resize events with grid alignment
        for event in pygame.event.get((pygame.VIDEORESIZE, pygame.QUIT, pygame.KEYDOWN)):
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.close()
                return False
            elif event.type == pygame.VIDEORESIZE:
                width, height = event.size
                # Ensure minimum window size and grid alignment
                width = max(400, width)
                height = max(320 + HEADER_HEIGHT, height)
                # Align dimensions to grid size
                width = (width // self.grid_size) * self.grid_size
                height = ((height - HEADER_HEIGHT) // self.grid_size) * self.grid_size + HEADER_HEIGHT
                # Update screen dimensions
                self.width = width
                self.height = height - HEADER_HEIGHT
                self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
                # Adjust positions to maintain grid alignment
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
        
        # Display improved scoreboard with snake names
        text1 = self.font.render(f'Bluessy: {self.score1}', True, SNAKE1_COLOR)
        text2 = self.font.render(f'Redish: {self.score2}', True, SNAKE2_COLOR)
        text_game = self.font.render(f'Game: {game_number}', True, (200, 200, 200))
        self.screen.blit(text1, (10, 5))
        self.screen.blit(text2, (200, 5))
        self.screen.blit(text_game, (self.width - 150, 5))
        
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
        
        # Draw snakes with enhanced effects
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
                    head_img = self.snake_head_img
                    if not is_snake1:
                        head_img = pygame.transform.flip(head_img, True, False)
                    self.screen.blit(head_img, rect.topleft)
                else:
                    pygame.draw.rect(self.screen, base_color, rect, border_radius=5)
            else:  # Body
                # Calculate gradient color
                factor = 1 - (idx / max(n - 1, 1)) * 0.7
                color = tuple(int(c * factor) for c in base_color[:3])
                pygame.draw.rect(self.screen, color, rect, border_radius=5)
=======
        # Trails for onion-skin effect (lists of previous head positions)
        self.trail1 = []
        self.trail2 = []
        pygame.event.pump()
        return self._get_state()

    def _place_food(self):
        while True:
            x = random.randint(0, (self.width - self.grid_size) // self.grid_size) * self.grid_size
            y = random.randint(0, (self.height - self.grid_size) // self.grid_size) * self.grid_size
            food_pos = (x, y)
            if food_pos not in self.snake1 and food_pos not in self.snake2:
                return food_pos

    def _get_state(self):
        # Generate state representations (unchanged from original)
        state1 = self._get_snake_state(self.snake1, self.snake1_direction, self.snake2)
        state2 = self._get_snake_state(self.snake2, self.snake2_direction, self.snake1)
        return state1, state2

    def _get_snake_state(self, snake, direction, other_snake):
        head = snake[0]
        point_l = point_r = point_u = point_d = head
        if direction == Direction.RIGHT:
            point_r = (head[0] + self.grid_size, head[1])
            point_u = (head[0], head[1] - self.grid_size)
            point_d = (head[0], head[1] + self.grid_size)
        elif direction == Direction.LEFT:
            point_l = (head[0] - self.grid_size, head[1])
            point_u = (head[0], head[1] - self.grid_size)
            point_d = (head[0], head[1] + self.grid_size)
        elif direction == Direction.UP:
            point_u = (head[0], head[1] - self.grid_size)
            point_l = (head[0] - self.grid_size, head[1])
            point_r = (head[0] + self.grid_size, head[1])
        elif direction == Direction.DOWN:
            point_d = (head[0], head[1] + self.grid_size)
            point_l = (head[0] - self.grid_size, head[1])
            point_r = (head[0] + self.grid_size, head[1])

        dir_l = direction == Direction.LEFT
        dir_r = direction == Direction.RIGHT
        dir_u = direction == Direction.UP
        dir_d = direction == Direction.DOWN
        food = self.food
        
        state = [
            (dir_r and self._is_collision(point_r)) or 
            (dir_l and self._is_collision(point_l)) or 
            (dir_u and self._is_collision(point_u)) or 
            (dir_d and self._is_collision(point_d)),

            (dir_u and self._is_collision(point_r)) or 
            (dir_d and self._is_collision(point_l)) or 
            (dir_l and self._is_collision(point_u)) or 
            (dir_r and self._is_collision(point_d)),

            (dir_d and self._is_collision(point_r)) or 
            (dir_u and self._is_collision(point_l)) or 
            (dir_r and self._is_collision(point_u)) or 
            (dir_l and self._is_collision(point_d)),

            dir_l, dir_r, dir_u, dir_d,
            food[0] < head[0], food[0] > head[0],
            food[1] < head[1], food[1] > head[1]
        ]
        return np.array(state, dtype=int)

    def _is_collision(self, pt):
        if pt[0] >= self.width or pt[0] < 0 or pt[1] >= self.height or pt[1] < 0:
            return True
        if pt in self.snake1[1:] or pt in self.snake2:
            return True
        return False

    def step(self, action1, action2):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None, 0, 0, True
        
        reward1, done1 = self._move_snake(1, action1)
        reward2, done2 = self._move_snake(2, action2)
        done = done1 or done2
        if done:
            if done1 and done2:
                reward1 = reward2 = -10
            elif done1:
                reward1, reward2 = -10, 10
            else:
                reward1, reward2 = 10, -10
        
        return self._get_state(), reward1, reward2, done
>>>>>>> e884ccb65626f5b0ab629333013f9ec33c7268db

    def _move_snake(self, snake_num, action):
        # Original turning logic using [straight, right, left]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        snake = self.snake1 if snake_num == 1 else self.snake2
        direction = self.snake1_direction if snake_num == 1 else self.snake2_direction
        idx = clock_wise.index(direction)
<<<<<<< HEAD
        
=======
>>>>>>> e884ccb65626f5b0ab629333013f9ec33c7268db
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = clock_wise[(idx + 1) % 4]
        else:  # [0, 0, 1]
            new_dir = clock_wise[(idx - 1) % 4]
<<<<<<< HEAD
            
=======
>>>>>>> e884ccb65626f5b0ab629333013f9ec33c7268db
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
        
<<<<<<< HEAD
        # Check for collisions
=======
>>>>>>> e884ccb65626f5b0ab629333013f9ec33c7268db
        if x < 0 or x >= self.width or y < 0 or y >= self.height or new_head in snake:
            return -10, True
        
        snake.insert(0, new_head)
<<<<<<< HEAD
        # Update onion-skin trail
=======
        # Update onion-skin trail for snake
>>>>>>> e884ccb65626f5b0ab629333013f9ec33c7268db
        if snake_num == 1:
            self.trail1.insert(0, new_head)
            self.trail1 = self.trail1[:10]
        else:
            self.trail2.insert(0, new_head)
            self.trail2 = self.trail2[:10]
        
<<<<<<< HEAD
        # Calculate rewards
        reward = 0
        # Base reward for staying alive
        reward += 0.1
        
        # Distance-based reward
        old_dist = abs(snake[1][0] - self.food[0]) + abs(snake[1][1] - self.food[1])
        new_dist = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])
        if new_dist < old_dist:
            reward += 0.5  # Reward for moving closer to food
        else:
            reward -= 0.2  # Small penalty for moving away
        
        # Food collision reward
        if new_head == self.food:
            reward += 15  # Increased reward for eating food
            if snake_num == 1:
                self.score1 += 1
                self._create_particles(new_head, SNAKE1_COLOR)
            else:
                self.score2 += 1
                self._create_particles(new_head, SNAKE2_COLOR)
            self.food = self._place_food()
            # Bonus reward for longer snake
            reward += len(snake) * 0.5
        else:
            snake.pop()
        
        # Penalty for getting too close to other snake
        other_snake = self.snake2 if snake_num == 1 else self.snake1
        for part in other_snake:
            if abs(new_head[0] - part[0]) + abs(new_head[1] - part[1]) <= self.grid_size:
                reward -= 0.3
        
=======
        reward = 0
        if new_head == self.food:
            reward = 10
            if snake_num == 1:
                self.score1 += 1
            else:
                self.score2 += 1
            self.food = self._place_food()
        else:
            snake.pop()
        
>>>>>>> e884ccb65626f5b0ab629333013f9ec33c7268db
        if snake_num == 1:
            self.snake1 = snake
        else:
            self.snake2 = snake
<<<<<<< HEAD
        
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
=======
        return reward, False

    def render(self, game_number):
        # Draw header with simple gradient
        for i in range(HEADER_HEIGHT):
            shade = 240 - int(40 * (i / HEADER_HEIGHT))
            pygame.draw.rect(self.screen, (shade, shade, shade), (0, i, self.width, 1))
        # Display improved scoreboard with snake names
        text1 = self.font.render('Bluessy: ' + str(self.score1), True, (0, 0, 255))
        text2 = self.font.render('Redish: ' + str(self.score2), True, (255, 0, 0))
        text_game = self.font.render(f'Game: {game_number}', True, (0, 0, 0))
        self.screen.blit(text1, (10, 5))
        self.screen.blit(text2, (200, 5))
        self.screen.blit(text_game, (self.width - 150, 5))
        
        # Fill playable area with white background
        arena_rect = pygame.Rect(0, HEADER_HEIGHT, self.width, self.height)
        self.screen.fill((255, 255, 255), arena_rect)
        
        # Draw food as an ellipse with Golden color
        food_rect = pygame.Rect(self.food[0], self.food[1] + HEADER_HEIGHT, self.grid_size - 2, self.grid_size - 2)
        pygame.draw.ellipse(self.screen, (255, 215, 0), food_rect)
        
        # Draw onion-skin trail for snake1
        for i, pos in enumerate(self.trail1):
            alpha = max(0, 255 - int(255 * (i / len(self.trail1))))
            trail_surf = pygame.Surface((self.grid_size - 2, self.grid_size - 2), pygame.SRCALPHA)
            trail_surf.fill((0, 0, 255, alpha))
            rect = pygame.Rect(pos[0], pos[1] + HEADER_HEIGHT, self.grid_size - 2, self.grid_size - 2)
            self.screen.blit(trail_surf, rect.topleft)
        
        # Draw onion-skin trail for snake2
        for i, pos in enumerate(self.trail2):
            alpha = max(0, 255 - int(255 * (i / len(self.trail2))))
            trail_surf = pygame.Surface((self.grid_size - 2, self.grid_size - 2), pygame.SRCALPHA)
            trail_surf.fill((255, 0, 0, alpha))
            rect = pygame.Rect(pos[0], pos[1] + HEADER_HEIGHT, self.grid_size - 2, self.grid_size - 2)
            self.screen.blit(trail_surf, rect.topleft)
        
        # Draw snake1 (Bluessy) with gradient body: head in full blue and body cells darkening gradually
        n1 = len(self.snake1)
        for idx, pt in enumerate(self.snake1):
            rect = pygame.Rect(pt[0], pt[1] + HEADER_HEIGHT, self.grid_size - 2, self.grid_size - 2)
            if idx == 0:
                if self.snake_head_img is not None:
                    self.screen.blit(self.snake_head_img, rect.topleft)
                else:
                    pygame.draw.rect(self.screen, (0, 0, 255), rect, border_radius=5)
            else:
                darkening = int((idx / max(n1 - 1, 1)) * 100)
                blue_val = max(0, 255 - darkening)
                cell_color = (0, 0, blue_val)
                pygame.draw.rect(self.screen, cell_color, rect, border_radius=5)
        
        # Draw snake2 (Redish) with gradient body: head in full red and body cells darkening gradually
        n2 = len(self.snake2)
        for idx, pt in enumerate(self.snake2):
            rect = pygame.Rect(pt[0], pt[1] + HEADER_HEIGHT, self.grid_size - 2, self.grid_size - 2)
            if idx == 0:
                if self.snake_head_img is not None:
                    head_img = pygame.transform.flip(self.snake_head_img, True, False)
                    self.screen.blit(head_img, rect.topleft)
                else:
                    pygame.draw.rect(self.screen, (255, 0, 0), rect, border_radius=5)
            else:
                darkening = int((idx / max(n2 - 1, 1)) * 100)
                red_val = max(0, 255 - darkening)
                cell_color = (red_val, 0, 0)
                pygame.draw.rect(self.screen, cell_color, rect, border_radius=5)
        
        pygame.display.flip()
        self.clock.tick(FPS)
>>>>>>> e884ccb65626f5b0ab629333013f9ec33c7268db

    def close(self):
        pygame.quit()

<<<<<<< HEAD
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

        # Create state array
        state = [
            danger_straight,
            danger_right,
            danger_left,
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            self.food[0] < head[0],  # food left
            self.food[0] > head[0],  # food right
            self.food[1] < head[1],  # food up
            self.food[1] > head[1]   # food down
        ]
        return np.array(state, dtype=int)

    def _is_collision(self, point, snake, other_snake):
        # Check if point is out of bounds
        if point[0] < 0 or point[0] >= self.width or \
           point[1] < 0 or point[1] >= self.height:
            return True
        # Check if point collides with snake body or other snake
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
        # Adjust snake positions
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
        
        # Adjust food position
        x, y = self.food
        x = min(max(0, x), self.width - self.grid_size)
        y = min(max(0, y), self.height - self.grid_size)
        self.food = (x, y)

=======
>>>>>>> e884ccb65626f5b0ab629333013f9ec33c7268db
# --- Main Loop for Testing ---
if __name__ == "__main__":
    env = SnakeGameEnv()
    game_number = 1
    done = False
    # Dummy actions: [1, 0, 0] (no change) for both snakes.
    action1 = np.array([1, 0, 0])
    action2 = np.array([1, 0, 0])
    while not done:
        state, reward1, reward2, done = env.step(action1, action2)
        env.render(game_number)
    env.close()
