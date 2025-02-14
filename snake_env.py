import pygame
import numpy as np
from enum import Enum
import random
import os

# Center the pygame window
os.environ['SDL_VIDEO_CENTERED'] = '1'

# Configuration Constants
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 520   # 40 pixels reserved for header
GRID_SIZE = 20
HEADER_HEIGHT = 40
FPS = 10

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
        
        self.reset()

    def reset(self):
        # Original snake initialization
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
        
        if x < 0 or x >= self.width or y < 0 or y >= self.height or new_head in snake:
            return -10, True
        
        snake.insert(0, new_head)
        # Update onion-skin trail for snake
        if snake_num == 1:
            self.trail1.insert(0, new_head)
            self.trail1 = self.trail1[:10]
        else:
            self.trail2.insert(0, new_head)
            self.trail2 = self.trail2[:10]
        
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
        
        if snake_num == 1:
            self.snake1 = snake
        else:
            self.snake2 = snake
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

    def close(self):
        pygame.quit()

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
