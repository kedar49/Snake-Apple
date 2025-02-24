import pygame
import os

# Asset loading and management
class GameAssets:
    def __init__(self):
        self.assets_dir = os.path.join(os.path.dirname(__file__), 'assets')
        self.textures = {}
        self.sounds = {}
        self.power_ups = {}
        
        # Initialize textures
        self._load_textures()
        # Initialize sounds
        self._load_sounds()
        # Initialize power-ups
        self._init_power_ups()
    
    def _load_textures(self):
        # Load background texture
        try:
            bg_texture = pygame.image.load(os.path.join(self.assets_dir, 'background.png')).convert_alpha()
            self.textures['background'] = bg_texture
        except:
            self.textures['background'] = None

        # Load snake textures
        try:
            snake_body = pygame.image.load(os.path.join(self.assets_dir, 'snake_body.png')).convert_alpha()
            self.textures['snake_body'] = snake_body
        except:
            self.textures['snake_body'] = None

    def _load_sounds(self):
        # Initialize pygame mixer
        pygame.mixer.init()
        
        # Load sound effects
        try:
            self.sounds['eat'] = pygame.mixer.Sound(os.path.join(self.assets_dir, 'eat.wav'))
            self.sounds['collision'] = pygame.mixer.Sound(os.path.join(self.assets_dir, 'collision.wav'))
            self.sounds['power_up'] = pygame.mixer.Sound(os.path.join(self.assets_dir, 'power_up.wav'))
        except:
            print("Warning: Sound files not found")

    def _init_power_ups(self):
        self.power_ups = {
            'speed_boost': {
                'duration': 5,  # seconds
                'effect': lambda snake: setattr(snake, 'speed', snake.speed * 1.5),
                'color': (255, 215, 0)  # Gold color
            },
            'score_multiplier': {
                'duration': 10,  # seconds
                'effect': lambda snake: setattr(snake, 'score_multiplier', 2),
                'color': (147, 112, 219)  # Purple color
            },
            'invincibility': {
                'duration': 3,  # seconds
                'effect': lambda snake: setattr(snake, 'invincible', True),
                'color': (255, 255, 255)  # White color
            }
        }

    def get_texture(self, name):
        return self.textures.get(name)

    def play_sound(self, name):
        if name in self.sounds:
            self.sounds[name].play()

    def get_power_up(self, name):
        return self.power_ups.get(name)