"""
Configuration file for Snake RL Training
"""
import os

# Training Configuration
TRAINING_CONFIG = {
    'input_size': 25,  # Enhanced state representation size
    'hidden_size': 512,
    'output_size': 3,
    'use_dueling': True,
    'use_double_dqn': True,
    'use_prioritized_replay': True,
    'learning_rate': 0.0001,
    'gamma': 0.99,
    'epsilon_start': 0.9,
    'epsilon_min': 0.01,
    'epsilon_decay': 0.995,
    'batch_size': 64,
    'memory_size': 100000,
    'target_update_frequency': 10,
    'min_memory_size': 1000,
    'save_frequency': 50,  # Save model every N games
    'plot_frequency': 10,  # Update plot every N games
    'summary_frequency': 100,  # Print summary every N games
}

# Environment Configuration
ENV_CONFIG = {
    'window_width': 800,
    'window_height': 640,
    'grid_size': 20,
    'fps': 10,
    'header_height': 40,
}

# Reward Configuration
REWARD_CONFIG = {
    'base_survival_reward': 0.1,
    'food_reward_base': 20.0,
    'food_reward_length_multiplier': 0.8,
    'food_reward_speed_bonus_max': 5.0,
    'competitive_bonus_leading': 5.0,
    'competitive_bonus_tie': 2.0,
    'distance_reward_multiplier': 1.0,
    'distance_penalty_multiplier': 0.3,
    'proximity_penalty_close': 1.0,
    'proximity_penalty_near': 0.2,
    'proximity_reward_distance': 0.1,
    'efficiency_reward_multiplier': 0.1,
    'survival_time_bonus': 0.05,
    'survival_time_threshold': 100,
}

# UI Configuration
UI_CONFIG = {
    'show_controls': True,
    'default_training_speed': 1.0,
    'max_training_speed': 5.0,
    'min_training_speed': 0.1,
    'speed_increment': 0.5,
    'colors': {
        'background': (255, 255, 255),
        'grid': (200, 200, 200),
        'header': (240, 240, 240),
        'apple': (255, 69, 58),
        'snake1': (0, 122, 255),
        'snake2': (255, 45, 85),
    }
}

# Logging Configuration
LOGGING_CONFIG = {
    'log_dir': 'logs',
    'models_dir': 'models',
    'save_plots': True,
    'plot_dpi': 300,
    'log_level': 'INFO',
}

# Device Configuration
DEVICE_CONFIG = {
    'use_gpu': True,
    'force_cpu': False,
}

def get_device():
    """Get the appropriate device for training"""
    if DEVICE_CONFIG['force_cpu']:
        return 'cpu'
    
    if DEVICE_CONFIG['use_gpu']:
        try:
            import torch
            if torch.cuda.is_available():
                return 'cuda'
        except ImportError:
            pass
    
    return 'cpu'

def create_directories():
    """Create necessary directories"""
    os.makedirs(LOGGING_CONFIG['log_dir'], exist_ok=True)
    os.makedirs(LOGGING_CONFIG['models_dir'], exist_ok=True)

def get_config_summary():
    """Get a summary of the current configuration"""
    return {
        'training': TRAINING_CONFIG,
        'environment': ENV_CONFIG,
        'rewards': REWARD_CONFIG,
        'ui': UI_CONFIG,
        'logging': LOGGING_CONFIG,
        'device': get_device()
    }
