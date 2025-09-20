# Snake's & The Golden Apple - Advanced RL System

A competitive multi-agent reinforcement learning environment implementing state-of-the-art Deep Q-Learning algorithms.

![Environment](https://img.shields.io/badge/Environment-Multi--Agent%20RL-blue)
![Algorithm](https://img.shields.io/badge/Algorithm-Double%20DQN-green)
![Framework](https://img.shields.io/badge/Framework-PyTorch-red)

## Game Environment

<svg width="600" height="300" viewBox="0 0 600 300" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
      <path d="M 20 0 L 0 0 0 20" fill="none" stroke="#e0e0e0" stroke-width="1"/>
    </pattern>
  </defs>
  
  <!-- Background -->
  <rect width="600" height="300" fill="#f8f9fa"/>
  <rect width="600" height="300" fill="url(#grid)"/>
  
  <!-- Snake 1 (Blue) -->
  <circle cx="120" cy="150" r="12" fill="#0066cc"/>
  <circle cx="100" cy="150" r="10" fill="#0066cc" opacity="0.8"/>
  <circle cx="80" cy="150" r="8" fill="#0066cc" opacity="0.6"/>
  <circle cx="60" cy="150" r="6" fill="#0066cc" opacity="0.4"/>
  
  <!-- Snake 2 (Red) -->
  <circle cx="480" cy="150" r="12" fill="#cc0000"/>
  <circle cx="500" cy="150" r="10" fill="#cc0000" opacity="0.8"/>
  <circle cx="520" cy="150" r="8" fill="#cc0000" opacity="0.6"/>
  <circle cx="540" cy="150" r="6" fill="#cc0000" opacity="0.4"/>
  
  <!-- Golden Apple -->
  <circle cx="300" cy="150" r="15" fill="#ffd700" stroke="#ff8c00" stroke-width="2"/>
  <circle cx="300" cy="150" r="8" fill="#ffff00" opacity="0.6"/>
  
  <!-- Labels -->
  <text x="120" y="130" text-anchor="middle" font-family="Arial" font-size="12" font-weight="bold" fill="#0066cc">Bluessy</text>
  <text x="480" y="130" text-anchor="middle" font-family="Arial" font-size="12" font-weight="bold" fill="#cc0000">Redish</text>
  <text x="300" y="130" text-anchor="middle" font-family="Arial" font-size="10" font-weight="bold" fill="#ff8c00">Golden Apple</text>
  
  <!-- Title -->
  <text x="300" y="30" text-anchor="middle" font-family="Arial" font-size="18" font-weight="bold" fill="#333">Competitive Snake Environment</text>
</svg>

### Game in Action

![Snake RL Training](Snake_RL.gif)

*Real-time training visualization showing competitive gameplay between AI agents*

## Technical Architecture

### Deep Q-Network (DQN) Implementation
- **Input**: 25-dimensional state vector
- **Architecture**: 256 → 128 → 4 neurons
- **Activation**: ReLU (hidden), Linear (output)
- **Optimizer**: Adam (lr=0.001)

### Double DQN Algorithm
```python
# Reduces overestimation bias
target_q_values = target_network(next_states).gather(1, next_actions.unsqueeze(1))
targets = rewards + (gamma * target_q_values * (1 - dones))
```

### Dueling DQN Architecture
```python
# Separate value and advantage streams
value_stream = self.value_stream(features)
advantage_stream = self.advantage_stream(features)
q_values = value_stream + (advantage_stream - advantage_stream.mean(dim=1, keepdim=True))
```

### Prioritized Experience Replay
- **Buffer Size**: 50,000 experiences
- **Alpha**: 0.6 (prioritization strength)
- **Beta**: 0.4 (importance sampling)
- **Sampling**: TD-error based priority

## State Representation (25D)

| Index | Feature | Description | Range |
|-------|---------|-------------|-------|
| 0-3 | Snake 1 direction | One-hot encoding | [0,1] |
| 4-7 | Snake 2 direction | One-hot encoding | [0,1] |
| 8-11 | Food direction | One-hot encoding | [0,1] |
| 12-15 | Wall proximity | One-hot encoding | [0,1] |
| 16-19 | Snake 1 body proximity | One-hot encoding | [0,1] |
| 20-23 | Snake 2 body proximity | One-hot encoding | [0,1] |
| 24 | Snake 1 length | Normalized | [0,1] |

## Reward Engineering

### Primary Rewards
- **Food Consumption**: `10 + length_bonus + speed_bonus + competitive_bonus`
- **Survival**: `0.1 * (1 - frame_iteration/1000)`
- **Collision**: `-10`

### Advanced Rewards
- **Proximity to Food**: `exp(-distance_to_food/10) * 2`
- **Competitive Advantage**: `2 if closer_to_food_than_opponent else 0`
- **Efficiency**: `length * 0.1 / max(frame_iteration, 1)`

## Hyperparameters

```python
LEARNING_RATE = 0.001
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 64
TARGET_UPDATE_FREQUENCY = 1000
MEMORY_CAPACITY = 50000
```

## Installation

```bash
# Clone repository
git clone <repository-url>
cd Snake-Apple

# Install dependencies
pip install torch pygame numpy matplotlib

# Run training
python train.py
```

## Usage

### Basic Training
```bash
python train.py
```

### Testing
```bash
python test_enhanced_snake.py
```

### Controls
- **SPACE**: Pause/Resume
- **H**: Toggle help
- **+/-**: Speed control
- **R**: Reset speed
- **ESC**: Exit

## File Structure

```
Snake-Apple/
├── model.py              # DQN architecture
├── snake_env.py          # Game environment
├── train.py              # Training loop
├── logger.py             # Metrics tracking
├── config.py             # Configuration
├── game_assets.py        # Asset loading
├── test_enhanced_snake.py # Test suite
├── requirements.txt      # Dependencies
├── models/               # Checkpoints
└── logs/                 # Training logs
```

## Performance Metrics

### Training Dashboard
- Loss tracking with gradient clipping
- Q-value monitoring per agent
- Epsilon decay visualization
- Win rate analysis
- Game length distribution

### Expected Performance
- **Convergence**: 2000-5000 games
- **Peak Score**: 15-25 average
- **Win Rate**: 60-70%
- **Training Time**: 2-4 hours (CPU), 30-60 min (GPU)

## Advanced Features

### Multi-Agent Competition
- Simultaneous learning
- Competitive dynamics
- Adaptive opponents

### Model Persistence
- Checkpoint system
- Resume training
- Version control

### Real-time Monitoring
- Live metrics
- Performance plots
- Interactive controls

## Troubleshooting

### Common Issues
1. **CUDA OOM**: Reduce batch size
2. **Slow Training**: Enable GPU
3. **Poor Performance**: Tune hyperparameters
4. **Memory Leaks**: Check buffer size

### Optimization
- GPU acceleration (3-5x speedup)
- Batch processing
- Memory management
- Parallel training

## Research Applications

- Multi-agent RL dynamics
- Algorithm comparison
- Strategic gameplay analysis
- Competitive learning

## License

MIT License

## Citation

```bibtex
@software{snake_rl_2024,
  title={Snake's & The Golden Apple: Advanced Multi-Agent RL},
  author={[Your Name]},
  year={2024}
}
```